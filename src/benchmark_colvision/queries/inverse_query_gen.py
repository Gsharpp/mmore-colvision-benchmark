"""Inverse query generation: ask an instruction-tuned LLM to author the queries
that a given page would uniquely answer.

The LLM is abstracted behind a `LLMClient` protocol so this module can be
exercised with a deterministic fake in unit tests and a real vLLM client in
production.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from benchmark_colvision.queries.schema import SyntheticQuery, query_id_for

_FEWSHOT_TEXT = """\
EXAMPLE (follow this output format exactly for the next page):
Medical article page (language: English):
---
Aspirin irreversibly inhibits cyclooxygenase-1 and COX-2, blocking thromboxane A2 synthesis \
and platelet aggregation. For acute ST-elevation myocardial infarction (STEMI), guidelines \
recommend a loading dose of 300 mg orally, followed by 75-100 mg daily. Dual antiplatelet \
therapy with a P2Y12 inhibitor reduces 30-day mortality by approximately 20%.
---
No visual hint available.

List exactly 2 clinical questions in English that can ONLY be answered by consulting THIS \
specific page. Each question must be precise and reference unique content from this page.

{{"queries": [{{"question": "What is the recommended aspirin loading dose for acute STEMI?", \
"expected_answer": "300 mg orally, followed by 75-100 mg daily", "requires_visual": false}}, \
{{"question": "By how much does dual antiplatelet therapy reduce 30-day mortality in STEMI?", \
"expected_answer": "Approximately 20%", "requires_visual": false}}]}}

END OF EXAMPLE — NOW PROCESS THE FOLLOWING PAGE:
"""

_FEWSHOT_MIXED = """\
EXAMPLE (follow this output format exactly for the next page):
Medical article page (language: English):
---
Figure 2 shows Kaplan-Meier survival curves for treatment A (n=142) vs placebo (n=138). \
At 36 months, survival was 78.4% vs 61.2% respectively (log-rank p=0.003). Table 1 reports \
baseline characteristics: mean age 65.2±8.1 years, 63% male, median follow-up 41 months.
---
This page contains at least one figure. This page contains at least one table.

List exactly 2 clinical questions in English that can ONLY be answered by consulting THIS \
specific page: 1 question requiring visual inspection of a figure or table \
(requires_visual: true), and 1 question answerable from the text (requires_visual: false).

{{"queries": [{{"question": \
"According to Figure 2, what is the 36-month survival rate for the treatment A group?", \
"expected_answer": "78.4%", "requires_visual": true}}, \
{{"question": "What is the mean age of patients enrolled in this study?", \
"expected_answer": "65.2 ± 8.1 years", "requires_visual": false}}]}}

END OF EXAMPLE — NOW PROCESS THE FOLLOWING PAGE:
"""

_FEWSHOT_VISUAL_ONLY = """\
EXAMPLE (follow this output format exactly for the next page):
Medical article page (language: English):
---
Figure 2 shows Kaplan-Meier survival curves for treatment A (n=142) vs placebo (n=138). \
At 36 months, survival was 78.4% vs 61.2% respectively (log-rank p=0.003). \
Figure 3 shows the distribution of adverse events by organ system as a bar chart.
---
This page contains at least one figure.

List exactly 2 clinical questions in English that can ONLY be answered by visually examining \
the figures on this page. Each question must require looking at the figure \
(set requires_visual to true for all).

{{"queries": [{{"question": \
"According to Figure 2, what is the 36-month survival rate for the treatment A group?", \
"expected_answer": "78.4%", "requires_visual": true}}, \
{{"question": "Which organ system shows the highest adverse event rate in Figure 3?", \
"expected_answer": "Gastrointestinal (tallest bar)", "requires_visual": true}}]}}

END OF EXAMPLE — NOW PROCESS THE FOLLOWING PAGE:
"""

# Per-mode instruction appended after the generic instruction line.
_MODE_INSTRUCTIONS: dict[str, str] = {
    "text": "",
    "visual": (
        " Each question must require visually examining a figure or table on this page."
        " Set requires_visual to true for all questions."
    ),
    "mixed": (
        " Generate 1 question requiring visual inspection of a figure or table"
        " (requires_visual: true) and 1 question answerable from the text"
        " (requires_visual: false)."
    ),
}

PROMPT_TEMPLATE = """{fewshot_prefix}Medical article page (language: {doc_lang}):
---
{page_text}
---
{visual_hint}

List exactly {n} clinical questions in {query_lang} that can ONLY be answered by consulting
THIS specific page.{mode_instruction}

{{"queries": ["""

# JSON schema for vLLM guided decoding (xgrammar backend).
# Root must be an object — xgrammar 0.2.x rejects array-root schemas.
GUIDED_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "expected_answer": {"type": "string"},
                    "requires_visual": {"type": "boolean"},
                },
                "required": ["question", "expected_answer"],
            },
        }
    },
    "required": ["queries"],
}

# Map ISO codes to the natural-language name used in the prompt instruction.
_LANG_NAME = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "zh": "Chinese",
    "ar": "Arabic",
}


class LLMClient(Protocol):
    def complete(self, prompt: str, **kwargs) -> str: ...


@dataclass
class PageInput:
    pdf_path: str  # relative path inside corpus
    page_number: int  # 0-based
    language: str
    text: str
    has_figures: bool = False
    has_tables: bool = False


def build_prompt(
    page: PageInput,
    n_queries: int = 2,
    few_shot: bool = False,
    query_mode: str = "mixed",
    query_language: str | None = None,
) -> str:
    hints = []
    if page.has_figures:
        hints.append("This page contains at least one figure.")
    if page.has_tables:
        hints.append("This page contains at least one table.")
    visual_hint = " ".join(hints) if hints else "No visual hint available."
    is_visual = page.has_figures or page.has_tables

    fewshot_prefix = ""
    if few_shot:
        if query_mode == "visual":
            fewshot_prefix = _FEWSHOT_VISUAL_ONLY
        elif query_mode == "mixed" and is_visual:
            fewshot_prefix = _FEWSHOT_MIXED
        else:
            fewshot_prefix = _FEWSHOT_TEXT

    # mixed mode only applies the visual instruction when the page has visual elements
    if query_mode == "mixed":
        mode_instruction = _MODE_INSTRUCTIONS["mixed"] if is_visual else ""
    else:
        mode_instruction = _MODE_INSTRUCTIONS.get(query_mode, "")

    # The document is described in its own language, but the questions can be
    # asked in a different language (cross-lingual retrieval: e.g. English queries
    # against a French/Chinese corpus). `query_language` overrides the question
    # language; when None it defaults to the document language.
    return PROMPT_TEMPLATE.format(
        fewshot_prefix=fewshot_prefix,
        page_text=page.text.strip()[:1500],
        visual_hint=visual_hint,
        n=n_queries,
        doc_lang=_LANG_NAME.get(page.language, "English"),
        query_lang=_LANG_NAME.get(query_language or page.language, "English"),
        mode_instruction=mode_instruction,
    )


_JSON_BLOCK = re.compile(r"\[.*\]", re.DOTALL)


_OBJ_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_completion(completion: str) -> list[dict]:
    """Tolerant JSON parser: extract query items from the completion.

    Handles three cases:
    - Wrapped object: {"queries": [...]} — produced by guided JSON (xgrammar).
    - Prompt ends with '{"queries": [', model returns the continuation.
    - Legacy flat array [...] for backwards compatibility.
    """
    text = completion.strip()

    def _extract(data: object) -> list[dict]:
        if isinstance(data, dict) and "queries" in data:
            items = data["queries"]
        elif isinstance(data, list):
            items = data
        else:
            return []
        return [d for d in items if isinstance(d, dict) and "question" in d]

    # Case 1: full object or array present in text
    for pattern in (_OBJ_BLOCK, _JSON_BLOCK):
        match = pattern.search(text)
        if match:
            try:
                result = _extract(json.loads(match.group(0)))
                if result:
                    return result
            except json.JSONDecodeError:
                pass

    # Case 2: prompt ended with '{"queries": [', response is the continuation
    # (items + closing brackets)
    for prefix in ('{"queries": [', '{"queries":['):
        for suffix in ("]}}", "]}", ""):
            candidate = prefix + text + suffix
            try:
                result = _extract(json.loads(candidate))
                if result:
                    return result
            except json.JSONDecodeError:
                pass

    # Case 3: legacy flat-array continuation (prompt ended with '[')
    if text.startswith("{") or text.startswith('"'):
        for candidate in (f"[{text}", f"[{text}]"):
            try:
                data = json.loads(candidate)
                if isinstance(data, list):
                    result = _extract(data)
                    if result:
                        return result
            except json.JSONDecodeError:
                pass

    return []


def generate_for_page(
    page: PageInput,
    llm: LLMClient,
    n_queries: int = 2,
    query_mode: str = "mixed",
    few_shot: bool = False,
    use_guided_json: bool = False,
    query_language: str | None = None,
) -> list[SyntheticQuery]:
    """Produce up to `n_queries` synthetic queries for one page.

    query_mode controls page selection and output filtering:
      "text"   — text questions only; visual pages also accepted.
      "visual" — skip pages without figures/tables; keep only requires_visual=True.
      "mixed"  — 1 visual + 1 text on visual pages; 2 text on others.

    query_language, when set, overrides the language the questions are written in
    (cross-lingual retrieval); the emitted query carries it as its language tag.
    """
    is_visual = page.has_figures or page.has_tables
    if query_mode == "visual" and not is_visual:
        return []

    prompt = build_prompt(page, n_queries=n_queries, few_shot=few_shot,
                          query_mode=query_mode, query_language=query_language)
    extra: dict = {}
    if use_guided_json:
        extra["guided_json"] = GUIDED_JSON_SCHEMA
    completion = llm.complete(prompt, **extra)
    items = parse_completion(completion)
    out: list[SyntheticQuery] = []
    for i, item in enumerate(items[:n_queries]):
        requires_visual = bool(item.get("requires_visual", is_visual))
        # Mode-specific output filter
        if query_mode == "text" and requires_visual:
            continue
        if query_mode == "visual":
            requires_visual = True  # override: visual pages, visual prompt → always visual
        out.append(
            SyntheticQuery(
                query_id=query_id_for(page.pdf_path, page.page_number, i),
                question=str(item["question"]).strip(),
                expected_answer=str(item.get("expected_answer", "")).strip(),
                requires_visual=requires_visual,
                source_pdf=page.pdf_path,
                source_page=page.page_number,
                language=query_language or page.language,
            )
        )
    return out


def generate_for_pages(
    pages: Iterable[PageInput],
    llm: LLMClient,
    n_per_page: int = 2,
    query_mode: str = "mixed",
    few_shot: bool = False,
    use_guided_json: bool = False,
    query_language: str | None = None,
) -> list[SyntheticQuery]:
    """Generate queries for an iterable of pages, flattening the result."""
    out: list[SyntheticQuery] = []
    for page in pages:
        try:
            out.extend(generate_for_page(page, llm, n_per_page, query_mode,
                                         few_shot=few_shot, use_guided_json=use_guided_json,
                                         query_language=query_language))
        except Exception as exc:  # noqa: BLE001
            print(f"[gen] skip {page.pdf_path}#{page.page_number}: {exc}", flush=True)
    return out
