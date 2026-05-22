"""Inverse query generation: ask an LLM (Meditron-70B) to author the queries
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


PROMPT_TEMPLATE = """Tu es un expert médical. Voici une page d'article médical.
Texte extrait :
---
{page_text}
---
{visual_hint}

Génère exactement {n} questions cliniques précises auxquelles on ne peut répondre QU'EN
consultant CETTE page (figures, tables, ou texte spécifique). Évite les questions
qui pourraient se répondre avec n'importe quelle page de littérature médicale.

Réponds en JSON strict, sans markdown, au format :
[{{"question": str, "expected_answer": str, "requires_visual": bool}}]
"""


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


def build_prompt(page: PageInput, n_queries: int = 2) -> str:
    hints = []
    if page.has_figures:
        hints.append("Cette page contient au moins une figure.")
    if page.has_tables:
        hints.append("Cette page contient au moins une table.")
    visual_hint = " ".join(hints) if hints else "Aucune indication visuelle disponible."
    return PROMPT_TEMPLATE.format(
        page_text=page.text.strip()[:3000],
        visual_hint=visual_hint,
        n=n_queries,
    )


_JSON_BLOCK = re.compile(r"\[.*\]", re.DOTALL)


def parse_completion(completion: str) -> list[dict]:
    """Tolerant JSON parser: extract the first JSON array in the completion."""
    match = _JSON_BLOCK.search(completion)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict) and "question" in d]


def generate_for_page(
    page: PageInput,
    llm: LLMClient,
    n_queries: int = 2,
    keep_visual_only: bool = True,
) -> list[SyntheticQuery]:
    """Produce up to `n_queries` synthetic queries for one page."""
    prompt = build_prompt(page, n_queries=n_queries)
    completion = llm.complete(prompt)
    items = parse_completion(completion)
    out: list[SyntheticQuery] = []
    for i, item in enumerate(items[:n_queries]):
        requires_visual = bool(item.get("requires_visual", page.has_figures or page.has_tables))
        if keep_visual_only and not requires_visual:
            continue
        out.append(
            SyntheticQuery(
                query_id=query_id_for(page.pdf_path, page.page_number, i),
                question=str(item["question"]).strip(),
                expected_answer=str(item.get("expected_answer", "")).strip(),
                requires_visual=requires_visual,
                source_pdf=page.pdf_path,
                source_page=page.page_number,
                language=page.language,
            )
        )
    return out


def generate_for_pages(
    pages: Iterable[PageInput],
    llm: LLMClient,
    n_per_page: int = 2,
    keep_visual_only: bool = True,
) -> list[SyntheticQuery]:
    """Generate queries for an iterable of pages, flattening the result."""
    out: list[SyntheticQuery] = []
    for page in pages:
        out.extend(generate_for_page(page, llm, n_per_page, keep_visual_only))
    return out
