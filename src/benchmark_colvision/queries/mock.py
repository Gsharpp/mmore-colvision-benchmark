"""Template-based fallback for synthetic query generation (no LLM).

Used when no LLM judge is available (self-hosted vLLM unavailable *and* no hosted
API). Each query is grounded on a single source page, so the retrieval ground
truth is exact, and only visually-dense pages are kept by default. Questions are
built from the most distinctive terms of the page (not a verbatim sentence) so
the task is not a trivial exact-match — but the phrasing is templated rather than
LLM-authored.

Questions are phrased in **English** to match the (English) corpus that ColVision
models encode, and the wording is steered toward the page's visual payload
(figures / tables) so the retrieval task exercises the multimodal pathway rather
than plain text matching.

Results from this path are PRELIMINARY: real inverse query generation with an
LLM judge (e.g. Meditron-70B, see ``inverse_query_gen``) is deferred to future
work. Performance metrics (throughput, latency, GPU memory) are unaffected by
query phrasing; only retrieval-quality numbers carry this caveat.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from benchmark_colvision.queries.inverse_query_gen import PageInput
from benchmark_colvision.queries.schema import SyntheticQuery, query_id_for

# Very small bilingual stop-list — enough to keep templated questions readable
# without pulling in an NLP dependency.
_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "des", "les", "une", "dans", "pour", "avec", "que", "qui", "sur", "par",
    "est", "sont", "aux", "ces", "cette", "leur", "nous", "vous", "plus",
    "patients", "patient", "study", "results", "method", "methods", "using",
    "figure", "table", "fig", "et", "la", "le", "of", "in", "to", "a",
}
_TOKEN = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\-]{4,}")


def _distinctive_terms(text: str, k: int) -> list[str]:
    """Return up to k distinctive content terms, deterministically.

    Heuristic: alphabetic tokens of length >= 5, not stop-words, ranked by
    length then first occurrence (longer, rarer-looking words first), keeping
    the original surface form of the first occurrence.
    """
    seen: dict[str, str] = {}
    for m in _TOKEN.finditer(text):
        surface = m.group(0)
        key = surface.lower()
        if key in _STOP or key in seen:
            continue
        seen[key] = surface
    terms = list(seen.values())
    terms.sort(key=lambda w: len(w), reverse=True)
    return terms[:k]


def _visual_question(joined: str, *, has_figures: bool, has_tables: bool) -> str:
    """Phrase an English question steered toward the page's visual payload.

    The wording references the figure/table actually present on the page so the
    retrieval task leans on the multimodal pathway rather than plain text. When
    the page has neither, fall back to a content-grounded phrasing.
    """
    if has_figures and has_tables:
        return f"Which page shows a figure and a table reporting {joined}?"
    if has_figures:
        return f"Which page shows a figure illustrating {joined}?"
    if has_tables:
        return f"Which page presents a table with data on {joined}?"
    return f"Which page jointly documents {joined}?"


def generate_mock_for_page(
    page: PageInput,
    n_queries: int = 2,
    keep_visual_only: bool = True,
    terms_per_query: int = 3,
) -> list[SyntheticQuery]:
    """Produce up to ``n_queries`` templated queries grounded on one page."""
    if keep_visual_only and not (page.has_figures or page.has_tables):
        return []
    terms = _distinctive_terms(page.text, n_queries * terms_per_query)
    out: list[SyntheticQuery] = []
    for i in range(n_queries):
        chunk = terms[i * terms_per_query : (i + 1) * terms_per_query]
        if len(chunk) < 2:  # not enough signal on this page for query #i
            break
        joined = ", ".join(chunk)
        out.append(
            SyntheticQuery(
                query_id=query_id_for(page.pdf_path, page.page_number, i),
                question=_visual_question(
                    joined,
                    has_figures=page.has_figures,
                    has_tables=page.has_tables,
                ),
                expected_answer=joined,
                requires_visual=page.has_figures or page.has_tables,
                source_pdf=page.pdf_path,
                source_page=page.page_number,
                language=page.language,
                judge_score=None,
                accepted=True,
            )
        )
    return out


def generate_mock_for_pages(
    pages: Iterable[PageInput],
    n_per_page: int = 2,
    keep_visual_only: bool = True,
    terms_per_query: int = 3,
) -> list[SyntheticQuery]:
    """Generate templated queries for an iterable of pages, flattening results."""
    out: list[SyntheticQuery] = []
    for page in pages:
        out.extend(
            generate_mock_for_page(
                page,
                n_queries=n_per_page,
                keep_visual_only=keep_visual_only,
                terms_per_query=terms_per_query,
            )
        )
    return out
