"""Second-pass filter: ask the judge whether a synthetic query is
non-ambiguous and truly requires the source page.

The judge returns a single numeric confidence in [0, 1]; queries below
`threshold` are dropped. The judge prompt is intentionally narrow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from benchmark_colvision.queries.inverse_query_gen import LLMClient
from benchmark_colvision.queries.schema import SyntheticQuery


AMBIGUITY_PROMPT = """Tu es un évaluateur. Voici une question médicale et la page
source qui a servi à la générer.

Question : {question}

Page source (texte) :
---
{page_text}
---

Évalue, sur une échelle de 0 à 1, si cette question :
1. est suffisamment précise pour avoir une seule bonne page-source,
2. nécessite réellement de consulter CETTE page (et pas une autre).

Réponds uniquement par un nombre entre 0 et 1, sans texte additionnel.
Confiance :"""


_NUMBER = re.compile(r"(\d+(?:\.\d+)?)")


def parse_score(completion: str) -> float:
    """Extract the first number in [0, 1] from the completion; clamp out-of-range."""
    match = _NUMBER.search(completion)
    if not match:
        return 0.0
    try:
        v = float(match.group(1))
    except ValueError:
        return 0.0
    return max(0.0, min(1.0, v))


@dataclass
class FilterReport:
    n_in: int
    n_kept: int
    n_dropped: int
    threshold: float
    mean_score: float


def score_query(
    query: SyntheticQuery,
    page_text: str,
    llm: LLMClient,
) -> float:
    prompt = AMBIGUITY_PROMPT.format(question=query.question, page_text=page_text.strip()[:3000])
    completion = llm.complete(prompt)
    return parse_score(completion)


def filter_queries(
    queries: list[SyntheticQuery],
    page_text_for: dict[str, str],
    llm: LLMClient,
    threshold: float = 0.8,
) -> tuple[list[SyntheticQuery], FilterReport]:
    """Score each query and drop those below `threshold`."""
    kept: list[SyntheticQuery] = []
    scores: list[float] = []
    for q in queries:
        key = f"{q.source_pdf}#page={q.source_page}"
        page_text = page_text_for.get(key, "")
        score = score_query(q, page_text, llm)
        scores.append(score)
        q.judge_score = score
        q.accepted = score >= threshold
        if q.accepted:
            kept.append(q)
    report = FilterReport(
        n_in=len(queries),
        n_kept=len(kept),
        n_dropped=len(queries) - len(kept),
        threshold=threshold,
        mean_score=(sum(scores) / len(scores)) if scores else 0.0,
    )
    return kept, report
