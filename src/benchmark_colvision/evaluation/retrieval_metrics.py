"""Standard IR metrics for evaluating ColVision retrieval against page-level relevance.

Each query has a set of relevant document ids (ground truth) and a ranked list
of retrieved ids. Functions accept either per-query data or batches.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be >= 1")
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in relevant) / k


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return sum(1 for d in top if d in relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: set[str]) -> float:
    """AP: mean of precision@i evaluated at each rank where a relevant doc is hit."""
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant)


def dcg_at_k(retrieved: Sequence[str], relevance: dict[str, float], k: int) -> float:
    """Binary or graded DCG@k using the log2(i+1) discount."""
    if k <= 0:
        raise ValueError("k must be >= 1")
    total = 0.0
    for i, d in enumerate(retrieved[:k], start=1):
        rel = relevance.get(d, 0.0)
        if rel:
            total += rel / math.log2(i + 1)
    return total


def ndcg_at_k(retrieved: Sequence[str], relevance: dict[str, float], k: int) -> float:
    """nDCG@k = DCG@k / IDCG@k. relevance maps doc_id -> graded relevance (>=0)."""
    dcg = dcg_at_k(retrieved, relevance, k)
    ideal_order = sorted(relevance.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_order[:k], start=1):
        if rel:
            idcg += rel / math.log2(i + 1)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def macro_average(per_query: Iterable[float]) -> float:
    values = list(per_query)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def evaluate_batch(
    retrieved_per_query: list[Sequence[str]],
    relevant_per_query: list[set[str]],
    relevance_per_query: list[dict[str, float]] | None = None,
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Compute the standard suite of retrieval metrics, macro-averaged across queries.

    If `relevance_per_query` is None, binary relevance is derived from
    `relevant_per_query` (all relevant docs get weight 1).
    """
    if len(retrieved_per_query) != len(relevant_per_query):
        raise ValueError("retrieved and relevant lists must have the same length")
    if relevance_per_query is None:
        relevance_per_query = [{d: 1.0 for d in r} for r in relevant_per_query]
    elif len(relevance_per_query) != len(retrieved_per_query):
        raise ValueError("relevance_per_query length mismatch")

    out: dict[str, float] = {}
    for k in ks:
        out[f"precision@{k}"] = macro_average(
            precision_at_k(r, rel, k) for r, rel in zip(retrieved_per_query, relevant_per_query)
        )
        out[f"recall@{k}"] = macro_average(
            recall_at_k(r, rel, k) for r, rel in zip(retrieved_per_query, relevant_per_query)
        )
        out[f"ndcg@{k}"] = macro_average(
            ndcg_at_k(r, grel, k) for r, grel in zip(retrieved_per_query, relevance_per_query)
        )
    out["mrr"] = macro_average(
        reciprocal_rank(r, rel) for r, rel in zip(retrieved_per_query, relevant_per_query)
    )
    out["map"] = macro_average(
        average_precision(r, rel) for r, rel in zip(retrieved_per_query, relevant_per_query)
    )
    return out
