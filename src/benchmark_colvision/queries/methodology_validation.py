"""Validate the auto-generation pipeline against an externally annotated subset.

Two correlations are computed between the per-query nDCG@k yielded by the
benchmark on (a) the auto-generated queries and (b) the human-annotated
queries: Pearson and Spearman. The benchmark Phase 4 gate is Pearson > 0.85.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class ValidationReport:
    subset: str
    pearson: float
    spearman: float
    n_queries: int
    passes_threshold: bool
    threshold: float


def correlate(
    auto_scores: list[float],
    human_scores: list[float],
    subset_name: str,
    threshold: float = 0.85,
) -> ValidationReport:
    """Pearson and Spearman between two per-query score lists of equal length."""
    if len(auto_scores) != len(human_scores):
        raise ValueError(
            f"auto and human score lists differ in length: {len(auto_scores)} vs {len(human_scores)}"
        )
    if len(auto_scores) < 3:
        raise ValueError("Need at least 3 paired scores to compute correlation")
    a = np.asarray(auto_scores, dtype=float)
    h = np.asarray(human_scores, dtype=float)
    # If either side has zero variance, Pearson is undefined.
    pearson_r = float(stats.pearsonr(a, h).statistic) if a.std() > 0 and h.std() > 0 else float("nan")
    spearman_r = float(stats.spearmanr(a, h).statistic) if a.std() > 0 and h.std() > 0 else float("nan")
    return ValidationReport(
        subset=subset_name,
        pearson=pearson_r,
        spearman=spearman_r,
        n_queries=len(auto_scores),
        passes_threshold=not np.isnan(pearson_r) and pearson_r >= threshold,
        threshold=threshold,
    )
