"""Bootstrap confidence intervals, paired tests and multiple-comparison correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class BootstrapCI:
    point_estimate: float
    lower: float
    upper: float
    confidence: float


def bootstrap_ci(
    values: list[float] | np.ndarray,
    statistic=np.mean,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI for an arbitrary statistic."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return BootstrapCI(float("nan"), float("nan"), float("nan"), confidence)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples)
    n = arr.size
    for i in range(n_resamples):
        sample = arr[rng.integers(0, n, size=n)]
        boot[i] = statistic(sample)
    alpha = 1 - confidence
    lower = float(np.quantile(boot, alpha / 2))
    upper = float(np.quantile(boot, 1 - alpha / 2))
    return BootstrapCI(
        point_estimate=float(statistic(arr)),
        lower=lower,
        upper=upper,
        confidence=confidence,
    )


@dataclass
class PairedTestResult:
    statistic: float
    pvalue: float
    n_pairs: int
    method: str


def wilcoxon_paired(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> PairedTestResult:
    """Wilcoxon signed-rank test on per-query paired scores."""
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Paired arrays must have the same shape: {arr_a.shape} vs {arr_b.shape}")
    res = stats.wilcoxon(arr_a, arr_b, zero_method="wilcox")
    return PairedTestResult(
        statistic=float(res.statistic),
        pvalue=float(res.pvalue),
        n_pairs=int(arr_a.size),
        method="wilcoxon-signed-rank",
    )


def bonferroni_correct(pvalues: list[float]) -> list[float]:
    """Bonferroni correction; capped at 1.0."""
    n = len(pvalues)
    return [min(p * n, 1.0) for p in pvalues]


def holm_correct(pvalues: list[float]) -> list[float]:
    """Holm step-down correction; less conservative than Bonferroni for large n."""
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((n - rank) * pvalues[idx], 1.0)
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted
