"""Unit tests for evaluation.statistical_tests."""

from __future__ import annotations

import numpy as np

from benchmark_colvision.evaluation.statistical_tests import (
    bonferroni_correct,
    bootstrap_ci,
    holm_correct,
    wilcoxon_paired,
)


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(42)
    sample = rng.normal(loc=0.8, scale=0.05, size=200)
    ci = bootstrap_ci(sample, n_resamples=500, confidence=0.95, seed=0)
    assert ci.lower < ci.point_estimate < ci.upper
    assert abs(ci.point_estimate - 0.8) < 0.02


def test_bootstrap_ci_empty():
    ci = bootstrap_ci([], n_resamples=100)
    assert np.isnan(ci.point_estimate)
    assert np.isnan(ci.lower)
    assert np.isnan(ci.upper)


def test_wilcoxon_paired_detects_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.05, size=100)
    b = a - 0.05  # b strictly worse
    res = wilcoxon_paired(a, b)
    assert res.pvalue < 0.001
    assert res.n_pairs == 100


def test_wilcoxon_paired_shape_mismatch():
    import pytest

    with pytest.raises(ValueError):
        wilcoxon_paired([1.0, 2.0], [1.0])


def test_bonferroni_caps_at_one():
    assert bonferroni_correct([0.5, 0.5, 0.5]) == [1.0, 1.0, 1.0]
    assert bonferroni_correct([0.01, 0.01, 0.01]) == [0.03, 0.03, 0.03]


def test_holm_monotone_and_caps():
    adjusted = holm_correct([0.04, 0.01, 0.03])
    # monotone-non-decreasing when sorted by original p
    assert all(0 <= a <= 1 for a in adjusted)
    assert adjusted[1] <= adjusted[2] <= adjusted[0]
