"""Unit tests for evaluation.performance_metrics."""

from __future__ import annotations

import time

from benchmark_colvision.evaluation.performance_metrics import (
    LatencySummary,
    percentile,
    timed,
)


def test_percentile_basic():
    assert percentile([10, 20, 30, 40, 50], 50) == 30
    assert percentile([10, 20, 30, 40, 50], 0) == 10
    assert percentile([10, 20, 30, 40, 50], 100) == 50


def test_percentile_empty_is_nan():
    import math

    assert math.isnan(percentile([], 50))


def test_timed_measures_duration():
    with timed("sleep") as t:
        time.sleep(0.02)
    assert t.label == "sleep"
    assert t.duration_s >= 0.02
    assert t.duration_s < 0.5


def test_latency_summary_from_seconds():
    summary = LatencySummary.from_seconds([0.010, 0.020, 0.030, 0.040, 0.050])
    assert summary.n == 5
    assert summary.min_ms == 10
    assert summary.max_ms == 50
    assert summary.mean_ms == 30
    assert summary.p50_ms == 30


def test_latency_summary_empty():
    import math

    summary = LatencySummary.from_seconds([])
    assert summary.n == 0
    assert math.isnan(summary.p50_ms)
