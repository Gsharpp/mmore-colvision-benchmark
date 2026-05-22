"""Tests for evaluation.generation_metrics aggregation logic.

The RAGAS call path itself is not exercised here — that needs a real LLM judge
and is covered separately by integration tests on RCP.
"""

from __future__ import annotations

from benchmark_colvision.evaluation.generation_metrics import (
    GenerationMetricsResult,
    RagSample,
    aggregate_results,
    score_samples,
)


def test_score_samples_empty_returns_zero_examples():
    res = score_samples([], judge_model="meditron-70b")
    assert res.n_examples == 0
    assert res.judge_model == "meditron-70b"


def test_aggregate_results_weighted_mean():
    a = GenerationMetricsResult(faithfulness=0.8, n_examples=10, judge_model="m")
    b = GenerationMetricsResult(faithfulness=0.6, n_examples=30, judge_model="m")
    agg = aggregate_results([a, b])
    # (0.8 * 10 + 0.6 * 30) / 40 = 0.65
    assert agg.faithfulness == 0.65
    assert agg.n_examples == 40


def test_aggregate_results_ignores_none_components():
    a = GenerationMetricsResult(faithfulness=None, n_examples=5, judge_model="m")
    b = GenerationMetricsResult(faithfulness=0.9, n_examples=5, judge_model="m")
    agg = aggregate_results([a, b])
    assert agg.faithfulness == 0.9
    assert agg.n_examples == 10


def test_aggregate_results_no_inputs():
    agg = aggregate_results([])
    assert agg.n_examples == 0
    assert agg.faithfulness is None


def test_ragsample_holds_data():
    s = RagSample(question="q?", answer="a.", contexts=["c1", "c2"], reference="r")
    assert s.question == "q?"
    assert len(s.contexts) == 2
