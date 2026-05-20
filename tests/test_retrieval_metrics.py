"""Unit tests for evaluation.retrieval_metrics.

Reference values computed by hand on tiny inputs.
"""

from __future__ import annotations

import math

import pytest

from benchmark_colvision.evaluation.retrieval_metrics import (
    average_precision,
    dcg_at_k,
    evaluate_batch,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, k=2) == 0.5
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, k=3) == pytest.approx(2 / 3)
    assert precision_at_k(["x"], set(), k=1) == 0.0


def test_recall_at_k():
    assert recall_at_k(["a", "b"], {"a", "c", "d"}, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(["a", "b", "c", "d"], {"a", "c"}, k=4) == 1.0
    assert recall_at_k([], {"a"}, k=10) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a", "b"], {"a"}) == 0.5
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_average_precision():
    # Relevant at positions 1 and 3 → AP = (1/1 + 2/3) / 2 = 5/6
    ap = average_precision(["a", "x", "b"], {"a", "b"})
    assert ap == pytest.approx(5 / 6)


def test_dcg_and_ndcg_binary():
    # Single relevant at position 2: DCG = 1 / log2(3)
    dcg = dcg_at_k(["x", "a", "y"], {"a": 1.0}, k=3)
    assert dcg == pytest.approx(1 / math.log2(3))
    # Ideal: relevant at position 1 → IDCG = 1 / log2(2) = 1
    ndcg = ndcg_at_k(["x", "a", "y"], {"a": 1.0}, k=3)
    assert ndcg == pytest.approx(1 / math.log2(3))


def test_ndcg_perfect_ranking_is_one():
    rels = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert ndcg_at_k(["a", "b", "c"], rels, k=3) == pytest.approx(1.0)


def test_ndcg_no_relevant_is_zero():
    assert ndcg_at_k(["a", "b"], {}, k=2) == 0.0


def test_evaluate_batch_keys_and_values():
    retrieved = [["a", "b", "c"], ["x", "y", "z"]]
    relevant = [{"a"}, {"y"}]
    out = evaluate_batch(retrieved, relevant, ks=(1, 3))
    assert set(out.keys()) == {
        "precision@1", "precision@3",
        "recall@1", "recall@3",
        "ndcg@1", "ndcg@3",
        "mrr", "map",
    }
    # mrr: 1.0 (q0) and 0.5 (q1) → 0.75
    assert out["mrr"] == pytest.approx(0.75)
    # recall@3: 1.0 and 1.0 → 1.0
    assert out["recall@3"] == pytest.approx(1.0)


def test_evaluate_batch_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate_batch([["a"]], [{"a"}, {"b"}])


def test_precision_at_k_invalid():
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, k=0)
