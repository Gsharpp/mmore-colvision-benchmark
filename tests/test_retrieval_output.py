"""Tests for the mmore retrieve output parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark_colvision.runners.retrieval_output import align_to_queries, load_retrieval


def test_load_retrieval_list_form(tmp_path: Path) -> None:
    data = [
        {"query_id": "q1", "results": [{"document_id": "d1", "score": 0.9}, {"document_id": "d2", "score": 0.5}]},
        {"query_id": "q2", "results": [{"document_id": "d3", "score": 0.7}]},
    ]
    p = tmp_path / "out.json"
    p.write_text(json.dumps(data))
    out = load_retrieval(p)
    assert out == {"q1": ["d1", "d2"], "q2": ["d3"]}


def test_load_retrieval_dict_form(tmp_path: Path) -> None:
    data = {"q1": [{"doc_id": "d1"}, {"doc_id": "d2"}]}
    p = tmp_path / "out.json"
    p.write_text(json.dumps(data))
    out = load_retrieval(p)
    assert out == {"q1": ["d1", "d2"]}


def test_load_retrieval_invalid_top_level(tmp_path: Path) -> None:
    p = tmp_path / "out.json"
    p.write_text(json.dumps(123))
    with pytest.raises(ValueError):
        load_retrieval(p)


def test_align_to_queries_by_id() -> None:
    retrieval = {"q2": ["d3"], "q1": ["d1", "d2"]}
    out = align_to_queries(retrieval, query_ids_in_order=["q1", "q2"])
    assert out == [["d1", "d2"], ["d3"]]


def test_align_to_queries_missing_returns_empty() -> None:
    retrieval = {"q1": ["d1"]}
    out = align_to_queries(retrieval, query_ids_in_order=["q1", "q3"])
    assert out == [["d1"], []]


def test_align_to_queries_fallback_to_questions() -> None:
    # mmore CLI sometimes echoes the question text as the key
    retrieval = {"What is X?": ["d1"], "Why Y?": ["d2"]}
    out = align_to_queries(
        retrieval,
        query_ids_in_order=["q1", "q2"],
        fallback_questions=["What is X?", "Why Y?"],
    )
    assert out == [["d1"], ["d2"]]
