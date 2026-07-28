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


def _mmore_context_item(question: str, pages: list[tuple[str, int]]) -> dict:
    """One entry in the real `mmore colvision retrieve` output format."""
    return {
        "query": question,
        "context": [
            {
                "page_content": "…",
                "metadata": {
                    "pdf_name": pdf,
                    "pdf_path": f"data/pdfs/{pdf}",
                    "page_number": page,  # mmore stores page_num + 1
                    "rank": rank,
                },
            }
            for rank, (pdf, page) in enumerate(pages, start=1)
        ],
    }


def test_perfect_run_on_real_mmore_format_scores_one(tmp_path: Path) -> None:
    """A run that ranks the gold page first must score nDCG@1 == 1.0.

    Regression test for the page-numbering mismatch: query generation numbers
    pages from 0, mmore emits `page_num + 1`, so scoring the raw `source_page`
    graded a perfect run at 0.0.
    """
    from benchmark_colvision.queries.schema import QuerySet, SyntheticQuery
    from benchmark_colvision.runners.run_track_a import _scores_from_retrieval

    queries = [
        SyntheticQuery(
            query_id="q1",
            question="what is shown on the third page?",
            expected_answer="",
            source_pdf="a.pdf",
            source_page=2,  # 0-based -> mmore emits page_number=3
            language="en",
        )
    ]
    queryset = QuerySet(name="t", language="en", queries=queries)

    p = tmp_path / "seed_0.json"
    p.write_text(
        json.dumps(
            [_mmore_context_item(queries[0].question, [("a.pdf", 3), ("a.pdf", 7)])]
        )
    )

    scores = _scores_from_retrieval(p, queryset)
    assert scores.ndcg_at_1 == 1.0
    assert scores.recall_at_5 == 1.0
    assert scores.mrr == 1.0


def test_vidore_style_queryset_is_not_shifted(tmp_path: Path) -> None:
    """ViDoRe querysets already carry mmore numbering; they must not shift."""
    from benchmark_colvision.queries.schema import MMORE_PAGE_BASE, SyntheticQuery

    q = SyntheticQuery(
        query_id="q1", question="?", expected_answer="",
        source_pdf="doc_0001.pdf", source_page=4,
        page_base=MMORE_PAGE_BASE, language="en",
    )
    assert q.mmore_doc_id() == "doc_0001.pdf#page=4"
