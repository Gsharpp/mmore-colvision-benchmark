"""Tests for the ViDoRe v2 corpus builder and graded-qrels scoring path."""

from __future__ import annotations

import json

import fitz
import pytest
from PIL import Image

from benchmark_colvision.corpus.corpus_manifest import CorpusManifest
from benchmark_colvision.corpus.vidore_v2 import (
    build_corpus_from_records,
    build_language_queryset,
    load_qrels,
)
from benchmark_colvision.queries.schema import QuerySet
from benchmark_colvision.runners.run_track_a import _load_qrels, _scores_from_retrieval


def _img(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (64, 48), color)


@pytest.fixture
def built(tmp_path):
    # Two documents; D1 has two pages (corpus-ids 10, 11), D2 one page (20).
    corpus = [
        {"corpus-id": 11, "doc-id": "D1", "image": _img((10, 20, 30))},
        {"corpus-id": 10, "doc-id": "D1", "image": _img((40, 50, 60))},
        {"corpus-id": 20, "doc-id": "D2", "image": _img((70, 80, 90))},
    ]
    queries = [
        {"query-id": 100, "query": "what does D1 page two show"},
        {"query-id": 101, "query": "what does D2 show"},
        {"query-id": 102, "query": "an unanswerable question"},
    ]
    qrels = [
        {"query-id": 100, "corpus-id": 11, "score": 2},
        {"query-id": 100, "corpus-id": 10, "score": 1},
        {"query-id": 100, "corpus-id": 12, "score": 0},  # below min_score → ignored
        {"query-id": 101, "corpus-id": 20, "score": 3},
        {"query-id": 102, "corpus-id": 999, "score": 1},  # page absent → dropped query
    ]
    result = build_corpus_from_records(
        corpus, queries, qrels, tmp_path, manifest_name="unit"
    )
    return tmp_path, result


def test_pdfs_reconstructed_in_corpus_id_order(built):
    tmp_path, result = built
    assert result.n_docs == 2
    assert result.n_pages == 3
    # doc-ids sorted → D1=doc_0000, D2=doc_0001.
    with fitz.open(tmp_path / "pdfs" / "doc_0000.pdf") as pdf:
        assert pdf.page_count == 2
    with fitz.open(tmp_path / "pdfs" / "doc_0001.pdf") as pdf:
        assert pdf.page_count == 1


def test_manifest_shape(built):
    tmp_path, result = built
    manifest = CorpusManifest.load(result.manifest_path)
    assert manifest.total_pages == 3
    assert [e.pdf_path for e in manifest.pdfs] == ["doc_0000.pdf", "doc_0001.pdf"]
    assert [e.source_id for e in manifest.pdfs] == ["D1", "D2"]
    assert all(e.source == "vidore" for e in manifest.pdfs)


def test_qrels_use_1based_page_ids_matching_mmore(built):
    _, result = built
    qrels = load_qrels(result.qrels_path)
    # corpus-id 10 is the lower id → page 1; corpus-id 11 → page 2 of doc_0000.
    assert qrels["vq-100"] == {"doc_0000.pdf#page=2": 2.0, "doc_0000.pdf#page=1": 1.0}
    assert qrels["vq-101"] == {"doc_0001.pdf#page=1": 3.0}
    # Unanswerable query (no in-corpus positive qrel) is dropped, not emitted.
    assert "vq-102" not in qrels
    assert result.n_queries == 2
    assert result.n_queries_dropped == 1
    assert result.n_qrels == 3


def test_queryset_fallback_fields_are_top_graded_page(built):
    _, result = built
    qs = QuerySet.load_jsonl(result.queries_path, name="unit", language="en")
    by_id = {q.query_id: q for q in qs.queries}
    # Highest-graded page for q100 is page 2 (score 2 > 1).
    assert by_id["vq-100"].source_pdf == "doc_0000.pdf"
    assert by_id["vq-100"].source_page == 2
    assert by_id["vq-101"].source_page == 1


def test_graded_scoring_end_to_end(built, tmp_path):
    """Feed a mmore-format retrieval JSON and check the graded qrels path."""
    _, result = built
    qs = QuerySet.load_jsonl(result.queries_path, name="unit", language="en")
    q_by_id = {q.query_id: q for q in qs.queries}

    def ctx(pdf_name: str, page: int) -> dict:
        return {"page_content": "", "metadata": {"pdf_name": pdf_name, "page_number": page}}

    # Perfect ranking for both queries → nDCG@k == 1.0 everywhere.
    retrieval = [
        {
            "query": q_by_id["vq-100"].question,
            "context": [ctx("doc_0000.pdf", 2), ctx("doc_0000.pdf", 1), ctx("distractor.pdf", 1)],
        },
        {
            "query": q_by_id["vq-101"].question,
            "context": [ctx("doc_0001.pdf", 1), ctx("distractor.pdf", 9)],
        },
    ]
    out = tmp_path / "retrieval.json"
    out.write_text(json.dumps(retrieval))

    qrels = _load_qrels(result.qrels_path)
    scores = _scores_from_retrieval(out, qs, qrels=qrels)
    assert scores.ndcg_at_1 == pytest.approx(1.0)
    assert scores.ndcg_at_5 == pytest.approx(1.0)
    assert scores.recall_at_5 == pytest.approx(1.0)
    assert scores.n_queries == 2


def test_graded_scoring_penalises_wrong_page(built, tmp_path):
    """An off-by-one page (page 1 instead of the graded page 2) must not match."""
    _, result = built
    qs = QuerySet.load_jsonl(result.queries_path, name="unit", language="en")
    q_by_id = {q.query_id: q for q in qs.queries}

    def ctx(pdf_name: str, page: int) -> dict:
        return {"page_content": "", "metadata": {"pdf_name": pdf_name, "page_number": page}}

    # q101's only relevant page is doc_0001.pdf#page=1; retrieve a neighbour page 2.
    retrieval = [
        {"query": q_by_id["vq-100"].question, "context": [ctx("doc_0000.pdf", 2)]},
        {"query": q_by_id["vq-101"].question, "context": [ctx("doc_0001.pdf", 2)]},
    ]
    out = tmp_path / "retrieval.json"
    out.write_text(json.dumps(retrieval))

    qrels = _load_qrels(result.qrels_path)
    scores = _scores_from_retrieval(out, qs, qrels=qrels)
    # q100 hits (1.0), q101 misses (0.0) → macro nDCG@1 == 0.5.
    assert scores.ndcg_at_1 == pytest.approx(0.5)


def test_language_queryset_aligns_with_english_corpus_pages(tmp_path):
    """A language-only build (no PDFs written) must resolve to the SAME doc-ids
    as `build_corpus_from_records` on a row-identical corpus config — this is
    what lets Track B re-retrieve on the Track A Milvus index unchanged."""
    corpus = [
        {"corpus-id": 11, "doc-id": "D1", "image": _img((10, 20, 30))},
        {"corpus-id": 10, "doc-id": "D1", "image": _img((40, 50, 60))},
        {"corpus-id": 20, "doc-id": "D2", "image": _img((70, 80, 90))},
    ]
    # Multilingual queries table: english block (query-id 100-101) + french block
    # (200-201), each carrying its own `language` field, same corpus-ids as `built`.
    queries = [
        {"query-id": 100, "query": "what does D1 page two show", "language": "english"},
        {"query-id": 101, "query": "what does D2 show", "language": "english"},
        {"query-id": 200, "query": "que montre la page deux de D1", "language": "french"},
        {"query-id": 201, "query": "que montre D2", "language": "french"},
    ]
    qrels = [
        {"query-id": 100, "corpus-id": 11, "score": 2},
        {"query-id": 100, "corpus-id": 10, "score": 1},
        {"query-id": 101, "corpus-id": 20, "score": 3},
        {"query-id": 200, "corpus-id": 11, "score": 2},
        {"query-id": 200, "corpus-id": 10, "score": 1},
        {"query-id": 201, "corpus-id": 20, "score": 3},
    ]
    fr = build_language_queryset(
        corpus, queries, qrels, tmp_path / "fr", language="french"
    )
    fr_qrels = load_qrels(fr.qrels_path)
    # Identical doc-ids to the English `build_corpus_from_records` fixture above:
    # doc-ids sorted → D1=doc_0000 (page 1=corpus-id 10, page 2=corpus-id 11).
    assert fr_qrels["vq-200"] == {"doc_0000.pdf#page=2": 2.0, "doc_0000.pdf#page=1": 1.0}
    assert fr_qrels["vq-201"] == {"doc_0001.pdf#page=1": 3.0}
    assert fr.n_queries == 2
    assert fr.n_qrels == 3
    assert fr.n_queries_dropped == 0
    # English-language rows are excluded from the french build.
    assert "vq-100" not in fr_qrels and "vq-101" not in fr_qrels


def test_language_queryset_drops_unanswerable_and_is_language_case_insensitive(tmp_path):
    corpus = [{"corpus-id": 1, "doc-id": "D1", "image": _img((1, 2, 3))}]
    queries = [
        {"query-id": 300, "query": "unanswerable", "language": "German"},
        {"query-id": 301, "query": "answerable", "language": "german"},
    ]
    qrels = [
        {"query-id": 300, "corpus-id": 999, "score": 2},  # page absent → dropped
        {"query-id": 301, "corpus-id": 1, "score": 2},
    ]
    de = build_language_queryset(corpus, queries, qrels, tmp_path / "de", language="german")
    assert de.n_queries == 1
    assert de.n_queries_dropped == 1
    de_qrels = load_qrels(de.qrels_path)
    assert de_qrels["vq-301"] == {"doc_0000.pdf#page=1": 2.0}
