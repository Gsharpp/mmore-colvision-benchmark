"""Native **mmore** text-retrieval baseline — hybrid dense+SPLADE.

This is the *actual* "mmore without the VLM" path: it drives mmore's own text
``Indexer`` (dense + sparse SPLADE, fused at search time) and ``Retriever`` over
the exact same corpora, queries and page-level relevance judgements as the
ColVision cells, scored with the identical metrics.

It complements two existing points of comparison:

* the ColVision encoders (``runners/run_track_a.py`` / ``run_track_b``), and
* the *direct dense* text baseline (``runners/text_baseline.py``, bge-m3 cosine).

Unlike the direct dense baseline, this runner exercises mmore's hybrid Milvus
index (``dense_embedding`` + ``sparse_embedding``) and its weighted-rank fusion,
so it measures the value of the full mmore text pipeline, not just a dense
encoder. The SPLADE sparse model is what forces a separate environment: its
``pymilvus.model`` implementation calls ``tokenizer.batch_encode_plus``, removed
in transformers 5.x. We therefore run this in a dedicated **tf4** venv
(``bcv-venv-mmoretext``, ``mmore[rag]`` *without* the ``colvision`` extra), exactly
as vLLM lives in its own ``bcv-venv-vllm``. ColVision (tf 5.3.0) is never imported.

Granularity alignment is identical to ``text_baseline.py``: one indexed unit per
page, id ``"<pdf>#page=<N>"`` matching the relevance ids. The id is stored both as
the Milvus primary key (truncated to the 128-char VARCHAR limit) and, in full, in
the dynamic ``file_path`` field so retrieval can always recover it.

Run (cluster, bcv-venv-mmoretext + PYTHONPATH=src):

    PYTHONPATH=/…/bcv-dev/src \
    /…/bcv-venv-mmoretext/bin/python -m benchmark_colvision.runners.mmore_text_baseline \
        --manifest /…/manifest-tb-en.json --corpus-root /…/pdfs-tb-en \
        --queryset data/track_b/queries/en.jsonl \
        --workdir data/mmore_text/B/en --record-out results/mmore_text/B/en/seed_0.json \
        --track B --language en
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmark_colvision.corpus.corpus_manifest import CorpusManifest
from benchmark_colvision.corpus.ocr_pages import pages_from_ocr
from benchmark_colvision.corpus.vidore_v2 import load_qrels
from benchmark_colvision.evaluation.retrieval_metrics import evaluate_batch
from benchmark_colvision.queries.pdf_pages import iter_manifest_pages
from benchmark_colvision.queries.schema import QuerySet
from benchmark_colvision.results.schema import (
    BenchmarkRecord,
    CellId,
    PerformanceScores,
    RetrievalScores,
)

# Recorded model id so this sits beside the ColVision + dense-baseline cells.
BASELINE_MODEL_ID = "mmore_text_hybrid"
_MILVUS_ID_MAX = 128


def collect_pages(
    manifest_path: Path,
    corpus_root: Path,
    *,
    min_text_chars: int = 80,
    page_base: int = 0,
) -> tuple[list[str], list[str]]:
    """Return (doc_ids, texts) for every page, ids as ``"<pdf>#page=<N>"``.

    `page_base=1` matches mmore colvision's output convention (needed for
    ViDoRe's graded qrels sidecar); the historical PMC/HAL baselines keep the
    0-based default, self-consistent with their own query-gen page numbering.
    """
    manifest = CorpusManifest.load(manifest_path)
    doc_ids: list[str] = []
    texts: list[str] = []
    for page in iter_manifest_pages(
        manifest, corpus_root, min_text_chars=min_text_chars, page_base=page_base
    ):
        doc_ids.append(f"{page.pdf_path}#page={page.page_number}")
        texts.append(page.text or "")
    return doc_ids, texts


def build_samples(doc_ids: list[str], texts: list[str]):
    """One mmore ``MultimodalSample`` per page; identity in pk id + dynamic file_path."""
    from mmore.type import DocumentMetadata, MultimodalSample

    samples = []
    for doc_id, text in zip(doc_ids, texts):
        samples.append(
            MultimodalSample(
                text=text,
                modalities=[],
                # file_path is emitted by to_dict() and stored as a *dynamic*
                # field (no VARCHAR limit) — the reliable place for the full id.
                metadata=DocumentMetadata(file_path=doc_id),
                id=doc_id[:_MILVUS_ID_MAX],
                document_id=doc_id[:_MILVUS_ID_MAX],
            )
        )
    return samples


def _hit_id(hit) -> str:
    """Best-effort extraction of the page id from one mmore retrieval hit."""
    if isinstance(hit, str):
        return hit
    if not isinstance(hit, dict):
        return str(getattr(hit, "file_path", "") or getattr(hit, "id", ""))
    # Prefer the full, untruncated id stored in file_path; fall back to pk id.
    for key in ("file_path", "id"):
        if hit.get(key):
            return str(hit[key])
    entity = hit.get("entity")
    if isinstance(entity, dict):
        for key in ("file_path", "id"):
            if entity.get(key):
                return str(entity[key])
    return ""


def run_cell(args: argparse.Namespace) -> BenchmarkRecord:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    db_path = str((workdir / "mmore_text.db").resolve())
    collection = args.collection

    qrels = load_qrels(args.qrels) if args.qrels else None
    if args.ocr_results:
        # ViDoRe: page images carry no text layer; use mmore's own Marker+Surya
        # OCR output instead of PyMuPDF extraction (see corpus.ocr_pages).
        doc_ids, texts = pages_from_ocr(args.ocr_results)
        print(f"[mmore-text] {len(doc_ids)} OCR'd page units from {args.ocr_results}", flush=True)
    else:
        page_base = 1 if qrels is not None else 0
        doc_ids, texts = collect_pages(
            Path(args.manifest), Path(args.corpus_root),
            min_text_chars=args.min_text_chars, page_base=page_base,
        )
        print(f"[mmore-text] {len(doc_ids)} page units from {args.manifest}", flush=True)
    over = [d for d in doc_ids if len(d) > _MILVUS_ID_MAX]
    if over:
        print(f"[mmore-text][warn] {len(over)} ids >128 chars; pk truncated, "
              f"file_path keeps the full id (e.g. {over[0]!r})", flush=True)
    samples = build_samples(doc_ids, texts)

    from mmore.index.indexer import DBConfig, Indexer, IndexerConfig
    from mmore.rag.model.dense.base import DenseModelConfig
    from mmore.rag.model.sparse.base import SparseModelConfig
    from mmore.rag.retriever import Retriever, RetrieverConfig

    idx_cfg = IndexerConfig(
        dense_model=DenseModelConfig(model_name=args.dense_model, is_multimodal=False),
        sparse_model=SparseModelConfig(model_name=args.sparse_model, is_multimodal=False),
        db=DBConfig(uri=db_path, name=collection),
    )
    t0 = time.monotonic()
    indexer = Indexer.from_config(idx_cfg)
    n_indexed = indexer.index_documents(
        samples, collection_name=collection, batch_size=args.batch_size
    )
    index_s = time.monotonic() - t0
    print(f"[mmore-text] indexed {n_indexed} pages (hybrid dense+SPLADE) -> {db_path}",
          flush=True)

    queryset = QuerySet.load_jsonl(Path(args.queryset), name=args.language, language="en")
    questions = [q.question for q in queryset.queries]
    if qrels is not None:
        # Graded, multi-relevant relevance (ViDoRe qrels), keyed by query_id.
        relevance = [qrels.get(q.query_id, {}) for q in queryset.queries]
        relevant = [set(r) for r in relevance]
    else:
        relevance = None
        relevant = [{f"{q.source_pdf}#page={q.source_page}"} for q in queryset.queries]

    ret_cfg = RetrieverConfig(
        db=DBConfig(uri=db_path, name=collection),
        hybrid_search_weight=args.hybrid_weight,  # 0.5 = mmore default dense/sparse mix
        k=args.top_k,
        collection_name=collection,
        reranker_model_name=None,  # isolate pure dense+SPLADE retrieval
    )
    retriever = Retriever.from_config(ret_cfg)

    t1 = time.monotonic()
    ranked: list[list[str]] = []
    for q in questions:
        hits = retriever.retrieve(
            q,
            collection_name=collection,
            k=args.top_k,
            output_fields=["id", "file_path"],
            search_type="hybrid",
        )
        ranked.append([_hit_id(h) for h in hits])
    retrieve_s = time.monotonic() - t1

    s = evaluate_batch(ranked, relevant, relevance_per_query=relevance, ks=(1, 5, 10))
    retrieval_scores = RetrievalScores(
        ndcg_at_1=s.get("ndcg@1"), ndcg_at_5=s.get("ndcg@5"), ndcg_at_10=s.get("ndcg@10"),
        recall_at_1=s.get("recall@1"), recall_at_5=s.get("recall@5"), recall_at_10=s.get("recall@10"),
        precision_at_1=s.get("precision@1"), precision_at_5=s.get("precision@5"),
        precision_at_10=s.get("precision@10"), mrr=s.get("mrr"), map=s.get("map"),
        n_queries=len(questions),
    )

    (workdir / "retrieval.json").write_text(
        json.dumps([{"query": q, "ranked": r} for q, r in zip(questions, ranked)], indent=2)
    )

    cell_id = CellId(
        track=args.track,
        model_id=BASELINE_MODEL_ID,
        palier_id=args.language if args.track == "A" else None,
        language=args.language if args.track == "B" else None,
        seed=args.seed,
    )
    record = BenchmarkRecord(
        cell=cell_id,
        mmore_commit=args.mmore_commit,
        benchmark_version=args.benchmark_version,
        corpus_manifest_sha256="",
        queries_sha256=queryset.sha256(),
        retrieval=retrieval_scores,
        performance=PerformanceScores(
            process_duration_s=None,
            index_duration_s=index_s,
            retrieve_duration_total_s=retrieve_s,
            throughput_pages_per_s=(n_indexed / index_s) if index_s else None,
        ),
        notes=(
            f"native mmore text pipeline, hybrid dense+SPLADE "
            f"(dense={args.dense_model}, sparse={args.sparse_model}, "
            f"hybrid_weight={args.hybrid_weight}, reranker=off); page-level units"
            + (" (OCR'd via mmore process, graded ViDoRe qrels)" if qrels is not None else "")
        ),
    )
    record_out = Path(args.record_out)
    record_out.parent.mkdir(parents=True, exist_ok=True)
    record_out.write_text(record.to_json())
    print(f"[mmore-text] nDCG@5={retrieval_scores.ndcg_at_5} -> {record_out}", flush=True)
    return record


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Native mmore hybrid (dense+SPLADE) text baseline.")
    p.add_argument("--manifest", default=None, help="Required unless --ocr-results is given.")
    p.add_argument("--corpus-root", default=None, help="Required unless --ocr-results is given.")
    p.add_argument(
        "--ocr-results", default=None,
        help="mmore `process` merged_results.jsonl (Marker+Surya OCR); use for "
        "image-only corpora (ViDoRe) instead of --manifest/--corpus-root.",
    )
    p.add_argument("--queryset", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--record-out", required=True)
    p.add_argument("--track", required=True, choices=["A", "B"])
    p.add_argument("--language", default="en")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--collection", default="bcv_mmore_text")
    p.add_argument("--dense-model", default="BAAI/bge-m3")
    p.add_argument("--sparse-model", default="splade")
    p.add_argument("--hybrid-weight", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--mmore-commit", default="")
    p.add_argument("--benchmark-version", default="0.1.0")
    p.add_argument(
        "--qrels", default=None,
        help="Graded qrels sidecar (ViDoRe); enables multi-relevant nDCG + 1-based page ids.",
    )
    p.add_argument(
        "--min-text-chars", type=int, default=80,
        help="Drop pages with less extracted text than this (0 for OCR'd figure-rich corpora).",
    )
    args = p.parse_args(argv)
    if not args.ocr_results and not (args.manifest and args.corpus_root):
        p.error("either --ocr-results or both --manifest/--corpus-root are required")
    run_cell(args)


if __name__ == "__main__":
    main()
