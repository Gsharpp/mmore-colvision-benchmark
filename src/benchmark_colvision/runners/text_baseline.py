"""Text-retrieval baseline (non-ColVision) for Track A and Track B.

Answers the core ViDoRe/ColPali question — *does visual retrieval beat a
standard text RAG pipeline?* — with a **dense multilingual text retriever**
(`BAAI/bge-m3`) run on the exact same corpora and queries as the ColVision
cells, scored with the identical metrics.

Granularity alignment (the hard part) is solved by retrieving over **one unit
per page**: we reuse `queries.pdf_pages.iter_manifest_pages` — the *same* page
enumerator the query generator uses — so each indexed unit's id is
`"<pdf>#page=<N>"`, exactly matching the relevance ids
(`run_track_a.py`: `f"{q.source_pdf}#page={q.source_page}"`).

Why a direct dense retriever rather than `mmore index`/`mmore retrieve`?
mmore's text Indexer mandates a SPLADE *sparse* model whose pymilvus
implementation calls `tokenizer.batch_encode_plus`, removed in transformers 5.x.
The benchmark pins `transformers==5.3.0` (required for correct ColVision
weights), so mmore's hybrid text path is unusable here. A normalized-embedding
cosine retriever over bge-m3 is the standard dense text-RAG baseline and keeps
the comparison clean and reproducible.

Run on the cluster (bcv-venv has sentence-transformers + torch):

    python -m benchmark_colvision.runners.text_baseline \
        --manifest    /…/manifest-tb-en.json \
        --corpus-root /…/pdfs-tb-en \
        --queryset    data/track_b/queries/en.jsonl \
        --workdir     data/baseline_text/B/en \
        --record-out  results/baseline_text/B/en/seed_0.json \
        --track B --language en --dense-model BAAI/bge-m3
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

# Baseline "model" id recorded in results, so it sits beside the ColVision cells.
BASELINE_MODEL_ID = "text_bge_m3"


def collect_pages(
    manifest_path: Path,
    corpus_root: Path,
    *,
    min_text_chars: int = 80,
    page_base: int = 0,
) -> tuple[list[str], list[str]]:
    """Return (doc_ids, texts) for every page, ids as `"<pdf>#page=<N>"`.

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
        texts.append(page.text)
    return doc_ids, texts


def dense_retrieve(
    page_texts: list[str],
    query_texts: list[str],
    *,
    dense_model: str,
    top_k: int,
) -> list[list[int]]:
    """Return, per query, the indices of the top-k pages by cosine similarity."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(dense_model, device=device)
    # Normalized embeddings -> dot product == cosine similarity.
    page_emb = model.encode(
        page_texts, batch_size=32, convert_to_tensor=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    query_emb = model.encode(
        query_texts, batch_size=32, convert_to_tensor=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    sims = query_emb @ page_emb.T  # (n_queries, n_pages)
    k = min(top_k, page_emb.shape[0])
    top = torch.topk(sims, k=k, dim=1).indices
    return top.cpu().tolist()


def run_baseline_cell(args: argparse.Namespace) -> BenchmarkRecord:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    qrels = load_qrels(args.qrels) if args.qrels else None
    if args.ocr_results:
        # ViDoRe: page images carry no text layer; use mmore's own Marker+Surya
        # OCR output instead of PyMuPDF extraction (see corpus.ocr_pages).
        doc_ids, page_texts = pages_from_ocr(args.ocr_results)
        print(f"[baseline] {len(doc_ids)} OCR'd page units from {args.ocr_results}", flush=True)
    else:
        page_base = 1 if qrels is not None else 0
        doc_ids, page_texts = collect_pages(
            Path(args.manifest), Path(args.corpus_root),
            min_text_chars=args.min_text_chars, page_base=page_base,
        )
        print(f"[baseline] {len(doc_ids)} page units from {args.manifest}", flush=True)

    queryset = QuerySet.load_jsonl(Path(args.queryset), name=args.language, language="en")
    questions = [q.question for q in queryset.queries]
    if qrels is not None:
        # Graded, multi-relevant relevance (ViDoRe qrels), keyed by query_id.
        relevance = [qrels.get(q.query_id, {}) for q in queryset.queries]
        relevant = [set(r) for r in relevance]
    else:
        relevance = None
        relevant = [{f"{q.source_pdf}#page={q.source_page}"} for q in queryset.queries]

    t0 = time.monotonic()
    top_idx = dense_retrieve(
        page_texts, questions, dense_model=args.dense_model, top_k=args.top_k
    )
    retrieve_s = time.monotonic() - t0
    ranked = [[doc_ids[i] for i in idxs] for idxs in top_idx]

    s = evaluate_batch(ranked, relevant, relevance_per_query=relevance, ks=(1, 5, 10))
    retrieval_scores = RetrievalScores(
        ndcg_at_1=s.get("ndcg@1"), ndcg_at_5=s.get("ndcg@5"), ndcg_at_10=s.get("ndcg@10"),
        recall_at_1=s.get("recall@1"), recall_at_5=s.get("recall@5"), recall_at_10=s.get("recall@10"),
        precision_at_1=s.get("precision@1"), precision_at_5=s.get("precision@5"),
        precision_at_10=s.get("precision@10"), mrr=s.get("mrr"), map=s.get("map"),
        n_queries=len(questions),
    )

    # Persist the ranked lists for auditing / re-scoring.
    (workdir / "retrieval.json").write_text(
        json.dumps(
            [{"query": q, "ranked": r} for q, r in zip(questions, ranked)], indent=2
        )
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
            index_duration_s=None,
            retrieve_duration_total_s=retrieve_s,
            throughput_pages_per_s=None,
        ),
        notes=(
            f"dense text baseline ({args.dense_model}); page-level units"
            + (" (OCR'd via mmore process, graded ViDoRe qrels)" if qrels is not None else "")
        ),
    )
    record_out = Path(args.record_out)
    record_out.parent.mkdir(parents=True, exist_ok=True)
    record_out.write_text(record.to_json())
    print(f"[baseline] nDCG@5={retrieval_scores.ndcg_at_5} -> {record_out}", flush=True)
    return record


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Dense text-retrieval baseline (bge-m3).")
    p.add_argument("--manifest", default=None, help="Required unless --ocr-results is given.")
    p.add_argument("--corpus-root", default=None, help="Required unless --ocr-results is given.")
    p.add_argument(
        "--ocr-results", default=None,
        help="mmore `process` merged_results.jsonl (Marker+Surya OCR); use for "
        "image-only corpora (ViDoRe) instead of --manifest/--corpus-root.",
    )
    p.add_argument("--queryset", required=True, help="SyntheticQuery JSONL (rich queryset).")
    p.add_argument("--workdir", required=True)
    p.add_argument("--record-out", required=True)
    p.add_argument("--track", required=True, choices=["A", "B"])
    p.add_argument("--language", default="en")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dense-model", default="BAAI/bge-m3")
    p.add_argument("--top-k", type=int, default=10)
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
    # Accepted for CLI compatibility; the dense baseline does not use a sparse model.
    p.add_argument("--sparse-model", default=None, help=argparse.SUPPRESS)
    p.add_argument("--reranker", default="none", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if not args.ocr_results and not (args.manifest and args.corpus_root):
        p.error("either --ocr-results or both --manifest/--corpus-root are required")
    run_baseline_cell(args)


if __name__ == "__main__":
    main()
