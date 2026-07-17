"""Track A orchestrator: scaling study on English medical corpus.

Iterates over (model, palier, seed), invokes mmore through the subprocess
wrapper, scores retrieval, and writes one `BenchmarkRecord` JSON per cell.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from benchmark_colvision.evaluation.retrieval_metrics import evaluate_batch
from benchmark_colvision.queries.schema import QuerySet
from benchmark_colvision.results.schema import (
    BenchmarkRecord,
    CellId,
    PerformanceScores,
    RetrievalScores,
)
from benchmark_colvision.runners.mmore_wrapper import MmoreRun, run_pipeline
from benchmark_colvision.runners.retrieval_output import (
    align_to_queries,
    load_retrieval,
)


@dataclass
class TrackACell:
    model_id: str
    model_hf_name: str
    palier_id: str
    seed: int
    process_config: Path
    index_config: Path
    retrieve_config: Path
    queries_file: Path  # the mmore-formatted queries file (txt/json per mmore CLI)
    queryset_jsonl: Path  # the rich queryset we generated, for relevance
    output_file: Path
    record_out: Path
    qrels_file: Path | None = None  # graded multi-relevant qrels sidecar (ViDoRe)


def _build_record(
    cell: TrackACell,
    mmore_commit: str,
    benchmark_version: str,
    corpus_manifest_sha256: str,
    queries_sha256: str,
    mmore_run: MmoreRun,
    retrieval: RetrievalScores,
    performance: PerformanceScores,
) -> BenchmarkRecord:
    return BenchmarkRecord(
        cell=CellId(track="A", model_id=cell.model_id, palier_id=cell.palier_id, seed=cell.seed),
        mmore_commit=mmore_commit,
        benchmark_version=benchmark_version,
        corpus_manifest_sha256=corpus_manifest_sha256,
        queries_sha256=queries_sha256,
        retrieval=retrieval,
        performance=performance,
        notes=None if mmore_run.process and mmore_run.process.succeeded
        else "mmore pipeline reported a non-zero exit; metrics may be partial",
    )


def _load_qrels(path: Path) -> dict[str, dict[str, float]]:
    """Load a graded qrels sidecar: ``{query_id: {doc_id: grade}}``."""
    return {
        str(qid): {str(doc): float(grade) for doc, grade in rel.items()}
        for qid, rel in json.loads(path.read_text()).items()
    }


def _scores_from_retrieval(
    retrieval_path: Path,
    queryset: QuerySet,
    top_ks: tuple[int, ...] = (1, 5, 10),
    qrels: dict[str, dict[str, float]] | None = None,
) -> RetrievalScores:
    retrieval = load_retrieval(retrieval_path)
    qids = [q.query_id for q in queryset.queries]
    questions = [q.question for q in queryset.queries]
    ranked = align_to_queries(retrieval, qids, fallback_questions=questions)
    if qrels is not None:
        # Graded, multi-relevant relevance (ViDoRe qrels). doc ids follow mmore's
        # `<pdf_basename>#page=<1-based>` output format.
        relevance = [qrels.get(qid, {}) for qid in qids]
        relevant = [set(r) for r in relevance]
        scores = evaluate_batch(ranked, relevant, relevance_per_query=relevance, ks=top_ks)
    else:
        relevant = [{f"{q.source_pdf}#page={q.source_page}"} for q in queryset.queries]
        scores = evaluate_batch(ranked, relevant, ks=top_ks)
    return RetrievalScores(
        ndcg_at_1=scores.get("ndcg@1"),
        ndcg_at_5=scores.get("ndcg@5"),
        ndcg_at_10=scores.get("ndcg@10"),
        recall_at_1=scores.get("recall@1"),
        recall_at_5=scores.get("recall@5"),
        recall_at_10=scores.get("recall@10"),
        precision_at_1=scores.get("precision@1"),
        precision_at_5=scores.get("precision@5"),
        precision_at_10=scores.get("precision@10"),
        mrr=scores.get("mrr"),
        map=scores.get("map"),
        n_queries=len(qids),
    )


def _performance_from_run(run: MmoreRun, n_pages: int) -> PerformanceScores:
    proc = run.process.duration_s if run.process else None
    idx = run.index.duration_s if run.index else None
    ret = run.retrieve.duration_s if run.retrieve else None
    throughput = (n_pages / proc) if proc and proc > 0 else None
    return PerformanceScores(
        process_duration_s=proc,
        index_duration_s=idx,
        retrieve_duration_total_s=ret,
        throughput_pages_per_s=throughput,
    )


def run_cell(
    cell: TrackACell,
    *,
    mmore_commit: str,
    benchmark_version: str,
    corpus_manifest_sha256: str,
    n_pages_in_palier: int,
) -> BenchmarkRecord:
    queryset = QuerySet.load_jsonl(cell.queryset_jsonl, name=cell.palier_id, language="en")
    qrels = (
        _load_qrels(cell.qrels_file)
        if cell.qrels_file is not None and cell.qrels_file.exists()
        else None
    )
    cell.output_file.parent.mkdir(parents=True, exist_ok=True)
    mmore_run = run_pipeline(
        model_name=cell.model_hf_name,
        process_config=cell.process_config,
        index_config=cell.index_config,
        retrieve_config=cell.retrieve_config,
        queries_file=cell.queries_file,
        output_file=cell.output_file,
    )

    if mmore_run.retrieve and mmore_run.retrieve.succeeded and cell.output_file.exists():
        retrieval_scores = _scores_from_retrieval(cell.output_file, queryset, qrels=qrels)
    else:
        retrieval_scores = RetrievalScores(n_queries=len(queryset.queries))

    perf = _performance_from_run(mmore_run, n_pages=n_pages_in_palier)
    record = _build_record(
        cell=cell,
        mmore_commit=mmore_commit,
        benchmark_version=benchmark_version,
        corpus_manifest_sha256=corpus_manifest_sha256,
        queries_sha256=queryset.sha256(),
        mmore_run=mmore_run,
        retrieval=retrieval_scores,
        performance=perf,
    )
    cell.record_out.parent.mkdir(parents=True, exist_ok=True)
    cell.record_out.write_text(record.to_json())
    return record
