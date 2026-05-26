"""Track B orchestrator: multilingual benchmark on 6 languages at fixed corpus size."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmark_colvision.queries.schema import QuerySet
from benchmark_colvision.results.schema import (
    BenchmarkRecord,
    CellId,
    RetrievalScores,
)
from benchmark_colvision.runners.mmore_wrapper import run_pipeline
from benchmark_colvision.runners.run_track_a import (
    _performance_from_run,
    _scores_from_retrieval,
)


@dataclass
class TrackBCell:
    model_id: str
    model_hf_name: str
    language: str
    seed: int
    process_config: Path
    index_config: Path
    retrieve_config: Path
    queries_file: Path
    queryset_jsonl: Path
    output_file: Path
    record_out: Path


def run_cell(
    cell: TrackBCell,
    *,
    mmore_commit: str,
    benchmark_version: str,
    corpus_manifest_sha256: str,
    n_pages_in_language: int,
) -> BenchmarkRecord:
    queryset = QuerySet.load_jsonl(cell.queryset_jsonl, name=cell.language, language=cell.language)
    mmore_run = run_pipeline(
        model_name=cell.model_hf_name,
        process_config=cell.process_config,
        index_config=cell.index_config,
        retrieve_config=cell.retrieve_config,
        queries_file=cell.queries_file,
        output_file=cell.output_file,
    )
    if mmore_run.retrieve and mmore_run.retrieve.succeeded and cell.output_file.exists():
        retrieval_scores = _scores_from_retrieval(cell.output_file, queryset)
    else:
        retrieval_scores = RetrievalScores(n_queries=len(queryset.queries))

    perf = _performance_from_run(mmore_run, n_pages=n_pages_in_language)
    record = BenchmarkRecord(
        cell=CellId(track="B", model_id=cell.model_id, language=cell.language, seed=cell.seed),
        mmore_commit=mmore_commit,
        benchmark_version=benchmark_version,
        corpus_manifest_sha256=corpus_manifest_sha256,
        queries_sha256=queryset.sha256(),
        retrieval=retrieval_scores,
        performance=perf,
        notes=None if mmore_run.process and mmore_run.process.succeeded
        else "mmore pipeline reported a non-zero exit; metrics may be partial",
    )
    cell.record_out.parent.mkdir(parents=True, exist_ok=True)
    cell.record_out.write_text(record.to_json())
    return record
