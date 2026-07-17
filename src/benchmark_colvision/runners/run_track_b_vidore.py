"""Track B ViDoRe orchestrator: re-retrieve translated queries on the FIXED
Track A Milvus index (no re-embedding).

Unlike the native `run_track_b.py` (one corpus + one Milvus collection per
language), the ViDoRe v2 multilingual release shares a single corpus across
languages — only the queries are translated (English/French/German/Spanish).
So the axis under test becomes *query language*, and Track A's index (already
built once per model) can be re-retrieved directly: this cell only runs
`mmore colvision retrieve`, never process/index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmark_colvision.queries.schema import QuerySet
from benchmark_colvision.results.schema import (
    BenchmarkRecord,
    CellId,
    RetrievalScores,
)
from benchmark_colvision.runners.mmore_wrapper import run_retrieve_only
from benchmark_colvision.runners.run_track_a import (
    _load_qrels,
    _performance_from_run,
    _scores_from_retrieval,
)


@dataclass
class TrackBViDoReCell:
    model_id: str
    model_hf_name: str
    language: str
    seed: int
    retrieve_config: Path  # Track A's already-rendered configs/mmore/cells/<model_id>/retrieve.yaml
    queries_file: Path  # this language's queries.jsonl (translated)
    qrels_file: Path  # this language's graded qrels.json (same doc-id space as Track A)
    output_file: Path
    record_out: Path


def run_cell(
    cell: TrackBViDoReCell,
    *,
    mmore_commit: str,
    benchmark_version: str,
    corpus_manifest_sha256: str,
    n_pages: int,
) -> BenchmarkRecord:
    queryset = QuerySet.load_jsonl(cell.queries_file, name=cell.language, language=cell.language)
    qrels = _load_qrels(cell.qrels_file) if cell.qrels_file.exists() else None

    mmore_run = run_retrieve_only(
        model_name=cell.model_hf_name,
        retrieve_config=cell.retrieve_config,
        queries_file=cell.queries_file,
        output_file=cell.output_file,
    )

    if mmore_run.retrieve and mmore_run.retrieve.succeeded and cell.output_file.exists():
        retrieval_scores = _scores_from_retrieval(cell.output_file, queryset, qrels=qrels)
    else:
        retrieval_scores = RetrievalScores(n_queries=len(queryset.queries))

    perf = _performance_from_run(mmore_run, n_pages=n_pages)
    record = BenchmarkRecord(
        cell=CellId(track="B", model_id=cell.model_id, language=cell.language, seed=cell.seed),
        mmore_commit=mmore_commit,
        benchmark_version=benchmark_version,
        corpus_manifest_sha256=corpus_manifest_sha256,
        queries_sha256=queryset.sha256(),
        retrieval=retrieval_scores,
        performance=perf,
        notes=None if mmore_run.retrieve and mmore_run.retrieve.succeeded
        else "mmore retrieve reported a non-zero exit; metrics may be partial",
    )
    cell.record_out.parent.mkdir(parents=True, exist_ok=True)
    cell.record_out.write_text(record.to_json())
    return record
