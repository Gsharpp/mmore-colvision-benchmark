"""Result JSON schema for one benchmark cell (model × corpus × seed).

The schema is intentionally close to MTEB so that runs can be ingested by
downstream MTEB tooling without conversion: a single `BenchmarkRecord` carries
the retrieval scores, generation scores, performance numbers, plus all the
identifiers needed to trace a cell back to its inputs (mmore commit, corpus
manifest hash, query set hash, seed, hardware).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class RetrievalScores(BaseModel):
    """Macro-averaged retrieval metrics produced by `evaluation.retrieval_metrics`."""

    ndcg_at_1: float | None = None
    ndcg_at_5: float | None = None
    ndcg_at_10: float | None = None
    recall_at_1: float | None = None
    recall_at_5: float | None = None
    recall_at_10: float | None = None
    precision_at_1: float | None = None
    precision_at_5: float | None = None
    precision_at_10: float | None = None
    mrr: float | None = None
    map: float | None = None
    n_queries: int = 0


class GenerationScores(BaseModel):
    """RAGAS-style generation metrics scored by an LLM judge."""

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    judge_model: str | None = None
    n_examples: int = 0


class PerformanceScores(BaseModel):
    """Resource usage and latency for the indexing and retrieval phases."""

    process_duration_s: float | None = None
    index_duration_s: float | None = None
    retrieve_duration_total_s: float | None = None
    throughput_pages_per_s: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None
    gpu_peak_mb: float | None = None
    db_size_mb: float | None = None


class HardwareInfo(BaseModel):
    gpu_name: str | None = None
    gpu_count: int | None = None
    cuda_version: str | None = None
    torch_version: str | None = None
    hostname: str | None = None


class CellId(BaseModel):
    """Identity of a single benchmark cell. Used as the JSON record key."""

    track: Literal["A", "B"]
    model_id: str
    palier_id: str | None = None  # Track A only
    language: str | None = None  # Track B only
    seed: int


class BenchmarkRecord(BaseModel):
    """One row in the results table: a single (track, model, palier|lang, seed)."""

    cell: CellId
    mmore_commit: str
    benchmark_version: str
    corpus_manifest_sha256: str
    queries_sha256: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    retrieval: RetrievalScores = Field(default_factory=RetrievalScores)
    generation: GenerationScores = Field(default_factory=GenerationScores)
    performance: PerformanceScores = Field(default_factory=PerformanceScores)
    hardware: HardwareInfo = Field(default_factory=HardwareInfo)
    notes: str | None = None

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)


class BenchmarkRunSet(BaseModel):
    """A collection of records produced by one orchestrator invocation."""

    name: str
    track: Literal["A", "B"]
    records: list[BenchmarkRecord] = Field(default_factory=list)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
