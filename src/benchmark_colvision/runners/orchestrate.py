"""Glue between the track YAML configs and the per-cell `run_cell` functions.

`bcv-run track-a` / `bcv-run track-b` are thin wrappers around the two
functions here. The orchestrator reads `configs/track_X.yaml` and
`configs/models.yaml`, resolves filesystem paths under the `paths` block,
iterates over paliers (Track~A) or languages (Track~B), and writes one
`BenchmarkRecord` JSON per cell.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from benchmark_colvision.corpus.corpus_manifest import CorpusManifest
from benchmark_colvision.results.schema import BenchmarkRecord
from benchmark_colvision.runners.run_track_a import TrackACell
from benchmark_colvision.runners.run_track_a import run_cell as run_cell_a
from benchmark_colvision.runners.run_track_b import TrackBCell
from benchmark_colvision.runners.run_track_b import run_cell as run_cell_b


@dataclass
class _ResolvedPaths:
    corpus_manifest: Path
    queries_dir: Path
    mmore_process_config: Path
    mmore_index_config: Path
    mmore_retrieve_config: Path
    mmore_output_dir: Path
    records_dir: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _resolve_paths(track_cfg: dict[str, Any], base_dir: Path) -> _ResolvedPaths:
    paths = track_cfg.get("paths") or {}
    required = (
        "corpus_manifest",
        "queries_dir",
        "mmore_process_config",
        "mmore_index_config",
        "mmore_retrieve_config",
        "mmore_output_dir",
        "records_dir",
    )
    missing = [k for k in required if k not in paths]
    if missing:
        raise ValueError(f"track config missing paths keys: {missing}")

    def _abs(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else base_dir / p

    return _ResolvedPaths(
        corpus_manifest=_abs(paths["corpus_manifest"]),
        queries_dir=_abs(paths["queries_dir"]),
        mmore_process_config=_abs(paths["mmore_process_config"]),
        mmore_index_config=_abs(paths["mmore_index_config"]),
        mmore_retrieve_config=_abs(paths["mmore_retrieve_config"]),
        mmore_output_dir=_abs(paths["mmore_output_dir"]),
        records_dir=_abs(paths["records_dir"]),
    )


def _lookup_model(models_cfg: dict[str, Any], model_id: str) -> dict[str, Any]:
    for entry in models_cfg.get("models", []):
        if entry.get("id") == model_id:
            return entry
    raise KeyError(f"model_id {model_id!r} not found in models.yaml")


def _manifest_sha256(manifest_path: Path) -> str:
    """SHA-256 of the manifest JSON itself (cheap; manifest already contains per-PDF hashes)."""
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def _render_cell_configs(
    model_id: str,
    hf_name: str,
    embed_dim: int,
    base_dir: Path,
    base_process_cfg: Path,
    base_index_cfg: Path,
    base_retrieve_cfg: Path,
) -> tuple[Path, Path, Path]:
    """Write per-model mmore configs with isolated output paths and return their paths.

    Each model gets its own process output dir, Milvus DB, and collection so that
    multiple models can run without overwriting each other's artefacts. The
    embedding dimension is injected per-model because it differs across families
    (128 for ColPali / ColQwen2 / ColGemma3, 320 for ColQwen3); a mismatch makes
    the Milvus insert fail the dim assertion.
    """
    cell_dir = base_dir / "configs" / "mmore" / "cells" / model_id
    cell_dir.mkdir(parents=True, exist_ok=True)

    process_out = base_dir / "data" / "track_a" / "process" / model_id
    process_out.mkdir(parents=True, exist_ok=True)
    milvus_db = base_dir / "data" / "track_a" / "milvus" / f"{model_id}.db"
    milvus_db.parent.mkdir(parents=True, exist_ok=True)
    collection = f"bcv_{model_id}_pages"

    proc_cfg = yaml.safe_load(base_process_cfg.read_text())
    proc_cfg["output_path"] = str(process_out)
    proc_path = cell_dir / "process.yaml"
    proc_path.write_text(yaml.dump(proc_cfg, default_flow_style=False, allow_unicode=True))

    idx_cfg = yaml.safe_load(base_index_cfg.read_text())
    idx_cfg["parquet_path"] = str(process_out / "pdf_page_objects.parquet")
    idx_cfg.setdefault("milvus", {})
    idx_cfg["milvus"]["db_path"] = str(milvus_db)
    idx_cfg["milvus"]["collection_name"] = collection
    idx_cfg["milvus"]["create_collection"] = True
    idx_cfg["milvus"]["dim"] = embed_dim
    idx_path = cell_dir / "index.yaml"
    idx_path.write_text(yaml.dump(idx_cfg, default_flow_style=False, allow_unicode=True))

    ret_cfg = yaml.safe_load(base_retrieve_cfg.read_text())
    ret_cfg["db_path"] = str(milvus_db)
    ret_cfg["collection_name"] = collection
    ret_cfg["model_name"] = hf_name
    ret_cfg["dim"] = embed_dim
    ret_cfg["text_parquet_path"] = str(process_out / "pdf_page_text.parquet")
    ret_path = cell_dir / "retrieve.yaml"
    ret_path.write_text(yaml.dump(ret_cfg, default_flow_style=False, allow_unicode=True))

    return proc_path, idx_path, ret_path


def run_track_a_for_model(
    model_id: str,
    seed: int,
    *,
    track_config: Path,
    models_config: Path,
    mmore_commit: str,
    benchmark_version: str,
    base_dir: Path | None = None,
    palier_filter: list[str] | None = None,
) -> list[BenchmarkRecord]:
    """Run every (palier) cell for `model_id` at `seed`. Returns the records written."""
    base_dir = base_dir or track_config.parent.parent
    track_cfg = _load_yaml(track_config)
    models_cfg = _load_yaml(models_config)
    paths = _resolve_paths(track_cfg, base_dir)
    model_entry = _lookup_model(models_cfg, model_id)
    hf_name = model_entry["hf_name"]
    paliers = track_cfg.get("scaling_paliers", [])
    if palier_filter is not None:
        paliers = [p for p in paliers if p["id"] in palier_filter]
    if not paliers:
        raise ValueError("no scaling_paliers selected — check the track config or filter")
    if not paths.corpus_manifest.exists():
        raise FileNotFoundError(
            f"corpus manifest not found: {paths.corpus_manifest} "
            "(build it with bcv-corpus or override paths.corpus_manifest)"
        )
    corpus_sha = _manifest_sha256(paths.corpus_manifest)
    # Sanity-check the manifest parses.
    CorpusManifest.load(paths.corpus_manifest)

    proc_cfg, idx_cfg, ret_cfg = _render_cell_configs(
        model_id=model_id,
        hf_name=hf_name,
        embed_dim=int(model_entry.get("embed_dim", 128)),
        base_dir=base_dir,
        base_process_cfg=paths.mmore_process_config,
        base_index_cfg=paths.mmore_index_config,
        base_retrieve_cfg=paths.mmore_retrieve_config,
    )

    records: list[BenchmarkRecord] = []
    for palier in paliers:
        palier_id = palier["id"]
        n_pages = int(palier["n_pages"])
        queries_jsonl = paths.queries_dir / f"{palier_id}.jsonl"
        if not queries_jsonl.exists():
            raise FileNotFoundError(f"queries JSONL not found: {queries_jsonl}")
        output_file = paths.mmore_output_dir / model_id / palier_id / f"seed_{seed}.json"
        record_out = paths.records_dir / model_id / palier_id / f"seed_{seed}.json"
        cell = TrackACell(
            model_id=model_id,
            model_hf_name=hf_name,
            palier_id=palier_id,
            seed=seed,
            process_config=proc_cfg,
            index_config=idx_cfg,
            retrieve_config=ret_cfg,
            queries_file=queries_jsonl,
            queryset_jsonl=queries_jsonl,
            output_file=output_file,
            record_out=record_out,
        )
        record = run_cell_a(
            cell,
            mmore_commit=mmore_commit,
            benchmark_version=benchmark_version,
            corpus_manifest_sha256=corpus_sha,
            n_pages_in_palier=n_pages,
        )
        records.append(record)
    return records


def run_track_b_for_model(
    model_id: str,
    language: str,
    seed: int,
    *,
    track_config: Path,
    models_config: Path,
    mmore_commit: str,
    benchmark_version: str,
    base_dir: Path | None = None,
) -> BenchmarkRecord:
    """Run a single (model, language, seed) cell of Track B."""
    base_dir = base_dir or track_config.parent.parent
    track_cfg = _load_yaml(track_config)
    models_cfg = _load_yaml(models_config)
    paths = _resolve_paths(track_cfg, base_dir)
    model_entry = _lookup_model(models_cfg, model_id)
    hf_name = model_entry["hf_name"]

    languages = track_cfg.get("languages", [])
    lang_entry = next((entry for entry in languages if entry.get("code") == language), None)
    if lang_entry is None:
        raise KeyError(f"language {language!r} not declared in track_b.yaml")
    n_pages = int(lang_entry.get("pages_target", 0)) or int(
        track_cfg.get("queries", {}).get("per_language", 0)
    )

    if not paths.corpus_manifest.exists():
        raise FileNotFoundError(f"corpus manifest not found: {paths.corpus_manifest}")
    corpus_sha = _manifest_sha256(paths.corpus_manifest)
    CorpusManifest.load(paths.corpus_manifest)

    queries_jsonl = paths.queries_dir / f"{language}.jsonl"
    if not queries_jsonl.exists():
        raise FileNotFoundError(f"queries JSONL not found: {queries_jsonl}")
    output_file = paths.mmore_output_dir / model_id / language / f"seed_{seed}.json"
    record_out = paths.records_dir / model_id / language / f"seed_{seed}.json"
    cell = TrackBCell(
        model_id=model_id,
        model_hf_name=hf_name,
        language=language,
        seed=seed,
        process_config=paths.mmore_process_config,
        index_config=paths.mmore_index_config,
        retrieve_config=paths.mmore_retrieve_config,
        queries_file=queries_jsonl,
        queryset_jsonl=queries_jsonl,
        output_file=output_file,
        record_out=record_out,
    )
    return run_cell_b(
        cell,
        mmore_commit=mmore_commit,
        benchmark_version=benchmark_version,
        corpus_manifest_sha256=corpus_sha,
        n_pages_in_language=n_pages,
    )
