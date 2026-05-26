"""CLI entry point: bcv-run."""

from __future__ import annotations

from pathlib import Path

import click

from benchmark_colvision.runners.mmore_wrapper import format_command, run_pipeline, save_run
from benchmark_colvision.runners.orchestrate import (
    run_track_a_for_model,
    run_track_b_for_model,
)


@click.group()
def main() -> None:
    """Benchmark runners (Track A scaling, Track B multilingual)."""


@main.command()
@click.option("--model-name", required=True, help="HuggingFace model id (e.g. vidore/colpali-v1.3)")
@click.option("--process-config", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--index-config", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--retrieve-config", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--queries-file", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output-file", required=True, type=click.Path(path_type=Path))
@click.option("--save-to", required=True, type=click.Path(path_type=Path), help="Where to write the run JSON.")
def pipeline(
    model_name: str,
    process_config: Path,
    index_config: Path,
    retrieve_config: Path,
    queries_file: Path,
    output_file: Path,
    save_to: Path,
) -> None:
    """Execute process → index → retrieve sequentially and save timings."""
    run = run_pipeline(
        model_name=model_name,
        process_config=process_config,
        index_config=index_config,
        retrieve_config=retrieve_config,
        queries_file=queries_file,
        output_file=output_file,
    )
    save_run(run, save_to)
    for step in ("process", "index", "retrieve"):
        result = getattr(run, step)
        if result is None:
            click.echo(f"{step}: SKIPPED (previous step failed)")
            continue
        status = "OK" if result.succeeded else f"FAIL ({result.returncode})"
        click.echo(f"{step}: {status} — {result.duration_s:.2f}s — {format_command(result.command)}")


@main.command(name="track-a")
@click.option("--model-id", required=True, help="Model id as declared in models.yaml")
@click.option("--seed", required=True, type=int)
@click.option("--config", "track_config", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--models", "models_config", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--mmore-commit", required=True)
@click.option("--benchmark-version", default="0.1.0", show_default=True)
@click.option("--palier", "palier_filter", multiple=True, help="Restrict to one or more paliers")
def track_a(
    model_id: str,
    seed: int,
    track_config: Path,
    models_config: Path,
    mmore_commit: str,
    benchmark_version: str,
    palier_filter: tuple[str, ...],
) -> None:
    """Run every palier for (model_id, seed) under Track A."""
    records = run_track_a_for_model(
        model_id=model_id,
        seed=seed,
        track_config=track_config,
        models_config=models_config,
        mmore_commit=mmore_commit,
        benchmark_version=benchmark_version,
        palier_filter=list(palier_filter) or None,
    )
    for r in records:
        click.echo(f"track-a: {r.cell.model_id} {r.cell.palier_id} seed={r.cell.seed} → ndcg@5={r.retrieval.ndcg_at_5}")


@main.command(name="track-b")
@click.option("--model-id", required=True)
@click.option("--language", required=True)
@click.option("--seed", required=True, type=int)
@click.option("--config", "track_config", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--models", "models_config", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--mmore-commit", required=True)
@click.option("--benchmark-version", default="0.1.0", show_default=True)
def track_b(
    model_id: str,
    language: str,
    seed: int,
    track_config: Path,
    models_config: Path,
    mmore_commit: str,
    benchmark_version: str,
) -> None:
    """Run a single (model, language, seed) cell of Track B."""
    record = run_track_b_for_model(
        model_id=model_id,
        language=language,
        seed=seed,
        track_config=track_config,
        models_config=models_config,
        mmore_commit=mmore_commit,
        benchmark_version=benchmark_version,
    )
    click.echo(
        f"track-b: {record.cell.model_id} {record.cell.language} seed={record.cell.seed} "
        f"→ ndcg@5={record.retrieval.ndcg_at_5}"
    )
