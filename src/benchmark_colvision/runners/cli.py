"""CLI entry point: bcv-run."""

from __future__ import annotations

from pathlib import Path

import click

from benchmark_colvision.runners.mmore_wrapper import format_command, run_pipeline, save_run


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
def track_a() -> None:
    """Run Track A (scaling study on EN medical corpus)."""
    raise NotImplementedError("Pending corpus assembly and queries; see plan Phase 5.")


@main.command(name="track-b")
def track_b() -> None:
    """Run Track B (multilingual benchmark on 6 languages)."""
    raise NotImplementedError("Pending corpus assembly and queries; see plan Phase 6.")
