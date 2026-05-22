"""CLI entry point: bcv-report."""

from __future__ import annotations

from pathlib import Path

import click

from benchmark_colvision.reporting.figures_track_a import (
    gpu_memory_curve,
    latency_curve,
    scaling_curve,
    throughput_bar,
)
from benchmark_colvision.reporting.figures_track_b import (
    heatmap_metric,
    language_gap_bars,
)
from benchmark_colvision.reporting.latex_tables import (
    save_table,
    track_a_table,
    track_b_table,
)
from benchmark_colvision.results.aggregate import load_records, records_to_frame


@click.group()
def main() -> None:
    """Figure and table generation from result JSONs."""


def _load_frame(results_dir: Path):
    records = load_records(results_dir)
    if not records:
        raise click.ClickException(f"no records found under {results_dir}")
    return records_to_frame(records)


@main.command("figures-a")
@click.option("--results-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--metric", default="retrieval.ndcg_at_5", show_default=True)
def figures_a(results_dir: Path, out_dir: Path, metric: str) -> None:
    """Render all Track A figures."""
    df = _load_frame(results_dir)
    df = df[df["track"] == "A"]
    scaling_curve(df, metric=metric, out_path=out_dir / "track_a_scaling.png", ylabel=metric)
    throughput_bar(df, out_path=out_dir / "track_a_throughput.png")
    latency_curve(df, out_path=out_dir / "track_a_latency.png")
    gpu_memory_curve(df, out_path=out_dir / "track_a_gpu_memory.png")
    click.echo(f"wrote 4 figures to {out_dir}")


@main.command("figures-b")
@click.option("--results-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--metric", default="retrieval.ndcg_at_5", show_default=True)
def figures_b(results_dir: Path, out_dir: Path, metric: str) -> None:
    """Render all Track B figures."""
    df = _load_frame(results_dir)
    df = df[df["track"] == "B"]
    heatmap_metric(df, metric=metric, out_path=out_dir / "track_b_heatmap.png")
    language_gap_bars(df, metric=metric, out_path=out_dir / "track_b_lang_gap.png")
    click.echo(f"wrote 2 figures to {out_dir}")


@main.command("tables")
@click.option("--results-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--metric", default="retrieval.ndcg_at_5", show_default=True)
def tables(results_dir: Path, out_dir: Path, metric: str) -> None:
    """Render LaTeX tables for Track A and Track B."""
    df = _load_frame(results_dir)
    a = df[df["track"] == "A"]
    b = df[df["track"] == "B"]
    if not a.empty:
        save_table(track_a_table(a, metric=metric), out_dir / "track_a.tex")
    if not b.empty:
        save_table(track_b_table(b, metric=metric), out_dir / "track_b.tex")
    click.echo(f"wrote tables to {out_dir}")
