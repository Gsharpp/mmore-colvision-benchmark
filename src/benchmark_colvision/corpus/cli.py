"""CLI entry point: bcv-corpus."""

from __future__ import annotations

import json
from pathlib import Path

import click

from benchmark_colvision.corpus.corpus_manifest import CorpusManifest, verify_manifest
from benchmark_colvision.corpus.visual_density_filter import compute_density, keep


@click.group()
def main() -> None:
    """Corpus construction and management."""


@main.command()
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--threshold", type=float, default=0.30, show_default=True)
def density(pdf_path: Path, threshold: float) -> None:
    """Compute visual-density score of a single PDF."""
    report = compute_density(pdf_path)
    payload = {
        "pdf_path": report.pdf_path,
        "page_count": report.page_count,
        "image_surface_ratio": report.image_surface_ratio,
        "table_surface_ratio": report.table_surface_ratio,
        "visual_density": report.visual_density,
        "kept": keep(report, threshold=threshold),
    }
    click.echo(json.dumps(payload, indent=2))


@main.command()
@click.argument("manifest_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("base_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def verify(manifest_path: Path, base_dir: Path) -> None:
    """Re-hash every PDF in a manifest and report drift."""
    manifest = CorpusManifest.load(manifest_path)
    drift = verify_manifest(manifest, base_dir)
    if not drift:
        click.echo(f"OK — {len(manifest.pdfs)} PDFs hashed cleanly.")
        return
    click.echo(f"DRIFT — {len(drift)} entries do not match:")
    for d in drift:
        click.echo(f"  {d}")
    raise SystemExit(1)
