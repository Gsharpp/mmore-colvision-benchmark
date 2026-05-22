"""CLI entry point: bcv-corpus."""

from __future__ import annotations

import json
from pathlib import Path

import click

from benchmark_colvision.corpus.corpus_manifest import CorpusManifest, verify_manifest
from benchmark_colvision.corpus.language_filter import detect_language
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


@main.command("detect-lang")
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--max-chars", type=int, default=4000, show_default=True)
def detect_lang(pdf_path: Path, max_chars: int) -> None:
    """Extract first-page text via PyMuPDF and report detected language code."""
    import fitz  # type: ignore[import-not-found]

    doc = fitz.open(pdf_path)
    try:
        text = "".join(page.get_text() for page in doc[:2])[:max_chars]
    finally:
        doc.close()
    lang = detect_language(text)
    click.echo(json.dumps({"pdf_path": str(pdf_path), "language": lang}, indent=2))


@main.command("download-pmc")
@click.argument("pmcids", nargs=-1, required=True)
@click.option("--cache-dir", type=click.Path(path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
def download_pmc(pmcids: tuple[str, ...], cache_dir: Path, out_dir: Path) -> None:
    """Download a set of PMC OA packages (idempotent) and extract their PDFs."""
    from benchmark_colvision.corpus.pmc_downloader import download_corpus

    pdfs = download_corpus(pmcids, cache_dir=cache_dir, pdf_out_dir=out_dir)
    click.echo(f"extracted {len(pdfs)} PDFs to {out_dir}")
