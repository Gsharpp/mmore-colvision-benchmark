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
@click.argument("pmcids", nargs=-1, required=False)
@click.option("--pmcids-json", type=click.Path(exists=True, path_type=Path), default=None,
              help="JSON list of PMCIDs (produced by sample-pmc-zh); alternative to positional args.")
@click.option("--cache-dir", type=click.Path(path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
def download_pmc(
    pmcids: tuple[str, ...],
    pmcids_json: Path | None,
    cache_dir: Path,
    out_dir: Path,
) -> None:
    """Download a set of PMC OA packages (idempotent) and extract their PDFs."""
    from benchmark_colvision.corpus.pmc_downloader import download_corpus

    all_pmcids: list[str] = list(pmcids)
    if pmcids_json is not None:
        all_pmcids.extend(json.loads(pmcids_json.read_text()))
    if not all_pmcids:
        raise click.UsageError("provide PMCIDs as positional arguments or via --pmcids-json")
    pdfs = download_corpus(all_pmcids, cache_dir=cache_dir, pdf_out_dir=out_dir)
    click.echo(f"extracted {len(pdfs)} PDFs to {out_dir}")


@main.command("sample-pmc")
@click.option("--n", type=int, default=60, show_default=True, help="How many PMCIDs to sample")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--scan-limit", type=int, default=400_000, show_default=True,
              help="Cap rows scanned from the OA index")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Write space-separated PMCIDs here (also echoed to stdout)")
@click.option("--packages-out", type=click.Path(path_type=Path), default=None,
              help="Write full package metadata as JSON (avoids re-streaming index on download)")
def sample_pmc(n: int, seed: int, scan_limit: int, out: Path | None, packages_out: Path | None) -> None:
    """Reservoir-sample valid PMCIDs from the PMC OA index (memory-safe stream)."""
    from benchmark_colvision.corpus.pmc_downloader import sample_packages

    pkgs = sample_packages(n, seed=seed, scan_limit=scan_limit)
    text = " ".join(p.pmcid for p in pkgs)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    if packages_out is not None:
        packages_out.parent.mkdir(parents=True, exist_ok=True)
        packages_out.write_text(
            json.dumps([{"pmcid": p.pmcid, "relative_path": p.relative_path, "license": p.license} for p in pkgs], indent=2)
        )
    click.echo(text)


@main.command("download-packages")
@click.option("--packages-json", type=click.Path(exists=True, path_type=Path), required=True,
              help="JSON produced by sample-pmc --packages-out (no index re-stream needed)")
@click.option("--cache-dir", type=click.Path(path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
def download_packages_cmd(packages_json: Path, cache_dir: Path, out_dir: Path) -> None:
    """Download PMC packages using pre-resolved paths (skips index streaming)."""
    from benchmark_colvision.corpus.pmc_downloader import PmcPackage, download_packages

    raw = json.loads(packages_json.read_text())
    pkgs = [PmcPackage(pmcid=r["pmcid"], relative_path=r["relative_path"], license=r["license"]) for r in raw]
    pdfs = download_packages(pkgs, cache_dir=cache_dir, pdf_out_dir=out_dir)
    click.echo(f"extracted {len(pdfs)} PDFs to {out_dir}")


@main.command("build-manifest")
@click.argument("pdf_root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out", type=click.Path(path_type=Path), required=True, help="Manifest JSON output path")
@click.option("--name", required=True, help="Manifest name (e.g. 'track_a_medium')")
@click.option("--track", required=True, type=click.Choice(["A", "B"]))
@click.option("--source", required=True, type=click.Choice(
    ["pmc-oa", "hal", "cairn", "scielo", "thieme-oa", "saudi-med", "cnki-oa", "other"]
))
@click.option("--density-threshold", type=float, default=0.30, show_default=True)
@click.option("--language-override", default=None, help="Skip language detection and use this code")
@click.option("--report-out", type=click.Path(path_type=Path), default=None,
              help="Optional skipped-PDFs report JSON")
def build_manifest_cmd(
    pdf_root: Path,
    out: Path,
    name: str,
    track: str,
    source: str,
    density_threshold: float,
    language_override: str | None,
    report_out: Path | None,
) -> None:
    """Walk PDF_ROOT, score each PDF, and write a CorpusManifest JSON."""
    from benchmark_colvision.corpus.build_manifest import build_manifest

    manifest, report = build_manifest(
        pdf_root,
        name=name,
        track=track,
        source=source,
        density_threshold=density_threshold,
        language_override=language_override,
    )
    manifest.save(out)
    click.echo(
        json.dumps(
            {
                "out": str(out),
                "n_kept": len(manifest.pdfs),
                "n_skipped": len(report.skipped),
                "languages": manifest.languages,
                "total_pages": manifest.total_pages,
            },
            indent=2,
        )
    )
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(
            json.dumps(
                {
                    "n_kept": len(manifest.pdfs),
                    "skipped": [{"pdf_path": s.pdf_path, "reason": s.reason} for s in report.skipped],
                },
                indent=2,
            )
        )


@main.command("sample-pmc-lang")
@click.option("--language", default="zh", show_default=True,
              help="ISO language code (zh, de, fr, …); native PMC OA literature")
@click.option("--n", type=int, default=60, show_default=True, help="Number of PMCIDs")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--out", type=click.Path(path_type=Path), required=True, help="JSON list of PMCIDs")
def sample_pmc_lang(language: str, n: int, seed: int, out: Path) -> None:
    """Sample native-language PMC OA article IDs via NCBI EUtils (no translation)."""
    from benchmark_colvision.corpus.pmc_zh_sampler import sample_and_save

    pmcids = sample_and_save(out, language=language, n=n, seed=seed)
    click.echo(json.dumps({"out": str(out), "language": language, "n_pmcids": len(pmcids)}))


@main.command("sample-hal")
@click.option("--n", type=int, default=100, show_default=True, help="Number of articles to fetch")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--language", default="fr", show_default=True, help="ISO language code")
@click.option("--out", type=click.Path(path_type=Path), required=True, help="UrlListManifest JSON output")
def sample_hal(n: int, seed: int, language: str, out: Path) -> None:
    """Sample open-access medical articles from HAL and write a UrlListManifest."""
    from benchmark_colvision.corpus.hal_downloader import sample_and_save

    n_items = sample_and_save(out, n=n, language=language, seed=seed)
    click.echo(json.dumps({"out": str(out), "n_items": n_items, "language": language}))


@main.command("download-urls")
@click.argument("manifest_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--report-out", type=click.Path(path_type=Path), default=None)
@click.option("--max-retries", type=int, default=3, show_default=True)
def download_urls(
    manifest_path: Path,
    out_dir: Path,
    report_out: Path | None,
    max_retries: int,
) -> None:
    """Bulk-download PDFs listed in a UrlListManifest (HAL, Thieme, SciELO, ...)."""
    from benchmark_colvision.corpus.url_list_downloader import (
        UrlListManifest,
        download_from_manifest,
    )

    manifest = UrlListManifest.load(manifest_path)
    report = download_from_manifest(manifest, out_dir, max_retries=max_retries)
    summary = {
        "source": report.source,
        "language": report.language,
        "n_total": report.n_total,
        "n_downloaded": report.n_downloaded,
        "n_cached": report.n_cached,
        "n_failed": report.n_failed,
    }
    click.echo(json.dumps(summary, indent=2))
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(
            json.dumps(
                {
                    **summary,
                    "outcomes": [
                        {
                            "source_id": o.source_id,
                            "url": o.url,
                            "status": o.status,
                            "pdf_path": str(o.pdf_path) if o.pdf_path else None,
                            "sha256": o.sha256,
                            "error": o.error,
                        }
                        for o in report.outcomes
                    ],
                },
                indent=2,
            )
        )


@main.command("build-vidore")
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--repo", default="vidore/biomedical_lectures_eng_v2", show_default=True)
@click.option("--split", default="test", show_default=True)
@click.option("--language", default="en", show_default=True)
@click.option("--manifest-name", default=None, help="Manifest name (defaults to the repo basename).")
@click.option("--cache-dir", type=click.Path(path_type=Path), default=None)
@click.option("--min-score", type=int, default=1, show_default=True,
              help="Minimum qrel score to treat a page as relevant.")
def build_vidore(
    out_dir: Path,
    repo: str,
    split: str,
    language: str,
    manifest_name: str | None,
    cache_dir: Path | None,
    min_score: int,
) -> None:
    """Download a ViDoRe v2 dataset and build a mmore-ready corpus.

    Writes {out_dir}/{pdfs/, corpus_manifest.json, queries.jsonl, qrels.json}.
    Reconstructs one PDF per document (page images), plus a graded multi-relevant
    qrels sidecar aligned to mmore's `<pdf>#page=<1-based>` output format.
    """
    from benchmark_colvision.corpus.vidore_v2 import download_and_build

    result = download_and_build(
        out_dir,
        repo=repo,
        split=split,
        language=language,
        manifest_name=manifest_name,
        cache_dir=str(cache_dir) if cache_dir else None,
        min_score=min_score,
    )
    click.echo(
        json.dumps(
            {
                "repo": repo,
                "out_dir": str(result.out_dir),
                "n_docs": result.n_docs,
                "n_pages": result.n_pages,
                "n_queries": result.n_queries,
                "n_qrels": result.n_qrels,
                "n_queries_dropped": result.n_queries_dropped,
            },
            indent=2,
        )
    )


@main.command("build-vidore-lang")
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--language", required=True, help="english, french, german, or spanish")
@click.option("--repo", default="vidore/biomedical_lectures_v2", show_default=True)
@click.option("--split", default="test", show_default=True)
@click.option("--manifest-name", default="vidore", show_default=True)
@click.option("--cache-dir", type=click.Path(path_type=Path), default=None)
@click.option("--min-score", type=int, default=1, show_default=True,
              help="Minimum qrel score to treat a page as relevant.")
def build_vidore_lang(
    out_dir: Path,
    language: str,
    repo: str,
    split: str,
    manifest_name: str,
    cache_dir: Path | None,
    min_score: int,
) -> None:
    """Build one query-language slice (queries.jsonl + qrels.json) of the ViDoRe
    v2 multilingual release, for Track B.

    Writes only {out_dir}/{queries.jsonl, qrels.json} — no PDFs or manifest. The
    corpus is shared with Track A's ``biomedical_lectures_eng_v2`` build (same
    corpus-id space); Track B re-retrieves these translated queries on Track A's
    existing Milvus index (see ``bcv-run track-b-vidore``).
    """
    from benchmark_colvision.corpus.vidore_v2 import download_and_build_language

    result = download_and_build_language(
        out_dir,
        language=language,
        repo=repo,
        split=split,
        manifest_name=manifest_name,
        cache_dir=str(cache_dir) if cache_dir else None,
        min_score=min_score,
    )
    click.echo(
        json.dumps(
            {
                "repo": repo,
                "language": language,
                "out_dir": str(result.out_dir),
                "n_queries": result.n_queries,
                "n_qrels": result.n_qrels,
                "n_queries_dropped": result.n_queries_dropped,
            },
            indent=2,
        )
    )
