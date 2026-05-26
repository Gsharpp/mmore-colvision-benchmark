"""Walk a directory of PDFs and emit a CorpusManifest.

For each PDF the walker computes the SHA-256 hash, the page count, the
visual density (figures/tables surface ratio) and the detected language.
PDFs whose visual density is below `density_threshold` are skipped --- the
benchmark is only meaningful on figure-rich documents.

The walker is intentionally tolerant: a PDF that PyMuPDF cannot open is
recorded in `skipped` with an error string instead of aborting the whole
build.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from benchmark_colvision.corpus.corpus_manifest import (
    CorpusManifest,
    PdfEntry,
    sha256_file,
)
from benchmark_colvision.corpus.language_filter import detect_language
from benchmark_colvision.corpus.visual_density_filter import compute_density


@dataclass
class SkippedPdf:
    pdf_path: str
    reason: str


@dataclass
class BuildReport:
    name: str
    track: str
    kept: list[PdfEntry] = field(default_factory=list)
    skipped: list[SkippedPdf] = field(default_factory=list)

    @property
    def total_pages(self) -> int:
        return sum(e.page_count for e in self.kept)


def iter_pdfs(root: Path) -> Iterable[Path]:
    """Yield every *.pdf under `root` in lexicographic order (stable across runs)."""
    yield from sorted(root.rglob("*.pdf"))


def _detect_language_for(pdf_path: Path, *, language_override: str | None) -> str:
    if language_override is not None:
        return language_override
    try:
        with fitz.open(pdf_path) as doc:
            text = "".join(page.get_text() for page in doc[:2])[:4000]
    except Exception:
        return "unknown"
    return detect_language(text)


def build_manifest(
    pdf_root: Path,
    *,
    name: str,
    track: str,
    source: str,
    density_threshold: float = 0.30,
    language_override: str | None = None,
) -> tuple[CorpusManifest, BuildReport]:
    """Walk `pdf_root`, score each PDF, return (manifest, BuildReport)."""
    report = BuildReport(name=name, track=track)
    languages_seen: set[str] = set()
    for pdf in iter_pdfs(pdf_root):
        try:
            density = compute_density(pdf)
        except Exception as e:
            report.skipped.append(SkippedPdf(str(pdf), f"density failed: {e}"))
            continue
        if density.visual_density < density_threshold:
            report.skipped.append(
                SkippedPdf(str(pdf), f"low density {density.visual_density:.3f}")
            )
            continue
        try:
            language = _detect_language_for(pdf, language_override=language_override)
        except Exception as e:
            report.skipped.append(SkippedPdf(str(pdf), f"lang detect failed: {e}"))
            continue
        rel = pdf.relative_to(pdf_root)
        entry = PdfEntry(
            pdf_path=str(rel),
            sha256=sha256_file(pdf),
            page_count=density.page_count,
            language=language,
            visual_density=density.visual_density,
            mesh_tags=[],
            source=source,  # type: ignore[arg-type]
        )
        report.kept.append(entry)
        languages_seen.add(language)

    manifest = CorpusManifest(
        name=name,
        track=track,  # type: ignore[arg-type]
        languages=sorted(languages_seen),
        total_pages=report.total_pages,
        pdfs=report.kept,
    )
    return manifest, report
