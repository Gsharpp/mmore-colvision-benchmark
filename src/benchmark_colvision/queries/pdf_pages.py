"""Walk a CorpusManifest and yield `PageInput` objects ready for query generation.

PDF parsing is done with PyMuPDF: per page we extract the raw text and detect
whether the page carries figures or tables (so the query-generation prompt can
include visual hints).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import fitz  # PyMuPDF

from benchmark_colvision.corpus.corpus_manifest import CorpusManifest, PdfEntry
from benchmark_colvision.queries.inverse_query_gen import PageInput


def iter_pdf_pages(
    pdf_path: Path,
    *,
    pdf_rel_path: str,
    language: str,
    min_text_chars: int = 80,
) -> Iterator[PageInput]:
    """Yield one PageInput per page of a single PDF; skip pages with too little text."""
    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc):
            text = page.get_text("text") or ""
            if len(text.strip()) < min_text_chars:
                continue
            has_figures = bool(page.get_image_info())
            has_tables = False
            try:
                tables = page.find_tables()
                has_tables = len(list(tables)) > 0
            except Exception:
                has_tables = False
            yield PageInput(
                pdf_path=pdf_rel_path,
                page_number=page_number,
                language=language,
                text=text,
                has_figures=has_figures,
                has_tables=has_tables,
            )


def iter_manifest_pages(
    manifest: CorpusManifest,
    corpus_root: Path,
    *,
    min_text_chars: int = 80,
) -> Iterator[PageInput]:
    """Yield PageInputs for every page of every PDF in the manifest."""
    for entry in manifest.pdfs:
        yield from _iter_entry_pages(entry, corpus_root, min_text_chars=min_text_chars)


def _iter_entry_pages(
    entry: PdfEntry,
    corpus_root: Path,
    *,
    min_text_chars: int,
) -> Iterator[PageInput]:
    absolute = corpus_root / entry.pdf_path
    if not absolute.exists():
        return
    yield from iter_pdf_pages(
        absolute,
        pdf_rel_path=entry.pdf_path,
        language=entry.language,
        min_text_chars=min_text_chars,
    )


def page_text_index(
    manifest: CorpusManifest,
    corpus_root: Path,
    *,
    needed_ids: set[str] | None = None,
) -> dict[str, str]:
    """Build a `{pdf#page=N: text}` lookup table for the ambiguity filter.

    If `needed_ids` is provided, only entries present in the set are returned —
    this lets the filter avoid loading pages that no query references.
    """
    out: dict[str, str] = {}
    for page in iter_manifest_pages(manifest, corpus_root):
        key = f"{page.pdf_path}#page={page.page_number}"
        if needed_ids is not None and key not in needed_ids:
            continue
        out[key] = page.text
    return out
