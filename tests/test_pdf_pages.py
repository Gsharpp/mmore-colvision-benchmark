"""Tests for `queries.pdf_pages` — corpus walk that yields PageInput objects."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from benchmark_colvision.corpus.corpus_manifest import (
    CorpusManifest,
    PdfEntry,
    sha256_file,
)
from benchmark_colvision.queries.pdf_pages import (
    iter_manifest_pages,
    iter_pdf_pages,
    page_text_index,
)


@pytest.fixture
def two_page_pdf(tmp_path: Path) -> Path:
    """A PDF with one text-rich page and one empty page (should be skipped)."""
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    page0 = doc.new_page(width=595, height=842)
    page0.insert_text(
        (72, 100),
        "Lorem ipsum dolor sit amet consectetur adipiscing elit "
        "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    )
    doc.new_page(width=595, height=842)  # empty page
    doc.save(pdf)
    doc.close()
    return pdf


def test_iter_pdf_pages_skips_empty(two_page_pdf: Path) -> None:
    pages = list(
        iter_pdf_pages(
            two_page_pdf, pdf_rel_path="doc.pdf", language="en", min_text_chars=80
        )
    )
    assert len(pages) == 1
    p = pages[0]
    assert p.pdf_path == "doc.pdf"
    assert p.page_number == 0
    assert p.language == "en"
    assert "Lorem" in p.text


def test_iter_pdf_pages_min_text_can_be_relaxed(two_page_pdf: Path) -> None:
    pages = list(
        iter_pdf_pages(
            two_page_pdf, pdf_rel_path="doc.pdf", language="en", min_text_chars=0
        )
    )
    # Empty page still yields nothing since text is "" → len(strip())==0,
    # but the threshold is now 0 so it should be kept. PyMuPDF returns "" for
    # a blank page → " ".strip() == "" → length 0 → still >= 0 → kept.
    assert len(pages) == 2


def test_iter_manifest_pages(two_page_pdf: Path, tmp_path: Path) -> None:
    entry = PdfEntry(
        pdf_path="doc.pdf",
        sha256=sha256_file(two_page_pdf),
        page_count=2,
        language="en",
        visual_density=0.0,
        mesh_tags=[],
        source="pmc-oa",
    )
    manifest = CorpusManifest(
        name="test", track="A", languages=["en"], total_pages=2, pdfs=[entry]
    )
    pages = list(iter_manifest_pages(manifest, tmp_path))
    assert len(pages) == 1
    assert pages[0].pdf_path == "doc.pdf"


def test_iter_manifest_pages_skips_missing(tmp_path: Path) -> None:
    entry = PdfEntry(
        pdf_path="missing.pdf",
        sha256="0" * 64,
        page_count=1,
        language="en",
        visual_density=0.0,
        mesh_tags=[],
        source="pmc-oa",
    )
    manifest = CorpusManifest(
        name="t", track="A", languages=["en"], total_pages=1, pdfs=[entry]
    )
    assert list(iter_manifest_pages(manifest, tmp_path)) == []


def test_page_text_index_respects_needed_ids(two_page_pdf: Path, tmp_path: Path) -> None:
    entry = PdfEntry(
        pdf_path="doc.pdf",
        sha256=sha256_file(two_page_pdf),
        page_count=2,
        language="en",
        visual_density=0.0,
        mesh_tags=[],
        source="pmc-oa",
    )
    manifest = CorpusManifest(
        name="t", track="A", languages=["en"], total_pages=2, pdfs=[entry]
    )
    idx_all = page_text_index(manifest, tmp_path)
    assert set(idx_all.keys()) == {"doc.pdf#page=0"}

    idx_filtered = page_text_index(manifest, tmp_path, needed_ids={"other#page=0"})
    assert idx_filtered == {}
