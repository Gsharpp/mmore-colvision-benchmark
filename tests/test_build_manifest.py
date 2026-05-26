"""Tests for the corpus-wide manifest builder."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from benchmark_colvision.corpus.build_manifest import build_manifest, iter_pdfs


def _write_text_pdf(path: Path, text: str = "Hello.") -> Path:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), text)
    doc.save(path)
    doc.close()
    return path


def _write_image_heavy_pdf(path: Path) -> Path:
    """A PDF where the figure covers nearly the whole page (high visual density)."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    # Insert a large red rectangle as a filled shape — PyMuPDF will detect it as
    # an image when we draw a pixmap. For density we need something
    # `page.get_image_info()` flags. Use an actual embedded image.
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 500, 700), 0)
    pix.clear_with(255)
    page.insert_image(fitz.Rect(40, 40, 540, 740), pixmap=pix)
    doc.save(path)
    doc.close()
    return path


def test_iter_pdfs_lexicographic(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    _write_text_pdf(tmp_path / "b.pdf")
    _write_text_pdf(tmp_path / "a.pdf")
    _write_text_pdf(tmp_path / "sub" / "c.pdf")
    names = [p.name for p in iter_pdfs(tmp_path)]
    assert names == ["a.pdf", "b.pdf", "c.pdf"]


def test_build_manifest_keeps_dense_and_skips_sparse(tmp_path: Path) -> None:
    _write_text_pdf(tmp_path / "text.pdf")  # density ~ 0
    _write_image_heavy_pdf(tmp_path / "image.pdf")  # density ~ 1
    manifest, report = build_manifest(
        tmp_path,
        name="test",
        track="A",
        source="pmc-oa",
        density_threshold=0.30,
        language_override="en",
    )
    paths = {e.pdf_path for e in manifest.pdfs}
    assert "image.pdf" in paths
    assert "text.pdf" not in paths
    assert any("text.pdf" in s.pdf_path for s in report.skipped)
    assert manifest.total_pages == 1
    assert manifest.languages == ["en"]


def test_build_manifest_uses_language_override(tmp_path: Path) -> None:
    _write_image_heavy_pdf(tmp_path / "a.pdf")
    manifest, _ = build_manifest(
        tmp_path,
        name="t",
        track="B",
        source="hal",
        density_threshold=0.30,
        language_override="fr",
    )
    assert all(e.language == "fr" for e in manifest.pdfs)


def test_build_manifest_unknown_source_rejected_by_schema(tmp_path: Path) -> None:
    _write_image_heavy_pdf(tmp_path / "a.pdf")
    with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError
        build_manifest(
            tmp_path,
            name="t",
            track="A",
            source="not-a-source",  # type: ignore[arg-type]
            density_threshold=0.30,
            language_override="en",
        )
