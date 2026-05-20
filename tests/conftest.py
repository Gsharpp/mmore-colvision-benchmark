"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_pdf(tmp_path: Path) -> Path:
    """Tiny one-page PDF generated via PyMuPDF for tests that need a real file."""
    import fitz

    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Hello, mmore.")
    doc.save(pdf_path)
    doc.close()
    return pdf_path
