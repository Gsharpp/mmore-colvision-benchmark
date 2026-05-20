"""Unit tests for corpus.corpus_manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark_colvision.corpus.corpus_manifest import (
    CorpusManifest,
    PdfEntry,
    build_entry,
    sha256_file,
    verify_manifest,
)


def test_sha256_file_is_deterministic(tmp_path: Path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello mmore")
    h1 = sha256_file(p)
    h2 = sha256_file(p)
    assert h1 == h2
    assert len(h1) == 64


def test_pdf_entry_visual_density_bounds():
    with pytest.raises(Exception):
        PdfEntry(
            pdf_path="x.pdf",
            sha256="0" * 64,
            page_count=1,
            language="en",
            visual_density=1.5,  # > 1.0 must fail
            source="pmc-oa",
        )


def test_manifest_roundtrip(tmp_path: Path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    entry = build_entry(
        pdf_path=pdf,
        language="en",
        visual_density=0.42,
        source="pmc-oa",
        page_count=3,
        mesh_tags=["A01"],
    )
    manifest = CorpusManifest(
        name="test",
        track="A",
        languages=["en"],
        total_pages=3,
        pdfs=[entry],
    )
    out = tmp_path / "manifest.json"
    manifest.save(out)
    loaded = CorpusManifest.load(out)
    assert loaded.name == "test"
    assert loaded.pdfs[0].sha256 == entry.sha256


def test_verify_manifest_detects_drift(tmp_path: Path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"original")
    entry = build_entry(pdf, "en", 0.5, "pmc-oa", page_count=1)
    manifest = CorpusManifest(
        name="t", track="A", languages=["en"], total_pages=1, pdfs=[entry]
    )
    assert verify_manifest(manifest, tmp_path) == []

    pdf.write_bytes(b"tampered")
    drift = verify_manifest(manifest, tmp_path)
    assert any("hash mismatch" in d for d in drift)
