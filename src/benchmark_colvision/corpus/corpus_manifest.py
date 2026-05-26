"""JSON manifest schema for the corpus, including SHA256 hashes for reproducibility."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class PdfEntry(BaseModel):
    pdf_path: str
    sha256: str
    page_count: int
    language: str
    visual_density: float = Field(ge=0.0, le=1.0)
    mesh_tags: list[str] = Field(default_factory=list)
    source: Literal["pmc-oa", "hal", "cairn", "scielo", "thieme-oa", "saudi-med", "cnki-oa", "other"]
    source_id: str | None = None


class CorpusManifest(BaseModel):
    name: str
    track: Literal["A", "B"]
    languages: list[str]
    total_pages: int
    pdfs: list[PdfEntry]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> CorpusManifest:
        return cls.model_validate_json(path.read_text())


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, computed in streaming."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def build_entry(
    pdf_path: Path,
    language: str,
    visual_density: float,
    source: str,
    page_count: int,
    mesh_tags: list[str] | None = None,
    source_id: str | None = None,
) -> PdfEntry:
    return PdfEntry(
        pdf_path=str(pdf_path),
        sha256=sha256_file(pdf_path),
        page_count=page_count,
        language=language,
        visual_density=visual_density,
        mesh_tags=mesh_tags or [],
        source=source,  # type: ignore[arg-type]
        source_id=source_id,
    )


def verify_manifest(manifest: CorpusManifest, base_dir: Path) -> list[str]:
    """Re-hash every PDF in the manifest and return paths whose hash drifted."""
    drift: list[str] = []
    for entry in manifest.pdfs:
        path = base_dir / entry.pdf_path
        if not path.exists():
            drift.append(f"missing: {entry.pdf_path}")
            continue
        actual = sha256_file(path)
        if actual != entry.sha256:
            drift.append(f"hash mismatch: {entry.pdf_path}")
    return drift
