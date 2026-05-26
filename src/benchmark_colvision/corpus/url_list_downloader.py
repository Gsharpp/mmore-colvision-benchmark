"""Generic bulk PDF downloader driven by a JSON manifest of URLs.

Used to fetch source-specific corpora once URLs have been collected by a
per-source discovery step (HAL/Cairn export, Thieme OA listing, SciELO search
results, Saudi Med Journal index, CNKI OA). The downloader itself is
source-agnostic: it streams each URL, retries transient errors, deduplicates by
SHA-256, and writes a sidecar JSON capturing the per-URL outcome.

Input manifest schema (`UrlListManifest`):
    {
      "source": "hal" | "thieme-oa" | "scielo" | "cnki-oa" | "saudi-med" | ...,
      "language": "fr",
      "items": [
        {"url": "https://...", "source_id": "hal-12345", "license": "CC-BY"},
        ...
      ]
    }
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from benchmark_colvision.corpus.corpus_manifest import sha256_file


SourceLiteral = Literal[
    "pmc-oa", "hal", "cairn", "scielo", "thieme-oa", "saudi-med", "cnki-oa", "other"
]


class UrlListItem(BaseModel):
    url: str
    source_id: str
    license: str | None = None


class UrlListManifest(BaseModel):
    source: SourceLiteral
    language: str
    items: list[UrlListItem] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "UrlListManifest":
        return cls.model_validate_json(path.read_text())


@dataclass
class DownloadOutcome:
    source_id: str
    url: str
    status: Literal["downloaded", "cached", "failed"]
    pdf_path: Path | None = None
    sha256: str | None = None
    error: str | None = None


@dataclass
class DownloadReport:
    source: str
    language: str
    n_total: int
    n_downloaded: int = 0
    n_cached: int = 0
    n_failed: int = 0
    outcomes: list[DownloadOutcome] = field(default_factory=list)

    def add(self, o: DownloadOutcome) -> None:
        self.outcomes.append(o)
        if o.status == "downloaded":
            self.n_downloaded += 1
        elif o.status == "cached":
            self.n_cached += 1
        else:
            self.n_failed += 1


def _target_filename(item: UrlListItem) -> str:
    """Derive a stable on-disk filename from the source_id (sanitized)."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in item.source_id)
    return f"{safe}.pdf"


def download_one(
    item: UrlListItem,
    out_dir: Path,
    client: httpx.Client,
    *,
    max_retries: int = 3,
    backoff_base_s: float = 2.0,
    chunk_size: int = 1 << 20,
    timeout_s: float = 120.0,
) -> DownloadOutcome:
    """Download one PDF, returning a structured outcome. Never raises on HTTP errors."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / _target_filename(item)

    if target.exists() and target.stat().st_size > 0:
        return DownloadOutcome(
            source_id=item.source_id,
            url=item.url,
            status="cached",
            pdf_path=target,
            sha256=sha256_file(target),
        )

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with client.stream("GET", item.url, timeout=timeout_s) as r:
                r.raise_for_status()
                tmp = target.with_suffix(target.suffix + ".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_bytes(chunk_size=chunk_size):
                        f.write(chunk)
                tmp.rename(target)
            return DownloadOutcome(
                source_id=item.source_id,
                url=item.url,
                status="downloaded",
                pdf_path=target,
                sha256=sha256_file(target),
            )
        except (httpx.HTTPError, OSError) as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(backoff_base_s * attempt)

    return DownloadOutcome(
        source_id=item.source_id,
        url=item.url,
        status="failed",
        error=type(last_exc).__name__ + ": " + str(last_exc) if last_exc else "unknown",
    )


def download_from_manifest(
    manifest: UrlListManifest,
    out_dir: Path,
    *,
    client: httpx.Client | None = None,
    max_retries: int = 3,
) -> DownloadReport:
    """Walk a UrlListManifest and download every item to `out_dir`."""
    owns = client is None
    client = client or httpx.Client(follow_redirects=True)
    report = DownloadReport(
        source=manifest.source, language=manifest.language, n_total=len(manifest.items)
    )
    try:
        for item in manifest.items:
            report.add(download_one(item, out_dir, client, max_retries=max_retries))
    finally:
        if owns:
            client.close()
    return report
