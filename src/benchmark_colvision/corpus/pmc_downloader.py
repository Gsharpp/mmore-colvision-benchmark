"""PubMed Central Open Access subset downloader.

PMC publishes the OA subset as a set of tar.gz packages indexed by an XML
listing. This module streams the listing, picks packages that match a MeSH
prefix filter, downloads each package to a cache directory, and extracts the
PDFs to a flat output directory.

Network calls go through `httpx`; failures are retried with exponential
backoff. Nothing on disk is overwritten if the SHA-256 matches an existing
entry — re-runs are idempotent.
"""

from __future__ import annotations

import tarfile
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx

PMC_OA_INDEX_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/oa_file_list.csv"
PMC_OA_PACKAGE_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/"


@dataclass
class PmcPackage:
    pmcid: str
    relative_path: str  # under PMC_OA_PACKAGE_BASE
    license: str

    @property
    def url(self) -> str:
        return PMC_OA_PACKAGE_BASE + self.relative_path


def iter_package_index(
    raw_csv: str,
    pmcid_filter: set[str] | None = None,
) -> Iterator[PmcPackage]:
    """Parse the oa_file_list.csv format and yield packages.

    Columns: File, Article Citation, Accession ID, Last Updated, PMID, License.
    """
    lines = raw_csv.splitlines()
    if not lines:
        return
    header = lines[0].split(",")
    try:
        col_file = header.index("File")
        col_pmcid = header.index("Accession ID")
        col_license = header.index("License")
    except ValueError:
        # The file format has been stable for years, but if columns shift
        # we'd rather fail loudly than silently return nothing.
        raise ValueError(f"unexpected PMC OA index header: {header}") from None
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) <= max(col_file, col_pmcid, col_license):
            continue
        pmcid = parts[col_pmcid].strip()
        if pmcid_filter is not None and pmcid not in pmcid_filter:
            continue
        yield PmcPackage(
            pmcid=pmcid,
            relative_path=parts[col_file].strip(),
            license=parts[col_license].strip(),
        )


def fetch_index(client: httpx.Client) -> str:
    """Download the PMC OA file list CSV (~tens of MB)."""
    resp = client.get(PMC_OA_INDEX_URL, timeout=60.0)
    resp.raise_for_status()
    return resp.text


def download_package(
    pkg: PmcPackage,
    cache_dir: Path,
    client: httpx.Client,
    max_retries: int = 3,
    backoff_s: float = 2.0,
) -> Path:
    """Download a single tar.gz package, caching to disk. Returns the local path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / pkg.relative_path.split("/")[-1]
    if local.exists() and local.stat().st_size > 0:
        return local
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with client.stream("GET", pkg.url, timeout=120.0) as r:
                r.raise_for_status()
                tmp = local.with_suffix(local.suffix + ".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_bytes(chunk_size=1 << 20):
                        f.write(chunk)
                tmp.rename(local)
            return local
        except (httpx.HTTPError, OSError) as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(backoff_s * attempt)
    raise RuntimeError(f"failed to download {pkg.url}") from last_exc


def extract_pdfs(archive_path: Path, out_dir: Path) -> list[Path]:
    """Extract every *.pdf inside a tar(.gz) package to `out_dir` (flat layout)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with tarfile.open(archive_path, "r:*") as tf:
        for member in tf.getmembers():
            if not member.isreg():
                continue
            if not member.name.lower().endswith(".pdf"):
                continue
            target = out_dir / Path(member.name).name
            with tf.extractfile(member) as src:
                if src is None:
                    continue
                target.write_bytes(src.read())
            extracted.append(target)
    return extracted


def parse_mesh_terms_from_nxml(nxml_text: str) -> list[str]:
    """Best-effort MeSH extraction from a PMC NXML article (used for stratification)."""
    try:
        root = ET.fromstring(nxml_text)
    except ET.ParseError:
        return []
    tags: list[str] = []
    for node in root.iter():
        if node.tag.endswith("kwd"):
            text = (node.text or "").strip()
            if text:
                tags.append(text)
    return tags


def download_corpus(
    pmcids: Iterable[str],
    cache_dir: Path,
    pdf_out_dir: Path,
    client: httpx.Client | None = None,
) -> list[Path]:
    """High-level entry point. Returns the list of extracted PDF paths."""
    owns_client = client is None
    client = client or httpx.Client(follow_redirects=True)
    pmc_set = set(pmcids)
    try:
        index_csv = fetch_index(client)
        pdfs: list[Path] = []
        for pkg in iter_package_index(index_csv, pmcid_filter=pmc_set):
            archive = download_package(pkg, cache_dir, client)
            pdfs.extend(extract_pdfs(archive, pdf_out_dir))
        return pdfs
    finally:
        if owns_client:
            client.close()
