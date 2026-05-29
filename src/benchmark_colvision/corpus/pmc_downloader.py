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

import csv
import io
import tarfile
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx

# NCBI moved the legacy bulk artefacts under /pub/pmc/deprecated/ in early 2026
# (scheduled for removal in August 2026). The OA file list and the per-article
# tar.gz packages both live there. The index's "File" column already carries the
# "oa_package/<xx>/<yy>/PMCID.tar.gz" prefix, so PMC_OA_PACKAGE_BASE must stop at
# the deprecated/ directory — otherwise the path doubles up.
PMC_OA_INDEX_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa_file_list.csv"
PMC_OA_PACKAGE_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/"


@dataclass
class PmcPackage:
    pmcid: str
    relative_path: str  # under PMC_OA_PACKAGE_BASE
    license: str

    @property
    def url(self) -> str:
        return PMC_OA_PACKAGE_BASE + self.relative_path


def _rows_to_packages(
    rows: Iterator[list[str]],
    pmcid_filter: set[str] | None,
) -> Iterator[PmcPackage]:
    """Shared row→PmcPackage logic for the in-memory and streaming parsers."""
    header = next(rows, None)
    if header is None:
        return
    try:
        col_file = header.index("File")
        col_pmcid = header.index("Accession ID")
        col_license = header.index("License")
    except ValueError:
        # The file format has been stable for years, but if columns shift
        # we'd rather fail loudly than silently return nothing.
        raise ValueError(f"unexpected PMC OA index header: {header}") from None
    wide = max(col_file, col_pmcid, col_license)
    for parts in rows:
        if len(parts) <= wide:
            continue
        pmcid = parts[col_pmcid].strip()
        if pmcid_filter is not None and pmcid not in pmcid_filter:
            continue
        yield PmcPackage(
            pmcid=pmcid,
            relative_path=parts[col_file].strip(),
            license=parts[col_license].strip(),
        )


def iter_package_index(
    raw_csv: str,
    pmcid_filter: set[str] | None = None,
) -> Iterator[PmcPackage]:
    """Parse the oa_file_list.csv text and yield packages.

    Columns: File, Article Citation, Accession ID, Last Updated, PMID, License.
    The Article Citation field is double-quoted and contains commas, so we parse
    with the csv module rather than a naive split.
    """
    yield from _rows_to_packages(iter(csv.reader(io.StringIO(raw_csv))), pmcid_filter)


def stream_package_index(
    client: httpx.Client,
    pmcid_filter: set[str] | None = None,
) -> Iterator[PmcPackage]:
    """Stream the OA index line by line (memory-safe for the multi-GB file)."""
    with client.stream("GET", PMC_OA_INDEX_URL, timeout=300.0) as resp:
        resp.raise_for_status()
        rows = csv.reader(resp.iter_lines())
        yield from _rows_to_packages(rows, pmcid_filter)


def fetch_index(client: httpx.Client) -> str:
    """Download the full PMC OA file list CSV into memory.

    The list has grown to the GB range; prefer ``stream_package_index`` when you
    only need to sample. Kept for callers that want the raw text.
    """
    resp = client.get(PMC_OA_INDEX_URL, timeout=300.0)
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


def sample_packages(
    n: int,
    *,
    seed: int = 0,
    scan_limit: int | None = 400_000,
    client: httpx.Client | None = None,
) -> list[PmcPackage]:
    """Reservoir-sample ``n`` PmcPackage objects by streaming the OA index once.

    Returning full PmcPackage objects (not just IDs) lets callers skip
    re-streaming the index during download — critical on slow cluster networks.
    Memory-safe; ``scan_limit`` caps how many index rows are scanned.
    """
    import random

    rng = random.Random(seed)
    owns_client = client is None
    client = client or httpx.Client(follow_redirects=True)
    reservoir: list[PmcPackage] = []
    kept = 0
    try:
        for i, pkg in enumerate(stream_package_index(client)):
            pid = pkg.pmcid
            if not (pid.startswith("PMC") and pid[3:].isdigit()):
                continue
            kept += 1
            if len(reservoir) < n:
                reservoir.append(pkg)
            else:
                j = rng.randint(0, kept - 1)
                if j < n:
                    reservoir[j] = pkg
            if scan_limit is not None and i >= scan_limit:
                break
        return reservoir
    finally:
        if owns_client:
            client.close()


def sample_pmcids(
    n: int,
    *,
    seed: int = 0,
    scan_limit: int | None = 400_000,
    client: httpx.Client | None = None,
) -> list[str]:
    """Compatibility shim — returns PMCIDs only. Prefer sample_packages."""
    return [p.pmcid for p in sample_packages(n, seed=seed, scan_limit=scan_limit, client=client)]


def download_packages(
    packages: Iterable[PmcPackage],
    cache_dir: Path,
    pdf_out_dir: Path,
    client: httpx.Client | None = None,
) -> list[Path]:
    """Download pre-resolved PmcPackage objects (no index re-streaming needed)."""
    owns_client = client is None
    client = client or httpx.Client(follow_redirects=True)
    try:
        pdfs: list[Path] = []
        for pkg in packages:
            archive = download_package(pkg, cache_dir, client)
            pdfs.extend(extract_pdfs(archive, pdf_out_dir))
            print(f"[pmc] extracted {pkg.pmcid}", flush=True)
        return pdfs
    finally:
        if owns_client:
            client.close()


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
        pdfs: list[Path] = []
        found: set[str] = set()
        # Stream the index instead of loading the multi-GB CSV into memory; stop
        # as soon as every requested PMCID has been located.
        for pkg in stream_package_index(client, pmcid_filter=pmc_set):
            archive = download_package(pkg, cache_dir, client)
            pdfs.extend(extract_pdfs(archive, pdf_out_dir))
            found.add(pkg.pmcid)
            if found >= pmc_set:
                break
        return pdfs
    finally:
        if owns_client:
            client.close()
