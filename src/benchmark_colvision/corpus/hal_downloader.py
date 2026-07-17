"""HAL (Hyper Articles en Ligne) open-access medical corpus builder.

Queries the HAL REST API for French-language medical articles with a
downloadable PDF, and produces a UrlListManifest consumable by
`bcv-corpus download-urls`.

HAL API reference: https://api.archives-ouvertes.fr/docs/search
Domain codes used:
  sdv.mhep  — Médecine humaine et pathologie
  sdv.spee  — Santé publique et épidémiologie
  sdv.imm   — Immunologie
  sdv.neu   — Neurosciences
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Iterator

import httpx

HAL_SEARCH_URL = "https://api.archives-ouvertes.fr/search/"

_MEDICAL_DOMAINS = [
    "1.sdv.mhep",   # Médecine humaine et pathologie
    "1.sdv.spee",   # Santé publique et épidémiologie
    "1.sdv.neu",    # Neurosciences
    "1.sdv.imm",    # Immunologie
]

_DEFAULT_QUERY = "medecine OR sante OR medical OR clinical"


def _build_domain_filter(domains: list[str]) -> str:
    return "(" + " OR ".join(f"domain_s:{d}" for d in domains) + ")"


def iter_hal_articles(
    client: httpx.Client,
    *,
    language: str = "fr",
    domains: list[str] | None = None,
    n: int = 100,
    rows_per_page: int = 50,
    seed: int = 0,
) -> Iterator[dict]:
    """Stream HAL article records with a PDF URL.

    Yields raw HAL document dicts (keys: halId_s, title_s, fileMain_s,
    licence_s, domain_s).
    """
    domains = domains or _MEDICAL_DOMAINS
    domain_fq = _build_domain_filter(domains)
    base_fq = [f"language_s:{language}", domain_fq, "openAccess_bool:true"]

    # HAL/Solr exposes no usable random-sort field (`random_<seed>` returns a
    # Solr error), so we sample reproducibly: count the result set once, then
    # pick a seeded start offset into a stable `docid asc` ordering and page
    # sequentially. Same seed → same window, different seed → different slice.
    count_params = {
        "q": _DEFAULT_QUERY,
        "fq": base_fq,
        "rows": 0,
        "wt": "json",
    }
    try:
        count_resp = client.get(HAL_SEARCH_URL, params=count_params, timeout=30.0)
        count_resp.raise_for_status()
        num_found = int(count_resp.json().get("response", {}).get("numFound", 0))
    except (httpx.HTTPError, ValueError) as e:
        print(f"[hal] count request failed: {e}", flush=True)
        num_found = 0

    # Over-fetch to absorb records that lack a downloadable PDF.
    window = min(num_found, n * 3)
    max_start = max(0, num_found - window)
    start = random.Random(seed).randint(0, max_start) if max_start > 0 else 0

    collected = 0
    fetched = 0
    while collected < n and fetched < window:
        rows = min(rows_per_page, window - fetched)
        params = {
            "q": _DEFAULT_QUERY,
            "fq": base_fq,
            "fl": "halId_s,title_s,files_s,licence_s,domain_s",
            "rows": rows,
            "start": start,
            "wt": "json",
            "sort": "docid asc",
        }
        try:
            resp = client.get(HAL_SEARCH_URL, params=params, timeout=30.0)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            print(f"[hal] HTTP error at start={start}: {e}", flush=True)
            time.sleep(2)
            break

        data = resp.json()
        docs = data.get("response", {}).get("docs", [])
        if not docs:
            break

        for doc in docs:
            # files_s is a list of direct PDF URLs; fileMain_s is an HTML landing page
            pdf_files = doc.get("files_s") or []
            pdf_url = next((u for u in pdf_files if u.lower().endswith(".pdf")), None)
            if pdf_url is None:
                continue
            yield {**doc, "fileMain_s": pdf_url}
            collected += 1
            if collected >= n:
                break

        start += len(docs)
        fetched += len(docs)
        if len(docs) < rows:
            break


def build_url_manifest(
    *,
    language: str = "fr",
    n: int = 100,
    domains: list[str] | None = None,
    seed: int = 0,
    client: httpx.Client | None = None,
) -> dict:
    """Return a UrlListManifest-compatible dict for HAL French medical PDFs."""
    owns_client = client is None
    client = client or httpx.Client(follow_redirects=True)
    items: list[dict] = []
    try:
        for doc in iter_hal_articles(
            client, language=language, domains=domains, n=n, seed=seed
        ):
            items.append(
                {
                    "url": doc["fileMain_s"],
                    "source_id": doc.get("halId_s", "unknown"),
                    "license": doc.get("licence_s"),
                }
            )
    finally:
        if owns_client:
            client.close()

    return {
        "source": "hal",
        "language": language,
        "items": items,
    }


def sample_and_save(
    out_manifest: Path,
    *,
    n: int = 100,
    language: str = "fr",
    seed: int = 0,
    domains: list[str] | None = None,
) -> int:
    """Fetch n article URLs from HAL and write a UrlListManifest JSON.

    Returns the number of items written.
    """
    manifest = build_url_manifest(n=n, language=language, seed=seed, domains=domains)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[hal] wrote {len(manifest['items'])} items → {out_manifest}", flush=True)
    return len(manifest["items"])
