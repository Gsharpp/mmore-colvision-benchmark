"""Sample native-language PMC OA articles using NCBI EUtils.

PMC OA hosts thousands of articles written *natively* in non-English
languages (Chinese ~6.8k, German ~13.5k, French ~10.5k). This module
queries the NCBI ESearch API with a `<Language>[Language]` filter to
collect their PMCIDs, then returns them for use as a filter in
`pmc_downloader.stream_package_index`.

These are NOT translations: each article is original native-language
literature, filtered by the article's recorded language in PMC.

Workflow:
    pmcids = fetch_pmc_ids_for_language("de", n=60, seed=0)
    pkgs   = [p for p in stream_package_index(client, pmcid_filter=set(pmcids))]
    download_packages(pkgs, cache_dir, pdf_out_dir)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import httpx

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

# NCBI's [Language] filter keys on the English language *name*, not the ISO code.
_LANGUAGE_NAMES = {
    "zh": "Chinese",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ru": "Russian",
    "ar": "Arabic",
}


def _query_for(language: str) -> str:
    name = _LANGUAGE_NAMES.get(language.lower())
    if name is None:
        raise ValueError(
            f"Unknown language code {language!r}; known: {sorted(_LANGUAGE_NAMES)}"
        )
    return f"{name}[Language] AND open access[filter]"


def fetch_pmc_ids_for_language(
    language: str = "zh",
    n: int = 60,
    *,
    seed: int = 0,
    client: httpx.Client | None = None,
    retmax_initial: int = 3000,
) -> list[str]:
    """Return up to `n` PMCIDs from native `language` OA articles.

    Fetches up to `retmax_initial` article UIDs from NCBI, converts to PMCIDs,
    and shuffles them deterministically (seeded) before taking the first `n`.
    """
    query = _query_for(language)
    owns = client is None
    client = client or httpx.Client(follow_redirects=True)
    try:
        resp = client.get(
            ESEARCH_URL,
            params={
                "db": "pmc",
                "term": query,
                "retmax": retmax_initial,
                "retmode": "json",
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        uids: list[str] = data.get("esearchresult", {}).get("idlist", [])
    except httpx.HTTPError as e:
        print(f"[pmc-{language}] EUtils error: {e}", flush=True)
        return []
    finally:
        if owns:
            client.close()

    pmcids = [f"PMC{uid}" for uid in uids]
    rng = random.Random(seed)
    rng.shuffle(pmcids)
    return pmcids[:n]


def fetch_chinese_pmc_ids(
    n: int = 60,
    *,
    seed: int = 0,
    client: httpx.Client | None = None,
    retmax_initial: int = 3000,
) -> list[str]:
    """Backward-compatible wrapper: Chinese-language OA PMCIDs."""
    return fetch_pmc_ids_for_language(
        "zh", n, seed=seed, client=client, retmax_initial=retmax_initial
    )


def sample_and_save(
    out_path: Path,
    *,
    language: str = "zh",
    n: int = 60,
    seed: int = 0,
) -> list[str]:
    """Fetch native `language` PMCIDs and save them as a JSON list."""
    pmcids = fetch_pmc_ids_for_language(language, n=n, seed=seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pmcids, indent=2))
    print(f"[pmc-{language}] saved {len(pmcids)} PMCIDs → {out_path}", flush=True)
    return pmcids
