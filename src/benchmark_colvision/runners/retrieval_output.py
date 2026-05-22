"""Read the JSON written by `mmore colpali retrieve -o <output>` and convert
it to the per-query ranked lists the metrics module consumes.

The exact JSON layout produced by mmore PR #305 is one of:

* a list of `{query, results: [{document_id|doc_id|page_id|page, score, ...}, ...]}`
* a dict `{query -> [{doc_id, score}, ...]}`

This module tolerates both. Any unknown shape raises `ValueError` so callers
can fail fast rather than silently scoring nothing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path


_DOC_KEYS = ("document_id", "doc_id", "page_id", "page", "id")


def _doc_id_of(item: dict) -> str:
    for k in _DOC_KEYS:
        if k in item:
            return str(item[k])
    raise ValueError(f"no document-id key found in retrieval item: {sorted(item)}")


def _normalize_results(items: Sequence[dict]) -> list[str]:
    return [_doc_id_of(it) for it in items]


def load_retrieval(path: Path) -> dict[str, list[str]]:
    """Return `{query_id_or_text -> [doc_id_in_rank_order]}`."""
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        out: dict[str, list[str]] = {}
        for entry in raw:
            if not isinstance(entry, dict):
                raise ValueError(f"unexpected list entry: {entry!r}")
            key = entry.get("query_id") or entry.get("query")
            if key is None:
                raise ValueError("retrieval entry missing 'query_id' and 'query'")
            results = entry.get("results") or entry.get("documents") or []
            out[str(key)] = _normalize_results(results)
        return out
    if isinstance(raw, dict):
        return {str(k): _normalize_results(v) for k, v in raw.items()}
    raise ValueError(f"unsupported retrieval JSON top-level type: {type(raw).__name__}")


def align_to_queries(
    retrieval: dict[str, list[str]],
    query_ids_in_order: list[str],
    fallback_questions: list[str] | None = None,
) -> list[list[str]]:
    """Reorder the retrieval dict to match a canonical list of query ids.

    Missing queries get an empty list. If keys in `retrieval` are the *question
    strings* rather than ids, pass `fallback_questions` so we can match them.
    """
    if all(qid in retrieval for qid in query_ids_in_order):
        return [retrieval[qid] for qid in query_ids_in_order]
    if fallback_questions is None:
        return [retrieval.get(qid, []) for qid in query_ids_in_order]
    return [retrieval.get(q, []) for q in fallback_questions]
