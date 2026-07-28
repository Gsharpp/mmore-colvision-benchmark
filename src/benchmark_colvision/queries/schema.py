"""Schema for synthetic queries produced from the corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

# mmore colvision stores `page_num + 1` (colvision/run_process.py), so every doc
# id that reaches the scorer is 1-based. Relevance judgements must be expressed
# in that same space or a perfect run scores zero.
MMORE_PAGE_BASE = 1


class SyntheticQuery(BaseModel):
    """One auto-generated query tied to a single source page."""

    query_id: str
    question: str
    expected_answer: str
    requires_visual: bool = True
    source_pdf: str  # path inside the corpus, relative
    source_page: int  # numbered from `page_base`
    language: str
    judge_score: float | None = None  # ambiguity filter confidence
    accepted: bool = True
    # Numbering convention of `source_page`. PMC/HAL query generation walks pages
    # with `iter_manifest_pages(..., page_base=0)`, ViDoRe emits 1-based numbers.
    # Defaults to 0 so JSONL written before this field existed still loads with
    # the convention it was actually generated under.
    page_base: int = 0

    def mmore_doc_id(self) -> str:
        """The gold doc id in mmore's `<pdf>#page=<1-based>` id space."""
        return f"{self.source_pdf}#page={self.source_page - self.page_base + MMORE_PAGE_BASE}"


class QuerySet(BaseModel):
    """A versioned bundle of synthetic queries."""

    name: str
    language: str
    queries: list[SyntheticQuery] = Field(default_factory=list)

    def to_jsonl(self) -> str:
        return "\n".join(q.model_dump_json() for q in self.queries)

    def save_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_jsonl())

    @classmethod
    def load_jsonl(cls, path: Path, name: str, language: str) -> QuerySet:
        queries = [
            SyntheticQuery.model_validate_json(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        return cls(name=name, language=language, queries=queries)

    def sha256(self) -> str:
        """Stable hash over the JSONL serialization, for provenance in results."""
        data = self.to_jsonl().encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def relevance(self) -> dict[str, dict[str, float]]:
        """Map query_id -> {doc_id -> 1.0} for use with retrieval metrics."""
        out: dict[str, dict[str, float]] = {}
        for q in self.queries:
            doc_id = f"{q.source_pdf}#page={q.source_page}"
            out[q.query_id] = {doc_id: 1.0}
        return out


def page_doc_id(pdf_path: str, page_number: int) -> str:
    return f"{pdf_path}#page={page_number}"


def query_id_for(pdf_path: str, page_number: int, idx: int) -> str:
    """Deterministic short id: hash(pdf_path+page) + ordinal."""
    h = hashlib.sha1(f"{pdf_path}:{page_number}".encode()).hexdigest()[:10]
    return f"q-{h}-{idx}"


def relevance_from_jsonl(path: Path) -> dict[str, dict[str, float]]:
    """Convenience: load a JSONL file and return the relevance map."""
    qs = QuerySet.load_jsonl(path, name=path.stem, language="unknown")
    return qs.relevance()


def queries_to_serialized(qset: QuerySet) -> bytes:
    """Canonical bytes for hashing — JSONL with sorted keys."""
    lines = [json.dumps(q.model_dump(), sort_keys=True) for q in qset.queries]
    return "\n".join(lines).encode("utf-8")
