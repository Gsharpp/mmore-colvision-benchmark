"""Build a mmore-ready corpus from a ViDoRe v2 HuggingFace dataset.

ViDoRe v2 ships page *images* plus human-validated, vision-grounded queries with
multi-page **graded** relevance judgements (qrels). Adopting it removes the two
biases of the home-grown PMC/HAL corpus at once: the documents are figure-rich
(lecture slides) and the queries were written by looking at the page, not the
extracted text.

Rather than teach mmore to ingest raw images, we reconstruct one PDF per
document (one image per page) so the existing `colvision process → index →
retrieve` pipeline runs unchanged. We then emit a graded qrels sidecar that the
scorer consumes for multi-relevant nDCG.

doc-id convention — the qrels sidecar uses the *exact* string mmore emits at
retrieval time so the two align:

* basename **with** extension: `colvision/retriever.py` sets
  `pdf_name = Path(pdf_path).name`;
* **1-based** page number: `colvision/run_process.py` stores `page_num + 1`.

Reconstructed pages are sized so that mmore's 200-DPI re-render reproduces the
original pixel dimensions (page_points = pixels * 72 / dpi), keeping the
round-trip visually faithful.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from PIL import Image

from benchmark_colvision.corpus.corpus_manifest import (
    CorpusManifest,
    PdfEntry,
    sha256_file,
)
from benchmark_colvision.queries.schema import QuerySet, SyntheticQuery

DEFAULT_REPO = "vidore/biomedical_lectures_eng_v2"
MMORE_RENDER_DPI = 200  # colvision/run_process.py PDFConverter default
MMORE_PAGE_BASE = 1  # colvision/run_process.py stores page_num + 1
_PDF_NAME_FMT = "doc_{:04d}.pdf"


DEFAULT_MULTILINGUAL_REPO = "vidore/biomedical_lectures_v2"


@dataclass
class BuildResult:
    out_dir: Path
    pdfs_dir: Path
    manifest_path: Path
    queries_path: Path
    qrels_path: Path
    n_docs: int
    n_pages: int
    n_queries: int
    n_qrels: int
    n_queries_dropped: int

    def summary(self) -> str:
        return (
            f"{self.n_docs} docs / {self.n_pages} pages, "
            f"{self.n_queries} queries ({self.n_qrels} qrels), "
            f"{self.n_queries_dropped} queries dropped (no positive qrel)"
        )


def _to_rgb(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGB" else image.convert("RGB")


def _add_image_page(pdf: fitz.Document, image: Image.Image, *, dpi: int) -> None:
    """Append `image` as a full-bleed page sized for a lossless dpi re-render."""
    image = _to_rgb(image)
    width_pt = image.width * 72.0 / dpi
    height_pt = image.height * 72.0 / dpi
    page = pdf.new_page(width=width_pt, height=height_pt)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    page.insert_image(page.rect, stream=buf.getvalue())


def build_corpus_from_records(
    corpus_rows: Iterable[Mapping[str, Any]],
    query_rows: Iterable[Mapping[str, Any]],
    qrel_rows: Iterable[Mapping[str, Any]],
    out_dir: Path | str,
    *,
    language: str = "en",
    manifest_name: str = "vidore",
    source_tag: str = "vidore",
    min_score: int = 1,
    dpi: int = MMORE_RENDER_DPI,
) -> BuildResult:
    """Materialise PDFs + manifest + queries JSONL + graded qrels sidecar.

    Rows follow the ViDoRe v2 schema:
      * corpus: ``{"corpus-id": int, "doc-id": str, "image": PIL.Image}``
      * queries: ``{"query-id": int, "query": str}``
      * qrels: ``{"query-id": int, "corpus-id": int, "score": int}``

    A query with no positive qrel (score >= ``min_score``) is dropped — it cannot
    be scored and would otherwise deflate the macro-averaged metrics.
    """
    out_dir = Path(out_dir)
    pdfs_dir = out_dir / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    # Group page images by document; preserve slide order via corpus-id.
    by_doc: dict[str, list[tuple[int, Image.Image]]] = {}
    for row in corpus_rows:
        by_doc.setdefault(str(row["doc-id"]), []).append(
            (int(row["corpus-id"]), row["image"])
        )

    corpusid_to_docid: dict[int, str] = {}  # corpus-id -> "<pdf>#page=<1-based>"
    entries: list[PdfEntry] = []
    total_pages = 0
    for idx, doc_id in enumerate(sorted(by_doc)):
        pdf_name = _PDF_NAME_FMT.format(idx)
        pages = sorted(by_doc[doc_id], key=lambda pair: pair[0])
        pdf = fitz.open()
        try:
            for page_pos, (corpus_id, image) in enumerate(pages):
                _add_image_page(pdf, image, dpi=dpi)
                page_number = page_pos + MMORE_PAGE_BASE
                corpusid_to_docid[corpus_id] = f"{pdf_name}#page={page_number}"
            pdf_path = pdfs_dir / pdf_name
            pdf.save(str(pdf_path))
        finally:
            pdf.close()
        total_pages += len(pages)
        entries.append(
            PdfEntry(
                pdf_path=pdf_name,
                sha256=sha256_file(pdf_path),
                page_count=len(pages),
                language=language,
                visual_density=1.0,  # full-page images: visual by construction
                mesh_tags=[],
                source=source_tag,  # type: ignore[arg-type]
                source_id=doc_id,
            )
        )

    manifest = CorpusManifest(
        name=manifest_name,
        track="A",
        languages=[language],
        total_pages=total_pages,
        pdfs=entries,
    )
    manifest_path = out_dir / "corpus_manifest.json"
    manifest.save(manifest_path)

    # Build graded relevance keyed by our string query ids.
    questions = {int(r["query-id"]): str(r["query"]) for r in query_rows}
    graded: dict[int, dict[str, float]] = {}
    for row in qrel_rows:
        score = float(row.get("score", 0) or 0)
        if score < min_score:
            continue
        doc_id = corpusid_to_docid.get(int(row["corpus-id"]))
        if doc_id is None:
            continue  # qrel points at a page absent from the corpus split
        graded.setdefault(int(row["query-id"]), {})[doc_id] = score

    syn_queries: list[SyntheticQuery] = []
    qrels_out: dict[str, dict[str, float]] = {}
    dropped = 0
    n_qrels = 0
    for qid in sorted(questions):
        relevance = graded.get(qid)
        if not relevance:
            dropped += 1
            continue
        query_id = f"vq-{qid}"
        top_doc = max(relevance.items(), key=lambda kv: kv[1])[0]
        top_pdf, _, top_page = top_doc.partition("#page=")
        syn_queries.append(
            SyntheticQuery(
                query_id=query_id,
                question=questions[qid],
                expected_answer="",
                requires_visual=True,
                source_pdf=top_pdf,
                source_page=int(top_page),
                language=language,
            )
        )
        qrels_out[query_id] = relevance
        n_qrels += len(relevance)

    queryset = QuerySet(name=manifest_name, language=language, queries=syn_queries)
    queries_path = out_dir / "queries.jsonl"
    queryset.save_jsonl(queries_path)
    qrels_path = out_dir / "qrels.json"
    qrels_path.write_text(json.dumps(qrels_out, indent=2, ensure_ascii=False))

    return BuildResult(
        out_dir=out_dir,
        pdfs_dir=pdfs_dir,
        manifest_path=manifest_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        n_docs=len(entries),
        n_pages=total_pages,
        n_queries=len(syn_queries),
        n_qrels=n_qrels,
        n_queries_dropped=dropped,
    )


def download_and_build(
    out_dir: Path | str,
    *,
    repo: str = DEFAULT_REPO,
    split: str = "test",
    language: str = "en",
    manifest_name: str | None = None,
    cache_dir: str | None = None,
    min_score: int = 1,
    dpi: int = MMORE_RENDER_DPI,
) -> BuildResult:
    """Download a ViDoRe v2 dataset from the Hub and build the mmore corpus."""
    from datasets import load_dataset

    corpus = load_dataset(repo, "corpus", split=split, cache_dir=cache_dir)
    queries = load_dataset(repo, "queries", split=split, cache_dir=cache_dir)
    qrels = load_dataset(repo, "qrels", split=split, cache_dir=cache_dir)
    return build_corpus_from_records(
        corpus_rows=corpus,
        query_rows=queries,
        qrel_rows=qrels,
        out_dir=out_dir,
        language=language,
        manifest_name=manifest_name or repo.split("/")[-1],
        min_score=min_score,
        dpi=dpi,
    )


def _corpusid_to_docid(corpus_rows: Iterable[Mapping[str, Any]]) -> dict[int, str]:
    """Recompute the deterministic ``corpus-id -> "<pdf>#page=<1-based>"`` mapping.

    Mirrors the PDF materialisation order in `build_corpus_from_records` (doc-ids
    sorted lexicographically, pages ordered by corpus-id) without touching the
    `image` column — so it stays cheap even on the un-decoded multilingual
    `corpus` config, and lets a language-only query/qrels build (no PDFs written)
    align with PDFs materialised earlier from a row-identical `corpus` config.
    """
    by_doc: dict[str, list[int]] = {}
    for row in corpus_rows:
        by_doc.setdefault(str(row["doc-id"]), []).append(int(row["corpus-id"]))
    mapping: dict[int, str] = {}
    for idx, doc_id in enumerate(sorted(by_doc)):
        pdf_name = _PDF_NAME_FMT.format(idx)
        for page_pos, corpus_id in enumerate(sorted(by_doc[doc_id])):
            mapping[corpus_id] = f"{pdf_name}#page={page_pos + MMORE_PAGE_BASE}"
    return mapping


@dataclass
class LanguageQueryResult:
    out_dir: Path
    queries_path: Path
    qrels_path: Path
    language: str
    n_queries: int
    n_qrels: int
    n_queries_dropped: int

    def summary(self) -> str:
        return (
            f"[{self.language}] {self.n_queries} queries ({self.n_qrels} qrels), "
            f"{self.n_queries_dropped} queries dropped (no positive qrel)"
        )


def build_language_queryset(
    corpus_rows: Iterable[Mapping[str, Any]],
    query_rows: Iterable[Mapping[str, Any]],
    qrel_rows: Iterable[Mapping[str, Any]],
    out_dir: Path | str,
    *,
    language: str,
    manifest_name: str = "vidore",
    min_score: int = 1,
) -> LanguageQueryResult:
    """Build ``queries.jsonl`` + ``qrels.json`` for one query-language of the ViDoRe
    v2 multilingual release, against a corpus **already materialised as PDFs** by
    `build_corpus_from_records` — no PDFs or manifest are (re)written here. This is
    what lets Track B re-retrieve the translated queries on the very same Milvus
    index built once for Track A, instead of re-embedding a per-language corpus.

    Rows follow the ViDoRe v2 multilingual schema:
      * corpus: ``{"corpus-id": int, "doc-id": str, "image": PIL.Image}`` (image
        column is never accessed here, only used to recompute the doc-id mapping)
      * queries: ``{"query-id": int, "query": str, "language": str, ...}``
      * qrels: ``{"query-id": int, "corpus-id": int, "score": int, ...}``

    ``language`` matches the HF ``language`` column values (case-insensitive):
    ``"english"``, ``"french"``, ``"german"``, ``"spanish"``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpusid_to_docid = _corpusid_to_docid(corpus_rows)

    questions = {
        int(r["query-id"]): str(r["query"])
        for r in query_rows
        if str(r.get("language", "")).lower() == language.lower()
    }
    graded: dict[int, dict[str, float]] = {}
    for row in qrel_rows:
        qid = int(row["query-id"])
        if qid not in questions:
            continue
        score = float(row.get("score", 0) or 0)
        if score < min_score:
            continue
        doc_id = corpusid_to_docid.get(int(row["corpus-id"]))
        if doc_id is None:
            continue  # qrel points at a page absent from the corpus split
        graded.setdefault(qid, {})[doc_id] = score

    syn_queries: list[SyntheticQuery] = []
    qrels_out: dict[str, dict[str, float]] = {}
    dropped = 0
    n_qrels = 0
    for qid in sorted(questions):
        relevance = graded.get(qid)
        if not relevance:
            dropped += 1
            continue
        query_id = f"vq-{qid}"
        top_doc = max(relevance.items(), key=lambda kv: kv[1])[0]
        top_pdf, _, top_page = top_doc.partition("#page=")
        syn_queries.append(
            SyntheticQuery(
                query_id=query_id,
                question=questions[qid],
                expected_answer="",
                requires_visual=True,
                source_pdf=top_pdf,
                source_page=int(top_page),
                language=language,
            )
        )
        qrels_out[query_id] = relevance
        n_qrels += len(relevance)

    queryset = QuerySet(name=manifest_name, language=language, queries=syn_queries)
    queries_path = out_dir / "queries.jsonl"
    queryset.save_jsonl(queries_path)
    qrels_path = out_dir / "qrels.json"
    qrels_path.write_text(json.dumps(qrels_out, indent=2, ensure_ascii=False))

    return LanguageQueryResult(
        out_dir=out_dir,
        queries_path=queries_path,
        qrels_path=qrels_path,
        language=language,
        n_queries=len(syn_queries),
        n_qrels=n_qrels,
        n_queries_dropped=dropped,
    )


def download_and_build_language(
    out_dir: Path | str,
    *,
    language: str,
    repo: str = DEFAULT_MULTILINGUAL_REPO,
    split: str = "test",
    manifest_name: str = "vidore",
    cache_dir: str | None = None,
    min_score: int = 1,
) -> LanguageQueryResult:
    """Download one query-language slice of the ViDoRe v2 multilingual release."""
    from datasets import load_dataset

    corpus = load_dataset(repo, "corpus", split=split, cache_dir=cache_dir)
    corpus = corpus.remove_columns([c for c in corpus.column_names if c == "image"])
    queries = load_dataset(repo, "queries", split=split, cache_dir=cache_dir)
    qrels = load_dataset(repo, "qrels", split=split, cache_dir=cache_dir)
    return build_language_queryset(
        corpus_rows=corpus,
        query_rows=queries,
        qrel_rows=qrels,
        out_dir=out_dir,
        language=language,
        manifest_name=manifest_name,
        min_score=min_score,
    )


def load_qrels(path: Path | str) -> dict[str, dict[str, float]]:
    """Load the graded qrels sidecar as ``{query_id: {doc_id: grade}}``."""
    return {
        str(qid): {str(doc): float(grade) for doc, grade in rel.items()}
        for qid, rel in json.loads(Path(path).read_text()).items()
    }
