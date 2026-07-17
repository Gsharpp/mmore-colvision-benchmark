"""Reconstruct page-level text from mmore's own `process` OCR output.

ViDoRe's reconstructed PDFs (`corpus.vidore_v2`) carry no embedded text layer —
each page is a full-bleed PNG. `queries.pdf_pages` (PyMuPDF `get_text()`) always
returns empty strings for them. To get a genuine "mmore without ColVision"
baseline we run the actual `mmore process` pipeline (Marker + Surya OCR, see
`mmore.process.processors.pdf_processor.PDFProcessor`) with
`use_fast_processors: false`, which OCRs every page regardless of an embedded
text layer.

`mmore process` emits one `MultimodalSample` per PDF (not per page): the OCR'd
text of every page concatenated, plus `metadata.paragraph_starts` — a list of
`(char_offset, page_id, paragraph_index)` triples with **1-based** `page_id`
(`PDFProcessor._parse_pagination`), terminated by a `(end_offset, -1, -1)`
sentinel. We slice `text` at the page boundaries to recover one text blob per
page, keyed the same way as ViDoRe's qrels sidecar: `"<pdf_basename>#page=<N>"`
(basename with extension, 1-based page number — see `corpus.vidore_v2`).

A page with no detected text block (e.g. a genuinely blank separator slide)
has no entry in `paragraph_starts` and is silently absent from the output —
same effect as `pdf_pages.iter_pdf_pages`'s `min_text_chars` filter dropping a
page, just driven by OCR content instead of a length threshold.
"""

from __future__ import annotations

import json
from pathlib import Path


def _split_pages(text: str, paragraph_starts: list[list[int]]) -> dict[int, str]:
    """Slice `text` into `{page_id: page_text}` using paragraph-start offsets."""
    entries = sorted(paragraph_starts, key=lambda t: t[0])
    pages: dict[int, str] = {}
    for i, (offset, page_id, _para_idx) in enumerate(entries):
        if page_id == -1:
            continue
        next_offset = entries[i + 1][0] if i + 1 < len(entries) else len(text)
        pages[page_id] = pages.get(page_id, "") + text[offset:next_offset]
    return pages


def pages_from_ocr(merged_results_path: Path | str) -> tuple[list[str], list[str]]:
    """Return (doc_ids, texts) for every OCR'd page in a `mmore process` output.

    `merged_results_path` is the `merged/merged_results.jsonl` file `mmore
    process` writes (see `mmore.run_process.merged_results_path`).
    """
    doc_ids: list[str] = []
    texts: list[str] = []
    with open(merged_results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            text = sample.get("text", "")
            meta = sample.get("metadata", {})
            pdf_name = Path(meta.get("file_path", "")).name
            paragraph_starts = meta.get("paragraph_starts", [])
            for page_id, page_text in sorted(_split_pages(text, paragraph_starts).items()):
                doc_ids.append(f"{pdf_name}#page={page_id}")
                texts.append(page_text)
    return doc_ids, texts
