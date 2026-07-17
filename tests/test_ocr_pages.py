"""Tests for reconstructing page-level text from mmore `process` OCR output."""

from __future__ import annotations

import json

from benchmark_colvision.corpus.ocr_pages import _split_pages, pages_from_ocr


def test_split_pages_contiguous_paragraphs():
    text = "page one text||page two text||page three text"
    paragraph_starts = [
        [0, 1, 0],
        [15, 2, 0],
        [30, 3, 0],
        [len(text), -1, -1],
    ]
    pages = _split_pages(text, paragraph_starts)
    assert pages == {
        1: "page one text||",
        2: "page two text||",
        3: "page three text",
    }


def test_split_pages_merges_multiple_paragraphs_per_page():
    text = "AABBCC"
    paragraph_starts = [
        [0, 1, 0],
        [2, 1, 1],  # second paragraph still on page 1
        [4, 2, 0],
        [len(text), -1, -1],
    ]
    pages = _split_pages(text, paragraph_starts)
    assert pages == {1: "AABB", 2: "CC"}


def test_split_pages_skips_pages_with_no_detected_text():
    # A blank/undetected page leaves a gap in page_id sequence (here page 2
    # is missing) — it should simply be absent, not raise or misalign.
    text = "onethree"
    paragraph_starts = [
        [0, 1, 0],
        [3, 3, 0],
        [len(text), -1, -1],
    ]
    pages = _split_pages(text, paragraph_starts)
    assert pages == {1: "one", 3: "three"}


def test_pages_from_ocr_reads_merged_results_jsonl(tmp_path):
    merged = tmp_path / "merged_results.jsonl"
    samples = [
        {
            "text": "slide1||slide2",
            "metadata": {
                "file_path": "/scratch/vidore/pdfs/doc_0000.pdf",
                "paragraph_starts": [[0, 1, 0], [8, 2, 0], [14, -1, -1]],
            },
        },
        {
            "text": "onlypage",
            "metadata": {
                "file_path": "/scratch/vidore/pdfs/doc_0001.pdf",
                "paragraph_starts": [[0, 1, 0], [8, -1, -1]],
            },
        },
    ]
    with merged.open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    doc_ids, texts = pages_from_ocr(merged)

    assert doc_ids == [
        "doc_0000.pdf#page=1",
        "doc_0000.pdf#page=2",
        "doc_0001.pdf#page=1",
    ]
    assert texts == ["slide1||", "slide2", "onlypage"]
