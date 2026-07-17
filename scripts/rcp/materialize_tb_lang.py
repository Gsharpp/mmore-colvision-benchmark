#!/usr/bin/env python
"""Materialize a capped Track B sub-corpus for one language.

Runs on the cluster (reads bcv-data, writes capped manifest + symlinked PDF dir,
repoints the repo's data/track_b_<lang> symlinks). Mirrors the one-off `bcv-tb-mat2`
job that built en/fr/zh, generalized to any language so de (and future langs) reuse it.

Cap: stop at PDF_CAP PDFs OR PAGE_BUDGET pages (whichever first), deterministic by
pdf_path sort — comparable search-space size across languages (isolates the language
effect). EN uses bare manifest.json/pdfs; other langs use manifest-<l>.json/pdfs-<l>.

Usage (on cluster):  python materialize_tb_lang.py de
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from benchmark_colvision.corpus.corpus_manifest import CorpusManifest

D = "/mloscratch/users/mbonnet/bcv-data"
R = "/mloscratch/users/mbonnet/bcv-dev/data"
PDF_CAP = 50
PAGE_BUDGET = 500


def repo_dir_for(lang: str) -> str:
    # EN keeps the original layout `track_b`; other langs use `track_b_<lang>`.
    return "track_b" if lang == "en" else f"track_b_{lang}"


def src_names(lang: str) -> tuple[str, str]:
    if lang == "en":
        return "manifest.json", "pdfs"
    return f"manifest-{lang}.json", f"pdfs-{lang}"


def materialize(lang: str) -> None:
    src_m, src_pdfs = src_names(lang)
    out_m = f"manifest-tb-{lang}.json"
    out_pdfs = f"pdfs-tb-{lang}"

    m = CorpusManifest.load(Path(os.path.join(D, src_m)))
    entries = sorted(m.pdfs, key=lambda e: e.pdf_path)  # deterministic
    sel, pages = [], 0
    for e in entries:
        if len(sel) >= PDF_CAP or pages >= PAGE_BUDGET:
            break
        sel.append(e)
        pages += e.page_count

    out_dir = os.path.join(D, out_pdfs)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    for e in sel:
        base = os.path.basename(e.pdf_path)
        src = os.path.join(D, src_pdfs, base)
        dst = os.path.join(out_dir, base)
        if not os.path.exists(src):
            print(f"  WARN missing {src}")
            continue
        os.symlink(src, dst)

    capped = CorpusManifest(
        name=f"track_b_{lang}_capped",
        track=m.track,
        languages=m.languages,
        total_pages=pages,
        pdfs=[e.model_copy(update={"pdf_path": os.path.basename(e.pdf_path)}) for e in sel],
    )
    capped.save(Path(os.path.join(D, out_m)))
    print(f"{lang}: selected {len(sel)} PDFs / {pages} pages -> {out_m}, {out_pdfs}/")

    # Repoint repo symlinks + purge stale process/milvus/retrieve artifacts.
    base = os.path.join(R, repo_dir_for(lang))
    os.makedirs(os.path.join(base, "queries"), exist_ok=True)
    for sub in ("process", "milvus", "retrieve"):
        p = os.path.join(base, sub)
        if os.path.exists(p):
            shutil.rmtree(p)
    for name, target in [
        ("pdfs", os.path.join(D, out_pdfs)),
        ("corpus_manifest.json", os.path.join(D, out_m)),
    ]:
        link = os.path.join(base, name)
        if os.path.islink(link) or os.path.exists(link):
            if os.path.isdir(link) and not os.path.islink(link):
                shutil.rmtree(link)
            else:
                os.remove(link)
        os.symlink(target, link)
    print(f"repointed {repo_dir_for(lang)}: pdfs->{out_pdfs}, manifest->{out_m}; purged artifacts")


if __name__ == "__main__":
    langs = sys.argv[1:] or ["de"]
    for lang in langs:
        materialize(lang)
    print("MATERIALIZE_OK")
