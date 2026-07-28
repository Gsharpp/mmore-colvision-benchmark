# mmore-colvision-benchmark

Does **visual document retrieval** actually beat a text RAG pipeline on real
documents — and does it hold up outside English?

This benchmark answers both inside [mmore](https://github.com/swiss-ai/mmore),
a multimodal RAG framework. Six ColVision multi-vector models are compared
against two text baselines, one of them mmore's own retrieval path. The suite
pins an exact mmore commit and drives its CLI as a subprocess, so every number
reflects the code path a real mmore user runs — not a reimplementation.

Academic project (Cassiopée), Télécom SudParis, run on EPFL's LiGHT cluster.

---

## Headline result

nDCG@5 — best ColVision model per cell, against both text baselines:

| | Track A | en | fr | zh | de | es |
| --- | --- | --- | --- | --- | --- | --- |
| **best ColVision** | **0.630** | **0.860** | **0.756** | **0.750** | **0.749** | **0.807** |
| dense `bge-m3` | 0.473 | 0.734 | 0.620 | 0.639 | 0.714 | 0.721 |
| mmore hybrid | 0.492 | 0.812 | 0.517 | 0.372 | 0.594 | 0.551 |

**Visual retrieval wins in every cell.** Three findings behind that table:

1. **The text baseline degrades badly outside English.** mmore's hybrid
   retriever fuses a multilingual dense encoder with an **English** SPLADE
   sparse model. On Chinese it scores 0.372 — *below* the plain dense retriever
   at 0.639. Hybrid fusion only pays off in English (0.812 vs 0.734). mmore's
   default text path is not cross-lingual-robust.
2. **No single ColVision model dominates.** ColQwen2.5 leads on English,
   French and Chinese, ColGemma3 on German and Spanish. Model choice is a
   language-dependent decision, not a ranking.
3. **Model size is not the axis that matters.** ColSmol-256M (0.655 en) is
   within reach of models several times larger, while the gap between the two
   ColSmol variants is larger than between some full-size families.

Full per-model tables: **[`results/SUMMARY.md`](results/SUMMARY.md)**.
Read **[`results/README.md`](results/README.md) before quoting any number** — it
lists the design's real limitations (single seed, linear-gain nDCG, MAP capped
at `top_k=10`, no latency instrumentation).

One caveat worth stating up front: Track B's queries are generated from page
**text**, which structurally favours text retrievers. The ColVision lead there
is therefore a *lower bound*. Track A uses ViDoRe's human-validated,
vision-grounded queries and is free of that bias.

## The two studies

| | Track A | Track B |
| --- | --- | --- |
| Corpus | ViDoRe v2 `biomedical_lectures_eng_v2` | Native-language biomedical literature |
| Languages | English | EN · FR · ZH · DE · ES |
| Queries | ViDoRe's own, human-validated, vision-grounded | Generated from page text, **always English** |
| Relevance | Graded, multi-page qrels | Single gold page |
| Cells | 6 models + 2 baselines | 6 models × 5 languages + 2 baselines |

Non-English corpora are **natively written** in their language — PMC OA filtered
on `<Language>[Language]`, French from HAL theses. Nothing is translated.
Queries stay English throughout Track B, making it a genuine *cross-lingual*
test: English query, non-English document.

A third slice re-queries Track A's fixed index with translated FR/DE/ES queries
(`track_b_vidore/`), which separates *corpus* effects from *query-language*
effects.

Metrics: nDCG@{1,5,10}, Recall@{1,5,10}, Precision@{1,5,10}, MRR, MAP@10.

## Models

| id | HF checkpoint | family | backbone |
| --- | --- | --- | --- |
| `colpali_v1_3` | `vidore/colpali-v1.3` | ColPali | PaliGemma |
| `colqwen2_v1_0` | `vidore/colqwen2-v1.0` | ColQwen2 | Qwen2-VL |
| `colqwen2_5_v0_2` | `vidore/colqwen2.5-v0.2` | ColQwen2.5 | Qwen2.5-VL |
| `colgemma3_colnetra` | `Cognitive-Lab/ColNetraEmbed` | ColGemma3 | Gemma 3 |
| `colsmol_500m` | `vidore/colSmol-500M` | ColSmol | SmolVLM |
| `colsmol_256m` | `vidore/colSmol-256M` | ColSmol | SmolVLM |

Baselines: **dense** — `BAAI/bge-m3`, page-level, cosine over normalised
embeddings. **mmore hybrid** — mmore's own `Indexer`/`Retriever` (dense +
SPLADE, `hybrid_search_weight=0.5`, reranker off), i.e. what a user gets from
mmore's text pipeline out of the box.

---

## Reproducing the benchmark

### 1. Local environment

Python 3.11 (pinned: `>=3.11,<3.12`) and [uv](https://docs.astral.sh/uv/).

```bash
scripts/setup.sh          # CPU-only: enough for tests, metrics, reporting
.venv/bin/python -m pytest tests -q
```

This is sufficient to re-derive every table from the committed records:

```bash
.venv/bin/python scripts/summarize_results.py   # rebuilds results/SUMMARY.md
```

Running the models themselves needs GPUs. `scripts/setup.sh --gpu` adds the CUDA
wheels for a single GPU machine; the cluster path below is what actually
produced the published numbers.

### 2. Cluster setup (EPFL RCP — Kubernetes + Run:AI, no SLURM)

```bash
./scripts/rcp/setup.sh            # builds + pushes the image, writes .rcp-env
./scripts/rcp/bootstrap-venv.sh --wait
```

Dependencies are **not** baked into the Docker image: they install once into a
venv on the scratch PVC, and every job activates it. Rebuilding the image is
therefore cheap.

Four venvs exist on scratch, because their dependencies genuinely conflict:

| venv | Purpose | Why separate |
| --- | --- | --- |
| `bcv-venv` | ColVision runs, metrics | pins `transformers==5.3.0` (required for correct ColVision weight loading) |
| `bcv-venv-vllm` | Query generation | latest vLLM is built for CUDA 13; the image is CUDA 12.x |
| `bcv-venv-mmoretext` | mmore hybrid baseline | SPLADE calls `batch_encode_plus`, removed in transformers 5.x → needs 4.x |
| `bcv-venv-mmoreprocess` | ViDoRe OCR (Marker+Surya) | heavy `mmore[process]` extra, unused elsewhere |

`bootstrap-venv.sh` creates the first two. The last two are built on demand, by
running `uv venv` against the same scratch path and installing `mmore[rag]`
(without the `colvision` extra, so `transformers` resolves to 4.x) or
`mmore[process]` respectively.

### 3. Build the corpora

```bash
./scripts/rcp/submit.sh corpus-vidore               # Track A: ViDoRe v2 → PDFs + graded qrels
./scripts/rcp/submit.sh corpus-lang <lang> 150 0   # Track B: PMC OA, native language
./scripts/rcp/submit.sh corpus-fr 100 0            # Track B French: HAL theses
```

Each language is then capped to a comparable sub-corpus (50 PDFs / 500 pages)
by `scripts/rcp/materialize_tb_lang.py`. Corpus sampling is seeded end to end,
and `bcv-corpus verify` re-hashes a local corpus against its manifest.

### 4. Generate Track B queries

```bash
./scripts/rcp/submit.sh gen-tb <lang> <manifest> <pdfs_dir> <out.jsonl>
```

Serves `Qwen/Qwen2.5-32B-Instruct` on vLLM and prompts it per page, with
`--query-language en` so questions stay English whatever the document language.
Track A needs no generation: ViDoRe ships its own validated queries.

### 5. Run the cells

```bash
./scripts/rcp/submit.sh smoke                          # one cell, sanity check
./scripts/rcp/submit.sh track-a <model> 0
./scripts/rcp/submit.sh track-b <model> <lang> 0
./scripts/rcp/submit.sh baseline B <lang>              # dense text baseline
./scripts/rcp/submit.sh baseline-vidore <dense|mmore>
./scripts/rcp/submit.sh all                            # smoke + every cell
```

Each cell runs mmore `process → index → retrieve`, then scores the ranked lists
into a `BenchmarkRecord` at `results/<study>/<model>/<axis>/seed_<n>.json`.

> **Re-running a cell:** purge it first —
> `rm -rf data/track_b_<l>/{milvus/<m>_<l>.db,process/<m>_<l>,retrieve/<m>_<l>}`.
> A partial Milvus database makes `retrieve` hang silently for hours. Cells are
> isolated by `cell_id=<model>_<lang>`, so purging one is safe while others run.

### 6. Tables and figures

```bash
.venv/bin/python scripts/summarize_results.py
bcv-report figures-a --results-dir results/track_a --out-dir report/figures
bcv-report figures-b --results-dir results/track_b --out-dir report/figures
```

## CLIs

| Command | Purpose |
| --- | --- |
| `bcv-corpus` | Corpus download, language + visual-density filters, manifest build/verify |
| `bcv-queries` | Query generation (inverse-query prompting) and ambiguity filtering |
| `bcv-run` | Orchestrate mmore `process → index → retrieve` and score a cell |
| `bcv-report` | Figures and LaTeX tables from the result JSONs |

## Repo layout

```
configs/          Model and per-track YAML (one file per Track B language)
results/          BenchmarkRecords + generated SUMMARY.md + caveats README
scripts/
  setup.sh                Local uv environment
  summarize_results.py    Rebuild results/SUMMARY.md from the records
  rcp/                    Run:AI image build, venv bootstrap, job submission
  rcp/debug/              One-off diagnostic probes
src/benchmark_colvision/
  corpus/         PMC OA / HAL / ViDoRe ingestion, filters, manifests, OCR
  queries/        LLM query generation, schema, page enumeration
  runners/        mmore CLI wrapper, orchestrators, text baselines
  evaluation/     Retrieval metrics, statistical tests
  reporting/      Figures and LaTeX tables
tests/            120 tests; subprocess and GPU calls are mocked
```

## Reproducibility notes

- `pyproject.toml` pins mmore to commit `4102f96` of `Gsharpp/mmore`
  (`colvision` extra). That branch carried PR #305 (ColVision support), since
  merged upstream into `swiss-ai/mmore` — the pin can move there directly.
- `uv.lock` is committed and resolves the full dependency graph.
- Every record stores the mmore commit, model id, seed, queryset SHA-256 and
  corpus manifest hash, so a number traces back to what produced it.
- Corpus sampling, query generation and cell execution are seeded.
- **One document-id convention end to end.** mmore emits
  `<pdf>#page=<1-based>`; `SyntheticQuery.mmore_doc_id()` maps every queryset
  into that space, so ColVision cells and text baselines are scored in the same
  id space and are directly comparable. A regression test asserts that a perfect
  run on mmore's *real* output format scores nDCG@1 = 1.0 — an earlier
  off-by-one here silently graded ColVision against the wrong page and inverted
  the headline finding.

## Limitations

Stated plainly, and in full, in [`results/README.md`](results/README.md). The
short version: one seed (so the bootstrap intervals in `results.aggregate`
collapse to zero width and are **not** confidence intervals), nDCG computed with
a linear rather than exponential gain (so Track A figures are not directly
comparable to the published ViDoRe leaderboard), MAP and Recall capped by
`top_k=10`, no latency or VRAM instrumentation, and no generation metrics.

## License

See [LICENSE](LICENSE).
