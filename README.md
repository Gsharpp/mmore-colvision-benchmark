# mmore-colvision-benchmark

Benchmark suite comparing ColVision models (ColPali, ColQwen2, ColQwen2.5, ColQwen3, ColGemma3) within the [mmore](https://github.com/swiss-ai/mmore) RAG pipeline (PR #305). The benchmark pins a specific commit of the mmore fork carrying PR #305 and invokes its CLI as a subprocess, so the measurements reflect the exact code path a real user would run.

## Two tracks

- **Track A — English medical scaling.** Same five models on a figure-rich English PMC-OA corpus, evaluated at four corpus sizes (100 / 1k / 10k / 50k pages). Reveals how each model degrades as the index grows.
- **Track B — Multilingual.** Six languages (EN, FR, DE, ES, ZH, AR) at a fixed corpus size, with per-language breakdowns. Includes one logographic (ZH) and one RTL (AR) script to stress models' cross-script ability.

Each run captures retrieval metrics (nDCG@k, Recall@k, MRR, MAP), generation metrics (RAGAS faithfulness and answer relevancy via Meditron-70B as judge), and resource metrics (latency p50/p95/p99, throughput, GPU memory peak). Each cell is run with three seeds and reported with bootstrap 95% CIs; pairwise model comparisons use Wilcoxon signed-rank with Holm correction.

## Repo layout

```
configs/             # YAML configs for models, tracks and the LLM judge
scripts/setup.sh     # uv-based environment bootstrap
scripts/slurm/       # SLURM job arrays for EPFL RCP
src/benchmark_colvision/
  corpus/            # PMC download, language and visual-density filters, manifest
  queries/           # Synthetic query generation (inverse queries via Meditron)
  runners/           # mmore CLI wrapper and benchmark orchestrators
  evaluation/        # Retrieval, generation, performance metrics, statistical tests
  reporting/         # Figures and LaTeX table generation
tests/               # pytest unit tests (subprocess and GPU calls are mocked)
results/             # Benchmark run outputs (JSON, MTEB-compatible schema)
report/              # LaTeX source for the technical report
```

## Quick start

```bash
# Install (CPU; GPU on RCP uses --gpu)
scripts/setup.sh

# Activate
source .venv/bin/activate

# Run unit tests (no GPU, no network)
uv run pytest -v

# Compute the visual-density score of a PDF (sanity check)
uv run bcv-corpus density path/to/some.pdf
```

## Available CLIs

| Command         | Purpose                                                 |
| --------------- | ------------------------------------------------------- |
| `bcv-corpus`    | Density filter, manifest build/verify                   |
| `bcv-queries`   | Synthetic query generation (Meditron) and validation    |
| `bcv-run`       | Orchestrate process → index → retrieve via mmore CLI    |
| `bcv-report`    | Render figures and LaTeX tables from result JSONs       |

## Reproducibility

- `pyproject.toml` pins mmore to commit `498560047e19ddb57ecc0fdb5b01f9f4ccd0d651` of `Gsharpp/mmore` (the PR #305 branch).
- `uv.lock` is committed and resolves the full dependency graph for Python 3.11.
- Every result JSON records the mmore commit hash, the model id, the seed, and the corpus manifest SHA-256.
- Corpus manifests (`corpus_manifest.py`) carry per-PDF SHA-256 hashes; `bcv-corpus verify` re-hashes the local corpus to detect drift.

## Status

Scaffolding complete:
- Project layout, dependencies, dev tooling (ruff, pre-commit), Apache-2.0 LICENSE.
- mmore CLI subprocess wrapper with mock-driven tests.
- Retrieval metrics (textbook implementations) with unit tests against hand-computed references.
- Generation metrics façade over RAGAS (faithfulness, answer relevancy, context precision/recall).
- Bootstrap CI, Wilcoxon, Holm / Bonferroni corrections.
- GPU memory tracker, latency percentile summary.
- Visual-density filter, corpus manifest with SHA-256 verification, language detector.
- PMC-OA index parser and tarball extractor.
- Generic URL-list downloader for any source whose article list can be exported
  (HAL, Thieme OA, SciELO, CNKI, Saudi Med Journal).
- Synthetic query pipeline: inverse-query prompt, ambiguity filter, methodology-validation correlator.
- vLLM client (`benchmark_colvision.clients.vllm_client`) wired into
  `bcv-queries generate` / `bcv-queries filter`.
- RAGAS judge bridge (`benchmark_colvision.clients.ragas_llm`) over the same vLLM endpoint.
- Track A and Track B cell orchestrators, mmore retrieve-output parser.
- Result schema (MTEB-compatible) and aggregator (per-cell bootstrap CI, pairwise Wilcoxon).
- Reporting: Track A scaling/latency/throughput/GPU-memory figures, Track B heatmap and language-gap, LaTeX tables with CIs.
- Config templates for the five models, both tracks, and the Meditron-based judge.
- SLURM job arrays for Track A and Track B on RCP, plus a long-lived
  `serve_meditron.sbatch` for the vLLM judge/query-generator.
- `docs/ARCHITECTURE.md` and a `report/main.tex` skeleton.

Pending (RCP-side):
- Source-specific URL discovery scripts (HAL OAI export, Thieme listing, SciELO
  search, CNKI OA, Saudi Med Journal index) producing the JSON manifests that
  feed `bcv-corpus download-urls`.
- Methodology validation run (Phase 4) against an annotated ViDoRe healthcare subset.
- End-to-end smoke run on RCP, then full Track A + Track B execution via SLURM.
- Filling in figures, tables, and prose for the LaTeX technical report.
