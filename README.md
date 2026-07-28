# mmore-colvision-benchmark

Benchmark comparing ColVision multi-vector retrieval models (ColPali, ColQwen2, ColQwen2.5, ColGemma3, ColSmol) inside the [mmore](https://github.com/swiss-ai/mmore) multimodal RAG pipeline. The suite pins a specific mmore commit and drives its CLI directly, so every measurement reflects the exact code path a real user would run.

## Studies

- **Biomedical retrieval** — 6 ColVision models plus 2 text baselines (dense `bge-m3`, mmore's native hybrid dense+SPLADE) on `vidore/biomedical_lectures_eng_v2`: figure-rich biomedical slides with human-validated, vision-grounded queries and graded, multi-page qrels.
- **Cross-lingual retrieval** — the same 6 models over native-language biomedical corpora (EN/FR/ZH/DE/ES, sourced from PMC OA and HAL), and over a fixed index queried with translated FR/DE/ES queries (ViDoRe multilingual slice), to separate corpus effects from query-language effects.

Metrics: nDCG@{1,5,10}, Recall@{1,5,10}, Precision@{1,5,10}, MRR, MAP.

## Results

All 61 reported cells live in [`results/`](results/) as one JSON record each;
the headline tables are in [`results/SUMMARY.md`](results/SUMMARY.md), generated
from those records by `scripts/summarize_results.py`. Read
[`results/README.md`](results/README.md) before quoting a number — it documents
the single-seed design, the linear-gain nDCG, the `top_k=10` cap on MAP/Recall,
and the missing latency instrumentation.

nDCG@5, best ColVision model per cell against the two text baselines:

| | Track A (ViDoRe) | en | fr | zh | de | es |
| --- | --- | --- | --- | --- | --- | --- |
| best ColVision | **0.630** | **0.860** | **0.756** | **0.750** | **0.749** | **0.807** |
| dense `bge-m3` | 0.473 | 0.734 | 0.620 | 0.639 | 0.714 | 0.721 |
| mmore hybrid | 0.492 | 0.812 | 0.517 | 0.372 | 0.594 | 0.551 |

Visual retrieval beats both text pipelines in every cell. The margin is widest
where the text path is weakest: mmore's hybrid retriever fuses a dense
multilingual encoder with an **English** SPLADE model, which actively degrades
Chinese and, to a lesser extent, the other non-English corpora.

Caveat on Track B: its queries are generated from page *text*, which favours
text retrievers — so the ColVision lead there is a lower bound. Track A uses
ViDoRe's human-validated, vision-grounded queries and is free of that bias.

## Models

| id | hf_name | family |
| --- | --- | --- |
| `colpali_v1_3` | `vidore/colpali-v1.3` | ColPali |
| `colqwen2_v1_0` | `vidore/colqwen2-v1.0` | ColQwen2 |
| `colqwen2_5_v0_2` | `vidore/colqwen2.5-v0.2` | ColQwen2.5 |
| `colgemma3_colnetra` | `Cognitive-Lab/ColNetraEmbed` | ColGemma3 |
| `colsmol_256m` | `vidore/colSmol-256M` | ColSmol |
| `colsmol_500m` | `vidore/colSmol-500M` | ColSmol |

## Repo layout

```
configs/               # Model and track YAML configs
results/               # Benchmark records (JSON) + generated SUMMARY.md
scripts/setup.sh       # Local uv-based environment bootstrap (CPU/GPU)
scripts/summarize_results.py  # Rebuild results/SUMMARY.md from the records
scripts/rcp/           # EPFL RCP (Run:AI / Kubernetes) build + submit tooling
scripts/rcp/debug/     # One-off diagnostic probes (model loading, key mapping)
src/benchmark_colvision/
  corpus/              # PMC OA / HAL download, language & density filters, manifests
  queries/              # Query generation (LLM-based, inverse-query prompting)
  runners/             # mmore CLI wrapper and benchmark orchestrators
  evaluation/          # Retrieval metrics, statistical tests
  clients/             # vLLM / judge clients
  reporting/           # Figures and LaTeX table generation
tests/                 # pytest unit tests (subprocess and GPU calls are mocked)
```

## Quick start (local)

```bash
scripts/setup.sh
source .venv/bin/activate
uv run pytest -v
```

Running the actual benchmark requires EPFL RCP GPUs; see `scripts/rcp/setup.sh` and `scripts/rcp/submit.sh`.

## CLIs

| Command | Purpose |
| --- | --- |
| `bcv-corpus` | Corpus download, density filter, manifest build/verify |
| `bcv-queries` | Query generation and validation |
| `bcv-run` | Orchestrate process → index → retrieve via mmore CLI |
| `bcv-report` | Render figures and tables from result JSONs |

## Reproducibility

- `pyproject.toml` pins mmore to a specific commit of `Gsharpp/mmore` (`colvision` extra). That branch carried PR #305 (ColVision support), since merged upstream into `swiss-ai/mmore` — the pin could move there directly.
- `uv.lock` is committed and resolves the full dependency graph.
- Result JSONs record the mmore commit hash, model id, seed, and corpus manifest SHA-256.
- `bcv-corpus verify` re-hashes the local corpus against its manifest to detect drift.
- Scoring runs on one id convention end to end: mmore emits `<pdf>#page=<1-based>`
  and `SyntheticQuery.mmore_doc_id()` maps every queryset into that space, so
  ColVision cells and text baselines are directly comparable. A regression test
  asserts that a perfect run on mmore's real output format scores nDCG@1 = 1.0.

---

Methodology and results write-up: in progress.
