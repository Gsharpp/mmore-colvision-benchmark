# mmore-colvision-benchmark

Benchmark comparing ColVision multi-vector retrieval models (ColPali, ColQwen2, ColQwen2.5, ColGemma3, ColSmol) inside the [mmore](https://github.com/swiss-ai/mmore) multimodal RAG pipeline. The suite pins a specific mmore commit and drives its CLI directly, so every measurement reflects the exact code path a real user would run.

## Studies

- **Biomedical retrieval** — 6 ColVision models plus 2 text baselines (dense `bge-m3`, mmore's native hybrid dense+SPLADE) on `vidore/biomedical_lectures_eng_v2`: figure-rich biomedical slides with human-validated, vision-grounded queries and graded, multi-page qrels.
- **Cross-lingual retrieval** — the same 6 models over native-language biomedical corpora (EN/FR/ZH/DE/ES, sourced from PMC OA and HAL), and over a fixed index queried with translated FR/DE/ES queries (ViDoRe multilingual slice), to separate corpus effects from query-language effects.

Metrics: nDCG@{1,5,10}, Recall@{1,5,10}, Precision@{1,5,10}, MRR, MAP.

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
configs/               # Model, track and judge YAML configs
results/               # Benchmark records (JSON) and aggregated metrics
scripts/setup.sh       # Local uv-based environment bootstrap (CPU/GPU)
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

---

Methodology and results write-up: in progress.
