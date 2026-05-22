# Architecture

This document describes how the benchmark code is organized, how a single
benchmark cell flows through it, and which extension points to use when adding
a model, a language, or a metric.

## Design principles

1. **Pin upstream, call upstream.** The benchmark depends on a specific commit
   of the `mmore` fork carrying PR #305 and invokes the `mmore` CLI as a
   subprocess. We never import `mmore` internals or reimplement its pipeline:
   the measurements reflect the same code path a real user would run.
2. **One JSON record per cell.** A "cell" is `(track, model, palier-or-language,
   seed)`. Every cell writes one `BenchmarkRecord` JSON with all the
   identifiers needed to trace it back (mmore commit, corpus manifest SHA-256,
   query set SHA-256, hardware). The records under `results/` are the
   ground truth — figures and tables are derivations.
3. **Layered, no global state.** Each module exposes pure functions that take
   typed inputs and return typed outputs. Side effects (subprocess, GPU memory
   polling, file writes) are confined to the runners and the CLI layer.
4. **Tests use fakes, not the real pipeline.** Subprocess calls and LLM calls
   are stubbed; the metrics modules are tested against textbook reference
   values; figures are smoke-tested by rendering to a temp path.

## Module map

```
src/benchmark_colvision/
├── corpus/
│   ├── corpus_manifest.py    schema: PdfEntry, CorpusManifest, SHA-256 verifier
│   ├── visual_density_filter.py    PyMuPDF heuristic (figures/tables surface)
│   ├── language_filter.py    langdetect wrapper, seeded for reproducibility
│   ├── pmc_downloader.py     PMC OA index parser + tarball extractor
│   └── cli.py                bcv-corpus
├── queries/
│   ├── schema.py             SyntheticQuery, QuerySet, deterministic ids/hash
│   ├── inverse_query_gen.py  LLMClient protocol, prompt, completion parser
│   ├── ambiguity_filter.py   judge prompt + score parser + filter loop
│   ├── methodology_validation.py  Pearson/Spearman vs annotated subset
│   └── cli.py                bcv-queries
├── runners/
│   ├── mmore_wrapper.py      subprocess CommandResult + MmoreRun aggregate
│   ├── retrieval_output.py   parses mmore retrieve JSON in tolerant form
│   ├── run_track_a.py        scaling-study cell orchestrator
│   ├── run_track_b.py        multilingual cell orchestrator
│   └── cli.py                bcv-run
├── evaluation/
│   ├── retrieval_metrics.py  precision@k, recall@k, MRR, MAP, nDCG@k
│   ├── generation_metrics.py RAGAS facade (faithfulness, answer relevancy, ...)
│   ├── performance_metrics.py timing context manager, GPU mem tracker, percentiles
│   └── statistical_tests.py  bootstrap CI, Wilcoxon, Bonferroni/Holm
├── results/
│   ├── schema.py             BenchmarkRecord, RetrievalScores, GenerationScores,
│   │                         PerformanceScores, HardwareInfo, CellId
│   └── aggregate.py          load_records, per_cell_ci, pairwise_wilcoxon
└── reporting/
    ├── figures_track_a.py    scaling curve, throughput bars, latency, GPU mem
    ├── figures_track_b.py    model × language heatmap, language-gap bars
    ├── latex_tables.py       tabular renderers with [point, lower, upper] CIs
    └── cli.py                bcv-report
```

## Data flow for one Track A cell

```
corpus/               queries/                 mmore CLI                results/
─────────             ────────                 ────────────             ────────
manifest.json    ────►  inverse_query_gen ─────► queries_file.jsonl
                                    │
                                    ▼
                            ambiguity_filter
                                    │
                                    ▼
                            queryset.jsonl (sha256)
                                                                          ▲
configs (process/index/retrieve YAML)                                     │
        │                                                                 │
        ▼                                                                 │
   runners/mmore_wrapper.run_pipeline()                                   │
        │   ── process → index → retrieve subprocesses                    │
        ▼                                                                 │
   runners/retrieval_output.load_retrieval()                              │
        │                                                                 │
        ▼                                                                 │
   evaluation/retrieval_metrics.evaluate_batch()                          │
        │                                                                 │
        ▼                                                                 │
   results/schema.BenchmarkRecord ──────────────────────────────────►  cell.json
```

For Track B the flow is identical except (a) one orchestrator iterates over
`language` instead of `palier`, and (b) `CellId.language` is set instead of
`palier_id`.

## Extension points

### Add a new ColVision model

1. Add an entry to `configs/models.yaml` with `id`, `hf_name`, `embed_dim`,
   `process_batch_size`. The id is what appears in `CellId.model_id`.
2. Confirm the model is recognized by `mmore`'s `model_utils.py` factory
   (it resolves names by regex on the HF id).
3. If the embedding dimension differs from 128, mmore will create a separate
   Milvus collection automatically — nothing to do on our side.

### Add a new language to Track B

1. Add a `languages:` entry in `configs/track_b.yaml` with code, name, source,
   `pages_target`. Add typological notes if the script is unusual (RTL, CJK).
2. If the source is a new PDF provider, add a thin downloader alongside
   `pmc_downloader.py`. The contract is: produce a `CorpusManifest` with the
   right `source` enum value and per-PDF SHA-256.

### Add a new metric

- Retrieval: extend `evaluation/retrieval_metrics.py` with a textbook
  implementation and a test asserting it against a hand-computed reference. Add
  the corresponding field to `results/schema.RetrievalScores`.
- Generation: extend `evaluation/generation_metrics.py` and `GenerationScores`
  in lock-step.
- Performance: same in `evaluation/performance_metrics.py` /
  `PerformanceScores`.

### Plug a real LLM into queries

`queries/inverse_query_gen.LLMClient` is a 1-method protocol. In `scripts/`
write an adapter (e.g. `VLLMClient`) that satisfies it and hand it to
`generate_for_pages`. The CLI subcommand `bcv-queries generate` is intentionally
guarded with a `ClickException`: the wiring of the live judge is an operational
choice the deployment script owns.

## Reproducibility checklist

Each `BenchmarkRecord` carries enough information to re-run the cell:

| Field                       | Source                                         |
| --------------------------- | ---------------------------------------------- |
| `mmore_commit`              | `pyproject.toml [tool.uv.sources]`             |
| `benchmark_version`         | `pyproject.toml project.version`               |
| `corpus_manifest_sha256`    | `CorpusManifest.sha256()` or shell command     |
| `queries_sha256`            | `QuerySet.sha256()`                            |
| `cell.seed`                 | from the SLURM array index                     |
| `hardware.*`                | populated by SLURM script before invoking      |

The combination is sufficient to identify "which mmore code ran on which
corpus, with which queries, under which seed and which hardware."

## Running on RCP

`scripts/slurm/run_track_a.sbatch` and `run_track_b.sbatch` set up a SLURM job
array. Each array index picks one cell from a generated list of cells and runs
it. Failures of individual cells are isolated — the array continues — and the
post-processing step (`bcv-report`) tolerates missing cells.
