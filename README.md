# Does visual document retrieval actually beat text retrieval?

Six **ColVision** multi-vector encoders (ColPali, ColQwen2, ColQwen2.5, ColGemma3,
ColSmol-256M/500M) benchmarked against two text retrieval pipelines inside a single
frozen RAG stack — [mmore](https://github.com/swiss-ai/mmore) — across three corpora
and 54 model cells. The suite pins an exact mmore commit and drives its CLI as a
subprocess, so every number reflects the code path a real mmore user runs, not a
reimplementation.

The headline is not a ranking. **Whether text or vision wins depends on how the query
set was written** — and most published comparisons never say.

Academic project (Cassiopée), Télécom SudParis, run on EPFL LiGHT's Run:AI cluster.

**Two experiments.** The first runs on figure-rich biomedical lecture slides, with
questions written by a model *looking at the page* and validated by humans (§1, §2). The
second runs on scientific literature natively written in five languages, with questions
drawn from the page *text* (§3). They bracket the problem from both ends, which is what
makes the query-grounding effect visible at all.

**What is held constant.** A comparison only means something if the encoder is the only
moving part. Across all 54 cells — one cell being a single (encoder, corpus) pair — the
mmore commit, the Milvus index (`FLAT`, inner product), the MaxSim reranker, the depth
(`top_k=10`), the 200-dpi page rendering and the retrieval contract are fixed. So are
precision (`bfloat16`), batch size (8) and token vector dimension (128); no encoder is
re-tuned in its own favour. Only the visual encoder and the corpus vary.

---

## 1. With vision-grounded queries, visual retrieval wins — and cost, not quality, separates the leaders

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/retrieval_quality-dark.png">
  <img alt="Retrieval quality on ViDoRe v2 biomedical lectures" src="assets/retrieval_quality.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/retrieval_cost-dark.png">
  <img alt="Retrieval cost per query on ViDoRe v2 biomedical lectures" src="assets/retrieval_cost.png">
</picture>

On [ViDoRe v2](https://huggingface.co/datasets/vidore) biomedical lectures — 1,016
figure-rich pages, 160 queries written by a model *looking at the page* and then
validated by human annotators, with graded multi-page relevance — all six
encoders beat both text pipelines. The best is **+28% nDCG@5** over mmore's own text
retriever; the smallest, ColSmol-256M, only **+7%**.

The top three sit within **0.006 nDCG@5** of each other, so quality does not decide
between them. Cost does: retrieval wall-clock per query spans **2.11 s (ColGemma3) to
11.59 s (ColSmol-256M)**, and it does not track model size — the *smallest* encoder is
the slowest to query. The cost broadly follows the number of vectors a model emits per
page, which is what mmore's MaxSim reranker iterates over, rather than the parameter
count — though not perfectly: ColPali emits the most vectors (1031/page) and is still
the second cheapest to query.
ColGemma3 delivers front-of-pack quality at a quarter of ColQwen2.5's cost.

<!-- generated:track-a-metrics -->
|  | nDCG@5 | nDCG@10 | MRR | Recall@10 | MAP@10 |
| --- | --- | --- | --- | --- | --- |
| ColQwen2.5 | **0.630** | **0.666** | 0.717 | 0.766 | **0.573** |
| ColGemma3 | 0.626 | 0.660 | 0.709 | **0.767** | 0.564 |
| ColQwen2 | 0.624 | 0.657 | **0.723** | 0.754 | 0.563 |
| ColPali | 0.591 | 0.627 | 0.683 | 0.719 | 0.536 |
| ColSmol-500M | 0.554 | 0.574 | 0.629 | 0.674 | 0.482 |
| ColSmol-256M | 0.526 | 0.556 | 0.615 | 0.661 | 0.457 |
| _dense bge-m3_ | 0.473 | 0.504 | 0.560 | 0.599 | 0.416 |
| _mmore hybrid_ | 0.492 | 0.521 | 0.562 | 0.625 | 0.433 |
<!-- /generated -->

nDCG here uses a **linear** gain, so these numbers are not comparable with the published
ViDoRe leaderboard; MAP and Recall are capped by `top_k=10`.

## 2. Translating the query only hurts the small encoders

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/query_language-dark.png">
  <img alt="Query-language robustness at a fixed index" src="assets/query_language.png">
</picture>

Re-asking the same 160 questions in four languages **against the index built for
English** (ViDoRe v2's official multilingual queries, not a single page re-embedded)
costs the four large encoders at most 0.09 nDCG@5. The two ColSmol models lose up to
0.38 — ColSmol-256M falls from 0.526 to 0.147. Multilingual query understanding is the
first thing model compression sacrifices.

## 3. On text-grounded queries the margin shrinks — but does not flip

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/multilingual_heatmap-dark.png">
  <img alt="nDCG@5 across five native-language corpora" src="assets/multilingual_heatmap.png">
</picture>

The second experiment uses natively-written biomedical literature in five languages
(PMC OA filtered on `<Language>[Language]`; French from HAL theses — **nothing is translated**),
with English queries throughout, so retrieval is genuinely cross-lingual. Here the
queries are generated by Qwen2.5-32B-Instruct from **extracted page text**, which makes
them answerable without the image and structurally favours a text retriever.

Even so, the best encoder beats the best text pipeline in all five languages. The
measured margin is a *floor*, not a ceiling.

<!-- generated:multilingual -->
| corpus | best encoder | dense bge-m3 | mmore hybrid |
| --- | --- | --- | --- |
| English | ColQwen2.5 (0.860) | 0.734 | 0.812 |
| French | ColQwen2.5 (0.756) | 0.620 | 0.517 |
| Chinese | ColQwen2.5 (0.750) | 0.639 | 0.372 |
| German | ColGemma3 (0.749) | 0.714 | 0.594 |
| Spanish | ColGemma3 (0.807) | 0.721 | 0.551 |
<!-- /generated -->

Three side findings:

- **No single encoder dominates.** ColQwen2.5 leads on EN/FR/ZH, ColGemma3 on DE/ES —
  and the gap between them reaches 0.158 on German, far wider than anything that
  separated them on the slides. Encoder choice becomes a per-corpus decision, not a
  global ranking.
- **Changing script is not the difficulty; German is.** One would expect encoders
  pre-trained on mostly Latin document images to struggle on Chinese. They don't:
  Chinese averages 0.637 across the six encoders, ahead of German for five of them and
  level with French (0.643), while **German is the hardest language at 0.590** — Latin
  alphabet notwithstanding. The encoders appear to lean on layout and figure structure,
  which transfer across scripts, more than on glyph recognition. (This language ranking
  is confounded with corpus construction: French is the only one not from PMC OA.)
- **mmore's out-of-the-box text path is not cross-lingual-robust.** Its hybrid fuses a
  multilingual dense encoder with an **English** SPLADE sparse model. That helps in
  English (0.812 vs 0.734 dense) and hurts everywhere else, collapsing on Chinese
  (0.372 vs 0.639). Routing by detected language — hybrid for English, dense elsewhere —
  is the obvious fix.

Full per-model tables: **[`results/SUMMARY.md`](results/SUMMARY.md)**, generated from
the records by `scripts/summarize_results.py` and never edited by hand.

## 4. So which one should you enable?

Quality does not decide: the top three sit within 0.006 nDCG@5. Cost and corpus
language do.

| Context | Enable |
| --- | --- |
| Default — quality at the best cost | **ColGemma3** |
| Maximum quality, cost no object | **ColQwen2.5** |
| German or Spanish corpus | **ColGemma3** |
| English, French or Chinese corpus | **ColQwen2.5** |
| Queries not in English | avoid **ColSmol** |
| Memory constraint (*not* a latency budget) | **ColSmol-256M** |

ColPali is the one hard to recommend: weakest of the four large encoders in quality
without being meaningfully cheaper. The two ColSmol are a memory play only — they are
the *most* expensive of the six per query, and they fall apart on non-English queries.

---

## What was built

The benchmark is the deliverable, but most of the work is the harness around it.

- **A fixed-pipeline harness.** Index, reranker, embedding loader and retrieval
  contract are held constant; only the encoder and the corpus vary. Each cell shells out
  to mmore's real `process → index → retrieve` CLI and scores the ranked lists into one
  `BenchmarkRecord` JSON.
- **Cluster execution without SLURM.** EPFL RCP is Kubernetes + Run:AI + Harbor.
  Dependencies are not baked into the image: they install once into a venv on the scratch
  PVC that every job activates, so image rebuilds stay cheap. Four venvs coexist because
  their dependency graphs genuinely conflict (see the table below).
- **Corpus connectors.** PMC OA (with a native-language filter and a visual-density
  filter), HAL open-access theses, and a ViDoRe v2 adapter that reconstructs one PDF per
  document — sized so mmore's 200-dpi re-render reproduces the original pixels — so an
  image-only dataset flows through a PDF pipeline unchanged.
- **Query generation served in-house.** Qwen2.5-32B-Instruct on vLLM with guided-JSON
  decoding and few-shot priming, prompted per page, with the document language and the
  query language as separate parameters. The generator has to follow an instruction and
  return valid JSON over thousands of calls, read Chinese, German and Spanish well enough
  to draw an English question out of them, and fit the available GPUs. Domain base models
  were tried first and dropped: Meditron continues the page text instead of asking a
  question about it.
- **OCR for the image-only corpus.** ViDoRe pages carry no text layer, so the text
  baselines read mmore's own Marker+Surya OCR output rather than a shortcut extraction.
- **Provenance in every number.** Each record stores the mmore commit, the queryset
  SHA-256 and the corpus manifest hash. Every table and figure in this README is
  regenerated from those records by a script.

## Methodological honesty

Two results in this repo had to be retracted and re-derived. Both are documented here
rather than quietly overwritten, and the superseded records are kept in
[`results/_superseded/`](results/_superseded).

**A scoring bug that inverted the headline.** An earlier version of this benchmark
reported that text retrieval beat every visual encoder by 2–4×. It was **a scoring bug**:
query generation numbered pages from 0 while
mmore emits `page_num + 1`, so ColVision runs were graded against the page *before* the
relevant one — a perfect run scored 0. The text baselines were self-consistent and
therefore correct, which is exactly what made the artefact look like a finding. Fixed in
`353c076`; all 30 multilingual cells were re-scored from the stored ranked lists (no
model was re-run — the ranking never depended on the id convention), and a regression
test now asserts that a perfect run on mmore's *real* output format scores nDCG@1 = 1.0.

**A silent partial index.** Auditing vectors-per-page — by counting Milvus entities
against ingested parquet rows — showed the two ColSmol cells had indexed only 707 and
777 of 1,016 pages. An interrupted ingestion, resumed with `skip_already_processed:
true`, had left a truncated parquet and reported success. Every query whose relevant
page was missing was lost outright, so the scores measured the run, not the model. Both
cells were re-ingested and re-scored on GPU: ColSmol-500M 0.390 → **0.554**,
ColSmol-256M 0.358 → **0.526** — enough to move both from *below* the text baselines to
above them, and to retract the earlier claim that visual retrieval loses at that model
scale. The lesson is that a benchmark needs a corpus-completeness assertion, not just a
zero exit code.

**The obvious objection, and what answers it.** Experiment 2's queries come from a single
generator, Qwen2.5-32B-Instruct, which shares a backbone with two of the encoders under
test — including ColQwen2.5, the one that leads on three languages out of five. With one
generator, kinship cannot be ruled out from that experiment alone. What makes it unlikely
is experiment 1: its questions are ViDoRe's, written against the page image and validated
by human annotators, so no generator of ours touches them — and **ColQwen2.5 still comes out
on top there**. The same encoder leading in both setups, one of which is immune to the
objection, is hard to explain by kinship. Settling it outright would mean regenerating
experiment 2's queries with a model from an unrelated family, at equal resources.

Other limits, stated in full in [`results/README.md`](results/README.md), should be read
before quoting any number here:

- **One seed.** The bootstrap intervals in `results.aggregate` collapse to zero width and
  are **not** confidence intervals. The 0.006 nDCG@5 separating the top three encoders is
  a point estimate, not a ranking.
- **Linear nDCG gain.** Not the exponential gain of the ViDoRe leaderboard, so these
  figures are not directly comparable to published ViDoRe scores. MAP and Recall are
  capped by `top_k=10`.
- **Cost is wall-clock on a shared GPU.** Latency and VRAM are not instrumented, so the
  ratios carry the result and the absolute values do not.
- **The French corpus is asymmetric.** Under the same page budget it comes down to a few
  long HAL theses against ~50 short articles elsewhere, so its scores are confounded with
  low document diversity and should not be read as a language effect.
- **Corpus and query grounding vary together** between the two experiments, so the
  grounding effect is not isolated at fixed corpus.

## Models and baselines

| id | HF checkpoint | family | backbone |
| --- | --- | --- | --- |
| `colqwen2_5_v0_2` | `vidore/colqwen2.5-v0.2` | ColQwen2.5 | Qwen2.5-VL |
| `colgemma3_colnetra` | `Cognitive-Lab/ColNetraEmbed` | ColGemma3 | Gemma 3 |
| `colqwen2_v1_0` | `vidore/colqwen2-v1.0` | ColQwen2 | Qwen2-VL |
| `colpali_v1_3` | `vidore/colpali-v1.3` | ColPali | PaliGemma |
| `colsmol_500m` | `vidore/colSmol-500M` | ColSmol | SmolVLM |
| `colsmol_256m` | `vidore/colSmol-256M` | ColSmol | SmolVLM |

**dense** — `BAAI/bge-m3`, one embedding per page. **mmore hybrid** — mmore's own
`Indexer`/`Retriever` (dense + SPLADE, `hybrid_search_weight=0.5`, reranker off): what a
user gets from mmore's text pipeline out of the box.

Metrics: nDCG@{1,5,10}, Recall@{1,5,10}, Precision@{1,5,10}, MRR, MAP@10.

---

## Reproducing

Python 3.11 and [uv](https://docs.astral.sh/uv/). Every table and figure re-derives from
the committed records on CPU alone:

```bash
scripts/setup.sh
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/summarize_results.py   # rebuilds results/SUMMARY.md + this README's tables
.venv/bin/python scripts/make_figures.py        # rebuilds assets/*.png, report/figures/*.pdf
```

Running the models needs GPUs. On EPFL RCP (Kubernetes + Run:AI, no SLURM):

```bash
./scripts/rcp/setup.sh                  # builds + pushes the image, writes .rcp-env
./scripts/rcp/bootstrap-venv.sh --wait

./scripts/rcp/submit.sh corpus-vidore           # ViDoRe v2 → PDFs + graded qrels
./scripts/rcp/submit.sh corpus-lang <lang> 150 0  # PMC OA, native language
./scripts/rcp/submit.sh corpus-fr 100 0         # HAL theses
./scripts/rcp/submit.sh gen-tb <lang> <manifest> <pdfs_dir> <out.jsonl>

./scripts/rcp/submit.sh all                     # smoke + every cell
```

Each language is capped to a comparable sub-corpus (50 PDFs / 500 pages) by
`scripts/rcp/materialize_tb_lang.py`; sampling is seeded end to end, and
`bcv-corpus verify` re-hashes a local corpus against its manifest.

> **Re-running a cell:** purge it first —
> `rm -rf data/track_b_<l>/{milvus/<m>_<l>.db,process/<m>_<l>,retrieve/<m>_<l>}`.
> A partial Milvus database makes `retrieve` hang silently for hours. Cells are isolated
> by `cell_id=<model>_<lang>`, so purging one is safe while others run.

### The four scratch venvs

| venv | Purpose | Why separate |
| --- | --- | --- |
| `bcv-venv` | ColVision runs, metrics | pins `transformers==5.3.0`, required for correct ColVision weight loading |
| `bcv-venv-vllm` | Query generation | current vLLM is built for CUDA 13; the image is CUDA 12.x |
| `bcv-venv-mmoretext` | mmore hybrid baseline | SPLADE calls `batch_encode_plus`, removed in transformers 5.x → needs 4.x |
| `bcv-venv-mmoreprocess` | ViDoRe OCR (Marker+Surya) | heavy `mmore[process]` extra, unused elsewhere |

### CLIs and layout

| Command | Purpose |
| --- | --- |
| `bcv-corpus` | Corpus download, language + visual-density filters, manifest build/verify |
| `bcv-queries` | Query generation (inverse-query prompting) and ambiguity filtering |
| `bcv-run` | Orchestrate mmore `process → index → retrieve` and score a cell |
| `bcv-report` | LaTeX tables and per-study figures from the result JSONs |

```
configs/          Model and per-corpus YAML
results/          BenchmarkRecords + generated SUMMARY.md + caveats README
assets/           README figures, generated by scripts/make_figures.py
scripts/          Local setup, summariser, figure generation, rcp/ job submission
src/benchmark_colvision/
  corpus/         PMC OA / HAL / ViDoRe ingestion, filters, manifests, OCR
  queries/        LLM query generation, schema, page enumeration
  runners/        mmore CLI wrapper, orchestrators, text baselines
  evaluation/     Retrieval metrics, statistical tests
  reporting/      LaTeX tables and per-study figures
tests/            120 tests; subprocess and GPU calls are mocked
```

`pyproject.toml` pins mmore to commit `4102f96` of `Gsharpp/mmore` (`colvision` extra).
That branch carried PR #305, ColVision support, since merged upstream into
`swiss-ai/mmore` — the pin can move there directly. `uv.lock` is committed.

## License

See [LICENSE](LICENSE).
