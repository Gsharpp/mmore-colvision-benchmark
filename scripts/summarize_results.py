#!/usr/bin/env python3
"""Render `results/SUMMARY.md` and the README tables from the BenchmarkRecords.

Single source of truth for the headline numbers: the tables are derived from the
records on disk, never typed by hand. The README blocks between
`<!-- generated:NAME -->` and `<!-- /generated -->` markers are rewritten in
place, so the claim that every table re-derives from the records holds
literally. Run after any re-scoring:

    python scripts/summarize_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
README = ROOT / "README.md"

MODELS = [
    "colqwen2_5_v0_2",
    "colgemma3_colnetra",
    "colqwen2_v1_0",
    "colpali_v1_3",
    "colsmol_500m",
    "colsmol_256m",
]
LANGS = ["en", "fr", "zh", "de", "es"]
VIDORE_LANGS = ["french", "german", "spanish"]
BASELINES = {"baseline_text": "dense bge-m3", "mmore_text": "mmore hybrid"}

# Reader-facing names, used in the README where the ids would be noise.
DISPLAY = {
    "colqwen2_5_v0_2": "ColQwen2.5",
    "colgemma3_colnetra": "ColGemma3",
    "colqwen2_v1_0": "ColQwen2",
    "colpali_v1_3": "ColPali",
    "colsmol_500m": "ColSmol-500M",
    "colsmol_256m": "ColSmol-256M",
}
LANG_NAMES = {"en": "English", "fr": "French", "zh": "Chinese", "de": "German", "es": "Spanish"}

# MAP and Recall are capped by top_k=10, hence the @10 labels; see results/README.md.
METRICS = [
    ("ndcg_at_5", "nDCG@5"),
    ("ndcg_at_10", "nDCG@10"),
    ("mrr", "MRR"),
    ("recall_at_10", "Recall@10"),
    ("map", "MAP@10"),
]


def retrieval(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("retrieval") or {}


def ndcg5(path: Path) -> float | None:
    return retrieval(path).get("ndcg_at_5")


def cell(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    sep = ["---"] * len(header)
    lines = [f"| {' | '.join(header)} |", f"| {' | '.join(sep)} |"]
    lines += [f"| {' | '.join(r)} |" for r in rows]
    return "\n".join(lines)


def best(values: dict[str, float | None]) -> str:
    scored = {k: v for k, v in values.items() if v is not None}
    if not scored:
        return "--"
    winner = max(scored, key=lambda k: scored[k])
    return f"{winner} ({scored[winner]:.3f})"


def track_a_metrics_table() -> str:
    """Full Track A metrics, encoders then text baselines, best per column in bold.

    Only the encoders compete for the bold: the baselines are a reference line,
    not contenders, and they never top a column anyway.
    """
    paths = {m: RESULTS / "track_a" / m / "full" / "seed_0.json" for m in MODELS}
    scores = {m: retrieval(p) for m, p in paths.items()}
    tops = {
        key: max((s[key] for s in scores.values() if s.get(key) is not None), default=None)
        for key, _ in METRICS
    }

    rows = []
    for m in MODELS:
        row = [DISPLAY[m]]
        for key, _ in METRICS:
            value = scores[m].get(key)
            text = cell(value)
            if value is not None and value == tops[key]:
                text = f"**{text}**"
            row.append(text)
        rows.append(row)
    for d, label in BASELINES.items():
        record = retrieval(RESULTS / d / "A" / "vidore" / "seed_0.json")
        rows.append([f"_{label}_"] + [cell(record.get(key)) for key, _ in METRICS])

    return table(["", *[label for _, label in METRICS]], rows)


def multilingual_table() -> str:
    """Best encoder per native-language corpus, against both text pipelines."""
    rows = []
    for lang in LANGS:
        scored = {m: ndcg5(RESULTS / "track_b" / m / lang / "seed_0.json") for m in MODELS}
        scored = {k: v for k, v in scored.items() if v is not None}
        winner = max(scored, key=lambda k: scored[k]) if scored else None
        rows.append(
            [
                LANG_NAMES[lang],
                f"{DISPLAY[winner]} ({scored[winner]:.3f})" if winner else "--",
                cell(ndcg5(RESULTS / "baseline_text" / "B" / lang / "seed_0.json")),
                cell(ndcg5(RESULTS / "mmore_text" / "B" / lang / "seed_0.json")),
            ]
        )
    return table(["corpus", "best encoder", "dense bge-m3", "mmore hybrid"], rows)


def inject(text: str, name: str, block: str) -> str:
    """Replace the README region delimited by the markers for `name`."""
    start, end = f"<!-- generated:{name} -->", "<!-- /generated -->"
    head, _, rest = text.partition(start)
    if not rest:
        raise SystemExit(f"README marker missing: {start}")
    _, _, tail = rest.partition(end)
    return f"{head}{start}\n{block}\n{end}{tail}"


def render_readme() -> None:
    text = README.read_text()
    text = inject(text, "track-a-metrics", track_a_metrics_table())
    text = inject(text, "multilingual", multilingual_table())
    README.write_text(text)
    print(f"wrote {README}")


def main() -> None:
    out: list[str] = [
        "# Results summary",
        "",
        "Generated by `scripts/summarize_results.py` from the records in "
        "`results/`. Metric: **nDCG@5**. Do not edit by hand.",
        "",
    ]

    # --- Track A: ViDoRe biomedical -------------------------------------
    out += ["## Track A — ViDoRe biomedical (English, vision-grounded queries)", ""]
    rows = [[m, cell(ndcg5(RESULTS / "track_a" / m / "full" / "seed_0.json"))] for m in MODELS]
    for d, label in BASELINES.items():
        rows.append([f"_{label}_", cell(ndcg5(RESULTS / d / "A" / "vidore" / "seed_0.json"))])
    out += [table(["model", "nDCG@5"], rows), ""]

    # --- Track B: native corpora ----------------------------------------
    out += [
        "## Track B — native-language corpora (English queries, cross-lingual)",
        "",
        "PMC OA for EN/ZH/DE/ES, HAL for FR. Natively written, never translated.",
        "",
    ]
    rows = []
    for m in MODELS:
        rows.append(
            [m] + [cell(ndcg5(RESULTS / "track_b" / m / lang / "seed_0.json")) for lang in LANGS]
        )
    for d, label in BASELINES.items():
        rows.append(
            [f"_{label}_"]
            + [cell(ndcg5(RESULTS / d / "B" / lang / "seed_0.json")) for lang in LANGS]
        )
    out += [table(["model", *LANGS], rows), ""]

    out += ["Best model per language:", ""]
    rows = [
        [
            lang,
            best({m: ndcg5(RESULTS / "track_b" / m / lang / "seed_0.json") for m in MODELS}),
            cell(ndcg5(RESULTS / "baseline_text" / "B" / lang / "seed_0.json")),
            cell(ndcg5(RESULTS / "mmore_text" / "B" / lang / "seed_0.json")),
        ]
        for lang in LANGS
    ]
    out += [table(["lang", "best ColVision", "dense bge-m3", "mmore hybrid"], rows), ""]

    # --- Track B ViDoRe multilingual ------------------------------------
    out += [
        "## Track B — ViDoRe multilingual slice (fixed index, translated queries)",
        "",
    ]
    rows = [
        [m] + [cell(ndcg5(RESULTS / "track_b_vidore" / m / lang / "seed_0.json")) for lang in VIDORE_LANGS]
        for m in MODELS
    ]
    out += [table(["model", *VIDORE_LANGS], rows), ""]

    # --- Track A, full metric set ---------------------------------------
    out += ["## Track A — all metrics", "", track_a_metrics_table(), ""]

    (RESULTS / "SUMMARY.md").write_text("\n".join(out))
    print(f"wrote {RESULTS / 'SUMMARY.md'}")
    render_readme()


if __name__ == "__main__":
    main()
