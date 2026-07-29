#!/usr/bin/env python3
"""Render the report figures from the BenchmarkRecords in `results/`.

Same contract as `summarize_results.py`: every value is read from a record on
disk, never typed by hand. Two outputs per figure --- a vector PDF for LaTeX
(`report/figures/`) and a raster PNG for the README (`assets/`).

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_results import BASELINES, LANGS, MODELS, RESULTS, VIDORE_LANGS, ndcg5  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "report" / "figures"
PNG_DIR = ROOT / "assets"

# Display names, and the two-way split the figures encode by colour: the four
# large encoders against the two ColSmol ones.
LABELS = {
    "colqwen2_5_v0_2": "ColQwen2.5",
    "colgemma3_colnetra": "ColGemma3",
    "colqwen2_v1_0": "ColQwen2",
    "colpali_v1_3": "ColPali",
    "colsmol_500m": "ColSmol-500M",
    "colsmol_256m": "ColSmol-256M",
}
SMALL = {"colsmol_256m", "colsmol_500m"}
LANG_LABELS = {"en": "EN", "fr": "FR", "zh": "ZH", "de": "DE", "es": "ES"}
VIDORE_LANG_LABELS = {"english": "EN", "french": "FR", "german": "DE", "spanish": "ES"}

# Light-surface values from the reference palette; slots 1 and 2 clear every
# all-pairs gate (worst CVD dE 24.7, normal-vision 33.6).
BLUE = "#2a78d6"
BLUE_SOFT = "#9ec5f4"
ORANGE = "#eb6834"
ORANGE_SOFT = "#f5b79c"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e6e5df"
SEQ = ["#eef5fe", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
       "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"]

plt.rcParams.update({
    # Serif, to sit with the report's Times body text.
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def colour(model: str, *, soft: bool = False) -> str:
    if soft:
        return ORANGE_SOFT if model in SMALL else BLUE_SOFT
    return ORANGE if model in SMALL else BLUE


def record(path: Path) -> dict:
    return json.loads(path.read_text())


def per_query_s(path: Path) -> float:
    """Retrieval wall-clock per query. The only timing the records expose."""
    rec = record(path)
    return rec["performance"]["retrieve_duration_total_s"] / rec["retrieval"]["n_queries"]


def bare(ax, *, grid_axis: str | None = "y", keep=("left", "bottom")) -> None:
    """Strip the frame down to the axes that carry meaning."""
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(True, axis=grid_axis, linewidth=0.5, color=GRID)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    for side in keep:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(length=2.5, width=0.5, pad=2)


def save(fig, stem: str, title: str) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    # The PNG stands alone in the README, with no LaTeX caption to name it.
    fig.suptitle(title, fontsize=9.5, fontweight="bold", color=INK,
                 x=0.0, y=1.05, ha="left")
    fig.savefig(PNG_DIR / f"{stem}.png", dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"wrote {stem}.pdf / {stem}.png")


# --- Figure 1: quality and cost on the biomedical lecture corpus -------------


def fig_quality_and_cost() -> None:
    scores = {m: ndcg5(RESULTS / "track_a" / m / "full" / "seed_0.json") for m in MODELS}
    costs = {m: per_query_s(RESULTS / "track_a" / m / "full" / "seed_0.json") for m in MODELS}
    text = {label: ndcg5(RESULTS / d / "A" / "vidore" / "seed_0.json")
            for d, label in BASELINES.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.7, 2.45),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    # (a) horizontal bars: names read left-to-right, no rotated tick labels.
    order = sorted(scores, key=lambda m: scores[m])
    ys = range(len(order))
    ax1.barh(ys, [scores[m] for m in order], height=0.58,
             color=[colour(m) for m in order], zorder=3)
    for y, m in zip(ys, order):
        ax1.text(scores[m] + 0.012, y, f"{scores[m]:.3f}", va="center",
                 fontsize=7.5, color=INK_2, zorder=5,
                 bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
    for label, value in text.items():
        ax1.axvline(value, color=MUTED, linewidth=0.9, linestyle=(0, (3, 2.5)), zorder=2)
    ax1.text(max(text.values()) + 0.012, -0.95,
             "  ·  ".join(f"{label} {value:.3f}"
                          for label, value in sorted(text.items(), key=lambda kv: -kv[1])),
             fontsize=6.8, color=MUTED, va="center")
    ax1.set_yticks(list(ys))
    ax1.set_yticklabels([LABELS[m] for m in order], color=INK_2)
    ax1.set_ylim(-1.5, len(order) - 0.4)
    ax1.set_xlim(0, 0.72)
    ax1.set_xlabel("nDCG@5")
    ax1.set_title("(a) Qualité de recherche", loc="left", color=INK, pad=6)
    bare(ax1, grid_axis="x", keep=("left",))

    # (b) the same six models, quality against what a query costs.
    offsets = {
        "colgemma3_colnetra": ((0, 10), "center"),
        "colpali_v1_3": ((0, -12), "center"),
        "colqwen2_v1_0": ((-6, 9), "center"),
        "colqwen2_5_v0_2": ((8, 1), "left"),
        "colsmol_500m": ((-7, 0), "right"),
        "colsmol_256m": ((7, 0), "left"),
    }
    for m in MODELS:
        offset, ha = offsets[m]
        ax2.scatter(costs[m], scores[m], s=40, color=colour(m), zorder=3,
                    edgecolor="white", linewidth=0.9)
        ax2.annotate(LABELS[m], (costs[m], scores[m]), textcoords="offset points",
                     xytext=offset, ha=ha, va="center", fontsize=7, color=INK_2)
    span = max(costs.values()) - min(costs.values())
    ax2.set_xlim(min(costs.values()) - 0.16 * span, max(costs.values()) + 0.44 * span)
    ax2.set_ylim(0.30, 0.72)
    ax2.set_xlabel("temps de recherche par requête (s)")
    ax2.set_ylabel("nDCG@5")
    ax2.set_title("(b) Coût d'une requête", loc="left", color=INK, pad=6)
    bare(ax2, grid_axis="both")

    fig.tight_layout(w_pad=2.4)
    save(fig, "quality_and_cost",
         "Four of six visual encoders beat text retrieval — at four times the cost spread")


# --- Figure 2: query language, index unchanged -------------------------------


def fig_query_language() -> None:
    steps = ["english", *VIDORE_LANGS]
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    for m in MODELS:
        ys = [ndcg5(RESULTS / "track_a" / m / "full" / "seed_0.json")] + [
            ndcg5(RESULTS / "track_b_vidore" / m / lang / "seed_0.json")
            for lang in VIDORE_LANGS
        ]
        ax.plot(range(len(steps)), ys, marker="o", markersize=3.8, linewidth=1.4,
                color=colour(m), zorder=3, markeredgecolor="white",
                markeredgewidth=0.7, clip_on=False)
    ax.annotate("4 gros encodeurs", (0.04, 0.665), fontsize=7.5, color=BLUE)
    ax.annotate("ColSmol", (0.72, 0.215), fontsize=7.5, color=ORANGE)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([VIDORE_LANG_LABELS[s] for s in steps], color=INK_2)
    ax.set_xlim(-0.08, len(steps) - 0.92)
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("nDCG@5")
    ax.set_xlabel("langue de la requête")
    bare(ax)
    save(fig, "query_language",
         "Only the small encoders lose the query language")


# --- Figure 3: native-language corpora ---------------------------------------


def fig_multilingual_heatmap() -> None:
    rows = [(LABELS[m], [ndcg5(RESULTS / "track_b" / m / lang / "seed_0.json") for lang in LANGS])
            for m in MODELS]
    rows += [(label, [ndcg5(RESULTS / d / "B" / lang / "seed_0.json") for lang in LANGS])
             for d, label in BASELINES.items()]
    values = [r[1] for r in rows]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    lo, hi = min(min(v) for v in values), max(max(v) for v in values)
    norm = matplotlib.colors.Normalize(vmin=lo - 0.08, vmax=hi + 0.02)

    fig, ax = plt.subplots(figsize=(3.35, 2.75))
    ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")

    for i, (_, vals) in enumerate(rows):
        for j, v in enumerate(vals):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="white", linewidth=1.6, zorder=2))
            r, g, b, _ = cmap(norm(v))
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.2,
                    zorder=3, color="white" if luminance < 0.45 else INK)

    ax.set_xticks(range(len(LANGS)))
    ax.set_xticklabels([LANG_LABELS[lang] for lang in LANGS], color=INK_2)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], color=INK_2)
    for label, model in zip(ax.get_yticklabels(), list(MODELS) + list(BASELINES.values())):
        if model not in MODELS:
            label.set_style("italic")
            label.set_color(MUTED)
    ax.tick_params(length=0, pad=3)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    # A visible gap between the encoders and the text reference rows.
    ax.axhline(len(MODELS) - 0.5, color="white", linewidth=4.5, zorder=4)

    save(fig, "multilingual_heatmap",
         "In all five languages the best encoder beats both text pipelines")


def main() -> None:
    fig_quality_and_cost()
    fig_query_language()
    fig_multilingual_heatmap()


if __name__ == "__main__":
    main()
