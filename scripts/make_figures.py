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
# Colour splits the two families; within a family the models share a hue, so
# line style and marker carry the identity. Without this a six-entry legend
# shows four identical blue swatches and names nothing.
DASHES = {
    "colqwen2_5_v0_2": (None, "o"),
    "colgemma3_colnetra": ((0, (4, 1.6)), "s"),
    "colqwen2_v1_0": ((0, (1.4, 1.4)), "^"),
    "colpali_v1_3": ((0, (5, 1.4, 1.2, 1.4)), "D"),
    "colsmol_500m": (None, "o"),
    "colsmol_256m": ((0, (1.4, 1.4)), "^"),
}
LANG_LABELS = {"en": "EN", "fr": "FR", "zh": "ZH", "de": "DE", "es": "ES"}
VIDORE_LANG_LABELS = {"english": "EN", "french": "FR", "german": "DE", "spanish": "ES"}

# The figures are rendered in English for BOTH outputs, on purpose: the same
# files are embedded in the French report and in the English README, and the
# report's eventual English translation will not regenerate them. Keeping one
# English set means no figure ever contradicts the document around it. The
# French strings are kept below in case that decision is revisited.
LANG = "en"
STRINGS = {
    "quality": {"fr": "(a) Qualité de recherche", "en": "(a) Retrieval quality"},
    "cost": {"fr": "(b) Coût d'une requête", "en": "(b) Cost of one query"},
    "cost_axis": {"fr": "temps de recherche par requête (s)",
                  "en": "retrieval time per query (s)"},
    "query_lang_axis": {"fr": "langue de la requête", "en": "query language"},
    "big_four": {"fr": "4 gros encodeurs", "en": "4 large encoders"},
    "colsmol": {"fr": "ColSmol", "en": "ColSmol"},
}


def t(key: str) -> str:
    return STRINGS[key][LANG]


def num(value: float, digits: int) -> str:
    """Format a number for the current language (French uses a decimal comma)."""
    out = f"{value:.{digits}f}"
    return out.replace(".", ",") if LANG == "fr" else out

# Calqué sur les figures du papier MMORE (report/examples/) : une seule teinte
# d'accent et des neutres, plutôt qu'une opposition de deux couleurs vives. Les
# six encodeurs portent l'orange — teinte pleine pour les quatre gros, teinte
# claire pour les deux ColSmol — et les pipelines textuels, qui sont une
# référence et non une série, restent en gris.
ACCENT = "#1d4f96"
ACCENT_SOFT = "#5599e0"
DARK = "#2f2f2f"
INK = "#111111"
INK_2 = "#333333"
MUTED = "#5a5a5a"
REF = "#7d7b76"
REF_SOFT = "#d7d5cf"
GRID = "#e6e5df"
# Rampe séquentielle sur la même teinte, clair -> foncé.
SEQ = ["#eff5fd", "#dde9fa", "#c9dcf6", "#b4cef2", "#9ec0ee", "#88b2e9",
       "#72a4e4", "#5c96de", "#4784cd", "#3671b8", "#2960a6", "#1d4f96"]

# Les six encodeurs partagent une teinte : le style de trait et le marqueur
# portent l'identité. Sans cela une légende de six entrées montre quatre
# pastilles identiques et ne nomme rien.
plt.rcParams.update({
    # TeX Gyre Termes is the Times clone newtxtext sets the body in, so the
    # figures and the text share one typeface instead of merely both being
    # serif. Math (the @ in nDCG@5, subscripts) follows with the STIX set.
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.edgecolor": INK_2,
    "axes.linewidth": 0.7,
    "legend.frameon": True,
    "legend.edgecolor": INK_2,
    "legend.facecolor": "white",
    "legend.framealpha": 1.0,
    "legend.fancybox": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def colour(model: str, *, soft: bool = False) -> str:
    """Full accent for the four large encoders, a tint for the two ColSmol."""
    return ACCENT_SOFT if model in SMALL else ACCENT


def record(path: Path) -> dict:
    return json.loads(path.read_text())


def per_query_s(path: Path) -> float:
    """Retrieval wall-clock per query. The only timing the records expose."""
    rec = record(path)
    return rec["performance"]["retrieve_duration_total_s"] / rec["retrieval"]["n_queries"]


def bare(ax, *, grid_axis: str | None = "y", keep=("left", "bottom")) -> None:
    """Closed rectangular frame, no gridlines — the house style of the MMORE
    paper this report is set against. `grid_axis` and `keep` are accepted and
    ignored so the call sites read the same."""
    ax.set_axisbelow(True)
    ax.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(length=2.6, width=0.7, pad=2.5, direction="out",
                   top=False, right=False)


def save(fig, stem: str, title: str, *, box: dict | None = None) -> None:
    """`box` is kept for figures whose labels sit outside the axes (the heatmap
    puts its column headers on top), but every output is cropped to its own
    content: with \centering and width=\columnwidth a tightly cropped figure
    fills the column exactly, so it is centred by construction. A shared axes
    rectangle was tried and abandoned — it aligned the plots with each other but
    left figures with short labels visibly off-centre in the column."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    if box:
        fig.subplots_adjust(**box)
    else:
        fig.tight_layout(pad=0.3)
    # The report names its figures in a LaTeX caption, so the PDF carries none.
    fig.savefig(PDF_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    # The PNG stands alone in the README, so it keeps its headline and is
    # cropped to fit it. The fixed box above exists only to align the figures
    # stacked in the report's columns, which carry LaTeX captions instead.
    fig.suptitle(title, fontsize=9, fontweight="bold", color=INK,
                 x=0.0, y=1.04, ha="left")
    fig.savefig(PNG_DIR / f"{stem}.png", dpi=200, bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {stem}.pdf / {stem}.png")


# --- Figure 1: quality and cost on the biomedical lecture corpus -------------


def fig_retrieval_quality() -> None:
    scores = {m: ndcg5(RESULTS / "track_a" / m / "full" / "seed_0.json") for m in MODELS}
    text = {label: ndcg5(RESULTS / d / "A" / "vidore" / "seed_0.json")
            for d, label in BASELINES.items()}

    fig, ax = plt.subplots(figsize=(3.35, 2.35))

    # Horizontal bars: names read left-to-right, no rotated tick labels. The two
    # text pipelines used to be dashed vertical lines; at 0.492 and 0.473 they
    # landed on top of each other and their shared label collided with the bars,
    # so they are drawn as two gray bars below a gap instead — same comparison,
    # read off the same axis, no overlap possible.
    order = sorted(scores, key=lambda m: scores[m])
    refs = sorted(text.items(), key=lambda kv: kv[1])
    gap = 0.9
    ys_models = [i + len(refs) + gap for i in range(len(order))]
    ys_refs = list(range(len(refs)))

    ax.barh(ys_models, [scores[m] for m in order], height=0.62,
            color=[colour(m) for m in order], zorder=3)
    ax.barh(ys_refs, [v for _, v in refs], height=0.62,
            color=REF_SOFT, edgecolor=REF, linewidth=0.6, zorder=3)

    for y, value in zip(ys_models, [scores[m] for m in order]):
        ax.text(value + 0.012, y, num(value, 3), va="center",
                fontsize=8, color=INK_2, zorder=5)
    for y, (_, value) in zip(ys_refs, refs):
        ax.text(value + 0.012, y, num(value, 3), va="center",
                fontsize=8, color=REF, zorder=5)

    ax.set_yticks(ys_refs + ys_models)
    ax.set_yticklabels([lab for lab, _ in refs] + [LABELS[m] for m in order],
                       color=INK_2)
    for label in ax.get_yticklabels()[:len(refs)]:
        label.set_style("italic")
        label.set_color(REF)
    ax.set_ylim(-0.7, len(order) + len(refs) + gap - 0.4)
    ax.set_xlim(0, 0.72)
    ax.set_xlabel("nDCG@5")
    bare(ax, grid_axis="x", keep=("left",))

    save(fig, "retrieval_quality",
         "Four of six visual encoders beat text retrieval")


def fig_retrieval_cost() -> None:
    scores = {m: ndcg5(RESULTS / "track_a" / m / "full" / "seed_0.json") for m in MODELS}
    costs = {m: per_query_s(RESULTS / "track_a" / m / "full" / "seed_0.json") for m in MODELS}

    fig, ax = plt.subplots(figsize=(3.35, 1.95))
    offsets = {
        "colgemma3_colnetra": ((0, 10), "center"),
        "colpali_v1_3": ((0, -12), "center"),
        "colqwen2_v1_0": ((-2, 10), "center"),
        "colqwen2_5_v0_2": ((0, -11), "center"),
        "colsmol_500m": ((-7, 0), "right"),
        "colsmol_256m": ((-7, 0), "right"),
    }
    for m in MODELS:
        offset, ha = offsets[m]
        ax.scatter(costs[m], scores[m], s=40, color=colour(m), zorder=3,
                   edgecolor="white", linewidth=0.9)
        ax.annotate(LABELS[m], (costs[m], scores[m]), textcoords="offset points",
                    xytext=offset, ha=ha, va="center", fontsize=7.5, color=INK_2)
    span = max(costs.values()) - min(costs.values())
    ax.set_xlim(min(costs.values()) - 0.22 * span, max(costs.values()) + 0.12 * span)
    ax.set_ylim(0.30, 0.74)
    ax.set_xlabel(t("cost_axis"))
    ax.set_ylabel("nDCG@5")
    bare(ax, grid_axis="both")

    save(fig, "retrieval_cost",
         "Cost per query spreads four-fold, and not by model size")


# --- Figure 2: query language, index unchanged -------------------------------


def fig_query_language() -> None:
    steps = ["english", *VIDORE_LANGS]
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    for m in MODELS:
        ys = [ndcg5(RESULTS / "track_a" / m / "full" / "seed_0.json")] + [
            ndcg5(RESULTS / "track_b_vidore" / m / lang / "seed_0.json")
            for lang in VIDORE_LANGS
        ]
        dash, marker = DASHES[m]
        # One reading key across every figure: the ColSmol pair always wears the
        # light tint. A tint is fainter than the full accent at the same weight,
        # so their lines are drawn thicker rather than recoloured.
        ax.plot(range(len(steps)), ys, marker=marker, markersize=3.6,
                linewidth=1.9 if m in SMALL else 1.3,
                linestyle=dash if dash else "-",
                color=colour(m), zorder=3, markeredgecolor="white",
                markeredgewidth=0.6, clip_on=False, label=LABELS[m])
    # Six lines in two colours cannot be told apart by colour alone, so the
    # group annotations are replaced by a real legend naming every model.
    # Below the axes, centred: inside the frame the box sat on top of the
    # ColSmol curves and hid the very data the figure is about.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2,
              handlelength=2.2, handletextpad=0.5, columnspacing=1.4,
              fontsize=7, borderpad=0.45, labelspacing=0.32, labelcolor=INK)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([VIDORE_LANG_LABELS[s] for s in steps], color=INK_2)
    ax.set_xlim(-0.08, len(steps) - 0.92)
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("nDCG@5")
    ax.set_xlabel(t("query_lang_axis"))
    bare(ax)
    save(fig, "query_language",
         "Only the small encoders lose the query language",
         box=dict(left=0.285, right=0.975, top=0.965, bottom=0.44))


# --- Figure 3: native-language corpora ---------------------------------------


def fig_multilingual_heatmap() -> None:
    rows = [(LABELS[m], [ndcg5(RESULTS / "track_b" / m / lang / "seed_0.json") for lang in LANGS])
            for m in MODELS]
    # Same order as the bar chart and the metrics table: mmore first.
    rows += [(label, [ndcg5(RESULTS / d / "B" / lang / "seed_0.json") for lang in LANGS])
             for d, label in reversed(list(BASELINES.items()))]
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
            ax.text(j, i, num(v, 2), ha="center", va="center", fontsize=7.6,
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
         "In all five languages the best encoder beats both text pipelines",
         box=dict(left=0.285, right=0.975, top=0.91, bottom=0.045))


def main() -> None:
    fig_retrieval_quality()
    fig_retrieval_cost()
    fig_query_language()
    fig_multilingual_heatmap()


if __name__ == "__main__":
    main()
