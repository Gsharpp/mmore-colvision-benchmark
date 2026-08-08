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

# Display names. Colour encodes the backbone family (see FAMILY below), so the
# two Qwen encoders share a hue and the two ColSmol ones share another.
LABELS = {
    "colqwen2_5_v0_2": "ColQwen2.5",
    "colgemma3_colnetra": "ColGemma3",
    "colqwen2_v1_0": "ColQwen2",
    "colpali_v1_3": "ColPali",
    "colsmol_500m": "ColSmol-500M",
    "colsmol_256m": "ColSmol-256M",
}
# The six encoders share one hue, so line style and marker carry the identity
# on their own: every model needs a combination no other model uses.
DASHES = {
    "colqwen2_5_v0_2": (None, "o"),
    "colgemma3_colnetra": ((0, (4, 1.6)), "s"),
    "colqwen2_v1_0": ((0, (1.4, 1.4)), "^"),
    "colpali_v1_3": ((0, (5, 1.4, 1.2, 1.4)), "D"),
    "colsmol_500m": ((0, (2.6, 1.2)), "v"),
    "colsmol_256m": ((0, (1, 1.2, 3.6, 1.2)), "P"),
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

# La couleur code la FAMILLE DE DORSALE, pas le rang ni la taille : les deux
# ColQwen partagent une teinte, les deux ColSmol une autre. C'est la seule
# partition des six qui soit une propriété des modèles et non de leurs scores,
# donc la seule qui ne se repeigne pas quand les chiffres bougent.
# Palette validée par scripts/validate_palette.js de la skill dataviz, en mode
# all-pairs (la figure 2 est un nuage : toutes les paires se voisinent) : pire
# écart CVD 11,0 et vision normale 16,3, au-dessus des planchers de 8 et 15.
# Le rouge est la seule quatrième teinte retenue : l'orange, le marron et le
# framboise ont été écartés par Mathieu, un second violet échoue (ΔE 3,8 contre
# celui de Gemma-3) et le teal aussi. L'aqua passe sous 3:1 de contraste, ce que
# couvrent l'étiquetage direct des figures 1 et 2 et les styles de trait de la
# figure 3. Les pipelines textuels restent en gris : une référence, pas une série.
FAM_BLUE = "#2a78d6"    # Qwen2-VL / Qwen2.5-VL
FAM_VIOLET = "#4a3aa7"  # Gemma-3
FAM_AQUA = "#1baf7a"    # PaliGemma
FAM_RED = "#d32f2f"     # SmolVLM
FAMILY = {
    "colqwen2_5_v0_2": FAM_BLUE,
    "colqwen2_v1_0": FAM_BLUE,
    "colgemma3_colnetra": FAM_VIOLET,
    "colpali_v1_3": FAM_AQUA,
    "colsmol_500m": FAM_RED,
    "colsmol_256m": FAM_RED,
}
ACCENT = FAM_BLUE
DARK = "#2f2f2f"
INK = "#111111"
INK_2 = "#333333"
MUTED = "#5a5a5a"
REF = "#7d7b76"
REF_SOFT = "#d7d5cf"
GRID = "#e6e5df"
# Rampe séquentielle bleue, clair -> foncé, pour la carte. Une rampe neutre a
# été essayée pour éviter que le bleu se lise comme « famille ColQwen » ; rendue,
# elle est nettement moins lisible, et la carte code une magnitude, pas une
# identité, donc la confusion reste théorique. Décision de Mathieu : bleu.
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
    """One hue per backbone family; the two Qwen and the two ColSmol pair up."""
    return FAMILY[model]


def record(path: Path) -> dict:
    return json.loads(path.read_text())


def per_query_s(path: Path) -> float:
    """Retrieval wall-clock per query. The only timing the records expose."""
    rec = record(path)
    return rec["performance"]["retrieve_duration_total_s"] / rec["retrieval"]["n_queries"]


_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def spell(n: int) -> str:
    """Number words for figure titles, so a title never contradicts its data."""
    return _NUMBER_WORDS.get(n, str(n))


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

    n_beat = sum(1 for v in scores.values() if v > max(text.values()))
    save(fig, "retrieval_quality",
         f"{spell(n_beat).capitalize()} of {spell(len(scores))} visual encoders "
         "beat text retrieval")


def fig_retrieval_cost() -> None:
    scores = {m: ndcg5(RESULTS / "track_a" / m / "full" / "seed_0.json") for m in MODELS}
    costs = {m: per_query_s(RESULTS / "track_a" / m / "full" / "seed_0.json") for m in MODELS}

    refs = {label: ndcg5(RESULTS / d / "A" / "vidore" / "seed_0.json")
            for d, label in BASELINES.items()}

    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    # The two text pipelines are the comparison the whole report turns on, so
    # the cost plot carries them too — as rules, not as points: they have no
    # cost on this axis.
    for label, value in refs.items():
        ax.axhline(value, color=REF, linewidth=0.8, linestyle=(0, (3, 2)),
                   zorder=1)
        ax.annotate(label, (0.35, value), textcoords="offset points",
                    xytext=(0, 3), ha="left", va="bottom", fontsize=6.8,
                    color=REF, style="italic")

    # One convention: the label sits to the right of its point, at a fixed
    # offset. It moves only where the right side is already taken, and then by
    # the same amount in the next free direction — above, below, or left for
    # the two ColSmol, which sit against the right edge of the axes.
    placement = {
        "colqwen2_v1_0": ((0, 9), "center"),      # ColQwen2.5 is just downstream
        "colpali_v1_3": ((0, -9), "center"),      # under ColQwen2
        "colsmol_500m": ((-7, 0), "right"),
        "colsmol_256m": ((-7, 0), "right"),
    }
    for m in MODELS:
        offset, ha = placement.get(m, ((7, 0), "left"))
        ax.scatter(costs[m], scores[m], s=40, color=colour(m), zorder=3,
                   edgecolor="white", linewidth=0.9)
        ax.annotate(LABELS[m], (costs[m], scores[m]), textcoords="offset points",
                    xytext=offset, ha=ha, va="center", fontsize=7.5, color=INK_2)

    ax.set_xlim(0, max(costs.values()) * 1.12)
    lo = min([*scores.values(), *refs.values()])
    hi = max(scores.values())
    ax.set_ylim(lo - 0.06, hi + 0.04)
    ax.set_xlabel(t("cost_axis"))
    ax.set_ylabel("nDCG@5")
    bare(ax, grid_axis="both")

    ratio = max(costs.values()) / min(costs.values())
    save(fig, "retrieval_cost",
         f"Cost per query spreads more than {spell(int(ratio))}-fold, "
         "and not by model size")


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
        # Colour names the family; within a family the two members are told
        # apart by dash pattern and marker, so every line carries one weight.
        ax.plot(range(len(steps)), ys, marker=marker, markersize=3.6,
                linewidth=1.5,
                linestyle=dash if dash else "-",
                color=colour(m), zorder=3, markeredgecolor="white",
                markeredgewidth=0.6, clip_on=False, label=LABELS[m])
    # Six lines cannot be told apart by colour alone — two families hold two
    # models each — so a real legend names every model. It sits inside the
    # frame, in the band above the curves: no line rises past 0.63, so the top
    # of the panel is free and the legend costs no figure height.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=3,
              handlelength=2.0, handletextpad=0.45, columnspacing=1.0,
              fontsize=6.6, borderpad=0.4, labelspacing=0.28, labelcolor=INK)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([VIDORE_LANG_LABELS[s] for s in steps], color=INK_2)
    ax.set_xlim(-0.08, len(steps) - 0.92)
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("nDCG@5")
    ax.set_xlabel(t("query_lang_axis"))
    bare(ax)
    save(fig, "query_language",
         "Only the small encoders lose the query language",
         box=dict(left=0.285, right=0.975, top=0.965, bottom=0.19))


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
            # Flip to white only where black ink would drop under ~4.5:1.
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

    # No colour bar: every cell prints its own value, so the scale would only
    # restate what the reader already has, and it costs a fifth of the width.

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
