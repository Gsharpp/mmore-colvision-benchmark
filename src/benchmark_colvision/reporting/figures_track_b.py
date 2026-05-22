"""Track B figures: model × language heatmaps and language-gap bars."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def heatmap_metric(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    *,
    cmap: str = "viridis",
    fmt: str = ".2f",
) -> Path:
    """Render a model × language heatmap, averaging over seeds."""
    pivot = (
        df.dropna(subset=[metric])
        .groupby(["model_id", "language"])[metric]
        .mean()
        .unstack("language")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(1.2 * pivot.shape[1] + 2, 0.55 * pivot.shape[0] + 1.5))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, ax=ax, cbar_kws={"label": metric})
    ax.set_title(f"Track B — {metric}")
    ax.set_xlabel("Language")
    ax.set_ylabel("Model")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def language_gap_bars(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    *,
    baseline: str = "en",
) -> Path:
    """Plot relative drop (Δ% vs baseline language) of `metric` per model and language."""
    means = (
        df.dropna(subset=[metric])
        .groupby(["model_id", "language"])[metric]
        .mean()
        .unstack("language")
    )
    if baseline not in means.columns:
        raise ValueError(f"baseline language {baseline!r} missing from results")
    base = means[baseline]
    gap = (means.subtract(base, axis=0).div(base, axis=0) * 100).drop(columns=[baseline])
    fig, ax = plt.subplots(figsize=(1.2 * gap.shape[1] + 3, 0.4 * gap.shape[0] * gap.shape[1] + 2))
    x = np.arange(len(gap.index))
    width = 0.8 / max(gap.shape[1], 1)
    for i, lang in enumerate(gap.columns):
        ax.bar(x + i * width, gap[lang].values, width, label=lang)
    ax.set_xticks(x + width * (gap.shape[1] - 1) / 2)
    ax.set_xticklabels(gap.index, rotation=15)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel(f"Δ% {metric} vs {baseline}")
    ax.set_title(f"Track B — language gap vs {baseline}")
    ax.legend(fontsize=8, title="lang")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
