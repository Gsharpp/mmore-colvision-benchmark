"""Track A figures: scaling curves, latency vs corpus size, GPU memory."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from benchmark_colvision.results.aggregate import per_cell_ci


PALIER_ORDER = ["tiny", "small", "medium", "large"]
PALIER_PAGES = {"tiny": 100, "small": 1000, "medium": 10000, "large": 50000}


def _palier_x(df: pd.DataFrame) -> pd.Series:
    return df["palier_id"].map(PALIER_PAGES)


def scaling_curve(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    *,
    ylabel: str | None = None,
    log_x: bool = True,
) -> Path:
    """Plot metric vs corpus size, one curve per model, with bootstrap CI bands."""
    ci = per_cell_ci(df, metric=metric, group_cols=["model_id", "palier_id"])
    ci["pages"] = ci["palier_id"].map(PALIER_PAGES)
    ci = ci.sort_values(["model_id", "pages"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model_id, sub in ci.groupby("model_id"):
        sub = sub.dropna(subset=["pages"])
        ax.plot(sub["pages"], sub["point"], marker="o", label=model_id)
        ax.fill_between(sub["pages"], sub["lower"], sub["upper"], alpha=0.18)
    ax.set_xlabel("Corpus size (pages)")
    ax.set_ylabel(ylabel or metric)
    if log_x:
        ax.set_xscale("log")
    ax.set_title(f"Track A — {metric} vs corpus size")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def throughput_bar(df: pd.DataFrame, out_path: Path) -> Path:
    """Bar chart of indexing throughput (pages/s) per model, faceted by palier."""
    metric = "performance.throughput_pages_per_s"
    pivot = (
        df.dropna(subset=[metric])
        .groupby(["model_id", "palier_id"])[metric]
        .mean()
        .unstack("palier_id")
        .reindex(columns=[c for c in PALIER_ORDER if c in df["palier_id"].unique()])
    )
    ax = pivot.plot(kind="bar", figsize=(8, 4.5))
    ax.set_ylabel("Throughput (pages / s)")
    ax.set_xlabel("Model")
    ax.set_title("Track A — indexing throughput by corpus size")
    ax.legend(title="palier", fontsize=8)
    fig = ax.get_figure()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def latency_curve(df: pd.DataFrame, out_path: Path) -> Path:
    """Latency p50/p95/p99 vs corpus size, faceted per model."""
    metrics = ("performance.latency_p50_ms", "performance.latency_p95_ms", "performance.latency_p99_ms")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, metric in zip(axes, metrics):
        for model_id, sub in df.groupby("model_id"):
            sub = sub.dropna(subset=[metric]).sort_values("palier_id")
            sub_means = sub.groupby("palier_id")[metric].mean().reindex(PALIER_ORDER).dropna()
            pages = [PALIER_PAGES[p] for p in sub_means.index]
            ax.plot(pages, sub_means.values, marker="o", label=model_id)
        ax.set_xscale("log")
        ax.set_title(metric.split(".")[-1])
        ax.set_xlabel("Corpus size (pages)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Latency (ms)")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Track A — retrieval latency vs corpus size")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def gpu_memory_curve(df: pd.DataFrame, out_path: Path) -> Path:
    """GPU peak memory (MB) vs corpus size per model."""
    metric = "performance.gpu_peak_mb"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model_id, sub in df.groupby("model_id"):
        means = (
            sub.dropna(subset=[metric])
            .groupby("palier_id")[metric]
            .mean()
            .reindex(PALIER_ORDER)
            .dropna()
        )
        pages = [PALIER_PAGES[p] for p in means.index]
        ax.plot(pages, means.values, marker="o", label=model_id)
    ax.set_xscale("log")
    ax.set_xlabel("Corpus size (pages)")
    ax.set_ylabel("GPU peak memory (MB)")
    ax.set_title("Track A — GPU memory peak vs corpus size")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
