"""Tests for the LaTeX tables and figures pipeline.

Figures are rendered to a temporary path with matplotlib's non-interactive Agg
backend (forced in conftest) and we only check that the file exists and is
non-empty — pixel-level rendering is not part of the contract.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from benchmark_colvision.reporting.latex_tables import (
    methodology_validation_table,
    track_a_table,
    track_b_table,
)


@pytest.fixture
def track_a_df() -> pd.DataFrame:
    rows = []
    for model in ("colpali", "colqwen3"):
        for palier in ("tiny", "small"):
            for seed in (0, 1, 2):
                rows.append(
                    {
                        "track": "A",
                        "model_id": model,
                        "palier_id": palier,
                        "language": None,
                        "seed": seed,
                        "retrieval.ndcg_at_5": 0.6 + 0.05 * (model == "colqwen3") + 0.01 * seed,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def track_b_df() -> pd.DataFrame:
    rows = []
    for model in ("colpali", "colgemma3"):
        for lang in ("en", "fr", "zh"):
            for seed in (0, 1, 2):
                rows.append(
                    {
                        "track": "B",
                        "model_id": model,
                        "palier_id": None,
                        "language": lang,
                        "seed": seed,
                        "retrieval.ndcg_at_5": 0.5
                        + 0.1 * (model == "colgemma3" and lang == "zh")
                        + 0.005 * seed,
                    }
                )
    return pd.DataFrame(rows)


def test_track_a_table_emits_latex(track_a_df: pd.DataFrame) -> None:
    tex = track_a_table(track_a_df, metric="retrieval.ndcg_at_5")
    assert "\\begin{tabular}" in tex
    assert "colpali" in tex
    assert "colqwen3" in tex
    assert "tiny" in tex


def test_track_b_table_emits_latex(track_b_df: pd.DataFrame) -> None:
    tex = track_b_table(track_b_df, metric="retrieval.ndcg_at_5")
    assert "\\begin{tabular}" in tex
    assert "colgemma3" in tex
    assert "zh" in tex


def test_methodology_validation_table_renders() -> None:
    rows = [
        {"subset": "ViDoRe-healthcare", "pearson": 0.91, "spearman": 0.89, "n_queries": 50},
        {"subset": "PMC-internal", "pearson": 0.87, "spearman": 0.85, "n_queries": 40},
    ]
    tex = methodology_validation_table(rows)
    assert "Pearson" in tex
    assert "0.910" in tex
    assert "ViDoRe-healthcare" in tex


def test_track_a_figures_save_files(track_a_df: pd.DataFrame, tmp_path: Path) -> None:
    from benchmark_colvision.reporting.figures_track_a import (
        gpu_memory_curve,
        latency_curve,
        scaling_curve,
        throughput_bar,
    )

    # Add the columns expected by the perf figures
    df = track_a_df.copy()
    df["performance.throughput_pages_per_s"] = 12.0
    df["performance.latency_p50_ms"] = 100.0
    df["performance.latency_p95_ms"] = 200.0
    df["performance.latency_p99_ms"] = 300.0
    df["performance.gpu_peak_mb"] = 8000.0

    out = scaling_curve(df, metric="retrieval.ndcg_at_5", out_path=tmp_path / "a_scaling.png")
    assert out.exists() and out.stat().st_size > 0
    out = throughput_bar(df, out_path=tmp_path / "a_throughput.png")
    assert out.exists() and out.stat().st_size > 0
    out = latency_curve(df, out_path=tmp_path / "a_latency.png")
    assert out.exists() and out.stat().st_size > 0
    out = gpu_memory_curve(df, out_path=tmp_path / "a_gpu.png")
    assert out.exists() and out.stat().st_size > 0


def test_track_b_figures_save_files(track_b_df: pd.DataFrame, tmp_path: Path) -> None:
    from benchmark_colvision.reporting.figures_track_b import (
        heatmap_metric,
        language_gap_bars,
    )

    out = heatmap_metric(track_b_df, metric="retrieval.ndcg_at_5", out_path=tmp_path / "b_heat.png")
    assert out.exists() and out.stat().st_size > 0
    out = language_gap_bars(
        track_b_df, metric="retrieval.ndcg_at_5", out_path=tmp_path / "b_gap.png"
    )
    assert out.exists() and out.stat().st_size > 0
