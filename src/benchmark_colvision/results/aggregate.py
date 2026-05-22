"""Aggregation helpers: load a directory of result JSONs into a pandas frame
and compute per-model bootstrap CIs and pairwise tests across seeds."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from benchmark_colvision.evaluation.statistical_tests import (
    BootstrapCI,
    bootstrap_ci,
    holm_correct,
    wilcoxon_paired,
)
from benchmark_colvision.results.schema import BenchmarkRecord


def load_records(root: Path) -> list[BenchmarkRecord]:
    """Recursively load every *.json benchmark record under `root`."""
    records: list[BenchmarkRecord] = []
    for p in sorted(root.rglob("*.json")):
        try:
            raw = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict) or "cell" not in raw:
            continue
        records.append(BenchmarkRecord.model_validate(raw))
    return records


def records_to_frame(records: Iterable[BenchmarkRecord]) -> pd.DataFrame:
    """Flatten records into a long-format dataframe."""
    rows: list[dict] = []
    for rec in records:
        base = {
            "track": rec.cell.track,
            "model_id": rec.cell.model_id,
            "palier_id": rec.cell.palier_id,
            "language": rec.cell.language,
            "seed": rec.cell.seed,
            "mmore_commit": rec.mmore_commit,
        }
        for k, v in rec.retrieval.model_dump().items():
            base[f"retrieval.{k}"] = v
        for k, v in rec.generation.model_dump().items():
            base[f"generation.{k}"] = v
        for k, v in rec.performance.model_dump().items():
            base[f"performance.{k}"] = v
        rows.append(base)
    return pd.DataFrame(rows)


def per_cell_ci(
    df: pd.DataFrame,
    metric: str,
    group_cols: list[str],
    n_resamples: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute bootstrap CIs for `metric` grouped by `group_cols` (e.g. model_id, palier_id)."""
    out: list[dict] = []
    for keys, sub in df.groupby(group_cols):
        values = sub[metric].dropna().to_numpy()
        ci: BootstrapCI = bootstrap_ci(values, n_resamples=n_resamples, seed=seed)
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(
            metric=metric,
            point=ci.point_estimate,
            lower=ci.lower,
            upper=ci.upper,
            n=len(values),
        )
        out.append(row)
    return pd.DataFrame(out)


def pairwise_wilcoxon(
    df: pd.DataFrame,
    metric: str,
    model_col: str = "model_id",
    pair_col: str = "seed",
) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank between models, paired on `pair_col`."""
    models = sorted(df[model_col].dropna().unique())
    rows: list[dict] = []
    pvals: list[float] = []
    pair_keys: list[tuple[str, str]] = []
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            sub_a = df[df[model_col] == a].sort_values(pair_col)[metric].to_numpy()
            sub_b = df[df[model_col] == b].sort_values(pair_col)[metric].to_numpy()
            if len(sub_a) != len(sub_b) or len(sub_a) < 2:
                continue
            res = wilcoxon_paired(sub_a, sub_b)
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "statistic": res.statistic,
                    "pvalue": res.pvalue,
                    "n_pairs": res.n_pairs,
                }
            )
            pvals.append(res.pvalue)
            pair_keys.append((a, b))
    if not rows:
        return pd.DataFrame(rows)
    adj = holm_correct(pvals)
    for r, p in zip(rows, adj):
        r["pvalue_holm"] = p
    return pd.DataFrame(rows)
