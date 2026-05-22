"""LaTeX table generation for the technical report (Track A, Track B, methodology validation)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmark_colvision.results.aggregate import per_cell_ci


def _fmt_ci(point: float, lower: float, upper: float, decimals: int = 3) -> str:
    if any(pd.isna(v) for v in (point, lower, upper)):
        return "--"
    return f"{point:.{decimals}f} [{lower:.{decimals}f}, {upper:.{decimals}f}]"


def track_a_table(df: pd.DataFrame, metric: str = "retrieval.ndcg_at_5") -> str:
    """One LaTeX table: rows = models, columns = paliers, cells = metric with 95% CI."""
    ci = per_cell_ci(df, metric=metric, group_cols=["model_id", "palier_id"])
    pivot = ci.pivot(index="model_id", columns="palier_id", values=["point", "lower", "upper"])
    paliers = sorted(set(ci["palier_id"].dropna()))
    cols = " & ".join(["Model"] + paliers)
    lines = [
        "\\begin{tabular}{l" + "c" * len(paliers) + "}",
        "\\toprule",
        cols + " \\\\",
        "\\midrule",
    ]
    for model_id, row in pivot.iterrows():
        cells = [model_id]
        for pal in paliers:
            try:
                cells.append(
                    _fmt_ci(
                        row[("point", pal)],
                        row[("lower", pal)],
                        row[("upper", pal)],
                    )
                )
            except KeyError:
                cells.append("--")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def track_b_table(df: pd.DataFrame, metric: str = "retrieval.ndcg_at_5") -> str:
    """Rows = models, columns = languages, cells = metric with 95% CI."""
    ci = per_cell_ci(df, metric=metric, group_cols=["model_id", "language"])
    pivot = ci.pivot(index="model_id", columns="language", values=["point", "lower", "upper"])
    langs = sorted(set(ci["language"].dropna()))
    cols = " & ".join(["Model"] + langs)
    lines = [
        "\\begin{tabular}{l" + "c" * len(langs) + "}",
        "\\toprule",
        cols + " \\\\",
        "\\midrule",
    ]
    for model_id, row in pivot.iterrows():
        cells = [model_id]
        for lang in langs:
            try:
                cells.append(
                    _fmt_ci(
                        row[("point", lang)],
                        row[("lower", lang)],
                        row[("upper", lang)],
                    )
                )
            except KeyError:
                cells.append("--")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def methodology_validation_table(rows: list[dict]) -> str:
    """`rows` is the JSON output of queries.methodology_validation.run()."""
    header = ["Subset", "Pearson", "Spearman", "N queries"]
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['subset']} & {r['pearson']:.3f} & {r['spearman']:.3f} & {r['n_queries']} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def save_table(latex: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex + "\n")
    return out_path
