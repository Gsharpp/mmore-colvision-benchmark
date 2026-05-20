"""Unit tests for corpus.visual_density_filter."""

from __future__ import annotations

from pathlib import Path

from benchmark_colvision.corpus.visual_density_filter import (
    DensityReport,
    compute_density,
    keep,
)


def test_keep_policy_threshold():
    low = DensityReport("a.pdf", 1, 0.10, 0.05)
    high = DensityReport("b.pdf", 1, 0.20, 0.20)
    assert not keep(low, threshold=0.30)
    assert keep(high, threshold=0.30)


def test_compute_density_text_only_pdf(tmp_pdf: Path):
    report = compute_density(tmp_pdf)
    assert report.page_count == 1
    # Pure text page → ratios are zero
    assert report.image_surface_ratio == 0.0
    assert report.table_surface_ratio == 0.0
    assert report.visual_density == 0.0
    assert not keep(report)
