"""Compute a visual-density score per PDF to filter for figure-rich documents.

Density = surface covered by images and tables, divided by total page surface,
averaged across pages. The intent is to retain documents where the figures
materially contribute to understanding (so VLM retrieval is meaningful).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class DensityReport:
    pdf_path: str
    page_count: int
    image_surface_ratio: float
    table_surface_ratio: float

    @property
    def visual_density(self) -> float:
        return min(1.0, self.image_surface_ratio + self.table_surface_ratio)


def _bbox_area(bbox) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def compute_density(pdf_path: Path) -> DensityReport:
    """Compute visual-density metrics for a single PDF."""
    image_area = 0.0
    table_area = 0.0
    total_area = 0.0
    with fitz.open(pdf_path) as doc:
        page_count = doc.page_count
        for page in doc:
            page_area = _bbox_area(page.rect)
            total_area += page_area
            for img in page.get_image_info():
                bbox = img.get("bbox")
                if bbox is not None:
                    image_area += _bbox_area(bbox)
            try:
                tables = page.find_tables()
                for t in tables:
                    table_area += _bbox_area(t.bbox)
            except Exception:  # noqa: BLE001 — PyMuPDF table extraction can raise
                pass
    if total_area == 0:
        return DensityReport(str(pdf_path), page_count, 0.0, 0.0)
    return DensityReport(
        pdf_path=str(pdf_path),
        page_count=page_count,
        image_surface_ratio=image_area / total_area,
        table_surface_ratio=table_area / total_area,
    )


def keep(report: DensityReport, threshold: float = 0.30) -> bool:
    """Default policy: keep PDFs whose visual density is at least 30%."""
    return report.visual_density >= threshold
