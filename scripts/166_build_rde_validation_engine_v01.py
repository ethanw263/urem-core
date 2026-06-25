#!/usr/bin/env python3

"""
166_build_rde_validation_engine_v01.py

Phase II RDE Validation Framework.

This script creates the reusable validation engine infrastructure.

It does NOT run Oregon validation yet.

It writes reusable modules into:

    src/validation/

Modules created:
    datasets.py
    metrics.py
    engine.py
    reporting.py

Purpose:
    Build the foundation for external geographic validation,
    statistical validation, null models, and cross-state validation.
"""

from pathlib import Path


SCRIPT_NAME = "166_build_rde_validation_engine_v01"

BASE_DIR = Path("src/validation")

FILES = {
    "datasets.py": r'''from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ValidationDataset:
    """
    Metadata wrapper for an independent validation dataset.

    This object describes the dataset.
    It does not perform validation by itself.
    """

    name: str
    path: str
    geometry_type: str
    source: str
    metrics: List[str]
    weight: float = 1.0
    id_column: Optional[str] = None
    category: Optional[str] = None
    independent_of_model: bool = True
    notes: str = ""

    def validate_metadata(self) -> None:
        allowed_geometry = {"point", "line", "polygon", "mixed"}
        allowed_metrics = {"intersects", "overlap", "distance", "density"}

        if self.geometry_type not in allowed_geometry:
            raise ValueError(
                f"Invalid geometry_type for {self.name}: {self.geometry_type}"
            )

        unknown_metrics = set(self.metrics) - allowed_metrics
        if unknown_metrics:
            raise ValueError(
                f"Invalid metrics for {self.name}: {unknown_metrics}"
            )

        if self.weight < 0:
            raise ValueError(
                f"Dataset weight must be nonnegative for {self.name}"
            )
''',

    "metrics.py": r'''import numpy as np
import geopandas as gpd


def safe_union(gdf: gpd.GeoDataFrame):
    """
    Compatibility wrapper for union_all / unary_union.
    """

    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()

    return gdf.geometry.unary_union


def intersection_count(geom, layer: gpd.GeoDataFrame) -> int:
    if layer.empty:
        return 0

    return int(layer.intersects(geom).sum())


def overlap_area_pct(geom, layer: gpd.GeoDataFrame) -> float:
    if layer.empty or geom is None or geom.area == 0:
        return 0.0

    hits = layer[layer.intersects(geom)]

    if hits.empty:
        return 0.0

    inter_area = hits.geometry.intersection(geom).area.sum()

    return float(inter_area / geom.area)


def nearest_distance(geom, layer: gpd.GeoDataFrame) -> float:
    if layer.empty:
        return np.nan

    centroid = geom.centroid
    distances = layer.geometry.distance(centroid)

    return float(distances.min())


def density_within_distance(
    geom,
    layer: gpd.GeoDataFrame,
    distance_m: float,
) -> int:
    if layer.empty:
        return 0

    centroid = geom.centroid
    distances = layer.geometry.distance(centroid)

    return int((distances <= distance_m).sum())


def support_from_distance(distance_m: float, thresholds=(1000, 5000, 10000)) -> str:
    if np.isnan(distance_m):
        return "no_data"

    if distance_m <= thresholds[0]:
        return "very_near"

    if distance_m <= thresholds[1]:
        return "near"

    if distance_m <= thresholds[2]:
        return "moderate_distance"

    return "far"


def support_from_overlap(overlap_pct: float) -> str:
    if overlap_pct >= 0.50:
        return "strong_overlap"

    if overlap_pct >= 0.25:
        return "moderate_overlap"

    if overlap_pct > 0:
        return "weak_overlap"

    return "no_overlap"
''',

    "engine.py": r'''from pathlib import Path
from typing import List, Dict, Any

import geopandas as gpd
import pandas as pd

from .datasets import ValidationDataset
from .metrics import (
    intersection_count,
    overlap_area_pct,
    nearest_distance,
    density_within_distance,
    support_from_distance,
    support_from_overlap,
)


class ValidationEngine:
    """
    Generic RDE validation engine.

    This class evaluates hotspots against registered validation datasets.
    It is intentionally study-area agnostic.
    """

    def __init__(
        self,
        hotspots_path: str,
        hotspot_id_column: str,
        hotspot_score_column: str,
    ):
        self.hotspots_path = Path(hotspots_path)
        self.hotspot_id_column = hotspot_id_column
        self.hotspot_score_column = hotspot_score_column
        self.datasets: List[ValidationDataset] = []

    def register_dataset(self, dataset: ValidationDataset) -> None:
        dataset.validate_metadata()
        self.datasets.append(dataset)

    def load_hotspots(self) -> gpd.GeoDataFrame:
        hotspots = gpd.read_file(self.hotspots_path)

        if self.hotspot_id_column not in hotspots.columns:
            raise ValueError(
                f"Missing hotspot id column: {self.hotspot_id_column}"
            )

        if self.hotspot_score_column not in hotspots.columns:
            raise ValueError(
                f"Missing hotspot score column: {self.hotspot_score_column}"
            )

        return hotspots

    def evaluate_dataset(
        self,
        hotspots: gpd.GeoDataFrame,
        dataset: ValidationDataset,
    ) -> pd.DataFrame:

        layer = gpd.read_file(dataset.path)

        if layer.crs != hotspots.crs:
            layer = layer.to_crs(hotspots.crs)

        records: List[Dict[str, Any]] = []

        for _, row in hotspots.iterrows():
            geom = row.geometry
            hotspot_id = row[self.hotspot_id_column]

            rec = {
                "hotspot_id": hotspot_id,
                "hotspot_score": row[self.hotspot_score_column],
                "validation_dataset": dataset.name,
                "validation_category": dataset.category,
                "validation_source": dataset.source,
                "validation_weight": dataset.weight,
                "validation_geometry_type": dataset.geometry_type,
                "independent_of_model": dataset.independent_of_model,
            }

            if "intersects" in dataset.metrics:
                rec["intersects_count"] = intersection_count(geom, layer)
                rec["intersects_any"] = rec["intersects_count"] > 0

            if "overlap" in dataset.metrics:
                rec["overlap_pct"] = overlap_area_pct(geom, layer)
                rec["overlap_support"] = support_from_overlap(rec["overlap_pct"])

            if "distance" in dataset.metrics:
                rec["nearest_distance_m"] = nearest_distance(geom, layer)
                rec["distance_support"] = support_from_distance(
                    rec["nearest_distance_m"]
                )

            if "density" in dataset.metrics:
                rec["density_1km"] = density_within_distance(geom, layer, 1000)
                rec["density_5km"] = density_within_distance(geom, layer, 5000)
                rec["density_10km"] = density_within_distance(geom, layer, 10000)

            records.append(rec)

        return pd.DataFrame(records)

    def run(self) -> pd.DataFrame:
        hotspots = self.load_hotspots()

        if not self.datasets:
            raise ValueError("No validation datasets registered.")

        outputs = []

        for dataset in self.datasets:
            print(f"[ValidationEngine] Evaluating: {dataset.name}")
            outputs.append(self.evaluate_dataset(hotspots, dataset))

        return pd.concat(outputs, ignore_index=True)

    def summarize(self, results: pd.DataFrame) -> pd.DataFrame:
        summary_rows = []

        for dataset_name, sub in results.groupby("validation_dataset"):
            row = {
                "validation_dataset": dataset_name,
                "n_hotspots": sub["hotspot_id"].nunique(),
            }

            if "intersects_any" in sub.columns:
                row["pct_intersects_any"] = sub["intersects_any"].mean()

            if "overlap_pct" in sub.columns:
                row["mean_overlap_pct"] = sub["overlap_pct"].mean()
                row["pct_any_overlap"] = (sub["overlap_pct"] > 0).mean()

            if "nearest_distance_m" in sub.columns:
                row["mean_nearest_distance_m"] = sub["nearest_distance_m"].mean()
                row["median_nearest_distance_m"] = sub["nearest_distance_m"].median()
                row["pct_within_1km"] = (sub["nearest_distance_m"] <= 1000).mean()
                row["pct_within_5km"] = (sub["nearest_distance_m"] <= 5000).mean()

            if "density_5km" in sub.columns:
                row["mean_density_5km"] = sub["density_5km"].mean()

            summary_rows.append(row)

        return pd.DataFrame(summary_rows)
''',

    "reporting.py": r'''from pathlib import Path
import pandas as pd


def write_markdown_report(
    summary: pd.DataFrame,
    output_path: str,
    title: str = "RDE Validation Report",
) -> None:
    output_path = Path(output_path)

    lines = []

    lines.append(f"# {title}")
    lines.append("")
    lines.append("This report was generated by the reusable RDE Validation Framework.")
    lines.append("")
    lines.append("## Validation Dataset Summary")
    lines.append("")

    if summary.empty:
        lines.append("No validation results available.")
    else:
        lines.append(summary.to_markdown(index=False))

    lines.append("")
    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append(
        "This report summarizes external or internal validation datasets "
        "registered with the validation engine. Strong validation should come "
        "from datasets that are independent of the model construction process."
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
''',
}


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    init_path = BASE_DIR / "__init__.py"
    if not init_path.exists():
        init_path.write_text("", encoding="utf-8")

    for filename, content in FILES.items():
        path = BASE_DIR / filename
        path.write_text(content, encoding="utf-8")
        print(f"[{SCRIPT_NAME}] Wrote: {path}")

    print()
    print(f"[{SCRIPT_NAME}] Validation framework created.")
    print()
    print("Created modules:")
    for filename in FILES:
        print(f"  - src/validation/{filename}")

    print()
    print("Next step:")
    print("  Build Script 167 to define the validation dataset registry.")
    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()