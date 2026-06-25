#!/usr/bin/env python3

from pathlib import Path

SCRIPT_NAME = "176_refactor_rde_validation_engine_v02"

VALIDATION_DIR = Path("src/validation")

DATASETS_PY = VALIDATION_DIR / "datasets.py"
LOADER_PY = VALIDATION_DIR / "loader.py"
ENGINE_PY = VALIDATION_DIR / "engine.py"


DATASETS_CONTENT = r'''from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ValidationDataset:
    """
    Lightweight metadata object for an independent validation dataset.

    This object describes the dataset.
    It does not store the GeoDataFrame.
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
        allowed_geometry = {
            "point",
            "line",
            "polygon",
            "mixed",
            "point_or_polygon",
        }

        allowed_metrics = {
            "intersects",
            "overlap",
            "distance",
            "density",
        }

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
'''


LOADER_CONTENT = r'''from pathlib import Path
import geopandas as gpd


def load_validation_layer(path: str, target_crs=None) -> gpd.GeoDataFrame:
    """
    Load and lightly clean a validation layer.

    Supports files readable by GeoPandas/pyogrio:
    - GPKG
    - SHP
    - GeoJSON
    - FileGDB layers when path points to a readable layer/file

    Assumption:
    Standardization scripts should already have converted complex raw data
    into clean GeoPackages before validation.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Validation dataset not found: {path}")

    gdf = gpd.read_file(path)

    if target_crs is not None and gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    return gdf
'''


ENGINE_CONTENT = r'''from pathlib import Path
from typing import List, Dict, Any

import geopandas as gpd
import pandas as pd

from .datasets import ValidationDataset
from .loader import load_validation_layer
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

    The engine is study-area agnostic.

    It receives:
    - a hotspot layer
    - one or more ValidationDataset metadata objects

    It loads each validation dataset internally and computes standardized
    metrics using the same mathematical procedure for every dataset.
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
        if not self.hotspots_path.exists():
            raise FileNotFoundError(f"Hotspots file not found: {self.hotspots_path}")

        hotspots = gpd.read_file(self.hotspots_path)

        if self.hotspot_id_column not in hotspots.columns:
            raise ValueError(
                f"Missing hotspot id column: {self.hotspot_id_column}"
            )

        if self.hotspot_score_column not in hotspots.columns:
            raise ValueError(
                f"Missing hotspot score column: {self.hotspot_score_column}"
            )

        hotspots = hotspots[
            hotspots.geometry.notna()
            & ~hotspots.geometry.is_empty
        ].copy()

        return hotspots

    def evaluate_dataset(
        self,
        hotspots: gpd.GeoDataFrame,
        dataset: ValidationDataset,
    ) -> pd.DataFrame:

        layer = load_validation_layer(
            dataset.path,
            target_crs=hotspots.crs,
        )

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
'''


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    DATASETS_PY.write_text(DATASETS_CONTENT, encoding="utf-8")
    LOADER_PY.write_text(LOADER_CONTENT, encoding="utf-8")
    ENGINE_PY.write_text(ENGINE_CONTENT, encoding="utf-8")

    print(f"[{SCRIPT_NAME}] Wrote: {DATASETS_PY}")
    print(f"[{SCRIPT_NAME}] Wrote: {LOADER_PY}")
    print(f"[{SCRIPT_NAME}] Wrote: {ENGINE_PY}")

    print()
    print(f"[{SCRIPT_NAME}] Validation engine refactor complete.")
    print()
    print("Next step:")
    print("  Re-run PAD-US validation script 169 to confirm compatibility.")
    print("  Then update and run estuary validation script 175.")
    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()