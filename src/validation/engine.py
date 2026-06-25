from pathlib import Path
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
