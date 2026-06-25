#!/usr/bin/env python3

"""
170_build_rde_null_model_validation_engine_v01.py

Creates reusable null-model validation infrastructure for Phase II RDE.

Writes:
    src/validation/null_models.py

This does NOT run Oregon PAD-US yet.
That comes next in Script 171.
"""

from pathlib import Path


SCRIPT_NAME = "170_build_rde_null_model_validation_engine_v01"

OUTPUT_PATH = Path("src/validation/null_models.py")

CONTENT = r'''import random
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import affinity


@dataclass
class NullModelConfig:
    """
    Configuration for spatial null-model validation.
    """

    n_simulations: int = 1000
    random_seed: int = 42
    null_model_type: str = "random_translation"
    max_attempts_per_feature: int = 250


def random_point_within_polygon(poly):
    minx, miny, maxx, maxy = poly.bounds

    for _ in range(10000):
        p = gpd.points_from_xy(
            [random.uniform(minx, maxx)],
            [random.uniform(miny, maxy)],
            crs=None,
        )[0]

        if poly.contains(p):
            return p

    return poly.representative_point()


def translate_geometry_to_point(geom, target_point):
    centroid = geom.centroid
    dx = target_point.x - centroid.x
    dy = target_point.y - centroid.y

    return affinity.translate(
        geom,
        xoff=dx,
        yoff=dy,
    )


def make_randomized_hotspots(
    hotspots: gpd.GeoDataFrame,
    study_area: gpd.GeoDataFrame,
    config: NullModelConfig,
) -> gpd.GeoDataFrame:
    """
    Randomly translates hotspot geometries inside the study area.

    Preserves hotspot shape and size.
    """

    study_union = study_area.geometry.union_all()

    randomized = []

    for _, row in hotspots.iterrows():
        geom = row.geometry

        placed = None

        for _ in range(config.max_attempts_per_feature):
            target = random_point_within_polygon(study_union)
            candidate = translate_geometry_to_point(geom, target)

            if study_union.contains(candidate):
                placed = candidate
                break

        if placed is None:
            # Fallback: use representative point buffer approximation.
            target = random_point_within_polygon(study_union)
            placed = translate_geometry_to_point(geom, target).intersection(study_union)

        rec = row.drop(labels="geometry").to_dict()
        rec["geometry"] = placed
        randomized.append(rec)

    return gpd.GeoDataFrame(
        randomized,
        geometry="geometry",
        crs=hotspots.crs,
    )


def validation_metrics_for_layer(
    hotspots: gpd.GeoDataFrame,
    validation_layer: gpd.GeoDataFrame,
) -> Dict[str, Any]:
    """
    Compute simple validation metrics for a hotspot set against a validation layer.
    """

    if validation_layer.crs != hotspots.crs:
        validation_layer = validation_layer.to_crs(hotspots.crs)

    layer_union = validation_layer.geometry.union_all()

    intersects = hotspots.geometry.intersects(layer_union)

    overlap_pcts = []

    distances = []

    for geom in hotspots.geometry:
        if geom is None or geom.is_empty:
            overlap_pcts.append(0.0)
            distances.append(np.nan)
            continue

        if geom.area > 0:
            inter_area = geom.intersection(layer_union).area
            overlap_pcts.append(float(inter_area / geom.area))
        else:
            overlap_pcts.append(0.0)

        distances.append(float(validation_layer.geometry.distance(geom.centroid).min()))

    overlap_pcts = np.array(overlap_pcts)
    distances = np.array(distances)

    return {
        "n_hotspots": len(hotspots),
        "pct_intersects_any": float(intersects.mean()),
        "mean_overlap_pct": float(np.nanmean(overlap_pcts)),
        "median_overlap_pct": float(np.nanmedian(overlap_pcts)),
        "pct_any_overlap": float((overlap_pcts > 0).mean()),
        "mean_nearest_distance_m": float(np.nanmean(distances)),
        "median_nearest_distance_m": float(np.nanmedian(distances)),
        "pct_within_1km": float((distances <= 1000).mean()),
        "pct_within_5km": float((distances <= 5000).mean()),
    }


def run_spatial_null_model(
    hotspots: gpd.GeoDataFrame,
    study_area: gpd.GeoDataFrame,
    validation_layer: gpd.GeoDataFrame,
    config: Optional[NullModelConfig] = None,
) -> pd.DataFrame:
    """
    Run repeated spatial null simulations.
    """

    if config is None:
        config = NullModelConfig()

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    results: List[Dict[str, Any]] = []

    for i in range(config.n_simulations):
        if (i + 1) % 50 == 0:
            print(f"[NullModel] Simulation {i + 1:,}/{config.n_simulations:,}")

        if config.null_model_type == "random_translation":
            randomized = make_randomized_hotspots(
                hotspots,
                study_area,
                config,
            )
        else:
            raise ValueError(
                f"Unsupported null_model_type: {config.null_model_type}"
            )

        metrics = validation_metrics_for_layer(
            randomized,
            validation_layer,
        )

        metrics["simulation_id"] = i + 1
        metrics["null_model_type"] = config.null_model_type

        results.append(metrics)

    return pd.DataFrame(results)


def compare_observed_to_null(
    observed_metrics: Dict[str, Any],
    null_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare observed validation metrics against null distributions.
    """

    rows = []

    for metric, observed_value in observed_metrics.items():
        if metric == "n_hotspots":
            continue

        if metric not in null_results.columns:
            continue

        null_values = null_results[metric].dropna()

        if len(null_values) == 0:
            continue

        null_mean = null_values.mean()
        null_std = null_values.std()

        if null_std == 0:
            z_score = np.nan
        else:
            z_score = (observed_value - null_mean) / null_std

        # One-sided p-value for metrics where higher support is better.
        if metric in [
            "pct_intersects_any",
            "mean_overlap_pct",
            "median_overlap_pct",
            "pct_any_overlap",
            "pct_within_1km",
            "pct_within_5km",
        ]:
            p_value = float((null_values >= observed_value).mean())
            direction = "higher_is_better"

        # One-sided p-value for distance metrics where lower is better.
        elif metric in [
            "mean_nearest_distance_m",
            "median_nearest_distance_m",
        ]:
            p_value = float((null_values <= observed_value).mean())
            direction = "lower_is_better"

        else:
            p_value = np.nan
            direction = "unknown"

        rows.append(
            {
                "metric": metric,
                "observed_value": observed_value,
                "null_mean": null_mean,
                "null_std": null_std,
                "z_score": z_score,
                "monte_carlo_p_value": p_value,
                "direction": direction,
                "effect_ratio_observed_to_null": (
                    observed_value / null_mean if null_mean != 0 else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)
'''


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(CONTENT, encoding="utf-8")

    print(f"[{SCRIPT_NAME}] Wrote: {OUTPUT_PATH}")
    print()
    print(f"[{SCRIPT_NAME}] Null-model engine created.")
    print()
    print("Next step:")
    print("  Script 171 should run Oregon PAD-US Monte Carlo validation.")
    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()