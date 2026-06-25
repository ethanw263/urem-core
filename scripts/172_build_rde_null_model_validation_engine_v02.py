#!/usr/bin/env python3

from pathlib import Path

SCRIPT_NAME = "172_build_rde_null_model_validation_engine_v02"

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
    progress_interval: int = 50


class PreparedValidationLayer:
    """
    Cached validation geometry to avoid recomputing expensive objects
    during every simulation.
    """

    def __init__(self, validation_layer: gpd.GeoDataFrame, target_crs):
        layer = validation_layer.copy()

        if layer.crs != target_crs:
            layer = layer.to_crs(target_crs)

        layer = layer[
            layer.geometry.notna()
            & ~layer.geometry.is_empty
        ].copy()

        try:
            layer["geometry"] = layer.geometry.make_valid()
        except Exception:
            layer["geometry"] = layer.geometry.buffer(0)

        layer = layer[
            layer.geometry.notna()
            & ~layer.geometry.is_empty
        ].copy()

        self.layer = layer
        self.union = layer.geometry.union_all()
        self.sindex = layer.sindex


def random_point_within_polygon(poly):
    minx, miny, maxx, maxy = poly.bounds

    for _ in range(10000):
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)

        point = gpd.points_from_xy([x], [y], crs=None)[0]

        if poly.contains(point):
            return point

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


def prepare_hotspot_shapes(
    hotspots: gpd.GeoDataFrame,
) -> List[Dict[str, Any]]:
    """
    Store hotspot attributes/geometries in a simple list to reduce GeoDataFrame
    overhead during repeated simulation.
    """

    prepared = []

    for _, row in hotspots.iterrows():
        prepared.append(
            {
                "attrs": row.drop(labels="geometry").to_dict(),
                "geometry": row.geometry,
            }
        )

    return prepared


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
    prepared_hotspots = prepare_hotspot_shapes(hotspots)

    randomized = []

    for item in prepared_hotspots:
        geom = item["geometry"]

        placed = None

        for _ in range(config.max_attempts_per_feature):
            target = random_point_within_polygon(study_union)
            candidate = translate_geometry_to_point(geom, target)

            if study_union.contains(candidate):
                placed = candidate
                break

        if placed is None:
            target = random_point_within_polygon(study_union)
            placed = translate_geometry_to_point(geom, target).intersection(study_union)

        rec = dict(item["attrs"])
        rec["geometry"] = placed
        randomized.append(rec)

    return gpd.GeoDataFrame(
        randomized,
        geometry="geometry",
        crs=hotspots.crs,
    )


def validation_metrics_for_prepared_layer(
    hotspots: gpd.GeoDataFrame,
    prepared_layer: PreparedValidationLayer,
) -> Dict[str, Any]:
    """
    Faster metrics against a prepared validation layer.
    """

    layer = prepared_layer.layer
    layer_union = prepared_layer.union

    intersects = hotspots.geometry.intersects(layer_union)

    overlap_pcts = []
    distances = []

    for geom in hotspots.geometry:
        if geom is None or geom.is_empty:
            overlap_pcts.append(0.0)
            distances.append(np.nan)
            continue

        if geom.area > 0 and geom.intersects(layer_union):
            inter_area = geom.intersection(layer_union).area
            overlap_pcts.append(float(inter_area / geom.area))
        else:
            overlap_pcts.append(0.0)

        centroid = geom.centroid

        possible_idx = list(
            prepared_layer.sindex.nearest(
                centroid,
                return_all=False,
            )
        )

        if possible_idx:
            nearest_geoms = layer.geometry.iloc[possible_idx]
            distances.append(float(nearest_geoms.distance(centroid).min()))
        else:
            distances.append(float(layer.geometry.distance(centroid).min()))

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


def validation_metrics_for_layer(
    hotspots: gpd.GeoDataFrame,
    validation_layer: gpd.GeoDataFrame,
) -> Dict[str, Any]:
    """
    Compatibility wrapper used by existing scripts.
    """

    prepared = PreparedValidationLayer(
        validation_layer=validation_layer,
        target_crs=hotspots.crs,
    )

    return validation_metrics_for_prepared_layer(
        hotspots=hotspots,
        prepared_layer=prepared,
    )


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

    if study_area.crs != hotspots.crs:
        study_area = study_area.to_crs(hotspots.crs)

    prepared_layer = PreparedValidationLayer(
        validation_layer=validation_layer,
        target_crs=hotspots.crs,
    )

    study_union = study_area.geometry.union_all()
    prepared_hotspots = prepare_hotspot_shapes(hotspots)

    results: List[Dict[str, Any]] = []

    for i in range(config.n_simulations):
        if config.progress_interval and (i + 1) % config.progress_interval == 0:
            print(f"[NullModel] Simulation {i + 1:,}/{config.n_simulations:,}")

        if config.null_model_type != "random_translation":
            raise ValueError(
                f"Unsupported null_model_type: {config.null_model_type}"
            )

        randomized = []

        for item in prepared_hotspots:
            geom = item["geometry"]
            placed = None

            for _ in range(config.max_attempts_per_feature):
                target = random_point_within_polygon(study_union)
                candidate = translate_geometry_to_point(geom, target)

                if study_union.contains(candidate):
                    placed = candidate
                    break

            if placed is None:
                target = random_point_within_polygon(study_union)
                placed = translate_geometry_to_point(geom, target).intersection(study_union)

            rec = dict(item["attrs"])
            rec["geometry"] = placed
            randomized.append(rec)

        randomized_gdf = gpd.GeoDataFrame(
            randomized,
            geometry="geometry",
            crs=hotspots.crs,
        )

        metrics = validation_metrics_for_prepared_layer(
            hotspots=randomized_gdf,
            prepared_layer=prepared_layer,
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
    print(f"[{SCRIPT_NAME}] Null-model engine v02 created.")
    print()
    print("Next step:")
    print("  Run Script 171 again to test speed and confirm similar results.")
    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()