#!/usr/bin/env python3
"""
69_urem_robustness_stability_v02.py

Fixed robustness test.

Baseline top 300 is taken directly from:
ranked_urem_candidates_v07_no_coast.gpkg

Alternative scenarios are computed on the full universe and compared against
that fixed baseline.
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

UNIVERSE_PATH = DATA / "urem_score_v07_no_coast.gpkg"
CANDIDATE_PATH = DATA / "ranked_urem_candidates_v07_no_coast.gpkg"
REGION_PATH = DATA / "v07_no_coast_discovery_regions.gpkg"

OUT_CELL_CSV = DATA / "urem_robustness_cell_scenarios_v02.csv"
OUT_CELL_GPKG = DATA / "urem_robustness_cell_scenarios_v02.gpkg"
OUT_SUMMARY_CSV = DATA / "urem_robustness_summary_v02.csv"
OUT_REGION_CSV = DATA / "urem_robustness_region_summary_v02.csv"
OUT_REGION_GPKG = DATA / "urem_robustness_region_summary_v02.gpkg"

TOP_N = 300

logging.basicConfig(
    level=logging.INFO,
    format="[69_urem_robustness_stability_v02] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


def minmax(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    lo = s.min()
    hi = s.max()
    if hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def safe_mean(df, cols):
    return df[cols].mean(axis=1)


def jaccard(a, b):
    return len(a & b) / len(a | b) if len(a | b) else 1.0


def main():
    log.info(f"Reading universe: {UNIVERSE_PATH}")
    gdf = gpd.read_file(UNIVERSE_PATH)

    log.info(f"Reading ranked baseline candidates: {CANDIDATE_PATH}")
    ranked = gpd.read_file(CANDIDATE_PATH)

    log.info(f"Universe rows: {len(gdf):,}")
    log.info(f"Ranked candidate rows: {len(ranked):,}")

    required = [
        "cell_id",
        "urem_score_v07_no_coast",
        "local_relief_norm_v07",
        "slope_norm_v07",
        "elevation_norm_v07",
        "terrain_drama_norm_v07",
        "terrain_only_exceptionality_v07",
        "under_recognition_component_v07",
        "confidence_component_v07",
        "passes_land_filter_v07",
        "passes_residual_filter_v07",
        "passes_confidence_filter_v07",
    ]

    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    baseline_top = ranked.head(TOP_N).copy()
    baseline_ids = set(baseline_top["cell_id"])

    log.info(f"Fixed baseline top-{TOP_N} cells: {len(baseline_ids):,}")

    gdf = gdf.copy()

    valid_mask = (
        gdf["passes_land_filter_v07"].astype(bool)
        & gdf["passes_residual_filter_v07"].astype(bool)
        & gdf["passes_confidence_filter_v07"].astype(bool)
    )

    gdf["is_baseline_top_300"] = gdf["cell_id"].isin(baseline_ids)

    terrain_cols = [
        "local_relief_norm_v07",
        "slope_norm_v07",
        "elevation_norm_v07",
        "terrain_drama_norm_v07",
    ]

    scenarios = {
        "drop_slope": (
            safe_mean(gdf, ["local_relief_norm_v07", "elevation_norm_v07", "terrain_drama_norm_v07"])
            * gdf["under_recognition_component_v07"]
            * gdf["confidence_component_v07"]
        ),
        "drop_relief": (
            safe_mean(gdf, ["slope_norm_v07", "elevation_norm_v07", "terrain_drama_norm_v07"])
            * gdf["under_recognition_component_v07"]
            * gdf["confidence_component_v07"]
        ),
        "drop_elevation": (
            safe_mean(gdf, ["local_relief_norm_v07", "slope_norm_v07", "terrain_drama_norm_v07"])
            * gdf["under_recognition_component_v07"]
            * gdf["confidence_component_v07"]
        ),
        "drop_terrain_drama": (
            safe_mean(gdf, ["local_relief_norm_v07", "slope_norm_v07", "elevation_norm_v07"])
            * gdf["under_recognition_component_v07"]
            * gdf["confidence_component_v07"]
        ),
        "terrain_heavy": (
            gdf["terrain_only_exceptionality_v07"] ** 1.5
            * gdf["under_recognition_component_v07"]
            * gdf["confidence_component_v07"]
        ),
        "recognition_gap_heavy": (
            gdf["terrain_only_exceptionality_v07"]
            * (gdf["under_recognition_component_v07"] ** 1.5)
            * gdf["confidence_component_v07"]
        ),
        "confidence_removed": (
            gdf["terrain_only_exceptionality_v07"]
            * gdf["under_recognition_component_v07"]
        ),
        "conservative_min_terrain": (
            gdf[terrain_cols].min(axis=1)
            * gdf["under_recognition_component_v07"]
            * gdf["confidence_component_v07"]
        ),
        "balanced_additive": (
            0.50 * gdf["terrain_only_exceptionality_v07"]
            + 0.40 * gdf["under_recognition_component_v07"]
            + 0.10 * gdf["confidence_component_v07"]
        ),
    }

    scenario_names = list(scenarios.keys())
    scenario_top_sets = {}

    for name, raw_score in scenarios.items():
        score = pd.to_numeric(raw_score, errors="coerce").fillna(0)
        score = score.where(valid_mask, 0)

        gdf[f"score_{name}"] = minmax(score)
        gdf[f"rank_{name}"] = gdf[f"score_{name}"].rank(
            ascending=False,
            method="min",
        )

        top_ids = set(
            gdf.sort_values(f"score_{name}", ascending=False)
            .head(TOP_N)["cell_id"]
        )

        scenario_top_sets[name] = top_ids
        gdf[f"in_top_{TOP_N}_{name}"] = gdf["cell_id"].isin(top_ids)

    gdf["robustness_top_count"] = gdf[
        [f"in_top_{TOP_N}_{name}" for name in scenario_names]
    ].sum(axis=1)

    gdf["robustness_share"] = gdf["robustness_top_count"] / len(scenario_names)

    summary_rows = []

    for name in scenario_names:
        top_ids = scenario_top_sets[name]
        overlap = len(top_ids & baseline_ids)

        summary_rows.append(
            {
                "scenario": name,
                "top_n": TOP_N,
                "overlap_with_baseline_top_n": overlap,
                "overlap_share": overlap / TOP_N,
                "jaccard_with_baseline": jaccard(top_ids, baseline_ids),
                "mean_score_top_n": gdf[gdf["cell_id"].isin(top_ids)][f"score_{name}"].mean(),
                "median_score_top_n": gdf[gdf["cell_id"].isin(top_ids)][f"score_{name}"].median(),
            }
        )

    summary = pd.DataFrame(summary_rows)

    if REGION_PATH.exists():
        log.info(f"Reading discovery regions: {REGION_PATH}")
        regions = gpd.read_file(REGION_PATH)

        if regions.crs != gdf.crs:
            regions = regions.to_crs(gdf.crs)

        joined = gpd.sjoin(gdf, regions, how="left", predicate="within")

        region_col = None
        for c in ["region_id", "discovery_region_id", "cluster_id", "id"]:
            if c in joined.columns:
                region_col = c
                break

        if region_col is None:
            raise KeyError("Could not identify region ID column.")

        agg_dict = {
            "cells": ("cell_id", "count"),
            "baseline_top_cells": ("is_baseline_top_300", "sum"),
            "mean_robustness_top_count": ("robustness_top_count", "mean"),
            "median_robustness_top_count": ("robustness_top_count", "median"),
            "mean_robustness_share": ("robustness_share", "mean"),
            "max_robustness_share": ("robustness_share", "max"),
        }

        for name in scenario_names:
            agg_dict[f"top_cells_{name}"] = (f"in_top_{TOP_N}_{name}", "sum")

        region_summary = (
            joined.dropna(subset=[region_col])
            .groupby(region_col)
            .agg(**agg_dict)
            .reset_index()
        )

        scenario_top_cols = [f"top_cells_{name}" for name in scenario_names]

        region_summary["scenario_presence_count"] = (
            region_summary[scenario_top_cols] > 0
        ).sum(axis=1)

        region_summary["scenario_presence_share"] = (
            region_summary["scenario_presence_count"] / len(scenario_names)
        )

        region_summary["total_top_cells_across_scenarios"] = (
            region_summary[scenario_top_cols].sum(axis=1)
        )

        regions_out = regions.merge(region_summary, on=region_col, how="left")

        log.info(f"Writing region robustness CSV: {OUT_REGION_CSV}")
        region_summary.to_csv(OUT_REGION_CSV, index=False)

        log.info(f"Writing region robustness GPKG: {OUT_REGION_GPKG}")
        regions_out.to_file(OUT_REGION_GPKG, driver="GPKG")

    keep_cols = [
        "cell_id",
        "is_baseline_top_300",
        "robustness_top_count",
        "robustness_share",
    ]

    keep_cols += [f"score_{name}" for name in scenario_names]
    keep_cols += [f"rank_{name}" for name in scenario_names]
    keep_cols += [f"in_top_{TOP_N}_{name}" for name in scenario_names]
    keep_cols += ["geometry"]

    cell_out = gdf[keep_cols].copy()

    log.info(f"Writing cell robustness GPKG: {OUT_CELL_GPKG}")
    cell_out.to_file(OUT_CELL_GPKG, driver="GPKG")

    log.info(f"Writing cell robustness CSV: {OUT_CELL_CSV}")
    cell_out.drop(columns="geometry").to_csv(OUT_CELL_CSV, index=False)

    log.info(f"Writing robustness summary CSV: {OUT_SUMMARY_CSV}")
    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    print("\nUREM Robustness / Stability Summary v02")
    print("---------------------------------------")
    print(summary.to_string(index=False))

    print("\nInterpretation")
    print("--------------")
    print("This compares alternative score formulations against the fixed ranked")
    print("baseline top 300 from ranked_urem_candidates_v07_no_coast.gpkg.")
    print()
    print("High overlap means cell-level stability.")
    print("Low cell overlap does not automatically mean failure; region-level")
    print("stability may matter more for discovery methodology.")


if __name__ == "__main__":
    main()