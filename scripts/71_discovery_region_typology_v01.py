#!/usr/bin/env python3
"""
71_discovery_region_typology_v01.py

Purpose
-------
Classify UREM discovery regions into interpretable recognition-disequilibrium
archetypes.

This script does not create a new UREM score. It synthesizes existing evidence.

Inputs
------
data/processed/v07_no_coast_discovery_regions.gpkg
data/processed/urem_score_v07_no_coast.gpkg
data/processed/matched_counterfactual_region_summary_v01.csv
data/processed/accessibility_friction_region_summary_v03.csv
data/processed/urem_region_robustness_summary_v01.csv

Outputs
-------
data/processed/urem_discovery_region_typology_v01.csv
data/processed/urem_discovery_region_typology_v01.gpkg
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

REGIONS_PATH = DATA / "v07_no_coast_discovery_regions.gpkg"
UNIVERSE_PATH = DATA / "urem_score_v07_no_coast.gpkg"

COUNTERFACTUAL_PATH = DATA / "matched_counterfactual_region_summary_v01.csv"
FRICTION_PATH = DATA / "accessibility_friction_region_summary_v03.csv"
ROBUSTNESS_PATH = DATA / "urem_region_robustness_summary_v01.csv"

OUT_CSV = DATA / "urem_discovery_region_typology_v01.csv"
OUT_GPKG = DATA / "urem_discovery_region_typology_v01.gpkg"

logging.basicConfig(
    level=logging.INFO,
    format="[71_discovery_region_typology_v01] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


def pct_rank(s):
    return pd.to_numeric(s, errors="coerce").rank(pct=True)


def classify_region(row):
    cf = row.get("mean_matched_gap_percentile", np.nan)
    robustness = row.get("scenario_presence_share", np.nan)
    friction = row.get("mean_accessibility_friction", np.nan)
    terrain = row.get("mean_terrain_drama", np.nan)
    exceptionality = row.get("mean_physical_exceptionality", np.nan)
    recognition = row.get("mean_observed_recognition", np.nan)
    dist_coast = row.get("mean_distance_to_coast_km", np.nan)

    high_cf = cf >= 0.90
    very_high_cf = cf >= 0.95
    high_robust = robustness >= 0.75
    moderate_robust = robustness >= 0.50
    high_friction = friction >= 0.70
    high_terrain = terrain >= 0.65
    high_exceptionality = exceptionality >= 0.85
    low_recognition = recognition <= 0.15
    inland = dist_coast >= 10 if not pd.isna(dist_coast) else False

    if very_high_cf and high_robust and high_friction:
        return "Core Recognition Disequilibrium Landscape"

    if high_cf and high_terrain and high_friction and low_recognition:
        return "Hidden Rugged Low-Access Landscape"

    if high_cf and high_robust and high_exceptionality:
        return "Robust High-Exceptionality Recognition Gap"

    if high_cf and inland:
        return "Interior Recognition Deficit"

    if high_cf and not high_robust:
        return "High-Gap Fragile Discovery"

    if moderate_robust and high_friction:
        return "Moderately Robust Friction-Limited Landscape"

    if high_exceptionality and low_recognition:
        return "High-Potential Low-Visibility Landscape"

    return "Secondary / Ambiguous Discovery"


def main():
    log.info(f"Reading regions: {REGIONS_PATH}")
    regions = gpd.read_file(REGIONS_PATH)

    log.info(f"Reading universe: {UNIVERSE_PATH}")
    universe = gpd.read_file(UNIVERSE_PATH)

    if regions.crs != universe.crs:
        universe = universe.to_crs(regions.crs)

    region_col = None
    for c in ["region_id", "discovery_region_id", "cluster_id", "id"]:
        if c in regions.columns:
            region_col = c
            break

    if region_col is None:
        raise KeyError("Could not identify region ID column.")

    log.info(f"Using region column: {region_col}")

    joined = gpd.sjoin(universe, regions[[region_col, "geometry"]], how="inner", predicate="within")

    base_summary = (
        joined.groupby(region_col)
        .agg(
            cells=("cell_id", "count"),
            mean_urem_score=("urem_score_v07_no_coast", "mean"),
            max_urem_score=("urem_score_v07_no_coast", "max"),
            mean_observed_recognition=("observed_recognition_v04", "mean"),
            mean_expected_recognition=("expected_recognition_v06", "mean"),
            mean_under_recognition_residual=("positive_under_recognition_residual_v06", "mean"),
            mean_physical_exceptionality=("physical_exceptionality_v03", "mean"),
            mean_terrain_drama=("terrain_drama_v03", "mean"),
            mean_local_relief_m=("local_relief_m", "mean"),
            mean_slope_deg=("slope_deg", "mean"),
            mean_elevation_m=("elevation_m", "mean"),
            mean_distance_to_coast_m=("distance_to_coast_m", "mean"),
            mean_land_area_share=("land_area_share", "mean"),
        )
        .reset_index()
    )

    base_summary["mean_distance_to_coast_km"] = base_summary["mean_distance_to_coast_m"] / 1000

    merged = base_summary.copy()

    for path, label in [
        (COUNTERFACTUAL_PATH, "counterfactual"),
        (FRICTION_PATH, "friction"),
        (ROBUSTNESS_PATH, "robustness"),
    ]:
        if path.exists():
            log.info(f"Reading {label}: {path}")
            df = pd.read_csv(path)
            if region_col not in df.columns:
                log.warning(f"{label} file missing {region_col}; skipping.")
                continue
            merged = merged.merge(df, on=region_col, how="left", suffixes=("", f"_{label}"))
        else:
            log.warning(f"Missing {label} file: {path}")

    # Normalize evidence dimensions.
    merged["exceptionality_strength_pct"] = pct_rank(merged["mean_physical_exceptionality"])
    merged["terrain_strength_pct"] = pct_rank(merged["mean_terrain_drama"])
    merged["recognition_gap_strength_pct"] = pct_rank(merged["mean_under_recognition_residual"])

    if "mean_matched_gap_percentile" in merged.columns:
        merged["counterfactual_strength_pct"] = merged["mean_matched_gap_percentile"]
    else:
        merged["counterfactual_strength_pct"] = np.nan

    if "mean_accessibility_friction" in merged.columns:
        merged["friction_strength_pct"] = pct_rank(merged["mean_accessibility_friction"])
    else:
        merged["friction_strength_pct"] = np.nan

    if "scenario_presence_share" in merged.columns:
        merged["robustness_strength_pct"] = merged["scenario_presence_share"]
    else:
        merged["robustness_strength_pct"] = np.nan

    merged["typology_confidence_score"] = merged[
        [
            "exceptionality_strength_pct",
            "recognition_gap_strength_pct",
            "counterfactual_strength_pct",
            "friction_strength_pct",
            "robustness_strength_pct",
        ]
    ].mean(axis=1)

    merged["discovery_archetype"] = merged.apply(classify_region, axis=1)

    merged["priority_class"] = pd.cut(
        merged["typology_confidence_score"],
        bins=[-0.01, 0.50, 0.70, 0.85, 1.01],
        labels=[
            "secondary",
            "promising",
            "strong",
            "core",
        ],
    )

    # Merge geometry back.
    out_gdf = regions.merge(merged, on=region_col, how="left")

    sort_cols = ["priority_class", "typology_confidence_score"]
    out_table = merged.sort_values("typology_confidence_score", ascending=False)

    log.info(f"Writing CSV: {OUT_CSV}")
    out_table.to_csv(OUT_CSV, index=False)

    log.info(f"Writing GPKG: {OUT_GPKG}")
    out_gdf.to_file(OUT_GPKG, driver="GPKG")

    print("\nUREM Discovery Region Typology")
    print("------------------------------")
    display_cols = [
        region_col,
        "discovery_archetype",
        "priority_class",
        "typology_confidence_score",
        "mean_physical_exceptionality",
        "mean_matched_gap_percentile",
        "mean_accessibility_friction",
        "scenario_presence_share",
        "mean_observed_recognition",
        "mean_distance_to_coast_km",
    ]

    display_cols = [c for c in display_cols if c in out_table.columns]
    print(out_table[display_cols].to_string(index=False))

    print("\nInterpretation")
    print("--------------")
    print("This converts UREM from a ranked hotspot output into a typology of")
    print("recognition-disequilibrium landscapes.")
    print()
    print("The strongest regions are those with:")
    print("- high physical exceptionality")
    print("- high matched recognition-gap percentile")
    print("- high accessibility/friction mechanism score")
    print("- high region-level robustness")


if __name__ == "__main__":
    main()