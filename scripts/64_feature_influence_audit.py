#!/usr/bin/env python3
"""
64_feature_influence_audit.py

Audit which feature groups are driving the current top UREM v06 exceptional-residual cells.

Purpose:
- Determine whether UREM is mostly a coastal detector.
- Compare top reviewed cells against all valid cells.
- Quantify whether top cells are extreme in coastality, terrain, relief, slope, or recognition gap.

Inputs:
- data/processed/expected_recognition_v06.gpkg
- data/processed/v06_review_package_exceptional_residual.gpkg

Outputs:
- data/processed/feature_influence_audit_v06.csv
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np


SCRIPT_NAME = "64_feature_influence_audit"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

ALL_CELLS_GPKG = PROCESSED_DIR / "expected_recognition_v06.gpkg"
TOP_REVIEW_GPKG = PROCESSED_DIR / "v06_review_package_exceptional_residual.gpkg"

OUT_CSV = PROCESSED_DIR / "feature_influence_audit_v06.csv"


FEATURES = [
    "distance_to_coast_m",
    "physical_exceptionality_v03",
    "terrain_drama_v03",
    "scenic_coast_v03",
    "flat_coastal_edge_penalty_v03",
    "complex_flat_shoreline_penalty_v03",
    "local_relief_m",
    "slope_deg",
    "elevation_m",
    "observed_recognition_v04",
    "expected_recognition_v06_raw",
    "positive_under_recognition_residual_v06",
    "recognition_cell_confidence_v04",
    "recognition_total_count_3km_v04",
]


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def bool_series(df, col):
    if col not in df.columns:
        return pd.Series(True, index=df.index)
    return df[col].astype(str).str.lower().isin(["true", "1", "yes"])


def percentile_of_top_mean(all_values, top_mean, reverse=False):
    """
    Returns where the top mean sits within the all-cell distribution.
    reverse=True means lower values are more extreme, useful for distance_to_coast_m.
    """
    s = pd.to_numeric(all_values, errors="coerce").dropna()

    if len(s) == 0 or pd.isna(top_mean):
        return np.nan

    if reverse:
        return (s >= top_mean).mean()
    return (s <= top_mean).mean()


def summarize_feature(all_valid, top, feature):
    all_s = pd.to_numeric(all_valid[feature], errors="coerce")
    top_s = pd.to_numeric(top[feature], errors="coerce")

    all_mean = all_s.mean()
    top_mean = top_s.mean()

    reverse = feature == "distance_to_coast_m"

    return {
        "feature": feature,
        "all_valid_mean": all_mean,
        "top100_mean": top_mean,
        "difference_top_minus_all": top_mean - all_mean,
        "ratio_top_to_all": top_mean / all_mean if all_mean not in [0, np.nan] else np.nan,
        "all_valid_median": all_s.median(),
        "top100_median": top_s.median(),
        "top_mean_percentile_vs_all": percentile_of_top_mean(
            all_s,
            top_mean,
            reverse=reverse,
        ),
        "all_valid_p10": all_s.quantile(0.10),
        "all_valid_p25": all_s.quantile(0.25),
        "all_valid_p75": all_s.quantile(0.75),
        "all_valid_p90": all_s.quantile(0.90),
        "top100_p10": top_s.quantile(0.10),
        "top100_p25": top_s.quantile(0.25),
        "top100_p75": top_s.quantile(0.75),
        "top100_p90": top_s.quantile(0.90),
    }


def main():
    log("Starting feature influence audit")

    if not ALL_CELLS_GPKG.exists():
        raise FileNotFoundError(f"Missing: {ALL_CELLS_GPKG}")

    if not TOP_REVIEW_GPKG.exists():
        raise FileNotFoundError(f"Missing: {TOP_REVIEW_GPKG}")

    all_cells = gpd.read_file(ALL_CELLS_GPKG)
    top = gpd.read_file(TOP_REVIEW_GPKG)

    log(f"All cells: {len(all_cells):,}")
    log(f"Top review cells: {len(top):,}")

    valid = bool_series(all_cells, "valid_land_v06")

    if "land_area_share" in all_cells.columns:
        land_share = pd.to_numeric(all_cells["land_area_share"], errors="coerce").fillna(0)
        valid = valid & (land_share >= 0.50)

    all_valid = all_cells[valid].copy()

    log(f"All valid land cells: {len(all_valid):,}")

    rows = []

    for feature in FEATURES:
        if feature not in all_valid.columns:
            log(f"Skipping missing feature in all cells: {feature}")
            continue

        if feature not in top.columns:
            log(f"Skipping missing feature in top review cells: {feature}")
            continue

        rows.append(summarize_feature(all_valid, top, feature))

    audit = pd.DataFrame(rows)

    audit["abs_standardized_difference_hint"] = (
        audit["difference_top_minus_all"].abs()
    )

    audit = audit.sort_values(
        "abs_standardized_difference_hint",
        ascending=False,
    )

    log(f"Writing CSV: {OUT_CSV}")
    audit.to_csv(OUT_CSV, index=False)

    log("Done")

    print("\nFeature influence audit:")
    print(
        audit[
            [
                "feature",
                "all_valid_mean",
                "top100_mean",
                "difference_top_minus_all",
                "ratio_top_to_all",
                "top_mean_percentile_vs_all",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()