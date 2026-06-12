#!/usr/bin/env python3
"""
13_build_physical_fingerprints.py

Builds the first formal UREM physical fingerprint layer.

Input:
- data/processed/coastal_grid_recognition_gap_v01.gpkg

Output:
- data/processed/coastal_grid_physical_fingerprints_v01.gpkg

Purpose:
This script creates normalized physical fingerprint variables for each
1 km grid cell. These fingerprints will be used in Phase 3 to find
physically comparable places and compute expected recognition.

This is the first step toward residual-based UREM.
"""

from pathlib import Path
import logging
import sys

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data/processed/coastal_grid_recognition_gap_v01.gpkg"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/coastal_grid_physical_fingerprints_v01.gpkg"

TARGET_CRS = "EPSG:3310"

RAW_FEATURES = [
    "coastal_proximity_score",
    "elevation_m",
    "local_relief_m",
    "slope_deg",
    "distance_to_coast_m",
    "physical_potential_v01",
]

RECOGNITION_FEATURES = [
    "golf_recognition_score",
    "golf_area_km2",
    "golf_course_count",
    "has_golf",
    "observed_recognition_v0",
]


def setup_logger():
    logger = logging.getLogger("13_build_physical_fingerprints")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def robust_minmax(series, lower_q=0.01, upper_q=0.99):
    """
    Robust min-max normalization using percentile clipping.

    This prevents extreme outliers from dominating fingerprint similarity.
    """
    s = series.astype(float).replace([np.inf, -np.inf], np.nan)

    valid = s.notna()

    if valid.sum() == 0:
        return pd.Series(0.0, index=s.index)

    lower = s[valid].quantile(lower_q)
    upper = s[valid].quantile(upper_q)

    if upper == lower:
        out = pd.Series(0.0, index=s.index)
        return out

    clipped = s.clip(lower=lower, upper=upper)
    normalized = (clipped - lower) / (upper - lower)

    return normalized.fillna(0).clip(0, 1)


def invert_score(series):
    """
    Converts a normalized distance-like score so higher means more favorable.
    """
    return (1.0 - series).clip(0, 1)


def main():
    logger = setup_logger()
    logger.info("Starting Script 13: Build physical fingerprints")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    logger.info(f"Reading input: {INPUT_PATH}")

    grid = gpd.read_file(INPUT_PATH).to_crs(TARGET_CRS)

    if grid.empty:
        raise ValueError("Input grid is empty.")

    if "cell_id" not in grid.columns:
        grid["cell_id"] = range(1, len(grid) + 1)

    missing_raw = [c for c in RAW_FEATURES if c not in grid.columns]
    if missing_raw:
        raise ValueError(f"Missing required physical fingerprint columns: {missing_raw}")

    logger.info(f"Cells loaded: {len(grid):,}")
    logger.info(f"Input CRS: {grid.crs}")

    # Core normalized fingerprint variables.
    grid["fp_coastal_proximity"] = robust_minmax(grid["coastal_proximity_score"])
    grid["fp_elevation"] = robust_minmax(grid["elevation_m"])
    grid["fp_relief"] = robust_minmax(grid["local_relief_m"])
    grid["fp_slope"] = robust_minmax(grid["slope_deg"])

    # distance_to_coast_m is inverted because lower distance means stronger coastal signal.
    distance_norm = robust_minmax(grid["distance_to_coast_m"])
    grid["fp_distance_to_coast_inverse"] = invert_score(distance_norm)

    # Preserve existing physical potential as a fingerprint/context feature.
    grid["fp_physical_potential"] = robust_minmax(grid["physical_potential_v01"])

    fingerprint_cols = [
        "fp_coastal_proximity",
        "fp_elevation",
        "fp_relief",
        "fp_slope",
        "fp_distance_to_coast_inverse",
        "fp_physical_potential",
    ]

    # Fingerprint completeness score.
    grid["fingerprint_valid_feature_count"] = 0

    for col in [
        "coastal_proximity_score",
        "elevation_m",
        "local_relief_m",
        "slope_deg",
        "distance_to_coast_m",
        "physical_potential_v01",
    ]:
        grid["fingerprint_valid_feature_count"] += grid[col].notna().astype(int)

    grid["fingerprint_completeness_score"] = (
        grid["fingerprint_valid_feature_count"] / len(RAW_FEATURES)
    )

    # Keep recognition fields if available.
    for col in RECOGNITION_FEATURES:
        if col not in grid.columns:
            if col == "has_golf":
                grid[col] = False
            else:
                grid[col] = 0.0

    grid["observed_recognition_v0"] = grid["observed_recognition_v0"].fillna(
        grid["golf_recognition_score"].fillna(0)
    )

    # Metadata.
    grid["fingerprint_version"] = "v01"
    grid["fingerprint_method"] = "robust_percentile_minmax_1pct_99pct"

    logger.info("QA summary")
    logger.info(f"Cells processed: {len(grid):,}")
    logger.info(f"Output CRS: {grid.crs}")
    logger.info(f"Fingerprint columns: {fingerprint_cols}")
    logger.info(
        f"Mean fingerprint completeness: "
        f"{grid['fingerprint_completeness_score'].mean():.4f}"
    )

    for col in fingerprint_cols:
        logger.info(
            f"{col} min/mean/max: "
            f"{grid[col].min():.4f} / "
            f"{grid[col].mean():.4f} / "
            f"{grid[col].max():.4f}"
        )

    if grid["fingerprint_completeness_score"].min() < 0.80:
        logger.warning("Some cells have low fingerprint completeness.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    grid.to_file(
        OUTPUT_PATH,
        layer="coastal_grid_physical_fingerprints_v01",
        driver="GPKG",
    )

    logger.info(f"Saved output: {OUTPUT_PATH}")
    logger.info("Script 13 completed successfully.")


if __name__ == "__main__":
    main()
    