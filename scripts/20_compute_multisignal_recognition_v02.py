#!/usr/bin/env python3
"""
20_compute_multisignal_recognition_v02.py

Computes multi-signal observed recognition v02 for each UREM grid cell.

Inputs:
- data/processed/coastal_grid_physical_fingerprints_v01.gpkg
- data/processed/osm_recognition_features_v02.gpkg
- data/processed/coastal_grid_golf_recognition.gpkg

Output:
- data/processed/coastal_grid_recognition_v02.gpkg

Recognition v02 combines:
- viewpoint
- peak
- attraction
- campground
- picnic_site
- information
- golf

This is the first broader observed-recognition layer beyond golf-only.
"""

from pathlib import Path
import logging
import sys

import geopandas as gpd
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRID_PATH = PROJECT_ROOT / "data/processed/coastal_grid_physical_fingerprints_v01.gpkg"
OSM_RECOGNITION_PATH = PROJECT_ROOT / "data/processed/osm_recognition_features_v02.gpkg"
GOLF_RECOGNITION_PATH = PROJECT_ROOT / "data/processed/coastal_grid_golf_recognition.gpkg"

OUTPUT_PATH = PROJECT_ROOT / "data/processed/coastal_grid_recognition_v02.gpkg"

TARGET_CRS = "EPSG:3310"

FEATURE_TYPES = [
    "viewpoint",
    "peak",
    "attraction",
    "campground",
    "picnic_site",
    "information",
]

WEIGHTS = {
    "viewpoint_score": 0.30,
    "peak_score": 0.25,
    "attraction_score": 0.20,
    "campground_score": 0.08,
    "picnic_site_score": 0.07,
    "information_score": 0.02,
    "golf_recognition_score": 0.08,
}


def setup_logger():
    logger = logging.getLogger("20_compute_multisignal_recognition_v02")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def robust_count_score(series):
    """
    Convert sparse count variables into 0–1 scores.

    Uses log1p to reduce dominance from cells with many mapped points.
    Then min-max normalizes.
    """
    s = series.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
    s_log = np.log1p(s)

    smin = s_log.min()
    smax = s_log.max()

    if smax == smin:
        return pd.Series(0.0, index=s.index)

    return ((s_log - smin) / (smax - smin)).clip(0, 1)


def safe_score(series):
    return (
        series.astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(0, 1)
    )


def main():
    logger = setup_logger()
    logger.info("Starting Script 20: Compute multi-signal recognition v02")

    for path in [GRID_PATH, OSM_RECOGNITION_PATH, GOLF_RECOGNITION_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    logger.info(f"Reading grid: {GRID_PATH}")
    grid = gpd.read_file(GRID_PATH).to_crs(TARGET_CRS)

    logger.info(f"Reading OSM recognition features: {OSM_RECOGNITION_PATH}")
    osm_features = gpd.read_file(OSM_RECOGNITION_PATH).to_crs(TARGET_CRS)

    logger.info(f"Reading golf recognition grid: {GOLF_RECOGNITION_PATH}")
    golf_grid = gpd.read_file(GOLF_RECOGNITION_PATH).to_crs(TARGET_CRS)

    if grid.empty:
        raise ValueError("Grid is empty.")
    if osm_features.empty:
        raise ValueError("OSM recognition features are empty.")
    if golf_grid.empty:
        raise ValueError("Golf recognition grid is empty.")

    if "cell_id" not in grid.columns:
        grid["cell_id"] = range(1, len(grid) + 1)

    if "cell_id" not in golf_grid.columns:
        golf_grid["cell_id"] = range(1, len(golf_grid) + 1)

    required_osm_cols = ["feature_type", "geometry"]
    missing_osm = [c for c in required_osm_cols if c not in osm_features.columns]
    if missing_osm:
        raise ValueError(f"Missing OSM recognition columns: {missing_osm}")

    logger.info(f"Grid cells loaded: {len(grid):,}")
    logger.info(f"OSM recognition features loaded: {len(osm_features):,}")

    logger.info("Spatially joining OSM point recognition features to grid cells")

    grid_cells = grid[["cell_id", "geometry"]].copy()

    joined = gpd.sjoin(
        osm_features,
        grid_cells,
        how="inner",
        predicate="within",
    )

    logger.info(f"OSM features assigned to grid cells: {len(joined):,}")

    count_table = (
        joined.groupby(["cell_id", "feature_type"])
        .size()
        .reset_index(name="count")
        .pivot(index="cell_id", columns="feature_type", values="count")
        .fillna(0)
        .reset_index()
    )

    count_table.columns.name = None

    for feature_type in FEATURE_TYPES:
        if feature_type not in count_table.columns:
            count_table[feature_type] = 0

    rename_map = {
        feature_type: f"{feature_type}_count"
        for feature_type in FEATURE_TYPES
    }

    count_table = count_table.rename(columns=rename_map)

    grid = grid.merge(count_table, on="cell_id", how="left")

    for feature_type in FEATURE_TYPES:
        count_col = f"{feature_type}_count"
        if count_col not in grid.columns:
            grid[count_col] = 0
        grid[count_col] = grid[count_col].fillna(0).astype(int)

    # Add golf recognition fields.
    golf_cols = [
        "cell_id",
        "golf_recognition_score",
        "golf_area_km2",
        "golf_course_count",
        "golf_area_share",
        "has_golf",
    ]

    golf_cols = [c for c in golf_cols if c in golf_grid.columns]

    grid = grid.merge(
        golf_grid[golf_cols],
        on="cell_id",
        how="left",
        suffixes=("", "_golf"),
    )

    if "golf_recognition_score" not in grid.columns:
        grid["golf_recognition_score"] = 0.0

    grid["golf_recognition_score"] = safe_score(grid["golf_recognition_score"])

    if "has_golf" not in grid.columns:
        grid["has_golf"] = False
    else:
        grid["has_golf"] = grid["has_golf"].fillna(False).astype(bool)

    # Convert counts into component scores.
    for feature_type in FEATURE_TYPES:
        count_col = f"{feature_type}_count"
        score_col = f"{feature_type}_score"
        grid[score_col] = robust_count_score(grid[count_col])

    # Weighted recognition v02.
    grid["observed_recognition_v02"] = 0.0

    for score_col, weight in WEIGHTS.items():
        if score_col not in grid.columns:
            grid[score_col] = 0.0
        grid["observed_recognition_v02"] += weight * safe_score(grid[score_col])

    grid["observed_recognition_v02"] = grid["observed_recognition_v02"].clip(0, 1)

    grid["recognition_v02_method"] = (
        "weighted_osm_point_recognition_plus_golf_log_count_scores"
    )

    grid["recognition_v02_weights"] = (
        "viewpoint .30; peak .25; attraction .20; campground .08; "
        "picnic_site .07; information .02; golf .08"
    )

    logger.info("QA summary")
    logger.info(f"Cells processed: {len(grid):,}")
    logger.info(f"Output CRS: {grid.crs}")

    for feature_type in FEATURE_TYPES:
        count_col = f"{feature_type}_count"
        score_col = f"{feature_type}_score"

        logger.info(
            f"{feature_type}: cells_with_feature="
            f"{(grid[count_col] > 0).sum():,}, "
            f"total_count={grid[count_col].sum():,}, "
            f"max_count={grid[count_col].max():,}, "
            f"max_score={grid[score_col].max():.4f}"
        )

    logger.info(
        f"golf: cells_with_golf={(grid['has_golf']).sum():,}, "
        f"max_score={grid['golf_recognition_score'].max():.4f}"
    )

    logger.info(
        f"Observed recognition v02 min/mean/max: "
        f"{grid['observed_recognition_v02'].min():.4f} / "
        f"{grid['observed_recognition_v02'].mean():.4f} / "
        f"{grid['observed_recognition_v02'].max():.4f}"
    )

    logger.info(
        f"Cells with nonzero recognition v02: "
        f"{(grid['observed_recognition_v02'] > 0).sum():,}"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    grid.to_file(
        OUTPUT_PATH,
        layer="coastal_grid_recognition_v02",
        driver="GPKG",
    )

    logger.info(f"Saved output: {OUTPUT_PATH}")
    logger.info("Script 20 completed successfully.")


if __name__ == "__main__":
    main()