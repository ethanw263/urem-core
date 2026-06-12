#!/usr/bin/env python3
"""
Script 36: Build Recognition Features v04

Purpose:
- Count expanded OSM v04 recognition features around each grid cell.
- Uses land validity mask so later UREM v04 can exclude water-only cells.

Inputs:
- data/processed/coastal_grid_1km.gpkg
- data/processed/osm_recognition_features_v04.gpkg
- data/processed/land_validity_mask_v01.gpkg

Outputs:
- data/processed/recognition_features_v04.gpkg
- data/processed/recognition_features_v04.csv
"""

from pathlib import Path
import warnings
import gc

import geopandas as gpd
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]

GRID_PATH = BASE_DIR / "data/processed/coastal_grid_1km.gpkg"
OSM_V04_PATH = BASE_DIR / "data/processed/osm_recognition_features_v04.gpkg"
LAND_MASK_PATH = BASE_DIR / "data/processed/land_validity_mask_v01.gpkg"

OUT_GPKG = BASE_DIR / "data/processed/recognition_features_v04.gpkg"
OUT_CSV = BASE_DIR / "data/processed/recognition_features_v04.csv"
TEMP_CSV = BASE_DIR / "data/processed/recognition_features_v04_temp.csv"

RADIUS_M = 3000


def log(msg: str) -> None:
    print(f"[36_build_recognition_features_v04] {msg}")


def safe_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    mn = s.min()
    mx = s.max()
    if mx == mn:
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)


def safe_log_score(series: pd.Series) -> pd.Series:
    return safe_minmax(np.log1p(pd.to_numeric(series, errors="coerce").fillna(0)))


def merge_land_mask(grid: gpd.GeoDataFrame, land: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Merging land validity mask")

    cols = [
        "cell_id",
        "land_area_share",
        "water_area_share",
        "centroid_on_land",
        "is_valid_land_candidate",
    ]

    land_cols = [c for c in cols if c in land.columns]

    grid = grid.merge(
        land[land_cols],
        on="cell_id",
        how="left",
    )

    grid["is_valid_land_candidate"] = grid["is_valid_land_candidate"].fillna(False)

    return grid


def count_category(grid: gpd.GeoDataFrame, features: gpd.GeoDataFrame, category: str) -> pd.Series:
    out_col = f"{category}_count_3km"

    subset = features[
        features["recognition_categories"].fillna("").str.contains(category, regex=False)
    ].copy()

    if subset.empty:
        return pd.Series(0, index=grid.index, name=out_col)

    log(f"{category}: {len(subset):,} source features")

    subset = subset.to_crs(grid.crs)

    # Use representative points to reduce spatial join complexity.
    subset["geometry"] = subset.geometry.representative_point()
    subset = subset[["geometry"]].copy()

    buffers = grid[["cell_id", "geometry"]].copy()
    buffers["geometry"] = buffers.geometry.centroid.buffer(RADIUS_M)

    joined = gpd.sjoin(
        buffers,
        subset,
        how="left",
        predicate="contains",
    )

    matched = joined[joined["index_right"].notna()]
    counts = matched.groupby("cell_id").size()

    out = grid["cell_id"].map(counts).fillna(0).astype(int)
    out.name = out_col

    del subset, buffers, joined, matched, counts
    gc.collect()

    return out


def compute_features(grid: gpd.GeoDataFrame, osm: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Computing expanded Recognition v04 feature counts")

    categories = [
        "trail_path",
        "beach",
        "park_recreation",
        "protected_area",
        "parking",
        "trailhead",
        "visitor_information",
        "viewpoint",
        "named_natural_feature",
        "tourism_recreation",
    ]

    for category in categories:
        log(f"Counting category: {category}")

        count_col = f"{category}_count_3km"
        score_col = f"{category}_score_v04"

        grid[count_col] = count_category(grid, osm, category)
        grid[score_col] = safe_log_score(grid[count_col])

        log(
            f"{category}: total={grid[count_col].sum():,}, "
            f"max={grid[count_col].max():,}"
        )

        grid.drop(columns="geometry").to_csv(TEMP_CSV, index=False)

    return grid


def compute_group_scores(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Computing Recognition v04 group scores")

    grid["recognition_access_v04"] = (
        0.60 * grid["parking_score_v04"]
        + 0.40 * grid["trailhead_score_v04"]
    )

    grid["recognition_recreation_v04"] = (
        0.35 * grid["trail_path_score_v04"]
        + 0.20 * grid["park_recreation_score_v04"]
        + 0.15 * grid["beach_score_v04"]
        + 0.15 * grid["tourism_recreation_score_v04"]
        + 0.15 * grid["visitor_information_score_v04"]
    )

    grid["recognition_natural_v04"] = (
        0.35 * grid["protected_area_score_v04"]
        + 0.25 * grid["beach_score_v04"]
        + 0.20 * grid["viewpoint_score_v04"]
        + 0.20 * grid["named_natural_feature_score_v04"]
    )

    grid["observed_recognition_preview_v04_raw"] = (
        0.40 * grid["recognition_recreation_v04"]
        + 0.30 * grid["recognition_natural_v04"]
        + 0.30 * grid["recognition_access_v04"]
    )

    grid["observed_recognition_preview_v04"] = safe_minmax(
        grid["observed_recognition_preview_v04_raw"]
    )

    count_cols = [c for c in grid.columns if c.endswith("_count_3km")]
    grid["recognition_total_count_3km_v04"] = grid[count_cols].sum(axis=1)
    grid["recognition_category_coverage_v04"] = (
        (grid[count_cols] > 0).sum(axis=1) / len(count_cols)
    )

    return grid


def main():
    log("Starting Script 36")

    if not GRID_PATH.exists():
        raise FileNotFoundError(f"Grid not found: {GRID_PATH}")
    if not OSM_V04_PATH.exists():
        raise FileNotFoundError(f"OSM v04 features not found: {OSM_V04_PATH}")
    if not LAND_MASK_PATH.exists():
        raise FileNotFoundError(f"Land mask not found: {LAND_MASK_PATH}")

    log(f"Reading grid: {GRID_PATH}")
    grid = gpd.read_file(GRID_PATH)

    log(f"Reading land mask: {LAND_MASK_PATH}")
    land = gpd.read_file(LAND_MASK_PATH)

    log(f"Reading OSM v04 features: {OSM_V04_PATH}")
    osm = gpd.read_file(OSM_V04_PATH)

    if "cell_id" not in grid.columns:
        grid["cell_id"] = range(len(grid))

    log(f"Grid rows: {len(grid):,}")
    log(f"OSM v04 rows: {len(osm):,}")

    grid = merge_land_mask(grid, land)
    grid = compute_features(grid, osm)
    grid = compute_group_scores(grid)

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

    log(f"Writing GeoPackage: {OUT_GPKG}")
    grid.to_file(OUT_GPKG, driver="GPKG")

    log(f"Writing CSV: {OUT_CSV}")
    grid.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")

    print("\nRecognition v04 summary:")
    print(
        grid[
            [
                "trail_path_count_3km",
                "beach_count_3km",
                "park_recreation_count_3km",
                "protected_area_count_3km",
                "parking_count_3km",
                "visitor_information_count_3km",
                "viewpoint_count_3km",
                "recognition_total_count_3km_v04",
                "observed_recognition_preview_v04",
                "land_area_share",
                "is_valid_land_candidate",
            ]
        ].describe(include="all")
    )

    print("\nLand-valid candidate counts:")
    print(grid["is_valid_land_candidate"].value_counts())


if __name__ == "__main__":
    main()