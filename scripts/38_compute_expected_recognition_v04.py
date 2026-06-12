#!/usr/bin/env python3
"""
Script 38: Compute Expected Recognition v04

Purpose:
- Compute expected recognition using Fingerprint v03 space.
- Uses Recognition v04 as the observed recognition target.
- Uses only valid land candidates for neighbor matching.

Inputs:
- data/processed/fingerprint_v03.gpkg
- data/processed/recognition_score_v04.gpkg

Outputs:
- data/processed/expected_recognition_v04.gpkg
- data/processed/expected_recognition_v04.csv
"""

from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]

FINGERPRINT_PATH = BASE_DIR / "data/processed/fingerprint_v03.gpkg"
RECOGNITION_PATH = BASE_DIR / "data/processed/recognition_score_v04.gpkg"

OUT_GPKG = BASE_DIR / "data/processed/expected_recognition_v04.gpkg"
OUT_CSV = BASE_DIR / "data/processed/expected_recognition_v04.csv"

K_NEIGHBORS = 50
EPSILON = 1e-9


def log(msg: str) -> None:
    print(f"[38_compute_expected_recognition_v04] {msg}")


def safe_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    mn = s.min()
    mx = s.max()
    if mx == mn:
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)


def require_columns(gdf: gpd.GeoDataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def merge_inputs(fp: gpd.GeoDataFrame, rec: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Merging fingerprint and Recognition v04 inputs")

    require_columns(fp, ["cell_id"])
    require_columns(rec, ["cell_id", "observed_recognition_v04"])

    rec_cols = [
        "cell_id",
        "observed_recognition_v04",
        "recognition_cell_confidence_v04",
        "recognition_dataset_confidence_v04",
        "recognition_total_count_3km_v04",
        "recognition_category_coverage_v04",
        "observed_recognition_tier_v04",
        "recognition_valid_land_flag_v04",
        "land_area_share",
        "water_area_share",
        "is_valid_land_candidate",
    ]

    rec_cols = [c for c in rec_cols if c in rec.columns]

    gdf = fp.merge(rec[rec_cols], on="cell_id", how="left")

    if gdf["observed_recognition_v04"].isna().any():
        raise ValueError("Some cells missing observed_recognition_v04 after merge")

    if "is_valid_land_candidate" not in gdf.columns:
        if "recognition_valid_land_flag_v04" in gdf.columns:
            gdf["is_valid_land_candidate"] = gdf["recognition_valid_land_flag_v04"]
        else:
            gdf["is_valid_land_candidate"] = True

    gdf["is_valid_land_candidate"] = gdf["is_valid_land_candidate"].astype(bool)

    return gdf


def compute_expected_recognition(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Computing expected recognition v04 using valid land neighbors only")

    fp_cols = [
        "fp_coastal_proximity_v03",
        "fp_elevation_v03",
        "fp_relief_v03",
        "fp_slope_v03",
        "fp_coastline_complexity_v03",
        "fp_exceptionality_v03",
        "fp_rarity_v03",
    ]

    require_columns(gdf, fp_cols)

    valid_neighbor_pool = gdf[gdf["is_valid_land_candidate"]].copy()

    log(f"Valid land neighbor pool: {len(valid_neighbor_pool):,}")

    X_pool = (
        valid_neighbor_pool[fp_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy()
    )

    y_pool = (
        pd.to_numeric(valid_neighbor_pool["observed_recognition_v04"], errors="coerce")
        .fillna(0)
        .to_numpy()
    )

    X_all = (
        gdf[fp_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy()
    )

    pool_cell_ids = valid_neighbor_pool["cell_id"].to_numpy()
    all_cell_ids = gdf["cell_id"].to_numpy()

    k_query = min(K_NEIGHBORS + 1, len(valid_neighbor_pool))

    nn = NearestNeighbors(
        n_neighbors=k_query,
        metric="euclidean",
        algorithm="auto",
    )
    nn.fit(X_pool)

    distances, indices = nn.kneighbors(X_all)

    expected = np.zeros(len(gdf))
    mean_dist = np.zeros(len(gdf))
    min_dist = np.zeros(len(gdf))
    max_dist = np.zeros(len(gdf))
    neighbor_count = np.zeros(len(gdf), dtype=int)

    for i in range(len(gdf)):
        if i % 5000 == 0:
            log(f"Processing expected recognition {i:,}/{len(gdf):,}")

        neigh_pool_idx = indices[i]
        neigh_dist = distances[i]

        # Remove self if the current cell is also in the valid pool.
        current_cell_id = all_cell_ids[i]
        neigh_cell_ids = pool_cell_ids[neigh_pool_idx]

        mask = neigh_cell_ids != current_cell_id
        neigh_pool_idx = neigh_pool_idx[mask]
        neigh_dist = neigh_dist[mask]

        if len(neigh_pool_idx) > K_NEIGHBORS:
            neigh_pool_idx = neigh_pool_idx[:K_NEIGHBORS]
            neigh_dist = neigh_dist[:K_NEIGHBORS]

        if len(neigh_pool_idx) == 0:
            expected[i] = np.nan
            mean_dist[i] = np.nan
            min_dist[i] = np.nan
            max_dist[i] = np.nan
            neighbor_count[i] = 0
            continue

        weights = 1 / (neigh_dist + EPSILON)
        expected[i] = np.sum(weights * y_pool[neigh_pool_idx]) / np.sum(weights)

        mean_dist[i] = np.mean(neigh_dist)
        min_dist[i] = np.min(neigh_dist)
        max_dist[i] = np.max(neigh_dist)
        neighbor_count[i] = len(neigh_pool_idx)

    gdf["expected_recognition_v04_raw"] = expected
    gdf["expected_recognition_v04"] = safe_minmax(gdf["expected_recognition_v04_raw"])

    gdf["expected_recognition_v04_neighbors"] = neighbor_count
    gdf["mean_neighbor_distance_v04"] = mean_dist
    gdf["min_neighbor_distance_v04"] = min_dist
    gdf["max_neighbor_distance_v04"] = max_dist

    return gdf


def compute_confidence(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Computing comparable confidence v04")

    distance_score = 1 - safe_minmax(gdf["mean_neighbor_distance_v04"])

    neighbor_score = (
        gdf["expected_recognition_v04_neighbors"] / K_NEIGHBORS
    ).clip(0, 1)

    fp_score = gdf.get(
        "fingerprint_completeness_v03",
        pd.Series(1.0, index=gdf.index),
    )

    recognition_conf = gdf.get(
        "recognition_cell_confidence_v04",
        pd.Series(0.65, index=gdf.index),
    )

    land_conf = gdf["is_valid_land_candidate"].astype(float)

    gdf["comparable_distance_confidence_v04"] = distance_score
    gdf["comparable_neighbor_count_confidence_v04"] = neighbor_score

    gdf["comparable_confidence_v04"] = (
        0.35 * distance_score
        + 0.25 * neighbor_score
        + 0.20 * fp_score
        + 0.15 * recognition_conf
        + 0.05 * land_conf
    ).clip(0, 1)

    return gdf


def compute_residual(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Computing recognition residual v04")

    gdf["recognition_residual_v04"] = (
        gdf["expected_recognition_v04"] - gdf["observed_recognition_v04"]
    )

    gdf["positive_under_recognition_residual_v04"] = (
        gdf["recognition_residual_v04"].clip(lower=0)
    )

    gdf["over_recognition_residual_v04"] = (
        (-gdf["recognition_residual_v04"]).clip(lower=0)
    )

    return gdf


def main():
    log("Starting Script 38")

    if not FINGERPRINT_PATH.exists():
        raise FileNotFoundError(f"Fingerprint not found: {FINGERPRINT_PATH}")
    if not RECOGNITION_PATH.exists():
        raise FileNotFoundError(f"Recognition v04 not found: {RECOGNITION_PATH}")

    log(f"Reading fingerprint: {FINGERPRINT_PATH}")
    fp = gpd.read_file(FINGERPRINT_PATH)

    log(f"Reading recognition v04: {RECOGNITION_PATH}")
    rec = gpd.read_file(RECOGNITION_PATH)

    log(f"Fingerprint rows: {len(fp):,}")
    log(f"Recognition rows: {len(rec):,}")

    gdf = merge_inputs(fp, rec)
    gdf = compute_expected_recognition(gdf)
    gdf = compute_confidence(gdf)
    gdf = compute_residual(gdf)

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

    log(f"Writing GeoPackage: {OUT_GPKG}")
    gdf.to_file(OUT_GPKG, driver="GPKG")

    log(f"Writing CSV: {OUT_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")

    print("\nExpected recognition v04 summary:")
    print(
        gdf[
            [
                "observed_recognition_v04",
                "expected_recognition_v04",
                "recognition_residual_v04",
                "positive_under_recognition_residual_v04",
                "mean_neighbor_distance_v04",
                "comparable_confidence_v04",
                "is_valid_land_candidate",
            ]
        ].describe(include="all")
    )

    print("\nLand-valid counts:")
    print(gdf["is_valid_land_candidate"].value_counts())


if __name__ == "__main__":
    main()