#!/usr/bin/env python3
"""
Script 30: Compute Expected Recognition v03

Purpose:
- Use k-nearest comparable places in Fingerprint v03 space.
- Predict expected recognition from physically similar places.
- Compute expected_recognition_v03.

Inputs:
- data/processed/fingerprint_v03.gpkg
- data/processed/recognition_score_v03.gpkg

Outputs:
- data/processed/expected_recognition_v03.gpkg
- data/processed/expected_recognition_v03.csv
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
RECOGNITION_PATH = BASE_DIR / "data/processed/recognition_score_v03.gpkg"

OUT_GPKG = BASE_DIR / "data/processed/expected_recognition_v03.gpkg"
OUT_CSV = BASE_DIR / "data/processed/expected_recognition_v03.csv"

K_NEIGHBORS = 50
EPSILON = 1e-9


def log(msg: str) -> None:
    print(f"[30_compute_expected_recognition_v03] {msg}")


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
    log("Merging fingerprint and recognition inputs")

    require_columns(fp, ["cell_id"])
    require_columns(rec, ["cell_id", "observed_recognition_v03"])

    rec_cols = [
        "cell_id",
        "observed_recognition_v03",
        "recognition_cell_confidence_v03",
        "recognition_dataset_confidence_v03",
        "recognition_total_count_3km_v03",
        "observed_recognition_tier_v03",
    ]

    rec_cols = [c for c in rec_cols if c in rec.columns]

    gdf = fp.merge(
        rec[rec_cols],
        on="cell_id",
        how="left",
        suffixes=("", "_rec"),
    )

    if gdf["observed_recognition_v03"].isna().any():
        raise ValueError("Some cells are missing observed_recognition_v03 after merge")

    return gdf


def compute_expected_recognition(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Computing kNN expected recognition v03")

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

    X = gdf[fp_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    y = pd.to_numeric(gdf["observed_recognition_v03"], errors="coerce").fillna(0).to_numpy()

    # k + 1 because each point finds itself as nearest neighbor.
    k_query = min(K_NEIGHBORS + 1, len(gdf))

    nn = NearestNeighbors(
        n_neighbors=k_query,
        metric="euclidean",
        algorithm="auto",
    )

    nn.fit(X)

    distances, indices = nn.kneighbors(X)

    expected = np.zeros(len(gdf))
    mean_dist = np.zeros(len(gdf))
    min_dist = np.zeros(len(gdf))
    max_dist = np.zeros(len(gdf))
    neighbor_count = np.zeros(len(gdf), dtype=int)

    for i in range(len(gdf)):
        if i % 5000 == 0:
            log(f"Processing expected recognition {i:,}/{len(gdf):,}")

        neigh_idx = indices[i]
        neigh_dist = distances[i]

        # Remove self-match.
        mask = neigh_idx != i
        neigh_idx = neigh_idx[mask]
        neigh_dist = neigh_dist[mask]

        if len(neigh_idx) > K_NEIGHBORS:
            neigh_idx = neigh_idx[:K_NEIGHBORS]
            neigh_dist = neigh_dist[:K_NEIGHBORS]

        if len(neigh_idx) == 0:
            expected[i] = np.nan
            mean_dist[i] = np.nan
            min_dist[i] = np.nan
            max_dist[i] = np.nan
            neighbor_count[i] = 0
            continue

        weights = 1 / (neigh_dist + EPSILON)
        expected[i] = np.sum(weights * y[neigh_idx]) / np.sum(weights)

        mean_dist[i] = np.mean(neigh_dist)
        min_dist[i] = np.min(neigh_dist)
        max_dist[i] = np.max(neigh_dist)
        neighbor_count[i] = len(neigh_idx)

    gdf["expected_recognition_v03_raw"] = expected
    gdf["expected_recognition_v03"] = safe_minmax(gdf["expected_recognition_v03_raw"])

    gdf["expected_recognition_v03_neighbors"] = neighbor_count
    gdf["mean_neighbor_distance_v03"] = mean_dist
    gdf["min_neighbor_distance_v03"] = min_dist
    gdf["max_neighbor_distance_v03"] = max_dist

    return gdf


def compute_comparable_confidence(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Computing comparable-place confidence v03")

    # Lower neighbor distance = higher confidence.
    distance_score = 1 - safe_minmax(gdf["mean_neighbor_distance_v03"])

    neighbor_score = (
        gdf["expected_recognition_v03_neighbors"] / K_NEIGHBORS
    ).clip(0, 1)

    fp_score = gdf.get(
        "fingerprint_completeness_v03",
        pd.Series(1.0, index=gdf.index),
    )

    recognition_conf = gdf.get(
        "recognition_cell_confidence_v03",
        pd.Series(0.5, index=gdf.index),
    )

    gdf["comparable_distance_confidence_v03"] = distance_score
    gdf["comparable_neighbor_count_confidence_v03"] = neighbor_score

    gdf["comparable_confidence_v03"] = (
        0.40 * distance_score
        + 0.25 * neighbor_score
        + 0.20 * fp_score
        + 0.15 * recognition_conf
    ).clip(0, 1)

    return gdf


def compute_residual_preview(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Computing recognition residual preview v03")

    gdf["recognition_residual_v03"] = (
        gdf["expected_recognition_v03"] - gdf["observed_recognition_v03"]
    )

    gdf["positive_under_recognition_residual_v03"] = (
        gdf["recognition_residual_v03"].clip(lower=0)
    )

    gdf["over_recognition_residual_v03"] = (
        (-gdf["recognition_residual_v03"]).clip(lower=0)
    )

    return gdf


def main():
    log("Starting Script 30")

    if not FINGERPRINT_PATH.exists():
        raise FileNotFoundError(f"Fingerprint not found: {FINGERPRINT_PATH}")

    if not RECOGNITION_PATH.exists():
        raise FileNotFoundError(f"Recognition score not found: {RECOGNITION_PATH}")

    log(f"Reading fingerprint: {FINGERPRINT_PATH}")
    fp = gpd.read_file(FINGERPRINT_PATH)

    log(f"Reading recognition: {RECOGNITION_PATH}")
    rec = gpd.read_file(RECOGNITION_PATH)

    log(f"Fingerprint rows: {len(fp):,}")
    log(f"Recognition rows: {len(rec):,}")

    gdf = merge_inputs(fp, rec)
    gdf = compute_expected_recognition(gdf)
    gdf = compute_comparable_confidence(gdf)
    gdf = compute_residual_preview(gdf)

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

    log(f"Writing GeoPackage: {OUT_GPKG}")
    gdf.to_file(OUT_GPKG, driver="GPKG")

    log(f"Writing CSV: {OUT_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")

    print("\nExpected recognition v03 summary:")
    print(
        gdf[
            [
                "observed_recognition_v03",
                "expected_recognition_v03",
                "recognition_residual_v03",
                "positive_under_recognition_residual_v03",
                "mean_neighbor_distance_v03",
                "comparable_confidence_v03",
            ]
        ].describe()
    )


if __name__ == "__main__":
    main()