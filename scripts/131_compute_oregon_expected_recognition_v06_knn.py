#!/usr/bin/env python3
"""
131_compute_oregon_expected_recognition_v06_knn.py

Compute Oregon Expected Recognition v06 using KNN/local-comparison matching.

Purpose:
Replace the v05 quantile-bin expected recognition model with a smoother
nearest-neighbor counterfactual estimate:

Expected Recognition = mean observed recognition among physically similar cells.

Inputs:
- data/processed/oregon_expected_recognition_v05.gpkg

Outputs:
- data/processed/oregon_expected_recognition_v06_knn.gpkg
- data/processed/oregon_expected_recognition_v06_knn.csv
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SCRIPT_NAME = "131_compute_oregon_expected_recognition_v06_knn"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = PROCESSED_DIR / "oregon_expected_recognition_v05.gpkg"

OUT_GPKG = PROCESSED_DIR / "oregon_expected_recognition_v06_knn.gpkg"
OUT_CSV = PROCESSED_DIR / "oregon_expected_recognition_v06_knn.csv"

K_NEIGHBORS = 75


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def norm01(s):
    s = pd.to_numeric(s, errors="coerce")
    lo = s.quantile(0.02)
    hi = s.quantile(0.98)

    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=s.index)

    return ((s - lo) / (hi - lo)).clip(0, 1)


def safe_num(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def main():
    log("Starting Oregon Expected Recognition v06 KNN")

    if not INPUT_GPKG.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_GPKG}")

    gdf = gpd.read_file(INPUT_GPKG)

    log(f"Rows: {len(gdf):,}")
    log(f"CRS: {gdf.crs}")

    feature_cols = [
        "physical_exceptionality_v03",
        "terrain_drama_v03",
        "scenic_coast_v03",
        "coast_proximity_v03",
        "coast_complexity_v03",
        "relief_norm_v03",
        "slope_norm_v03",
    ]

    feature_cols = [c for c in feature_cols if c in gdf.columns]

    if len(feature_cols) < 3:
        raise ValueError(f"Too few KNN feature columns available: {feature_cols}")

    log(f"KNN feature columns: {feature_cols}")

    gdf["observed_recognition_v04"] = safe_num(gdf, "observed_recognition_v04")
    gdf["recognition_cell_confidence_v04"] = safe_num(
        gdf,
        "recognition_cell_confidence_v04",
        0.5,
    )
    gdf["land_area_share"] = safe_num(gdf, "land_area_share", 1.0)
    gdf["is_valid_land_candidate"] = safe_num(gdf, "is_valid_land_candidate", 1.0)

    valid_pool = (
        (gdf["is_valid_land_candidate"] == 1)
        & (gdf["land_area_share"] >= 0.50)
        & (gdf["recognition_cell_confidence_v04"] >= 0.25)
        & gdf["observed_recognition_v04"].notna()
    )

    pool = gdf.loc[valid_pool].copy()

    log(f"Comparable KNN pool rows: {len(pool):,}")

    if len(pool) < K_NEIGHBORS:
        raise ValueError("Comparable pool too small for KNN expected recognition.")

    X_pool = pool[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    X_all = gdf[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    scaler = StandardScaler()
    X_pool_scaled = scaler.fit_transform(X_pool)
    X_all_scaled = scaler.transform(X_all)

    k = min(K_NEIGHBORS, len(pool))

    log(f"Fitting KNN model with k={k}")

    nn = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean",
        algorithm="auto",
    )

    nn.fit(X_pool_scaled)

    log("Querying nearest physical-geographic neighbors")

    distances, indices = nn.kneighbors(X_all_scaled)

    observed_pool = pool["observed_recognition_v04"].to_numpy()
    confidence_pool = pool["recognition_cell_confidence_v04"].to_numpy()

    expected_raw = []
    expected_median = []
    expected_std = []
    expected_confidence = []
    neighbor_mean_distance = []

    for row_dist, row_idx in zip(distances, indices):
        vals = observed_pool[row_idx]
        conf = confidence_pool[row_idx]

        # Inverse-distance weighting with small epsilon.
        weights = 1 / (row_dist + 1e-6)
        weights = weights / weights.sum()

        expected_raw.append(float(np.sum(vals * weights)))
        expected_median.append(float(np.median(vals)))
        expected_std.append(float(np.std(vals)))
        expected_confidence.append(float(np.mean(conf)))
        neighbor_mean_distance.append(float(np.mean(row_dist)))

    gdf["expected_recognition_v06_raw"] = expected_raw
    gdf["expected_recognition_v06_median"] = expected_median
    gdf["expected_recognition_v06_neighbor_std"] = expected_std
    gdf["expected_recognition_v06_neighbor_mean_distance"] = neighbor_mean_distance

    gdf["expected_recognition_v06"] = norm01(gdf["expected_recognition_v06_raw"])

    gdf["recognition_residual_v06"] = (
        gdf["expected_recognition_v06"] - gdf["observed_recognition_v04"]
    )

    gdf["positive_under_recognition_residual_v06"] = (
        gdf["recognition_residual_v06"].clip(lower=0)
    )

    gdf["over_recognition_residual_v06"] = (
        (-gdf["recognition_residual_v06"]).clip(lower=0)
    )

    # Confidence combines recognition confidence and neighbor compactness.
    gdf["expected_recognition_v06_neighbor_compactness"] = (
        1 - norm01(gdf["expected_recognition_v06_neighbor_mean_distance"])
    ).clip(0, 1)

    gdf["expected_recognition_confidence_v06"] = (
        0.60 * pd.Series(expected_confidence, index=gdf.index).fillna(0)
        + 0.40 * gdf["expected_recognition_v06_neighbor_compactness"].fillna(0)
    ).clip(0, 1)

    gdf["expected_recognition_method_v06"] = (
        f"KNN physical-geographic similarity; k={k}; features={','.join(feature_cols)}"
    )

    log("Summary:")
    log(f"Mean observed_recognition_v04: {gdf['observed_recognition_v04'].mean():.4f}")
    log(f"Mean expected_recognition_v05: {gdf['expected_recognition_v05'].mean():.4f}")
    log(f"Mean expected_recognition_v06: {gdf['expected_recognition_v06'].mean():.4f}")
    log(f"Mean recognition_residual_v06: {gdf['recognition_residual_v06'].mean():.4f}")
    log(
        "Mean positive_under_recognition_residual_v06: "
        f"{gdf['positive_under_recognition_residual_v06'].mean():.4f}"
    )

    if OUT_GPKG.exists():
        OUT_GPKG.unlink()

    log(f"Writing GPKG: {OUT_GPKG}")
    gdf.to_file(
        OUT_GPKG,
        layer="oregon_expected_recognition_v06_knn",
        driver="GPKG",
    )

    log(f"Writing CSV: {OUT_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")


if __name__ == "__main__":
    main()