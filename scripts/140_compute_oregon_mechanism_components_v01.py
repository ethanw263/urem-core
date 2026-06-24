#!/usr/bin/env python3
"""
140_compute_oregon_mechanism_components_v01.py

Compute first-generation Oregon RDE mechanism components.

Purpose:
Move from UREM discovery to RDE explanation by estimating:

- Opportunity Failure
- Transmission Failure
- Recognition Inefficiency
- Comparative Shadowing

Inputs:
- data/processed/oregon_urem_score_v06.gpkg

Outputs:
- data/processed/oregon_rde_mechanism_components_v01.gpkg
- data/processed/oregon_rde_mechanism_components_v01.csv
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


SCRIPT_NAME = "140_compute_oregon_mechanism_components_v01"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = PROCESSED_DIR / "oregon_urem_score_v06.gpkg"

OUT_GPKG = PROCESSED_DIR / "oregon_rde_mechanism_components_v01.gpkg"
OUT_CSV = PROCESSED_DIR / "oregon_rde_mechanism_components_v01.csv"

SHADOW_RADIUS_M = 25_000
EPSILON = 1e-9


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def safe_num(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def norm01(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    mn = s.min()
    mx = s.max()

    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.0, index=s.index)

    return ((s - mn) / (mx - mn)).clip(0, 1)


def compute_shadow_pressure(gdf):
    log("Computing comparative shadow pressure")

    centroids = gdf.geometry.centroid
    coords = np.column_stack([centroids.x, centroids.y])

    tree = cKDTree(coords)

    recognition = safe_num(gdf, "observed_recognition_v04").to_numpy()
    potential = safe_num(gdf, "physical_exceptionality_v03").to_numpy()

    source_strength = recognition * potential

    shadow_pressure = np.zeros(len(gdf), dtype=float)
    neighbor_counts = np.zeros(len(gdf), dtype=int)

    neighbor_lists = tree.query_ball_point(coords, r=SHADOW_RADIUS_M)

    for i, neighbors in enumerate(neighbor_lists):
        if i % 2500 == 0:
            log(f"Shadow scan {i:,}/{len(gdf):,}")

        neighbors = [j for j in neighbors if j != i]

        if not neighbors:
            continue

        neighbor_coords = coords[neighbors]
        dists = np.linalg.norm(neighbor_coords - coords[i], axis=1)

        valid = dists > 0

        if not valid.any():
            continue

        dists = dists[valid]
        neighbor_idx = np.array(neighbors)[valid]

        values = source_strength[neighbor_idx] / (dists + 1.0)

        shadow_pressure[i] = float(np.sum(values))
        neighbor_counts[i] = len(values)

    return (
        pd.Series(shadow_pressure, index=gdf.index),
        pd.Series(neighbor_counts, index=gdf.index),
    )


def dominant_mechanism(row):
    shares = {
        "opportunity_failure": row["opportunity_failure_share_v01"],
        "transmission_failure": row["transmission_failure_share_v01"],
        "recognition_inefficiency": row["recognition_inefficiency_share_v01"],
        "comparative_shadowing": row["comparative_shadowing_share_v01"],
    }

    best = max(shares, key=shares.get)
    best_value = shares[best]

    if best_value < 0.40:
        return "mixed_mechanism"

    return best


def main():
    log("Starting Oregon RDE mechanism components v01")

    if not INPUT_GPKG.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_GPKG}")

    gdf = gpd.read_file(INPUT_GPKG)

    log(f"Rows: {len(gdf):,}")
    log(f"CRS: {gdf.crs}")

    # Core RDE terms
    gdf["P_physical_potential_v01"] = safe_num(
        gdf,
        "physical_exceptionality_v03",
    ).clip(0, 1)

    gdf["R_observed_recognition_v01"] = safe_num(
        gdf,
        "observed_recognition_v04",
    ).clip(0, 1)

    gdf["R_expected_recognition_v01"] = safe_num(
        gdf,
        "expected_recognition_v06",
    ).clip(0, 1)

    gdf["D_positive_disequilibrium_v01"] = safe_num(
        gdf,
        "positive_under_recognition_residual_v06",
    ).clip(lower=0)

    # --------------------------------------------------
    # Opportunity proxy v01
    # --------------------------------------------------

    recognition_total_count = safe_num(
        gdf,
        "recognition_total_count_3km_v04",
    )

    category_coverage = safe_num(
        gdf,
        "recognition_category_coverage_v04",
    ).clip(0, 1)

    recognition_total_norm = norm01(recognition_total_count)

    gdf["opportunity_proxy_v01"] = (
        0.60 * category_coverage
        + 0.40 * recognition_total_norm
    ).clip(0, 1)

    # --------------------------------------------------
    # Transmission proxy v01
    # --------------------------------------------------

    gdf["transmission_proxy_v01"] = (
        0.50 * category_coverage
        + 0.50 * gdf["R_observed_recognition_v01"]
    ).clip(0, 1)

    # --------------------------------------------------
    # Comparative shadow pressure v01
    # --------------------------------------------------

    shadow_raw, shadow_neighbor_count = compute_shadow_pressure(gdf)

    gdf["shadow_pressure_raw_v01"] = shadow_raw
    gdf["shadow_neighbor_count_v01"] = shadow_neighbor_count
    gdf["shadow_pressure_v01"] = norm01(shadow_raw)

    # --------------------------------------------------
    # Mechanism equations
    # --------------------------------------------------

    P = gdf["P_physical_potential_v01"]
    D = gdf["D_positive_disequilibrium_v01"]
    O = gdf["opportunity_proxy_v01"]
    T = gdf["transmission_proxy_v01"]
    S = gdf["shadow_pressure_v01"]

    gdf["opportunity_failure_v01"] = (
        P * D * (1 - O)
    ).clip(lower=0)

    gdf["transmission_failure_v01"] = (
        P * D * O * (1 - T)
    ).clip(lower=0)

    gdf["recognition_inefficiency_v01"] = (
        P * D * O * T
    ).clip(lower=0)

    gdf["comparative_shadowing_v01"] = (
        P * D * S
    ).clip(lower=0)

    mechanism_cols = [
        "opportunity_failure_v01",
        "transmission_failure_v01",
        "recognition_inefficiency_v01",
        "comparative_shadowing_v01",
    ]

    gdf["mechanism_total_v01"] = gdf[mechanism_cols].sum(axis=1)

    denom = gdf["mechanism_total_v01"] + EPSILON

    gdf["opportunity_failure_share_v01"] = (
        gdf["opportunity_failure_v01"] / denom
    ).clip(0, 1)

    gdf["transmission_failure_share_v01"] = (
        gdf["transmission_failure_v01"] / denom
    ).clip(0, 1)

    gdf["recognition_inefficiency_share_v01"] = (
        gdf["recognition_inefficiency_v01"] / denom
    ).clip(0, 1)

    gdf["comparative_shadowing_share_v01"] = (
        gdf["comparative_shadowing_v01"] / denom
    ).clip(0, 1)

    gdf["dominant_mechanism_v01"] = gdf.apply(
        dominant_mechanism,
        axis=1,
    )

    gdf["dominant_mechanism_share_v01"] = gdf[
        [
            "opportunity_failure_share_v01",
            "transmission_failure_share_v01",
            "recognition_inefficiency_share_v01",
            "comparative_shadowing_share_v01",
        ]
    ].max(axis=1)

    confidence_base = safe_num(
        gdf,
        "recognition_cell_confidence_v04",
        0.5,
    ).clip(0, 1)

    gdf["mechanism_confidence_v01"] = (
        confidence_base
        * norm01(gdf["D_positive_disequilibrium_v01"])
        * gdf["dominant_mechanism_share_v01"]
    ).clip(0, 1)

    gdf["mechanism_vector_v01"] = (
        gdf["opportunity_failure_share_v01"].round(4).astype(str)
        + ","
        + gdf["transmission_failure_share_v01"].round(4).astype(str)
        + ","
        + gdf["recognition_inefficiency_share_v01"].round(4).astype(str)
        + ","
        + gdf["comparative_shadowing_share_v01"].round(4).astype(str)
    )

    log("Summary:")
    log(f"Mean opportunity proxy: {gdf['opportunity_proxy_v01'].mean():.4f}")
    log(f"Mean transmission proxy: {gdf['transmission_proxy_v01'].mean():.4f}")
    log(f"Mean shadow pressure: {gdf['shadow_pressure_v01'].mean():.4f}")

    print("\nDominant mechanism counts:")
    print(gdf["dominant_mechanism_v01"].value_counts())

    print("\nMean mechanism shares:")
    print(
        gdf[
            [
                "opportunity_failure_share_v01",
                "transmission_failure_share_v01",
                "recognition_inefficiency_share_v01",
                "comparative_shadowing_share_v01",
            ]
        ].mean()
    )

    if OUT_GPKG.exists():
        OUT_GPKG.unlink()

    log(f"Writing GPKG: {OUT_GPKG}")
    gdf.to_file(
        OUT_GPKG,
        layer="oregon_rde_mechanism_components_v01",
        driver="GPKG",
    )

    log(f"Writing CSV: {OUT_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")


if __name__ == "__main__":
    main()