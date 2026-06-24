#!/usr/bin/env python3
"""
142_compute_oregon_recognition_disequilibrium_surface_v01.py

Compute Oregon Recognition Disequilibrium Surface v01.

Purpose:
Move RDE from residual scoring toward field theory by treating
recognition disequilibrium as a spatial surface.

Inputs:
- data/processed/oregon_rde_mechanism_components_v01.gpkg

Outputs:
- data/processed/oregon_recognition_disequilibrium_surface_v01.gpkg
- data/processed/oregon_recognition_disequilibrium_surface_v01.csv
"""

from pathlib import Path
import math

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


SCRIPT_NAME = "142_compute_oregon_recognition_disequilibrium_surface_v01"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = PROCESSED_DIR / "oregon_rde_mechanism_components_v01.gpkg"

OUT_GPKG = PROCESSED_DIR / "oregon_recognition_disequilibrium_surface_v01.gpkg"
OUT_CSV = PROCESSED_DIR / "oregon_recognition_disequilibrium_surface_v01.csv"

NEIGHBOR_RADIUS_M = 3000
MIN_NEIGHBORS = 4
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


def compute_gradient(gdf, value_col):
    log("Computing disequilibrium gradient field")

    centroids = gdf.geometry.centroid
    coords = np.column_stack([centroids.x, centroids.y])
    values = safe_num(gdf, value_col).to_numpy(dtype=float)

    tree = cKDTree(coords)
    neighbor_lists = tree.query_ball_point(coords, r=NEIGHBOR_RADIUS_M)

    dx_values = np.full(len(gdf), np.nan)
    dy_values = np.full(len(gdf), np.nan)
    grad_mag = np.full(len(gdf), np.nan)
    grad_dir = np.full(len(gdf), np.nan)
    neighbor_count = np.zeros(len(gdf), dtype=int)

    for i, neighbors in enumerate(neighbor_lists):
        if i % 2500 == 0:
            log(f"Gradient scan {i:,}/{len(gdf):,}")

        neighbors = [j for j in neighbors if j != i]

        if len(neighbors) < MIN_NEIGHBORS:
            continue

        x0, y0 = coords[i]
        z0 = values[i]

        rows = []
        targets = []

        for j in neighbors:
            xj, yj = coords[j]
            zj = values[j]

            dx = xj - x0
            dy = yj - y0

            dist = math.sqrt(dx * dx + dy * dy)

            if dist <= EPSILON:
                continue

            rows.append([dx, dy])
            targets.append(zj - z0)

        if len(rows) < MIN_NEIGHBORS:
            continue

        A = np.array(rows, dtype=float)
        b = np.array(targets, dtype=float)

        try:
            # Least-squares local plane:
            # dz ≈ gx * dx + gy * dy
            gx, gy = np.linalg.lstsq(A, b, rcond=None)[0]

            dx_values[i] = gx
            dy_values[i] = gy

            magnitude = math.sqrt(gx * gx + gy * gy)
            grad_mag[i] = magnitude

            # Direction of increasing disequilibrium.
            angle = math.degrees(math.atan2(gy, gx))
            if angle < 0:
                angle += 360

            grad_dir[i] = angle
            neighbor_count[i] = len(rows)

        except Exception:
            continue

    return (
        pd.Series(dx_values, index=gdf.index),
        pd.Series(dy_values, index=gdf.index),
        pd.Series(grad_mag, index=gdf.index),
        pd.Series(grad_dir, index=gdf.index),
        pd.Series(neighbor_count, index=gdf.index),
    )


def classify_disequilibrium(row):
    d = row["recognition_disequilibrium_v01"]
    gm = row["recognition_gradient_magnitude_norm_v01"]

    if d >= 0.40 and gm >= 0.60:
        return "high_disequilibrium_high_gradient"

    if d >= 0.40 and gm < 0.60:
        return "high_disequilibrium_basin"

    if d >= 0.20:
        return "moderate_disequilibrium"

    if d > 0:
        return "low_positive_disequilibrium"

    if d == 0:
        return "equilibrium_or_no_gap"

    return "over_recognized"


def main():
    log("Starting Oregon Recognition Disequilibrium Surface v01")

    if not INPUT_GPKG.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_GPKG}")

    gdf = gpd.read_file(INPUT_GPKG)

    log(f"Rows: {len(gdf):,}")
    log(f"CRS: {gdf.crs}")

    gdf["R_observed_v01"] = safe_num(
        gdf,
        "observed_recognition_v04",
    ).clip(0, 1)

    gdf["R_expected_v01"] = safe_num(
        gdf,
        "expected_recognition_v06",
    ).clip(0, 1)

    gdf["recognition_disequilibrium_raw_v01"] = (
        gdf["R_expected_v01"] - gdf["R_observed_v01"]
    )

    gdf["recognition_disequilibrium_v01"] = (
        gdf["recognition_disequilibrium_raw_v01"].clip(lower=0)
    )

    gdf["over_recognition_v01"] = (
        (-gdf["recognition_disequilibrium_raw_v01"]).clip(lower=0)
    )

    (
        grad_dx,
        grad_dy,
        grad_mag,
        grad_dir,
        grad_neighbors,
    ) = compute_gradient(
        gdf,
        "recognition_disequilibrium_v01",
    )

    gdf["recognition_gradient_dx_v01"] = grad_dx
    gdf["recognition_gradient_dy_v01"] = grad_dy
    gdf["recognition_gradient_magnitude_v01"] = grad_mag
    gdf["recognition_gradient_direction_deg_v01"] = grad_dir
    gdf["recognition_gradient_neighbor_count_v01"] = grad_neighbors

    gdf["recognition_gradient_magnitude_norm_v01"] = norm01(
        gdf["recognition_gradient_magnitude_v01"]
    )

    gdf["recognition_disequilibrium_norm_v01"] = norm01(
        gdf["recognition_disequilibrium_v01"]
    )

    gdf["rde_surface_energy_v01"] = (
        gdf["recognition_disequilibrium_norm_v01"]
        * (0.50 + 0.50 * gdf["recognition_gradient_magnitude_norm_v01"])
    ).clip(0, 1)

    gdf["rde_surface_class_v01"] = gdf.apply(
        classify_disequilibrium,
        axis=1,
    )

    log("Summary:")
    log(
        f"Mean disequilibrium: "
        f"{gdf['recognition_disequilibrium_v01'].mean():.4f}"
    )
    log(
        f"Mean gradient magnitude: "
        f"{gdf['recognition_gradient_magnitude_v01'].mean():.8f}"
    )
    log(
        f"Mean RDE surface energy: "
        f"{gdf['rde_surface_energy_v01'].mean():.4f}"
    )

    print("\nRDE surface class counts:")
    print(gdf["rde_surface_class_v01"].value_counts())

    if OUT_GPKG.exists():
        OUT_GPKG.unlink()

    log(f"Writing GPKG: {OUT_GPKG}")
    gdf.to_file(
        OUT_GPKG,
        layer="oregon_recognition_disequilibrium_surface_v01",
        driver="GPKG",
    )

    log(f"Writing CSV: {OUT_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")


if __name__ == "__main__":
    main()