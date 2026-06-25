#!/usr/bin/env python3
"""
150_compute_oregon_recognition_flow_field_v01.py

Compute Oregon Recognition Flow Field v01.

Purpose:
Move RDE from disequilibrium surfaces to flow theory.

Concept:
Recognition disequilibrium D(x) behaves like a potential field.
The recognition flow field is defined as the direction of decreasing
disequilibrium pressure:

Flow = -∇D

Inputs:
- data/processed/oregon_recognition_disequilibrium_surface_v01.gpkg

Outputs:
- data/processed/oregon_recognition_flow_field_v01.gpkg
- data/processed/oregon_recognition_flow_field_v01.csv
"""

from pathlib import Path
import math

import geopandas as gpd
import pandas as pd


SCRIPT_NAME = "150_compute_oregon_recognition_flow_field_v01"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = PROCESSED_DIR / "oregon_recognition_disequilibrium_surface_v01.gpkg"

OUT_GPKG = PROCESSED_DIR / "oregon_recognition_flow_field_v01.gpkg"
OUT_CSV = PROCESSED_DIR / "oregon_recognition_flow_field_v01.csv"


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def norm01(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    mn = s.min()
    mx = s.max()

    if mx == mn:
        return pd.Series(0.0, index=s.index)

    return ((s - mn) / (mx - mn)).clip(0, 1)


def direction_label(angle):
    if pd.isna(angle):
        return "unknown"

    if angle >= 337.5 or angle < 22.5:
        return "east"
    if angle < 67.5:
        return "northeast"
    if angle < 112.5:
        return "north"
    if angle < 157.5:
        return "northwest"
    if angle < 202.5:
        return "west"
    if angle < 247.5:
        return "southwest"
    if angle < 292.5:
        return "south"
    return "southeast"


def flow_class(row):
    mag = row["recognition_flow_magnitude_norm_v01"]
    d = row["recognition_disequilibrium_v01"]

    if d >= 0.40 and mag >= 0.60:
        return "strong_flow_from_high_disequilibrium"

    if d >= 0.40 and mag < 0.60:
        return "high_disequilibrium_low_flow"

    if d >= 0.20 and mag >= 0.60:
        return "moderate_disequilibrium_strong_flow"

    if d >= 0.20:
        return "moderate_disequilibrium_flow"

    if mag >= 0.60:
        return "low_disequilibrium_strong_flow"

    return "weak_flow"


def main():
    log("Starting Oregon Recognition Flow Field v01")

    if not INPUT_GPKG.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_GPKG}")

    gdf = gpd.read_file(INPUT_GPKG)

    log(f"Rows: {len(gdf):,}")
    log(f"CRS: {gdf.crs}")

    required = [
        "recognition_gradient_dx_v01",
        "recognition_gradient_dy_v01",
        "recognition_gradient_magnitude_v01",
        "recognition_disequilibrium_v01",
        "rde_surface_energy_v01",
    ]

    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Flow is negative gradient: recognition pressure moves away from
    # high disequilibrium toward lower disequilibrium.
    gdf["recognition_flow_x_v01"] = -pd.to_numeric(
        gdf["recognition_gradient_dx_v01"],
        errors="coerce",
    )

    gdf["recognition_flow_y_v01"] = -pd.to_numeric(
        gdf["recognition_gradient_dy_v01"],
        errors="coerce",
    )

    gdf["recognition_flow_magnitude_v01"] = (
        gdf["recognition_flow_x_v01"] ** 2
        + gdf["recognition_flow_y_v01"] ** 2
    ) ** 0.5

    gdf["recognition_flow_magnitude_norm_v01"] = norm01(
        gdf["recognition_flow_magnitude_v01"]
    )

    directions = []

    for x, y in zip(
        gdf["recognition_flow_x_v01"],
        gdf["recognition_flow_y_v01"],
    ):
        if pd.isna(x) or pd.isna(y):
            directions.append(float("nan"))
            continue

        angle = math.degrees(math.atan2(y, x))

        if angle < 0:
            angle += 360

        directions.append(angle)

    gdf["recognition_flow_direction_deg_v01"] = directions

    gdf["recognition_flow_direction_label_v01"] = (
        gdf["recognition_flow_direction_deg_v01"].apply(direction_label)
    )

    gdf["recognition_flow_energy_v01"] = (
        gdf["rde_surface_energy_v01"].fillna(0)
        * gdf["recognition_flow_magnitude_norm_v01"].fillna(0)
    ).clip(0, 1)

    gdf["recognition_flow_class_v01"] = gdf.apply(flow_class, axis=1)

    log("Summary:")
    log(
        f"Mean flow magnitude: "
        f"{gdf['recognition_flow_magnitude_v01'].mean():.8f}"
    )
    log(
        f"Mean normalized flow magnitude: "
        f"{gdf['recognition_flow_magnitude_norm_v01'].mean():.4f}"
    )
    log(
        f"Mean flow energy: "
        f"{gdf['recognition_flow_energy_v01'].mean():.4f}"
    )

    print("\nFlow class counts:")
    print(gdf["recognition_flow_class_v01"].value_counts())

    print("\nFlow direction counts:")
    print(gdf["recognition_flow_direction_label_v01"].value_counts())

    if OUT_GPKG.exists():
        OUT_GPKG.unlink()

    log(f"Writing GPKG: {OUT_GPKG}")
    gdf.to_file(
        OUT_GPKG,
        layer="oregon_recognition_flow_field_v01",
        driver="GPKG",
    )

    log(f"Writing CSV: {OUT_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")


if __name__ == "__main__":
    main()