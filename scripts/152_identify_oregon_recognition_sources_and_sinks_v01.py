#!/usr/bin/env python3
"""
152_identify_oregon_recognition_sources_and_sinks_v01.py

Identify Oregon Recognition Sources and Sinks.

Purpose:
Use the recognition flow field to classify cells as:
- recognition sinks
- recognition sources
- transfer zones
- neutral zones

Inputs:
- data/processed/oregon_recognition_flow_field_v01.gpkg

Outputs:
- data/processed/oregon_recognition_sources_sinks_v01.gpkg
- data/processed/oregon_recognition_sources_sinks_v01.csv
"""

from pathlib import Path
import math

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


SCRIPT_NAME = "152_identify_oregon_recognition_sources_and_sinks_v01"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = PROCESSED_DIR / "oregon_recognition_flow_field_v01.gpkg"

OUT_GPKG = PROCESSED_DIR / "oregon_recognition_sources_sinks_v01.gpkg"
OUT_CSV = PROCESSED_DIR / "oregon_recognition_sources_sinks_v01.csv"

NEIGHBOR_RADIUS_M = 3000
EPSILON = 1e-9


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def safe_num(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def norm_signed(values):
    s = pd.Series(values, dtype=float)

    max_abs = np.nanmax(np.abs(s))

    if max_abs == 0:
        return pd.Series(
            np.zeros(len(s)),
            index=s.index,
            dtype=float,
        )

    return (s / max_abs).clip(-1, 1)


def classify_source_sink(row):
    net = row["recognition_net_flow_norm_v01"]
    mag = row["recognition_flow_magnitude_norm_v01"]
    d = row["recognition_disequilibrium_v01"]

    if net >= 0.40 and d >= 0.20:
        return "recognition_sink"

    if net <= -0.40 and mag >= 0.20:
        return "recognition_source"

    if abs(net) < 0.15 and mag >= 0.30:
        return "recognition_transfer_zone"

    if d >= 0.40 and mag < 0.20:
        return "stored_potential_zone"

    return "neutral_or_weak_flow"


def main():
    log("Starting Oregon recognition sources and sinks")

    if not INPUT_GPKG.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_GPKG}")

    gdf = gpd.read_file(INPUT_GPKG)

    log(f"Rows: {len(gdf):,}")
    log(f"CRS: {gdf.crs}")

    required = [
        "recognition_flow_x_v01",
        "recognition_flow_y_v01",
        "recognition_flow_magnitude_norm_v01",
        "recognition_disequilibrium_v01",
    ]

    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    centroids = gdf.geometry.centroid
    coords = np.column_stack([centroids.x, centroids.y])

    flow_x = safe_num(gdf, "recognition_flow_x_v01").to_numpy(dtype=float)
    flow_y = safe_num(gdf, "recognition_flow_y_v01").to_numpy(dtype=float)
    flow_mag = safe_num(gdf, "recognition_flow_magnitude_norm_v01").to_numpy(dtype=float)

    tree = cKDTree(coords)
    neighbor_lists = tree.query_ball_point(coords, r=NEIGHBOR_RADIUS_M)

    incoming = np.zeros(len(gdf), dtype=float)
    outgoing = np.zeros(len(gdf), dtype=float)
    net_flow = np.zeros(len(gdf), dtype=float)
    neighbor_count = np.zeros(len(gdf), dtype=int)

    log("Computing local incoming/outgoing recognition flow")

    for i, neighbors in enumerate(neighbor_lists):
        if i % 2500 == 0:
            log(f"Flow balance scan {i:,}/{len(gdf):,}")

        xi, yi = coords[i]

        for j in neighbors:
            if j == i:
                continue

            xj, yj = coords[j]
            dx = xj - xi
            dy = yj - yi

            dist = math.sqrt(dx * dx + dy * dy)

            if dist <= EPSILON:
                continue

            ux = dx / dist
            uy = dy / dist

            # Flow at i projected toward neighbor j.
            out_proj = flow_x[i] * ux + flow_y[i] * uy

            if out_proj > 0:
                outgoing[i] += out_proj * flow_mag[i]

            # Flow at neighbor j projected toward i.
            # Direction from neighbor to current cell is -u.
            in_proj = flow_x[j] * (-ux) + flow_y[j] * (-uy)

            if in_proj > 0:
                incoming[i] += in_proj * flow_mag[j]

            neighbor_count[i] += 1

    net_flow = incoming - outgoing

    gdf["recognition_incoming_flow_v01"] = incoming
    gdf["recognition_outgoing_flow_v01"] = outgoing
    gdf["recognition_net_flow_v01"] = net_flow
    gdf["recognition_net_flow_norm_v01"] = norm_signed(net_flow)
    gdf["recognition_flow_balance_neighbor_count_v01"] = neighbor_count

    gdf["recognition_source_sink_class_v01"] = gdf.apply(
        classify_source_sink,
        axis=1,
    )

    gdf["recognition_sink_strength_v01"] = (
        gdf["recognition_net_flow_norm_v01"].clip(lower=0)
        * gdf["recognition_disequilibrium_v01"].clip(lower=0)
    )

    gdf["recognition_source_strength_v01"] = (
        (-gdf["recognition_net_flow_norm_v01"]).clip(lower=0)
        * gdf["recognition_flow_magnitude_norm_v01"].clip(lower=0)
    )

    gdf["recognition_transfer_strength_v01"] = (
        (1 - gdf["recognition_net_flow_norm_v01"].abs()).clip(0, 1)
        * gdf["recognition_flow_magnitude_norm_v01"].clip(lower=0)
    )

    log("Summary:")
    print("\nSource / sink class counts:")
    print(gdf["recognition_source_sink_class_v01"].value_counts())

    print("\nMean strengths:")
    print(
        gdf[
            [
                "recognition_sink_strength_v01",
                "recognition_source_strength_v01",
                "recognition_transfer_strength_v01",
            ]
        ].mean()
    )

    if OUT_GPKG.exists():
        OUT_GPKG.unlink()

    log(f"Writing GPKG: {OUT_GPKG}")
    gdf.to_file(
        OUT_GPKG,
        layer="oregon_recognition_sources_sinks_v01",
        driver="GPKG",
    )

    log(f"Writing CSV: {OUT_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")


if __name__ == "__main__":
    main()