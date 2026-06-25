#!/usr/bin/env python3

"""
153_compute_oregon_recognition_divergence_field_v01.py

Recognition Disequilibrium Evolution (RDE)

Compute a mathematically defensible recognition-flow divergence field:

    div(F) = dFx/dx + dFy/dy

where:

    F = recognition flow field

Positive divergence:
    recognition source behavior

Negative divergence:
    recognition sink behavior

Near zero:
    transfer / neutral behavior

Author:
    Ethan Wilson / RDE Project

Version:
    v01
"""

from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd

from scipy.spatial import cKDTree


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

SCRIPT_NAME = "153_compute_oregon_recognition_divergence_field_v01"

INPUT_GPKG = (
    "data/processed/"
    "oregon_recognition_flow_field_v01.gpkg"
)

OUTPUT_GPKG = (
    "data/processed/"
    "oregon_recognition_divergence_field_v01.gpkg"
)

OUTPUT_CSV = (
    "data/processed/"
    "oregon_recognition_divergence_field_v01.csv"
)

SEARCH_RADIUS_M = 3000.0
MIN_NEIGHBORS = 8


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def normalize_signed(series):
    s = pd.Series(series)

    p95 = np.nanpercentile(np.abs(s), 95)

    if p95 <= 0:
        return np.zeros(len(s))

    out = s / p95
    out = np.clip(out, -1, 1)

    return out


def fit_local_plane(x, y, z):
    """
    Fit:

        z = a*x + b*y + c

    Returns:
        a, b, c
    """

    A = np.column_stack(
        [
            x,
            y,
            np.ones(len(x))
        ]
    )

    coef, *_ = np.linalg.lstsq(
        A,
        z,
        rcond=None
    )

    return coef


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print(f"[{SCRIPT_NAME}] Starting")

    gdf = gpd.read_file(INPUT_GPKG)

    print(
        f"[{SCRIPT_NAME}] Rows: "
        f"{len(gdf):,}"
    )

    print(
        f"[{SCRIPT_NAME}] CRS: "
        f"{gdf.crs}"
    )

    required = [
        "recognition_flow_x_v01",
        "recognition_flow_y_v01"
    ]

    missing = [
        c for c in required
        if c not in gdf.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    centroids = gdf.geometry.centroid

    xcoord = centroids.x.values
    ycoord = centroids.y.values

    coords = np.column_stack(
        [xcoord, ycoord]
    )

    tree = cKDTree(coords)

    fx = (
        gdf["recognition_flow_x_v01"]
        .fillna(0)
        .values
    )

    fy = (
        gdf["recognition_flow_y_v01"]
        .fillna(0)
        .values
    )

    divergence = np.full(
        len(gdf),
        np.nan
    )

    neighbor_count = np.zeros(
        len(gdf),
        dtype=int
    )

    print(
        f"[{SCRIPT_NAME}] "
        f"Computing local divergence field"
    )

    for i in range(len(gdf)):

        idx = tree.query_ball_point(
            coords[i],
            SEARCH_RADIUS_M
        )

        if len(idx) < MIN_NEIGHBORS:
            continue

        neighbor_count[i] = len(idx)

        x_local = xcoord[idx]
        y_local = ycoord[idx]

        fx_local = fx[idx]
        fy_local = fy[idx]

        ax, bx, cx = fit_local_plane(
            x_local,
            y_local,
            fx_local
        )

        ay, by, cy = fit_local_plane(
            x_local,
            y_local,
            fy_local
        )

        dfx_dx = ax
        dfy_dy = by

        divergence[i] = (
            dfx_dx
            + dfy_dy
        )

    gdf[
        "recognition_divergence_v01"
    ] = divergence

    gdf[
        "recognition_divergence_neighbor_count_v01"
    ] = neighbor_count

    norm_div = normalize_signed(
        divergence
    )

    gdf[
        "recognition_divergence_norm_v01"
    ] = norm_div

    gdf[
        "recognition_source_strength_v01"
    ] = np.where(
        norm_div > 0,
        norm_div,
        0
    )

    gdf[
        "recognition_sink_strength_v01"
    ] = np.where(
        norm_div < 0,
        np.abs(norm_div),
        0
    )

    gdf[
        "recognition_transfer_strength_v01"
    ] = 1 - np.abs(norm_div)

    # -------------------------------------------------------------
    # Classification
    # -------------------------------------------------------------

    q95 = np.nanpercentile(
        norm_div,
        95
    )

    q75 = np.nanpercentile(
        norm_div,
        75
    )

    q05 = np.nanpercentile(
        norm_div,
        5
    )

    q25 = np.nanpercentile(
        norm_div,
        25
    )

    classes = []

    for div, ndiv in zip(
        divergence,
        norm_div
    ):

        if np.isnan(div):

            classes.append(
                "insufficient_neighbors"
            )

        elif abs(ndiv) < 0.10:

            classes.append(
                "weak_or_neutral"
            )

        elif ndiv >= q95:

            classes.append(
                "strong_source"
            )

        elif ndiv >= q75:

            classes.append(
                "moderate_source"
            )

        elif ndiv <= q05:

            classes.append(
                "strong_sink"
            )

        elif ndiv <= q25:

            classes.append(
                "moderate_sink"
            )

        else:

            classes.append(
                "transfer_zone"
            )

    gdf[
        "recognition_divergence_class_v01"
    ] = classes

    # -------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------

    valid = (
        gdf[
            "recognition_divergence_v01"
        ]
        .dropna()
    )

    print()

    print(
        f"[{SCRIPT_NAME}] "
        f"Divergence summary"
    )

    print(
        f"Count: {len(valid):,}"
    )

    print(
        f"Mean: {valid.mean():.8f}"
    )

    print(
        f"Std: {valid.std():.8f}"
    )

    print(
        f"Min: {valid.min():.8f}"
    )

    print(
        f"Max: {valid.max():.8f}"
    )

    print()

    print(
        gdf[
            "recognition_divergence_class_v01"
        ]
        .value_counts(
            dropna=False
        )
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    print()

    print(
        f"[{SCRIPT_NAME}] "
        f"Writing GPKG"
    )

    gdf.to_file(
        OUTPUT_GPKG,
        driver="GPKG"
    )

    print(
        f"[{SCRIPT_NAME}] "
        f"Writing CSV"
    )

    gdf.drop(
        columns="geometry"
    ).to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        f"[{SCRIPT_NAME}] Done"
    )


if __name__ == "__main__":
    main()