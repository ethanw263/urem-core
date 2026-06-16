#!/usr/bin/env python3
"""
80_transmission_artifact_validation_v01.py

Purpose
-------
Determine whether transmission-limited discoveries represent:

A)
genuinely distinct recognition-transmission failures

or

B)
merely remote zero-recognition landscapes.

Scientific Question
-------------------
Do transmission-limited discoveries differ from physically comparable
zero-recognition places?

Outputs
-------
data/processed/transmission_artifact_validation_v01.csv
data/processed/transmission_artifact_validation_summary_v01.csv
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = DATA / "recognition_transmission_index_v01.gpkg"

OUT_DETAIL = DATA / "transmission_artifact_validation_v01.csv"
OUT_SUMMARY = DATA / "transmission_artifact_validation_summary_v01.csv"

TOP_N = 300
N_MATCHES = 50

logging.basicConfig(
    level=logging.INFO,
    format="[80_transmission_artifact_validation_v01] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


def pct_rank(series):
    return pd.to_numeric(series, errors="coerce").rank(pct=True)


def main():

    log.info(f"Reading: {INPUT_PATH}")

    gdf = gpd.read_file(INPUT_PATH)

    required = [
        "cell_id",
        "physical_exceptionality_v03",
        "terrain_drama_v03",
        "local_relief_m",
        "slope_deg",
        "elevation_m",
        "observed_recognition_v04",
        "recognition_transmission_index_v01",
        "recognition_transmission_deficit_v01",
        "transmission_limited_disequilibrium_v01",
        "is_valid_land_candidate",
    ]

    missing = [c for c in required if c not in gdf.columns]

    if missing:
        raise KeyError(f"Missing columns: {missing}")

    gdf = gdf[gdf["is_valid_land_candidate"] == True].copy()

    log.info(f"Valid rows: {len(gdf):,}")

    # ---------------------------------------------------------
    # Group A
    # Transmission discoveries
    # ---------------------------------------------------------

    group_a = (
        gdf.sort_values(
            "transmission_limited_disequilibrium_v01",
            ascending=False,
        )
        .head(TOP_N)
        .copy()
    )

    log.info(f"Transmission discovery cells: {len(group_a):,}")

    # ---------------------------------------------------------
    # Group B
    # Zero-recognition controls
    # ---------------------------------------------------------

    controls = gdf[
        (gdf["observed_recognition_v04"] <= 0.001)
    ].copy()

    controls = controls[
        ~controls["cell_id"].isin(group_a["cell_id"])
    ]

    log.info(f"Zero-recognition control pool: {len(controls):,}")

    features = [
        "physical_exceptionality_v03",
        "terrain_drama_v03",
        "local_relief_m",
        "slope_deg",
        "elevation_m",
    ]

    group_a = group_a.dropna(subset=features)
    controls = controls.dropna(subset=features)

    if len(controls) < 100:
        raise RuntimeError(
            "Insufficient zero-recognition control cells."
        )

    X_controls = controls[features].to_numpy()

    nn = NearestNeighbors(
        n_neighbors=min(N_MATCHES, len(controls)),
        metric="euclidean",
    )

    nn.fit(X_controls)

    X_targets = group_a[features].to_numpy()

    distances, indices = nn.kneighbors(X_targets)

    rows = []

    for i, (_, target_row) in enumerate(group_a.iterrows()):

        matched = controls.iloc[indices[i]]

        rows.append(
            {
                "cell_id": target_row["cell_id"],
                "target_transmission_index":
                    target_row["recognition_transmission_index_v01"],

                "target_transmission_deficit":
                    target_row["recognition_transmission_deficit_v01"],

                "control_mean_transmission_index":
                    matched["recognition_transmission_index_v01"].mean(),

                "control_mean_transmission_deficit":
                    matched["recognition_transmission_deficit_v01"].mean(),

                "delta_transmission_index":
                    target_row["recognition_transmission_index_v01"]
                    - matched["recognition_transmission_index_v01"].mean(),

                "delta_transmission_deficit":
                    target_row["recognition_transmission_deficit_v01"]
                    - matched["recognition_transmission_deficit_v01"].mean(),

                "physical_exceptionality":
                    target_row["physical_exceptionality_v03"],
            }
        )

    result = pd.DataFrame(rows)

    result.to_csv(OUT_DETAIL, index=False)

    summary_rows = []

    metrics = [
        "delta_transmission_index",
        "delta_transmission_deficit",
    ]

    for metric in metrics:

        summary_rows.append(
            {
                "metric": metric,
                "mean": result[metric].mean(),
                "median": result[metric].median(),
                "p90": result[metric].quantile(0.90),
                "p95": result[metric].quantile(0.95),
                "positive_share":
                    (result[metric] > 0).mean(),
            }
        )

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(OUT_SUMMARY, index=False)

    print("\nTransmission Artifact Validation")
    print("--------------------------------")

    print(f"Transmission discoveries: {len(group_a):,}")
    print(f"Matched controls: {N_MATCHES}")

    print("\nSummary")
    print(summary.to_string(index=False))

    print("\nInterpretation")
    print("--------------")
    print("If transmission discoveries show consistently")
    print("higher transmission deficit than physically")
    print("matched zero-recognition controls,")
    print("transmission is not merely remoteness.")
    print()
    print("If differences disappear,")
    print("the current transmission layer is mostly")
    print("capturing remote landscapes.")

if __name__ == "__main__":
    main()