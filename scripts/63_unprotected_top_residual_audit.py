#!/usr/bin/env python3

from pathlib import Path
import geopandas as gpd
import pandas as pd

SCRIPT_NAME = "63_unprotected_top_residual_audit"

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_GPKG = (
    BASE_DIR
    / "data/processed/protected_area_audit_top100.gpkg"
)

OUT_CSV = (
    BASE_DIR
    / "data/processed/unprotected_top50_v06.csv"
)

OUT_GPKG = (
    BASE_DIR
    / "data/processed/unprotected_top50_v06.gpkg"
)

OUT_KML = (
    BASE_DIR
    / "data/processed/unprotected_top50_v06.kml"
)

TOP_N = 50


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def main():

    log("Starting unprotected audit")

    gdf = gpd.read_file(INPUT_GPKG)

    # Keep only cells with no PAD-US match
    unprotected = gdf[
        gdf["inside_protected_area"] == False
    ].copy()

    log(f"Unprotected rows: {len(unprotected):,}")

    unprotected = unprotected.sort_values(
        "positive_under_recognition_residual_v06",
        ascending=False,
    )

    top50 = unprotected.head(TOP_N).copy()

    log(f"Top unprotected cells: {len(top50):,}")

    if top50.crs.to_epsg() != 4326:
        top50 = top50.to_crs(4326)

    centroids = top50.geometry.centroid
    top50["longitude"] = centroids.x
    top50["latitude"] = centroids.y

    cols = [
        c
        for c in [
            "cell_id",
            "longitude",
            "latitude",
            "distance_to_coast_m",
            "physical_exceptionality_v03",
            "observed_recognition_v04",
            "expected_recognition_v06_raw",
            "positive_under_recognition_residual_v06",
        ]
        if c in top50.columns
    ]

    log(f"Writing CSV: {OUT_CSV}")
    top50[cols].to_csv(
        OUT_CSV,
        index=False,
    )

    log(f"Writing GPKG: {OUT_GPKG}")
    top50.to_file(
        OUT_GPKG,
        driver="GPKG",
    )

    try:
        log(f"Writing KML: {OUT_KML}")
        top50.to_file(
            OUT_KML,
            driver="KML",
        )
    except Exception as e:
        print(f"KML export skipped: {e}")

    print("\nTop 20 Unprotected Cells:")
    print(
        top50[
            [
                "cell_id",
                "positive_under_recognition_residual_v06",
                "physical_exceptionality_v03",
                "observed_recognition_v04",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    log("Done")


if __name__ == "__main__":
    main()