#!/usr/bin/env python3
"""
135_oregon_discovery_region_interpretation_v01.py

Add interpretation fields to Oregon discovery regions.

Purpose
-------
Prepare discovery regions for manual geographic review.

Inputs
------
data/processed/oregon_discovery_regions_v06.gpkg
data/processed/oregon_discovery_region_centroids_v06.gpkg

Outputs
-------
data/processed/oregon_discovery_region_interpretation_v01.gpkg
data/processed/oregon_discovery_region_interpretation_v01.csv
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

SCRIPT_NAME = "135_oregon_discovery_region_interpretation_v01"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_REGIONS = PROCESSED_DIR / "oregon_discovery_regions_v06.gpkg"
INPUT_CENTROIDS = PROCESSED_DIR / "oregon_discovery_region_centroids_v06.gpkg"

OUT_GPKG = (
    PROCESSED_DIR /
    "oregon_discovery_region_interpretation_v01.gpkg"
)

OUT_CSV = (
    PROCESSED_DIR /
    "oregon_discovery_region_interpretation_v01.csv"
)


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def region_type(area_km2):

    if area_km2 >= 75:
        return "large_landscape"

    if area_km2 >= 25:
        return "medium_landscape"

    return "localized_feature"


def main():

    log("Starting Oregon discovery region interpretation")

    regions = gpd.read_file(INPUT_REGIONS)

    centroids = gpd.read_file(INPUT_CENTROIDS)

    log(f"Regions loaded: {len(regions):,}")
    log(f"Centroids loaded: {len(centroids):,}")

    centroid_cols = [
        c for c in centroids.columns
        if c != "geometry"
    ]

    regions = regions.merge(
        centroids[centroid_cols],
        on="discovery_region_id_v06",
        how="left",
        suffixes=("", "_centroid")
    )

    # --------------------------------------------------
    # Geographic coordinates
    # --------------------------------------------------

    centroids_wgs84 = centroids.to_crs(4326)

    coord_df = pd.DataFrame({
        "discovery_region_id_v06":
            centroids_wgs84["discovery_region_id_v06"],
        "longitude":
            centroids_wgs84.geometry.x,
        "latitude":
            centroids_wgs84.geometry.y,
    })

    regions = regions.merge(
        coord_df,
        on="discovery_region_id_v06",
        how="left"
    )

    # --------------------------------------------------
    # Region classification
    # --------------------------------------------------

    regions["manual_region_type_v01"] = (
        regions["region_area_km2"]
        .apply(region_type)
    )

    # --------------------------------------------------
    # Manual interpretation fields
    # --------------------------------------------------

    regions["manual_region_name_v01"] = ""

    regions["manual_nearest_town_v01"] = ""

    regions["manual_nearest_highway_v01"] = ""

    regions["manual_nearest_park_v01"] = ""

    regions["manual_review_notes_v01"] = ""

    regions["manual_validation_status_v01"] = ""

    regions["manual_hidden_gem_score_v01"] = None

    # --------------------------------------------------
    # Rank group
    # --------------------------------------------------

    def rank_group(rank):

        if rank <= 10:
            return "top_10"

        if rank <= 20:
            return "top_20"

        if rank <= 50:
            return "top_50"

        return "other"

    regions["review_priority_group_v01"] = (
        regions["discovery_region_rank_v06"]
        .apply(rank_group)
    )

    # --------------------------------------------------
    # Review summary
    # --------------------------------------------------

    summary = (
        regions.groupby(
            "manual_region_type_v01"
        )
        .agg(
            region_count=(
                "discovery_region_id_v06",
                "count"
            ),
            mean_area_km2=(
                "region_area_km2",
                "mean"
            ),
            mean_score=(
                "mean_urem_score_v06_raw",
                "mean"
            )
        )
        .reset_index()
    )

    log("Region type summary:")
    print(summary)

    # --------------------------------------------------
    # Write outputs
    # --------------------------------------------------

    log(f"Writing GPKG: {OUT_GPKG}")

    regions.to_file(
        OUT_GPKG,
        layer="oregon_region_interpretation_v01",
        driver="GPKG"
    )

    log(f"Writing CSV: {OUT_CSV}")

    regions.drop(
        columns="geometry",
        errors="ignore"
    ).to_csv(
        OUT_CSV,
        index=False
    )

    log("Done")


if __name__ == "__main__":
    main()