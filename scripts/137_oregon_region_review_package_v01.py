#!/usr/bin/env python3

from pathlib import Path
import geopandas as gpd

SCRIPT_NAME = "137_oregon_region_review_package_v01"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT = (
    PROCESSED_DIR /
    "oregon_region_validation_template_v01.gpkg"
)

TOP10_GPKG = (
    PROCESSED_DIR /
    "oregon_top10_regions_v01.gpkg"
)

TOP20_GPKG = (
    PROCESSED_DIR /
    "oregon_top20_regions_v01.gpkg"
)

TOP50_GPKG = (
    PROCESSED_DIR /
    "oregon_top50_regions_v01.gpkg"
)


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def export_subset(gdf, n, out_path):

    subset = (
        gdf
        .sort_values("discovery_region_rank_v06")
        .head(n)
        .copy()
    )

    keep_cols = [
        "discovery_region_rank_v06",
        "discovery_region_tier_v06",
        "manual_region_type_v01",
        "cell_count",
        "region_area_km2",
        "longitude",
        "latitude",
        "mean_urem_score_v06_raw",
        "mean_physical_exceptionality_v03",
        "mean_observed_recognition_v04",
        "mean_expected_recognition_v06",
        "mean_positive_under_recognition_residual_v06",
        "geometry",
    ]

    keep_cols = [c for c in keep_cols if c in subset.columns]

    subset = subset[keep_cols]

    subset.to_file(
        out_path,
        driver="GPKG"
    )

    log(f"Wrote {len(subset)} regions -> {out_path}")


def main():

    log("Starting Oregon review package export")

    gdf = gpd.read_file(INPUT)

    export_subset(
        gdf,
        10,
        TOP10_GPKG
    )

    export_subset(
        gdf,
        20,
        TOP20_GPKG
    )

    export_subset(
        gdf,
        50,
        TOP50_GPKG
    )

    log("Done")


if __name__ == "__main__":
    main()