#!/usr/bin/env python3
"""
66_cluster_v07_no_coast_discovery_regions.py

Cluster top v07 no-coast UREM candidate cells into discovery regions.

Purpose:
- Convert hundreds of adjacent high-scoring cells into meaningful regions.
- Identify distinct coastal UREM discoveries.
- Avoid over-counting neighboring grid cells.

Inputs:
- data/processed/ranked_urem_candidates_v07_no_coast.gpkg

Outputs:
- data/processed/v07_no_coast_discovery_regions.gpkg
- data/processed/v07_no_coast_discovery_regions.csv
- data/processed/v07_no_coast_discovery_region_centroids.csv
- data/processed/v07_no_coast_discovery_region_points.gpkg
"""

from pathlib import Path
import warnings

import geopandas as gpd
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

SCRIPT_NAME = "66_cluster_v07_no_coast_discovery_regions"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = PROCESSED_DIR / "ranked_urem_candidates_v07_no_coast.gpkg"

OUT_REGIONS_GPKG = PROCESSED_DIR / "v07_no_coast_discovery_regions.gpkg"
OUT_REGIONS_CSV = PROCESSED_DIR / "v07_no_coast_discovery_regions.csv"
OUT_CENTROIDS_CSV = PROCESSED_DIR / "v07_no_coast_discovery_region_centroids.csv"
OUT_POINTS_GPKG = PROCESSED_DIR / "v07_no_coast_discovery_region_points.gpkg"

TOP_N_CELLS = 300
BUFFER_M = 2500
MIN_CELLS_PER_REGION = 2


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def main():
    log("Starting v07 no-coast discovery region clustering")

    if not INPUT_GPKG.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_GPKG}")

    candidates = gpd.read_file(INPUT_GPKG)

    if candidates.empty:
        raise ValueError("Input candidate layer is empty.")

    if candidates.crs is None:
        raise ValueError("Input candidate layer has no CRS.")

    log(f"Candidate rows: {len(candidates):,}")
    log(f"CRS: {candidates.crs}")

    candidates = candidates.sort_values(
        "urem_score_v07_no_coast",
        ascending=False,
    ).head(TOP_N_CELLS).copy()

    log(f"Selected top cells: {len(candidates):,}")

    working = candidates.copy()
    working["geometry"] = working.geometry.buffer(BUFFER_M)

    dissolved = working.dissolve()
    exploded = dissolved.explode(index_parts=False).reset_index(drop=True)

    regions = gpd.GeoDataFrame(
        {"region_id": range(1, len(exploded) + 1)},
        geometry=exploded.geometry,
        crs=candidates.crs,
    )

    log(f"Initial regions: {len(regions):,}")

    joined = gpd.sjoin(
        candidates,
        regions[["region_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    stats = (
        joined.groupby("region_id")
        .agg(
            cell_count=("cell_id", "count"),
            mean_urem_score_v07_no_coast=("urem_score_v07_no_coast", "mean"),
            max_urem_score_v07_no_coast=("urem_score_v07_no_coast", "max"),
            mean_terrain_only_exceptionality_v07=(
                "terrain_only_exceptionality_v07",
                "mean",
            ),
            max_terrain_only_exceptionality_v07=(
                "terrain_only_exceptionality_v07",
                "max",
            ),
            mean_positive_residual_v06=(
                "positive_under_recognition_residual_v06",
                "mean",
            ),
            max_positive_residual_v06=(
                "positive_under_recognition_residual_v06",
                "max",
            ),
            mean_observed_recognition_v04=("observed_recognition_v04", "mean"),
            mean_expected_recognition_v06_raw=(
                "expected_recognition_v06_raw",
                "mean",
            ),
            mean_distance_to_coast_m=("distance_to_coast_m", "mean"),
            min_distance_to_coast_m=("distance_to_coast_m", "min"),
            max_distance_to_coast_m=("distance_to_coast_m", "max"),
            mean_local_relief_m=("local_relief_m", "mean"),
            mean_slope_deg=("slope_deg", "mean"),
            mean_elevation_m=("elevation_m", "mean"),
            best_cell_id=("cell_id", "first"),
        )
        .reset_index()
    )

    regions = regions.merge(stats, on="region_id", how="left")
    regions = regions[regions["cell_count"] >= MIN_CELLS_PER_REGION].copy()

    if regions.empty:
        raise ValueError("No regions remain after minimum cell filter.")

    regions["region_area_m2"] = regions.geometry.area
    regions["region_area_km2"] = regions["region_area_m2"] / 1_000_000

    regions["support_score"] = np.log1p(regions["cell_count"])

    regions["discovery_region_score_raw"] = (
        0.35 * regions["max_urem_score_v07_no_coast"]
        + 0.25 * regions["mean_urem_score_v07_no_coast"]
        + 0.20 * regions["mean_terrain_only_exceptionality_v07"]
        + 0.15 * regions["mean_positive_residual_v06"]
        + 0.05 * regions["support_score"]
    )

    mn = regions["discovery_region_score_raw"].min()
    mx = regions["discovery_region_score_raw"].max()

    if mx == mn:
        regions["discovery_region_score"] = 1.0
    else:
        regions["discovery_region_score"] = (
            (regions["discovery_region_score_raw"] - mn) / (mx - mn)
        )

    regions = regions.sort_values(
        "discovery_region_score",
        ascending=False,
    ).reset_index(drop=True)

    regions["discovery_region_rank"] = range(1, len(regions) + 1)

    # Attach region rank back to points
    region_lookup = regions[["region_id", "discovery_region_rank"]].copy()
    points = joined.merge(region_lookup, on="region_id", how="inner")
    points = points.sort_values(
        ["discovery_region_rank", "urem_score_v07_no_coast"],
        ascending=[True, False],
    ).copy()

    # Centroid output
    centroids = regions.copy()
    centroids["geometry"] = centroids.geometry.centroid
    centroids_wgs = centroids.to_crs("EPSG:4326")

    centroids_out = pd.DataFrame(
        {
            "discovery_region_rank": centroids_wgs["discovery_region_rank"],
            "region_id": centroids_wgs["region_id"],
            "longitude": centroids_wgs.geometry.x,
            "latitude": centroids_wgs.geometry.y,
            "cell_count": centroids_wgs["cell_count"],
            "region_area_km2": centroids_wgs["region_area_km2"],
            "discovery_region_score": centroids_wgs["discovery_region_score"],
            "mean_urem_score_v07_no_coast": centroids_wgs[
                "mean_urem_score_v07_no_coast"
            ],
            "max_urem_score_v07_no_coast": centroids_wgs[
                "max_urem_score_v07_no_coast"
            ],
            "mean_terrain_only_exceptionality_v07": centroids_wgs[
                "mean_terrain_only_exceptionality_v07"
            ],
            "mean_positive_residual_v06": centroids_wgs[
                "mean_positive_residual_v06"
            ],
            "mean_observed_recognition_v04": centroids_wgs[
                "mean_observed_recognition_v04"
            ],
            "mean_expected_recognition_v06_raw": centroids_wgs[
                "mean_expected_recognition_v06_raw"
            ],
            "mean_distance_to_coast_m": centroids_wgs["mean_distance_to_coast_m"],
            "best_cell_id": centroids_wgs["best_cell_id"],
        }
    )

    log(f"Writing regions GPKG: {OUT_REGIONS_GPKG}")
    regions.to_file(
        OUT_REGIONS_GPKG,
        layer="v07_no_coast_discovery_regions",
        driver="GPKG",
    )

    log(f"Writing regions CSV: {OUT_REGIONS_CSV}")
    regions.drop(columns="geometry").to_csv(OUT_REGIONS_CSV, index=False)

    log(f"Writing centroids CSV: {OUT_CENTROIDS_CSV}")
    centroids_out.to_csv(OUT_CENTROIDS_CSV, index=False)

    log(f"Writing region points GPKG: {OUT_POINTS_GPKG}")
    points.to_file(
        OUT_POINTS_GPKG,
        layer="v07_no_coast_discovery_region_points",
        driver="GPKG",
    )

    log("Done")

    print("\nDiscovery region summary:")
    print(
        regions[
            [
                "discovery_region_rank",
                "cell_count",
                "region_area_km2",
                "discovery_region_score",
                "mean_urem_score_v07_no_coast",
                "mean_terrain_only_exceptionality_v07",
                "mean_positive_residual_v06",
                "mean_distance_to_coast_m",
                "best_cell_id",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()