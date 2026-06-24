#!/usr/bin/env python3
"""
130_oregon_candidate_geographic_interpretation_v01.py

Create Oregon candidate geographic interpretation package.

Purpose:
Convert top Oregon UREM candidate cells into an interpretable review layer
with WGS84 latitude/longitude for manual geographic inspection.

Inputs:
- data/processed/oregon_ranked_urem_candidates_v05.gpkg

Outputs:
- data/processed/oregon_top100_candidate_review_v01.gpkg
- data/processed/oregon_top100_candidate_review_v01.csv
- data/processed/oregon_top500_candidate_review_v01.gpkg
- data/processed/oregon_top500_candidate_review_v01.csv
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd


SCRIPT_NAME = "130_oregon_candidate_geographic_interpretation_v01"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = PROCESSED_DIR / "oregon_ranked_urem_candidates_v05.gpkg"

OUT_TOP100_GPKG = PROCESSED_DIR / "oregon_top100_candidate_review_v01.gpkg"
OUT_TOP100_CSV = PROCESSED_DIR / "oregon_top100_candidate_review_v01.csv"

OUT_TOP500_GPKG = PROCESSED_DIR / "oregon_top500_candidate_review_v01.gpkg"
OUT_TOP500_CSV = PROCESSED_DIR / "oregon_top500_candidate_review_v01.csv"

OUT_CLUSTER_HINTS_CSV = PROCESSED_DIR / "oregon_candidate_cluster_hints_v01.csv"


def log(msg: str) -> None:
    print(f"[{SCRIPT_NAME}] {msg}")


def add_wgs84_coordinates(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    points = gdf.copy()
    points["geometry"] = points.geometry.centroid

    wgs = points.to_crs("EPSG:4326")

    gdf["longitude"] = wgs.geometry.x
    gdf["latitude"] = wgs.geometry.y

    return gdf


def assign_manual_review_group(row) -> str:
    """
    Rough first-pass geography labels based only on latitude.
    These are NOT final place names.
    They are review aids for manual interpretation.
    """
    lat = row.get("latitude")

    if pd.isna(lat):
        return "unknown"

    if lat >= 45.8:
        return "north_oregon_coast"
    if lat >= 44.7:
        return "central_north_oregon_coast"
    if lat >= 43.6:
        return "central_south_oregon_coast"
    return "south_oregon_coast"


def build_review_table(gdf: gpd.GeoDataFrame, n: int) -> gpd.GeoDataFrame:
    review = gdf.sort_values("candidate_rank_v05").head(n).copy()

    review = add_wgs84_coordinates(review)

    review["manual_review_group_v01"] = review.apply(
        assign_manual_review_group,
        axis=1,
    )

    keep_cols = [
        "candidate_rank_v05",
        "urem_rank_v05",
        "urem_tier_v05",
        "urem_score_v05_raw",
        "urem_score_v05",
        "physical_exceptionality_v03",
        "observed_recognition_v04",
        "expected_recognition_v05",
        "recognition_residual_v05",
        "positive_under_recognition_residual_v05",
        "terrain_drama_v03",
        "scenic_coast_v03",
        "relief_norm_v03",
        "slope_norm_v03",
        "coast_proximity_v03",
        "coast_complexity_v03",
        "recognition_total_count_3km_v04",
        "recognition_category_coverage_v04",
        "recognition_cell_confidence_v04",
        "expected_recognition_confidence_v05",
        "land_area_share",
        "water_area_share",
        "is_valid_land_candidate",
        "longitude",
        "latitude",
        "manual_review_group_v01",
        "geometry",
    ]

    keep_cols = [c for c in keep_cols if c in review.columns]

    return review[keep_cols].copy()


def make_cluster_hints(top500: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Produces rough summary counts by broad latitude group.
    This is not spatial clustering yet.
    It helps identify where manual review should focus.
    """
    rows = []

    for group, sub in top500.groupby("manual_review_group_v01"):
        rows.append(
            {
                "manual_review_group_v01": group,
                "candidate_count": len(sub),
                "best_rank": sub["candidate_rank_v05"].min(),
                "mean_urem_score_v05_raw": sub["urem_score_v05_raw"].mean(),
                "mean_exceptionality": sub["physical_exceptionality_v03"].mean(),
                "mean_observed_recognition": sub["observed_recognition_v04"].mean(),
                "mean_under_recognition_residual": sub[
                    "positive_under_recognition_residual_v05"
                ].mean(),
                "mean_latitude": sub["latitude"].mean(),
                "mean_longitude": sub["longitude"].mean(),
            }
        )

    return pd.DataFrame(rows).sort_values("best_rank")


def write_outputs(gdf: gpd.GeoDataFrame, gpkg_path: Path, csv_path: Path, layer: str):
    if gpkg_path.exists():
        gpkg_path.unlink()

    gdf.to_file(
        gpkg_path,
        layer=layer,
        driver="GPKG",
    )

    gdf.drop(columns="geometry").to_csv(csv_path, index=False)


def main():
    log("Starting Oregon candidate geographic interpretation")

    if not INPUT_GPKG.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_GPKG}")

    gdf = gpd.read_file(INPUT_GPKG)

    log(f"Loaded ranked candidates: {len(gdf):,}")
    log(f"CRS: {gdf.crs}")

    if "candidate_rank_v05" not in gdf.columns:
        raise ValueError("Missing required column: candidate_rank_v05")

    top100 = build_review_table(gdf, 100)
    top500 = build_review_table(gdf, 500)

    log("Writing top 100 review outputs")
    write_outputs(
        top100,
        OUT_TOP100_GPKG,
        OUT_TOP100_CSV,
        "oregon_top100_candidate_review_v01",
    )

    log("Writing top 500 review outputs")
    write_outputs(
        top500,
        OUT_TOP500_GPKG,
        OUT_TOP500_CSV,
        "oregon_top500_candidate_review_v01",
    )

    cluster_hints = make_cluster_hints(top500)
    cluster_hints.to_csv(OUT_CLUSTER_HINTS_CSV, index=False)

    log("Cluster hint summary:")
    print(cluster_hints)

    log(f"Wrote: {OUT_TOP100_GPKG}")
    log(f"Wrote: {OUT_TOP100_CSV}")
    log(f"Wrote: {OUT_TOP500_GPKG}")
    log(f"Wrote: {OUT_TOP500_CSV}")
    log(f"Wrote: {OUT_CLUSTER_HINTS_CSV}")
    log("Done")


if __name__ == "__main__":
    main()
    