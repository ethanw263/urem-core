#!/usr/bin/env python3
"""
55_generate_top_urem_cells_v05b_review.py

Generate top individual UREM v05b cells for manual review.

Purpose:
- Review individual high-scoring cells, not giant hotspot polygons.
- Determine whether v05b is finding obscure exceptional places inside broader famous regions.
- Create CSV, GeoPackage, and KML for Google Earth/QGIS review.

Inputs:
- data/processed/ranked_urem_candidates_v05b.gpkg
- data/processed/urem_hotspots_v05b.gpkg

Outputs:
- data/processed/top_urem_cells_v05b_review.csv
- data/processed/top_urem_cells_v05b_review.gpkg
- data/processed/top_urem_cells_v05b_review.kml
"""

from pathlib import Path
import warnings

import geopandas as gpd
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_NAME = "55_generate_top_urem_cells_v05b_review"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CANDIDATES_GPKG = PROCESSED_DIR / "ranked_urem_candidates_v05b.gpkg"
HOTSPOTS_GPKG = PROCESSED_DIR / "urem_hotspots_v05b.gpkg"

OUT_CSV = PROCESSED_DIR / "top_urem_cells_v05b_review.csv"
OUT_GPKG = PROCESSED_DIR / "top_urem_cells_v05b_review.gpkg"
OUT_KML = PROCESSED_DIR / "top_urem_cells_v05b_review.kml"

TOP_N = 100


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def main():
    log("Starting top UREM v05b cell review package")

    require_file(CANDIDATES_GPKG)

    log(f"Reading candidates: {CANDIDATES_GPKG}")
    candidates = gpd.read_file(CANDIDATES_GPKG)

    if candidates.empty:
        raise ValueError("Candidate file is empty.")

    if candidates.crs is None:
        raise ValueError("Candidate file has no CRS.")

    candidates = candidates.sort_values("urem_score_v05b", ascending=False).head(TOP_N).copy()
    candidates["top_cell_rank_v05b"] = range(1, len(candidates) + 1)

    # Attach hotspot membership if available
    if HOTSPOTS_GPKG.exists():
        log(f"Reading hotspots: {HOTSPOTS_GPKG}")
        hotspots = gpd.read_file(HOTSPOTS_GPKG).to_crs(candidates.crs)

        if "hotspot_rank_v05b" not in hotspots.columns:
            hotspots = hotspots.reset_index(drop=True)
            hotspots["hotspot_rank_v05b"] = range(1, len(hotspots) + 1)

        joined = gpd.sjoin(
            candidates,
            hotspots[["hotspot_id", "hotspot_rank_v05b", "geometry"]],
            how="left",
            predicate="intersects",
        )

        candidates["v05b_hotspot_id"] = joined["hotspot_id"].values
        candidates["v05b_hotspot_rank"] = joined["hotspot_rank_v05b"].values
    else:
        candidates["v05b_hotspot_id"] = pd.NA
        candidates["v05b_hotspot_rank"] = pd.NA

    points = candidates.copy()
    points["geometry"] = points.geometry.centroid
    points_wgs = points.to_crs("EPSG:4326")

    points_wgs["longitude"] = points_wgs.geometry.x
    points_wgs["latitude"] = points_wgs.geometry.y

    manual_cols = [
        "manual_place_name",
        "manual_region_quality_1_5",
        "manual_point_quality_1_5",
        "manual_scenic_quality_1_5",
        "manual_existing_recognition_1_5",
        "manual_under_recognized_exceptionality_1_5",
        "dominant_landscape_type",
        "is_success",
        "is_false_positive",
        "failure_mode",
        "reviewer_notes",
    ]

    for c in manual_cols:
        points_wgs[c] = ""

    preferred_cols = [
        "top_cell_rank_v05b",
        "cell_id",
        "longitude",
        "latitude",
        "v05b_hotspot_rank",
        "v05b_hotspot_id",
        "candidate_rank_v05b",
        "urem_score_v05b",
        "urem_score_v05b_raw",
        "urem_score_v04",
        "physical_exceptionality_v03",
        "physical_exceptionality_score_v02",
        "observed_recognition_v04",
        "expected_recognition_v04",
        "positive_under_recognition_residual_v04",
        "recognition_cell_confidence_v04",
        "land_area_share",
        "terrain_drama_v03",
        "scenic_coast_v03",
        "flat_coastal_edge_penalty_v03",
        "complex_flat_shoreline_penalty_v03",
        "v03_refinement_bonus_v05b",
        "v03_flat_edge_penalty_v05b",
    ]

    preferred_cols = [c for c in preferred_cols if c in points_wgs.columns]
    out_cols = preferred_cols + manual_cols + ["geometry"]

    review = points_wgs[out_cols].copy()

    log(f"Writing CSV: {OUT_CSV}")
    review.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log(f"Writing GeoPackage: {OUT_GPKG}")
    review.to_file(OUT_GPKG, layer="top_urem_cells_v05b_review", driver="GPKG")

    log(f"Writing KML: {OUT_KML}")
    kml = review.copy()
    kml["Name"] = (
        "UREM v05b Cell "
        + kml["top_cell_rank_v05b"].astype(str)
        + " | "
        + kml["cell_id"].astype(str)
    )
    kml["Description"] = (
        "Rank: " + kml["top_cell_rank_v05b"].astype(str)
        + " | UREM v05b: " + kml["urem_score_v05b"].round(4).astype(str)
        + " | Exceptionality v03: " + kml["physical_exceptionality_v03"].round(4).astype(str)
        + " | Observed recognition v04: " + kml["observed_recognition_v04"].round(4).astype(str)
    )

    try:
        kml[["Name", "Description", "geometry"]].to_file(OUT_KML, driver="KML")
    except Exception as exc:
        log(f"KML export failed: {exc}")

    log("Done")

    print("\nTop UREM v05b cell review package summary:")
    print(f"Rows: {len(review):,}")
    print("\nTop 25:")
    print(
        review[
            [
                "top_cell_rank_v05b",
                "cell_id",
                "longitude",
                "latitude",
                "v05b_hotspot_rank",
                "urem_score_v05b",
                "physical_exceptionality_v03",
                "observed_recognition_v04",
                "positive_under_recognition_residual_v04",
            ]
        ]
        .head(25)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()