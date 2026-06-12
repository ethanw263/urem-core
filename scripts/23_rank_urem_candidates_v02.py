#!/usr/bin/env python3
"""
23_rank_urem_candidates_v02.py

Ranks residual-based UREM v02 candidate cells.

Input:
- data/processed/coastal_grid_urem_algorithm_v02.gpkg

Outputs:
- data/processed/ranked_urem_candidates_v02.gpkg
- outputs/validation/top_urem_candidates_v02.csv

Ranks by:
- urem_score_v02_norm
- urem_score_v02
- positive_under_recognition_residual_v02
- physical_potential_v01
"""

from pathlib import Path
import logging
import sys

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data/processed/coastal_grid_urem_algorithm_v02.gpkg"
OUTPUT_GPKG_PATH = PROJECT_ROOT / "data/processed/ranked_urem_candidates_v02.gpkg"
OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs/validation/top_urem_candidates_v02.csv"

TARGET_CRS = "EPSG:3310"
TOP_N = 500


def setup_logger():
    logger = logging.getLogger("23_rank_urem_candidates_v02")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def main():
    logger = setup_logger()
    logger.info("Starting Script 23: Rank UREM candidates v02")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    grid = gpd.read_file(INPUT_PATH).to_crs(TARGET_CRS)

    if grid.empty:
        raise ValueError("Input grid is empty.")

    if "cell_id" not in grid.columns:
        grid["cell_id"] = range(1, len(grid) + 1)

    required_cols = [
        "urem_score_v02",
        "urem_score_v02_norm",
        "positive_under_recognition_residual_v02",
        "recognition_residual_v02",
        "expected_recognition_v02",
        "observed_recognition_v02",
        "physical_potential_v01",
        "comparable_confidence_v02",
    ]

    missing = [c for c in required_cols if c not in grid.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    candidates = grid.copy()

    # Keep only true under-recognition candidates.
    candidates = candidates[
        candidates["positive_under_recognition_residual_v02"] > 0
    ].copy()

    # Optional: exclude existing golf cells if available.
    # For v02, we do NOT exclude all recognized places generally, because
    # low-but-nonzero recognition can still be under-recognized.
    if "has_golf" in candidates.columns:
        candidates = candidates[candidates["has_golf"] == False].copy()

    candidates = candidates.sort_values(
        by=[
            "urem_score_v02_norm",
            "urem_score_v02",
            "positive_under_recognition_residual_v02",
            "physical_potential_v01",
            "comparable_confidence_v02",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    candidates["urem_rank_v02"] = candidates.index + 1

    top_candidates = candidates.head(TOP_N).copy()

    centroids = top_candidates.geometry.centroid.to_crs("EPSG:4326")
    top_candidates["centroid_lon"] = centroids.x
    top_candidates["centroid_lat"] = centroids.y

    logger.info("QA summary")
    logger.info(f"Input cells: {len(grid):,}")
    logger.info(f"Candidate cells after filters: {len(candidates):,}")
    logger.info(f"Top candidates saved: {len(top_candidates):,}")

    if not top_candidates.empty:
        logger.info(
            f"Best normalized UREM v02 score: "
            f"{top_candidates['urem_score_v02_norm'].max():.4f}"
        )
        logger.info(
            f"Best raw UREM v02 score: "
            f"{top_candidates['urem_score_v02'].max():.6f}"
        )
        logger.info(
            f"Best under-recognition residual v02: "
            f"{top_candidates['positive_under_recognition_residual_v02'].max():.4f}"
        )
        logger.info(
            f"Best physical potential: "
            f"{top_candidates['physical_potential_v01'].max():.4f}"
        )

    OUTPUT_GPKG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_GPKG_PATH.exists():
        OUTPUT_GPKG_PATH.unlink()

    top_candidates.to_file(
        OUTPUT_GPKG_PATH,
        layer="ranked_urem_candidates_v02",
        driver="GPKG",
    )

    csv_cols = [
        "urem_rank_v02",
        "cell_id",
        "centroid_lat",
        "centroid_lon",
        "urem_score_v02_norm",
        "urem_score_v02",
        "positive_under_recognition_residual_v02",
        "recognition_residual_v02",
        "expected_recognition_v02",
        "observed_recognition_v02",
        "physical_potential_v01",
        "comparable_confidence_v02",
        "mean_neighbor_distance_v02",
        "coastal_proximity_score",
        "elevation_score",
        "relief_score",
        "slope_score",
        "viewpoint_count",
        "peak_count",
        "attraction_count",
        "campground_count",
        "picnic_site_count",
        "information_count",
        "viewpoint_score",
        "peak_score",
        "attraction_score",
        "campground_score",
        "picnic_site_score",
        "information_score",
        "golf_recognition_score",
        "golf_area_km2",
        "golf_course_count",
        "has_golf",
    ]

    existing_csv_cols = [c for c in csv_cols if c in top_candidates.columns]
    top_candidates[existing_csv_cols].to_csv(OUTPUT_CSV_PATH, index=False)

    logger.info(f"Saved GeoPackage: {OUTPUT_GPKG_PATH}")
    logger.info(f"Saved CSV: {OUTPUT_CSV_PATH}")
    logger.info("Script 23 completed successfully.")


if __name__ == "__main__":
    main()