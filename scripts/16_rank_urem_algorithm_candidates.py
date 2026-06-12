#!/usr/bin/env python3
"""
16_rank_urem_algorithm_candidates.py

Ranks the first formal residual-based UREM candidate cells.

Input:
- data/processed/coastal_grid_urem_algorithm_v01.gpkg

Outputs:
- data/processed/ranked_urem_algorithm_candidates_v01.gpkg
- outputs/validation/top_urem_algorithm_candidates_v01.csv
"""

from pathlib import Path
import logging
import sys

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data/processed/coastal_grid_urem_algorithm_v01.gpkg"
OUTPUT_GPKG_PATH = PROJECT_ROOT / "data/processed/ranked_urem_algorithm_candidates_v01.gpkg"
OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs/validation/top_urem_algorithm_candidates_v01.csv"

TARGET_CRS = "EPSG:3310"
TOP_N = 500


def setup_logger():
    logger = logging.getLogger("16_rank_urem_algorithm_candidates")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def main():
    logger = setup_logger()
    logger.info("Starting Script 16: Rank residual-based UREM candidates")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    grid = gpd.read_file(INPUT_PATH).to_crs(TARGET_CRS)

    if grid.empty:
        raise ValueError("Input grid is empty.")

    if "cell_id" not in grid.columns:
        grid["cell_id"] = range(1, len(grid) + 1)

    required_cols = [
        "urem_score_v01",
        "urem_score_v01_norm",
        "positive_under_recognition_residual_v01",
        "recognition_residual_v01",
        "expected_recognition_v01",
        "observed_recognition_v0",
        "physical_potential_v01",
        "comparable_confidence_v01",
        "has_golf",
    ]

    missing = [c for c in required_cols if c not in grid.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    candidates = grid.copy()

    # Still exclude existing golf cells for this golf-specific MVF test.
    candidates = candidates[candidates["has_golf"] == False].copy()

    # Keep only actual under-recognition candidates.
    candidates = candidates[
        candidates["positive_under_recognition_residual_v01"] > 0
    ].copy()

    candidates = candidates.sort_values(
        by=[
            "urem_score_v01_norm",
            "urem_score_v01",
            "positive_under_recognition_residual_v01",
            "physical_potential_v01",
            "comparable_confidence_v01",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    candidates["urem_algorithm_rank_v01"] = candidates.index + 1

    top_candidates = candidates.head(TOP_N).copy()

    centroids = top_candidates.geometry.centroid.to_crs("EPSG:4326")
    top_candidates["centroid_lon"] = centroids.x
    top_candidates["centroid_lat"] = centroids.y

    logger.info("QA summary")
    logger.info(f"Input cells: {len(grid):,}")
    logger.info(f"Candidates after excluding golf: {len(candidates):,}")
    logger.info(f"Top candidates saved: {len(top_candidates):,}")

    if not top_candidates.empty:
        logger.info(
            f"Best normalized UREM score: "
            f"{top_candidates['urem_score_v01_norm'].max():.4f}"
        )
        logger.info(
            f"Best raw UREM score: "
            f"{top_candidates['urem_score_v01'].max():.6f}"
        )
        logger.info(
            f"Best under-recognition residual: "
            f"{top_candidates['positive_under_recognition_residual_v01'].max():.4f}"
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
        layer="ranked_urem_algorithm_candidates_v01",
        driver="GPKG",
    )

    csv_cols = [
        "urem_algorithm_rank_v01",
        "cell_id",
        "centroid_lat",
        "centroid_lon",
        "urem_score_v01_norm",
        "urem_score_v01",
        "positive_under_recognition_residual_v01",
        "recognition_residual_v01",
        "expected_recognition_v01",
        "observed_recognition_v0",
        "physical_potential_v01",
        "comparable_confidence_v01",
        "mean_neighbor_distance",
        "coastal_proximity_score",
        "elevation_score",
        "relief_score",
        "slope_score",
        "golf_recognition_score",
        "golf_area_km2",
        "golf_course_count",
        "has_golf",
    ]

    existing_csv_cols = [c for c in csv_cols if c in top_candidates.columns]

    top_candidates[existing_csv_cols].to_csv(OUTPUT_CSV_PATH, index=False)

    logger.info(f"Saved GeoPackage: {OUTPUT_GPKG_PATH}")
    logger.info(f"Saved CSV: {OUTPUT_CSV_PATH}")
    logger.info("Script 16 completed successfully.")


if __name__ == "__main__":
    main()