#!/usr/bin/env python3

from pathlib import Path
import logging
import sys

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data/processed/coastal_grid_recognition_gap_v01.gpkg"
OUTPUT_GPKG_PATH = PROJECT_ROOT / "data/processed/ranked_urem_candidate_cells_v01.gpkg"
OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs/validation/top_urem_candidate_cells_v01.csv"

TARGET_CRS = "EPSG:3310"
TOP_N = 500


def setup_logger():
    logger = logging.getLogger("12_rank_candidate_cells")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def main():
    logger = setup_logger()
    logger.info("Starting Script 12: Rank UREM candidate cells")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    grid = gpd.read_file(INPUT_PATH).to_crs(TARGET_CRS)

    if grid.empty:
        raise ValueError("Input grid is empty.")

    required_cols = [
        "recognition_gap_v01",
        "physical_potential_v0",
        "observed_recognition_v0",
        "coastal_proximity_score",
        "golf_recognition_score",
        "golf_area_km2",
        "has_golf",
    ]

    missing = [c for c in required_cols if c not in grid.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    candidates = grid.copy()

    # For this first pass, exclude cells already containing golf.
    candidates = candidates[candidates["has_golf"] == False].copy()

    candidates = candidates.sort_values(
        by=[
            "recognition_gap_v01",
            "physical_potential_v0",
            "coastal_proximity_score",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    candidates["urem_rank_v01"] = candidates.index + 1

    top_candidates = candidates.head(TOP_N).copy()

    centroids = top_candidates.geometry.centroid.to_crs("EPSG:4326")
    top_candidates["centroid_lon"] = centroids.x
    top_candidates["centroid_lat"] = centroids.y

    logger.info("QA summary")
    logger.info(f"Input cells: {len(grid):,}")
    logger.info(f"Candidate cells after excluding golf: {len(candidates):,}")
    logger.info(f"Top candidates saved: {len(top_candidates):,}")
    logger.info(f"Best recognition gap: {top_candidates['recognition_gap_v01'].max():.4f}")

    OUTPUT_GPKG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_GPKG_PATH.exists():
        OUTPUT_GPKG_PATH.unlink()

    top_candidates.to_file(
        OUTPUT_GPKG_PATH,
        layer="ranked_urem_candidate_cells_v01",
        driver="GPKG",
    )

    csv_cols = [
        "urem_rank_v01",
        "centroid_lat",
        "centroid_lon",
        "recognition_gap_v01",
        "physical_potential_v0",
        "observed_recognition_v0",
        "coastal_proximity_score",
        "golf_recognition_score",
        "golf_area_km2",
        "has_golf",
    ]

    existing_csv_cols = [c for c in csv_cols if c in top_candidates.columns]

    top_candidates[existing_csv_cols].to_csv(OUTPUT_CSV_PATH, index=False)

    logger.info(f"Saved GeoPackage: {OUTPUT_GPKG_PATH}")
    logger.info(f"Saved CSV: {OUTPUT_CSV_PATH}")
    logger.info("Script 12 completed successfully.")


if __name__ == "__main__":
    main()