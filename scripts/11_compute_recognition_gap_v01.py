#!/usr/bin/env python3

from pathlib import Path
import logging
import sys

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data/processed/coastal_grid_golf_recognition.gpkg"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/coastal_grid_recognition_gap_v01.gpkg"

TARGET_CRS = "EPSG:3310"


def setup_logger():
    logger = logging.getLogger("11_compute_recognition_gap_v01")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def main():
    logger = setup_logger()
    logger.info("Starting Script 11: Compute Recognition Gap v0.1")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    grid = gpd.read_file(INPUT_PATH).to_crs(TARGET_CRS)

    if grid.empty:
        raise ValueError("Input grid is empty.")

    required_cols = [
        "coastal_proximity_score",
        "golf_recognition_score",
    ]

    missing = [c for c in required_cols if c not in grid.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    grid["physical_potential_v0"] = grid["coastal_proximity_score"].fillna(0)
    grid["observed_recognition_v0"] = grid["golf_recognition_score"].fillna(0)

    grid["recognition_gap_v01"] = (
        grid["physical_potential_v0"] - grid["observed_recognition_v0"]
    )

    grid["recognition_gap_v01"] = grid["recognition_gap_v01"].clip(lower=0)

    grid["has_golf"] = grid["golf_area_m2"].fillna(0) > 0

    logger.info("QA summary")
    logger.info(f"Cells processed: {len(grid):,}")
    logger.info(f"Max physical potential: {grid['physical_potential_v0'].max():.4f}")
    logger.info(f"Max observed recognition: {grid['observed_recognition_v0'].max():.4f}")
    logger.info(f"Max recognition gap: {grid['recognition_gap_v01'].max():.4f}")
    logger.info(f"Mean recognition gap: {grid['recognition_gap_v01'].mean():.4f}")
    logger.info(f"Cells with nonzero gap: {(grid['recognition_gap_v01'] > 0).sum():,}")

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    grid.to_file(
        OUTPUT_PATH,
        layer="coastal_grid_recognition_gap_v01",
        driver="GPKG",
    )

    logger.info(f"Saved output: {OUTPUT_PATH}")
    logger.info("Script 11 completed successfully.")


if __name__ == "__main__":
    main()