#!/usr/bin/env python3

from pathlib import Path
import logging
import sys

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PHYSICAL_PATH = PROJECT_ROOT / "data/processed/coastal_grid_physical_potential_v01.gpkg"
RECOGNITION_PATH = PROJECT_ROOT / "data/processed/coastal_grid_golf_recognition.gpkg"
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

    if not PHYSICAL_PATH.exists():
        raise FileNotFoundError(f"Missing physical potential file: {PHYSICAL_PATH}")

    if not RECOGNITION_PATH.exists():
        raise FileNotFoundError(f"Missing recognition file: {RECOGNITION_PATH}")

    logger.info(f"Reading physical potential: {PHYSICAL_PATH}")
    physical = gpd.read_file(PHYSICAL_PATH).to_crs(TARGET_CRS)

    logger.info(f"Reading golf recognition: {RECOGNITION_PATH}")
    recognition = gpd.read_file(RECOGNITION_PATH).to_crs(TARGET_CRS)

    if physical.empty:
        raise ValueError("Physical potential grid is empty.")

    if recognition.empty:
        raise ValueError("Recognition grid is empty.")

    if "cell_id" not in physical.columns:
        physical["cell_id"] = range(1, len(physical) + 1)

    if "cell_id" not in recognition.columns:
        recognition["cell_id"] = range(1, len(recognition) + 1)

    required_physical_cols = [
        "cell_id",
        "physical_potential_v01",
    ]

    required_recognition_cols = [
        "cell_id",
        "golf_recognition_score",
        "golf_area_m2",
        "golf_area_km2",
    ]

    missing_physical = [c for c in required_physical_cols if c not in physical.columns]
    missing_recognition = [c for c in required_recognition_cols if c not in recognition.columns]

    if missing_physical:
        raise ValueError(f"Missing required physical columns: {missing_physical}")

    if missing_recognition:
        raise ValueError(f"Missing required recognition columns: {missing_recognition}")

    recognition_cols = [
        "cell_id",
        "golf_recognition_score",
        "golf_area_m2",
        "golf_area_km2",
        "golf_course_count",
        "golf_area_share",
    ]

    recognition_cols = [c for c in recognition_cols if c in recognition.columns]

    grid = physical.merge(
        recognition[recognition_cols],
        on="cell_id",
        how="left",
    )

    grid["observed_recognition_v0"] = grid["golf_recognition_score"].fillna(0)
    grid["physical_potential_v01"] = grid["physical_potential_v01"].fillna(0)

    grid["recognition_gap_v01"] = (
        grid["physical_potential_v01"] - grid["observed_recognition_v0"]
    ).clip(lower=0)

    grid["has_golf"] = grid["golf_area_m2"].fillna(0) > 0

    logger.info("QA summary")
    logger.info(f"Cells processed: {len(grid):,}")
    logger.info(f"Max physical potential v01: {grid['physical_potential_v01'].max():.4f}")
    logger.info(f"Mean physical potential v01: {grid['physical_potential_v01'].mean():.4f}")
    logger.info(f"Max observed recognition: {grid['observed_recognition_v0'].max():.4f}")
    logger.info(f"Mean observed recognition: {grid['observed_recognition_v0'].mean():.4f}")
    logger.info(f"Max recognition gap: {grid['recognition_gap_v01'].max():.4f}")
    logger.info(f"Mean recognition gap: {grid['recognition_gap_v01'].mean():.4f}")
    logger.info(f"Cells with nonzero gap: {(grid['recognition_gap_v01'] > 0).sum():,}")
    logger.info(f"Cells with golf: {grid['has_golf'].sum():,}")

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