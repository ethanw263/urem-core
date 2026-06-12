#!/usr/bin/env python3
"""
15_compute_urem_score_v01.py

Computes the first formal residual-based UREM score.

Input:
- data/processed/coastal_grid_expected_recognition_v01.gpkg

Output:
- data/processed/coastal_grid_urem_algorithm_v01.gpkg

Core idea:
UREM is not simply physical potential minus recognition.

Instead:
1. Estimate expected recognition from physically comparable places.
2. Compare expected recognition to observed recognition.
3. Keep only positive under-recognition residuals.
4. Weight by physical potential and comparable-place confidence.

Formula:
recognition_residual_v01 =
    expected_recognition_v01 - observed_recognition_v0

positive_under_recognition_residual_v01 =
    max(recognition_residual_v01, 0)

urem_score_v01 =
    positive_under_recognition_residual_v01
    * physical_potential_v01
    * comparable_confidence_v01
"""

from pathlib import Path
import logging
import sys

import geopandas as gpd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data/processed/coastal_grid_expected_recognition_v01.gpkg"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/coastal_grid_urem_algorithm_v01.gpkg"

TARGET_CRS = "EPSG:3310"


def setup_logger():
    logger = logging.getLogger("15_compute_urem_score_v01")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def clean_numeric(series, fill_value=0.0):
    return (
        series.astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(fill_value)
    )


def main():
    logger = setup_logger()
    logger.info("Starting Script 15: Compute UREM Score v0.1")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    logger.info(f"Reading input: {INPUT_PATH}")

    grid = gpd.read_file(INPUT_PATH).to_crs(TARGET_CRS)

    if grid.empty:
        raise ValueError("Input grid is empty.")

    required_cols = [
        "expected_recognition_v01",
        "observed_recognition_v0",
        "physical_potential_v01",
        "comparable_confidence_v01",
    ]

    missing = [c for c in required_cols if c not in grid.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Cells loaded: {len(grid):,}")
    logger.info(f"Input CRS: {grid.crs}")

    expected = clean_numeric(grid["expected_recognition_v01"])
    observed = clean_numeric(grid["observed_recognition_v0"])
    physical = clean_numeric(grid["physical_potential_v01"]).clip(0, 1)
    confidence = clean_numeric(grid["comparable_confidence_v01"]).clip(0, 1)

    grid["recognition_residual_v01"] = expected - observed

    grid["positive_under_recognition_residual_v01"] = (
        grid["recognition_residual_v01"].clip(lower=0)
    )

    grid["over_recognition_residual_v01"] = (
        (-grid["recognition_residual_v01"]).clip(lower=0)
    )

    grid["urem_score_v01"] = (
        grid["positive_under_recognition_residual_v01"]
        * physical
        * confidence
    )

    # Normalized score for easier QGIS display/ranking.
    max_score = grid["urem_score_v01"].max()

    if max_score > 0:
        grid["urem_score_v01_norm"] = grid["urem_score_v01"] / max_score
    else:
        grid["urem_score_v01_norm"] = 0.0

    grid["urem_score_method"] = (
        "positive_expected_minus_observed_residual_times_physical_potential_times_confidence"
    )

    grid["urem_algorithm_version"] = "v01"

    logger.info("QA summary")
    logger.info(f"Cells processed: {len(grid):,}")
    logger.info(f"Output CRS: {grid.crs}")

    logger.info(
        f"Expected recognition min/mean/max: "
        f"{expected.min():.4f} / {expected.mean():.4f} / {expected.max():.4f}"
    )
    logger.info(
        f"Observed recognition min/mean/max: "
        f"{observed.min():.4f} / {observed.mean():.4f} / {observed.max():.4f}"
    )
    logger.info(
        f"Recognition residual min/mean/max: "
        f"{grid['recognition_residual_v01'].min():.4f} / "
        f"{grid['recognition_residual_v01'].mean():.4f} / "
        f"{grid['recognition_residual_v01'].max():.4f}"
    )
    logger.info(
        f"Positive under-recognition residual min/mean/max: "
        f"{grid['positive_under_recognition_residual_v01'].min():.4f} / "
        f"{grid['positive_under_recognition_residual_v01'].mean():.4f} / "
        f"{grid['positive_under_recognition_residual_v01'].max():.4f}"
    )
    logger.info(
        f"UREM score min/mean/max: "
        f"{grid['urem_score_v01'].min():.6f} / "
        f"{grid['urem_score_v01'].mean():.6f} / "
        f"{grid['urem_score_v01'].max():.6f}"
    )
    logger.info(
        f"Normalized UREM score min/mean/max: "
        f"{grid['urem_score_v01_norm'].min():.4f} / "
        f"{grid['urem_score_v01_norm'].mean():.4f} / "
        f"{grid['urem_score_v01_norm'].max():.4f}"
    )
    logger.info(
        f"Cells with positive under-recognition residual: "
        f"{(grid['positive_under_recognition_residual_v01'] > 0).sum():,}"
    )
    logger.info(
        f"Cells with nonzero UREM score: "
        f"{(grid['urem_score_v01'] > 0).sum():,}"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    grid.to_file(
        OUTPUT_PATH,
        layer="coastal_grid_urem_algorithm_v01",
        driver="GPKG",
    )

    logger.info(f"Saved output: {OUTPUT_PATH}")
    logger.info("Script 15 completed successfully.")


if __name__ == "__main__":
    main()