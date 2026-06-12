#!/usr/bin/env python3
"""
08_compute_physical_potential_v01.py

Computes the first terrain-based UREM Physical Potential surface.

Input:
- data/processed/coastal_grid_slope.gpkg

Output:
- data/processed/coastal_grid_physical_potential_v01.gpkg

Uses:
- coastal_proximity_score
- elevation_m
- relief_score
- slope_score

Formula:
physical_potential_v01 =
    0.40 * coastal_proximity_score
  + 0.35 * relief_score
  + 0.15 * elevation_score
  + 0.10 * slope_score
"""

from pathlib import Path
import logging
import sys

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data/processed/coastal_grid_slope.gpkg"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/coastal_grid_physical_potential_v01.gpkg"

TARGET_CRS = "EPSG:3310"

ELEVATION_CAP_M = 1000.0

WEIGHTS = {
    "coastal_proximity_score": 0.40,
    "relief_score": 0.35,
    "elevation_score": 0.15,
    "slope_score": 0.10,
}


def setup_logger():
    logger = logging.getLogger("08_compute_physical_potential_v01")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def safe_fill_score(series):
    return series.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)


def main():
    logger = setup_logger()
    logger.info("Starting Script 08: Compute Physical Potential v0.1")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    logger.info(f"Reading grid: {INPUT_PATH}")

    grid = gpd.read_file(INPUT_PATH).to_crs(TARGET_CRS)

    if grid.empty:
        raise ValueError("Input grid is empty.")

    required_cols = [
        "coastal_proximity_score",
        "elevation_m",
        "relief_score",
        "slope_score",
    ]

    missing = [c for c in required_cols if c not in grid.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Cells loaded: {len(grid):,}")
    logger.info(f"Input CRS: {grid.crs}")

    # Normalize / clean input scores.
    grid["coastal_component_score"] = safe_fill_score(grid["coastal_proximity_score"])
    grid["relief_component_score"] = safe_fill_score(grid["relief_score"])
    grid["slope_component_score"] = safe_fill_score(grid["slope_score"])

    # Elevation score is capped so very high terrain does not dominate.
    elevation_clean = (
        grid["elevation_m"]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(lower=0, upper=ELEVATION_CAP_M)
    )

    grid["elevation_score"] = elevation_clean / ELEVATION_CAP_M
    grid["elevation_score"] = grid["elevation_score"].clip(0, 1)

    grid["physical_potential_v01"] = (
        WEIGHTS["coastal_proximity_score"] * grid["coastal_component_score"]
        + WEIGHTS["relief_score"] * grid["relief_component_score"]
        + WEIGHTS["elevation_score"] * grid["elevation_score"]
        + WEIGHTS["slope_score"] * grid["slope_component_score"]
    )

    grid["physical_potential_v01"] = grid["physical_potential_v01"].clip(0, 1)

    grid["physical_potential_version"] = "v01"
    grid["physical_potential_formula"] = (
        "0.40 coastal + 0.35 relief + 0.15 capped elevation + 0.10 slope"
    )
    grid["elevation_cap_m"] = ELEVATION_CAP_M

    logger.info("QA summary")
    logger.info(f"Cells processed: {len(grid):,}")
    logger.info(f"Output CRS: {grid.crs}")

    logger.info(
        f"Coastal component min/mean/max: "
        f"{grid['coastal_component_score'].min():.4f} / "
        f"{grid['coastal_component_score'].mean():.4f} / "
        f"{grid['coastal_component_score'].max():.4f}"
    )

    logger.info(
        f"Relief component min/mean/max: "
        f"{grid['relief_component_score'].min():.4f} / "
        f"{grid['relief_component_score'].mean():.4f} / "
        f"{grid['relief_component_score'].max():.4f}"
    )

    logger.info(
        f"Elevation score min/mean/max: "
        f"{grid['elevation_score'].min():.4f} / "
        f"{grid['elevation_score'].mean():.4f} / "
        f"{grid['elevation_score'].max():.4f}"
    )

    logger.info(
        f"Slope component min/mean/max: "
        f"{grid['slope_component_score'].min():.4f} / "
        f"{grid['slope_component_score'].mean():.4f} / "
        f"{grid['slope_component_score'].max():.4f}"
    )

    logger.info(
        f"Physical potential v01 min/mean/max: "
        f"{grid['physical_potential_v01'].min():.4f} / "
        f"{grid['physical_potential_v01'].mean():.4f} / "
        f"{grid['physical_potential_v01'].max():.4f}"
    )

    top_1pct_threshold = grid["physical_potential_v01"].quantile(0.99)

    logger.info(f"Top 1% threshold: {top_1pct_threshold:.4f}")
    logger.info(
        f"Cells in top 1%: "
        f"{(grid['physical_potential_v01'] >= top_1pct_threshold).sum():,}"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    grid.to_file(
        OUTPUT_PATH,
        layer="coastal_grid_physical_potential_v01",
        driver="GPKG",
    )

    logger.info(f"Saved output: {OUTPUT_PATH}")
    logger.info("Script 08 completed successfully.")


if __name__ == "__main__":
    main()