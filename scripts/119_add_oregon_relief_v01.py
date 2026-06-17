#!/usr/bin/env python3
"""
119_add_oregon_relief_v01.py

Adds local relief metrics to the Oregon Coast grid.

Input:
- data/processed/oregon_coast_grid_elevation_v01.gpkg

Output:
- data/processed/oregon_coast_grid_relief_v01.gpkg
"""

from pathlib import Path
import logging
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm


SCRIPT_NAME = "119_add_oregon_relief_v01"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data/processed/oregon_coast_grid_elevation_v01.gpkg"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/oregon_coast_grid_relief_v01.gpkg"

TARGET_CRS = "EPSG:5070"
RELIEF_RADIUS_M = 3000


def setup_logger():
    logger = logging.getLogger(SCRIPT_NAME)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def minmax_score(series):
    s = series.copy()
    valid = s.notna()

    if valid.sum() == 0:
        return pd.Series(np.nan, index=s.index)

    smin = s[valid].min()
    smax = s[valid].max()

    if smax == smin:
        out = pd.Series(0.0, index=s.index)
        out[~valid] = np.nan
        return out

    out = (s - smin) / (smax - smin)
    return out.clip(lower=0, upper=1)


def main():
    logger = setup_logger()
    logger.info("Starting Oregon local relief computation")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    logger.info(f"Reading grid: {INPUT_PATH}")

    grid = gpd.read_file(INPUT_PATH).to_crs(TARGET_CRS)

    if grid.empty:
        raise ValueError("Input grid is empty.")

    if "elevation_m" not in grid.columns:
        raise ValueError("Missing required column: elevation_m")

    if "cell_id" not in grid.columns:
        grid["cell_id"] = range(1, len(grid) + 1)

    logger.info(f"Cells loaded: {len(grid):,}")
    logger.info(f"Input CRS: {grid.crs}")
    logger.info(f"Relief radius: {RELIEF_RADIUS_M:,} m")

    centroids = grid.geometry.centroid
    coords = np.column_stack([centroids.x.values, centroids.y.values])

    elevations = grid["elevation_m"].to_numpy(dtype=float)
    valid_elev = ~np.isnan(elevations)

    logger.info(f"Cells with valid elevation: {valid_elev.sum():,}")
    logger.info(f"Cells missing elevation: {(~valid_elev).sum():,}")

    logger.info("Building spatial KDTree")
    tree = cKDTree(coords)

    logger.info("Computing local relief")

    local_relief = np.full(len(grid), np.nan)
    neighborhood_count = np.zeros(len(grid), dtype=int)

    neighbor_lists = tree.query_ball_point(coords, r=RELIEF_RADIUS_M)

    for i, neighbors in tqdm(
        enumerate(neighbor_lists),
        total=len(neighbor_lists),
        desc="Computing relief",
    ):
        neighbor_idx = np.array(neighbors, dtype=int)

        neighbor_elev = elevations[neighbor_idx]
        neighbor_elev = neighbor_elev[~np.isnan(neighbor_elev)]

        neighborhood_count[i] = len(neighbor_elev)

        if len(neighbor_elev) == 0:
            continue

        local_relief[i] = float(np.max(neighbor_elev) - np.min(neighbor_elev))

    grid["relief_radius_m"] = RELIEF_RADIUS_M
    grid["relief_neighbor_count"] = neighborhood_count
    grid["local_relief_m"] = local_relief
    grid["relief_score"] = minmax_score(grid["local_relief_m"])

    valid_relief = grid["local_relief_m"].notna()

    logger.info("QA summary")
    logger.info(f"Cells processed: {len(grid):,}")
    logger.info(f"Cells with relief: {valid_relief.sum():,}")
    logger.info(f"Missing relief count: {(~valid_relief).sum():,}")
    logger.info(f"Output CRS: {grid.crs}")

    if valid_relief.any():
        logger.info(f"Min local relief: {grid.loc[valid_relief, 'local_relief_m'].min():,.2f} m")
        logger.info(f"Mean local relief: {grid.loc[valid_relief, 'local_relief_m'].mean():,.2f} m")
        logger.info(f"Max local relief: {grid.loc[valid_relief, 'local_relief_m'].max():,.2f} m")
        logger.info(f"Mean neighbor count: {grid['relief_neighbor_count'].mean():,.2f}")
        logger.info(f"Max relief score: {grid['relief_score'].max():.4f}")
    else:
        raise ValueError("No valid relief values were computed.")

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    logger.info(f"Writing output: {OUTPUT_PATH}")

    grid.to_file(
        OUTPUT_PATH,
        layer="oregon_coast_grid_relief_v01",
        driver="GPKG",
    )

    logger.info("Done")


if __name__ == "__main__":
    main()