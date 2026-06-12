#!/usr/bin/env python3
"""
21_compute_expected_recognition_v02.py

Computes expected recognition v02 using physically comparable places.

Input:
- data/processed/coastal_grid_recognition_v02.gpkg

Output:
- data/processed/coastal_grid_expected_recognition_v02.gpkg

Method:
- Uses normalized physical fingerprint columns.
- Finds k-nearest neighbors in fingerprint space.
- Computes distance-weighted expected recognition using observed_recognition_v02.
"""

from pathlib import Path
import logging
import sys

import geopandas as gpd
import numpy as np
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data/processed/coastal_grid_recognition_v02.gpkg"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/coastal_grid_expected_recognition_v02.gpkg"

TARGET_CRS = "EPSG:3310"

FINGERPRINT_COLS = [
    "fp_coastal_proximity",
    "fp_elevation",
    "fp_relief",
    "fp_slope",
    "fp_distance_to_coast_inverse",
    "fp_physical_potential",
]

OBSERVED_RECOGNITION_COL = "observed_recognition_v02"

K_NEIGHBORS = 50
EPSILON = 1e-9


def setup_logger():
    logger = logging.getLogger("21_compute_expected_recognition_v02")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def main():
    logger = setup_logger()
    logger.info("Starting Script 21: Compute expected recognition v02")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    logger.info(f"Reading input: {INPUT_PATH}")

    grid = gpd.read_file(INPUT_PATH).to_crs(TARGET_CRS)

    if grid.empty:
        raise ValueError("Input grid is empty.")

    if "cell_id" not in grid.columns:
        grid["cell_id"] = range(1, len(grid) + 1)

    missing_fp = [c for c in FINGERPRINT_COLS if c not in grid.columns]
    if missing_fp:
        raise ValueError(f"Missing fingerprint columns: {missing_fp}")

    if OBSERVED_RECOGNITION_COL not in grid.columns:
        raise ValueError(f"Missing observed recognition column: {OBSERVED_RECOGNITION_COL}")

    logger.info(f"Cells loaded: {len(grid):,}")
    logger.info(f"Input CRS: {grid.crs}")
    logger.info(f"Fingerprint columns: {FINGERPRINT_COLS}")
    logger.info(f"Observed recognition column: {OBSERVED_RECOGNITION_COL}")
    logger.info(f"K neighbors: {K_NEIGHBORS}")

    X = (
        grid[FINGERPRINT_COLS]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy()
    )

    observed = (
        grid[OBSERVED_RECOGNITION_COL]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy()
    )

    logger.info("Fitting nearest-neighbor model in fingerprint space")

    nn = NearestNeighbors(
        n_neighbors=K_NEIGHBORS + 1,
        algorithm="auto",
        metric="euclidean",
    )

    nn.fit(X)

    logger.info("Querying comparable places")

    distances, indices = nn.kneighbors(X)

    neighbor_distances = distances[:, 1:]
    neighbor_indices = indices[:, 1:]

    neighbor_recognition = observed[neighbor_indices]

    weights = 1.0 / (neighbor_distances + EPSILON)
    weight_sums = weights.sum(axis=1)

    expected_recognition = (
        (neighbor_recognition * weights).sum(axis=1) / weight_sums
    )

    mean_neighbor_distance = neighbor_distances.mean(axis=1)
    min_neighbor_distance = neighbor_distances.min(axis=1)
    max_neighbor_distance = neighbor_distances.max(axis=1)

    max_mean_distance = np.nanmax(mean_neighbor_distance)

    if max_mean_distance == 0:
        comparable_confidence = np.ones(len(grid))
    else:
        comparable_confidence = 1.0 - (mean_neighbor_distance / max_mean_distance)
        comparable_confidence = np.clip(comparable_confidence, 0, 1)

    grid["expected_recognition_v02"] = expected_recognition
    grid["expected_recognition_v02_method"] = "fingerprint_knn_inverse_distance"
    grid["expected_recognition_v02_neighbors"] = K_NEIGHBORS

    grid["mean_neighbor_distance_v02"] = mean_neighbor_distance
    grid["min_neighbor_distance_v02"] = min_neighbor_distance
    grid["max_neighbor_distance_v02"] = max_neighbor_distance
    grid["comparable_confidence_v02"] = comparable_confidence

    logger.info("QA summary")
    logger.info(f"Cells processed: {len(grid):,}")
    logger.info(f"Output CRS: {grid.crs}")

    logger.info(
        f"Observed recognition v02 min/mean/max: "
        f"{observed.min():.4f} / {observed.mean():.4f} / {observed.max():.4f}"
    )

    logger.info(
        f"Expected recognition v02 min/mean/max: "
        f"{grid['expected_recognition_v02'].min():.4f} / "
        f"{grid['expected_recognition_v02'].mean():.4f} / "
        f"{grid['expected_recognition_v02'].max():.4f}"
    )

    logger.info(
        f"Mean neighbor distance v02 min/mean/max: "
        f"{grid['mean_neighbor_distance_v02'].min():.4f} / "
        f"{grid['mean_neighbor_distance_v02'].mean():.4f} / "
        f"{grid['mean_neighbor_distance_v02'].max():.4f}"
    )

    logger.info(
        f"Comparable confidence v02 min/mean/max: "
        f"{grid['comparable_confidence_v02'].min():.4f} / "
        f"{grid['comparable_confidence_v02'].mean():.4f} / "
        f"{grid['comparable_confidence_v02'].max():.4f}"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    grid.to_file(
        OUTPUT_PATH,
        layer="coastal_grid_expected_recognition_v02",
        driver="GPKG",
    )

    logger.info(f"Saved output: {OUTPUT_PATH}")
    logger.info("Script 21 completed successfully.")


if __name__ == "__main__":
    main()