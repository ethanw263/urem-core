#!/usr/bin/env python3
"""
14_compute_expected_recognition.py

Computes expected recognition using physically comparable places.

Input:
- data/processed/coastal_grid_physical_fingerprints_v01.gpkg

Output:
- data/processed/coastal_grid_expected_recognition_v01.gpkg

Method:
- Build fingerprint matrix X from normalized physical fingerprint columns.
- Use k-nearest neighbors in fingerprint space.
- For each cell, expected recognition is the distance-weighted average
  observed recognition of physically similar cells.
"""

from pathlib import Path
import logging
import sys

import geopandas as gpd
import numpy as np
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data/processed/coastal_grid_physical_fingerprints_v01.gpkg"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/coastal_grid_expected_recognition_v01.gpkg"

TARGET_CRS = "EPSG:3310"

FINGERPRINT_COLS = [
    "fp_coastal_proximity",
    "fp_elevation",
    "fp_relief",
    "fp_slope",
    "fp_distance_to_coast_inverse",
    "fp_physical_potential",
]

OBSERVED_RECOGNITION_COL = "observed_recognition_v0"

K_NEIGHBORS = 50
EPSILON = 1e-9


def setup_logger():
    logger = logging.getLogger("14_compute_expected_recognition")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def main():
    logger = setup_logger()
    logger.info("Starting Script 14: Compute expected recognition")

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

    # +1 because the nearest neighbor of each point is itself.
    nn = NearestNeighbors(
        n_neighbors=K_NEIGHBORS + 1,
        algorithm="auto",
        metric="euclidean",
    )

    nn.fit(X)

    logger.info("Querying comparable places")

    distances, indices = nn.kneighbors(X)

    # Drop self-neighbor at position 0.
    neighbor_distances = distances[:, 1:]
    neighbor_indices = indices[:, 1:]

    neighbor_recognition = observed[neighbor_indices]

    # Inverse-distance weighting.
    weights = 1.0 / (neighbor_distances + EPSILON)

    weight_sums = weights.sum(axis=1)

    expected_recognition = (
        (neighbor_recognition * weights).sum(axis=1) / weight_sums
    )

    mean_neighbor_distance = neighbor_distances.mean(axis=1)
    min_neighbor_distance = neighbor_distances.min(axis=1)
    max_neighbor_distance = neighbor_distances.max(axis=1)

    # Comparable confidence:
    # Lower mean distance = stronger comparability.
    # Convert to a 0-1 score where 1 means very close fingerprint neighbors.
    max_mean_distance = np.nanmax(mean_neighbor_distance)

    if max_mean_distance == 0:
        comparable_confidence = np.ones(len(grid))
    else:
        comparable_confidence = 1.0 - (mean_neighbor_distance / max_mean_distance)
        comparable_confidence = np.clip(comparable_confidence, 0, 1)

    grid["expected_recognition_v01"] = expected_recognition
    grid["expected_recognition_method"] = "fingerprint_knn_inverse_distance"
    grid["comparable_neighbor_count"] = K_NEIGHBORS
    grid["mean_neighbor_distance"] = mean_neighbor_distance
    grid["min_neighbor_distance"] = min_neighbor_distance
    grid["max_neighbor_distance"] = max_neighbor_distance
    grid["comparable_confidence_v01"] = comparable_confidence

    logger.info("QA summary")
    logger.info(f"Cells processed: {len(grid):,}")
    logger.info(f"Output CRS: {grid.crs}")
    logger.info(
        f"Observed recognition min/mean/max: "
        f"{observed.min():.4f} / {observed.mean():.4f} / {observed.max():.4f}"
    )
    logger.info(
        f"Expected recognition min/mean/max: "
        f"{grid['expected_recognition_v01'].min():.4f} / "
        f"{grid['expected_recognition_v01'].mean():.4f} / "
        f"{grid['expected_recognition_v01'].max():.4f}"
    )
    logger.info(
        f"Mean neighbor distance min/mean/max: "
        f"{grid['mean_neighbor_distance'].min():.4f} / "
        f"{grid['mean_neighbor_distance'].mean():.4f} / "
        f"{grid['mean_neighbor_distance'].max():.4f}"
    )
    logger.info(
        f"Comparable confidence min/mean/max: "
        f"{grid['comparable_confidence_v01'].min():.4f} / "
        f"{grid['comparable_confidence_v01'].mean():.4f} / "
        f"{grid['comparable_confidence_v01'].max():.4f}"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    grid.to_file(
        OUTPUT_PATH,
        layer="coastal_grid_expected_recognition_v01",
        driver="GPKG",
    )

    logger.info(f"Saved output: {OUTPUT_PATH}")
    logger.info("Script 14 completed successfully.")


if __name__ == "__main__":
    main()