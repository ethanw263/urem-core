#!/usr/bin/env python3
"""
Phase 1 Script 04: Compute Initial Proxy Features

Purpose:
    Create the first testable UREM feature layer using only data we already have:
    - 1 km grid
    - coastline

Outputs:
    data/processed/coastal_grid_proxy_features.gpkg

Features:
    - distance_to_coast_m
    - coastal_proximity_score
    - edge_area_ratio
    - preliminary_physical_potential

This is NOT the final MVF.
It is only the first runnable proxy layer so we can test the algorithm pipeline.
"""

from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
from shapely.ops import unary_union


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
COASTLINE_DIR = RAW_DIR / "coastline"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

GRID_PATH = PROCESSED_DIR / "coastal_grid_1km.gpkg"
OUTPUT_PATH = PROCESSED_DIR / "coastal_grid_proxy_features.gpkg"

CRS_CA_ALBERS = "EPSG:3310"


def log(message: str) -> None:
    print(f"[04_compute_phase1_proxy_features] {message}")


def find_vector_files(folder: Path) -> list[Path]:
    files = []
    for ext in [".gpkg", ".shp", ".geojson", ".json"]:
        files.extend(folder.rglob(f"*{ext}"))
    return files


def read_first_vector_file(folder: Path, label: str) -> gpd.GeoDataFrame:
    files = find_vector_files(folder)

    if not files:
        raise FileNotFoundError(f"No vector file found for {label} in {folder}")

    priority = {".gpkg": 0, ".shp": 1, ".geojson": 2, ".json": 3}
    files = sorted(files, key=lambda p: priority.get(p.suffix.lower(), 99))

    selected = files[0]
    log(f"Reading {label}: {selected}")

    gdf = gpd.read_file(selected)

    if gdf.empty:
        raise ValueError(f"{label} file is empty.")

    if gdf.crs is None:
        raise ValueError(f"{label} CRS is missing.")

    return gdf


def minmax_inverse(values: np.ndarray) -> np.ndarray:
    """
    Lower raw values receive higher scores.
    Used for distance-to-coast.
    """
    min_val = np.nanmin(values)
    max_val = np.nanmax(values)

    if max_val == min_val:
        return np.ones_like(values)

    return 1 - ((values - min_val) / (max_val - min_val))


def minmax_positive(values: np.ndarray) -> np.ndarray:
    """
    Higher raw values receive higher scores.
    """
    min_val = np.nanmin(values)
    max_val = np.nanmax(values)

    if max_val == min_val:
        return np.ones_like(values)

    return (values - min_val) / (max_val - min_val)


def load_grid() -> gpd.GeoDataFrame:
    if not GRID_PATH.exists():
        raise FileNotFoundError(
            f"Missing grid file: {GRID_PATH}. Run scripts/02_create_grid.py first."
        )

    log(f"Reading grid: {GRID_PATH}")
    grid = gpd.read_file(GRID_PATH)

    if grid.crs is None:
        raise ValueError("Grid CRS is missing.")

    if grid.crs.to_string() != CRS_CA_ALBERS:
        grid = grid.to_crs(CRS_CA_ALBERS)

    grid["geometry"] = grid.geometry.make_valid()
    grid = grid[~grid.geometry.is_empty].copy()

    return grid


def load_coastline() -> gpd.GeoDataFrame:
    coast = read_first_vector_file(COASTLINE_DIR, "coastline")

    if coast.crs.to_string() != CRS_CA_ALBERS:
        coast = coast.to_crs(CRS_CA_ALBERS)

    coast["geometry"] = coast.geometry.make_valid()
    coast = coast[~coast.geometry.is_empty].copy()

    return coast[["geometry"]]


def compute_features(grid: gpd.GeoDataFrame, coast: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Unioning coastline geometry...")
    coast_union = unary_union(coast.geometry)

    log("Computing centroid distance to coastline...")
    centroids = grid.geometry.centroid
    grid["distance_to_coast_m"] = centroids.distance(coast_union)

    log("Computing coastal proximity score...")
    grid["coastal_proximity_score"] = minmax_inverse(
        grid["distance_to_coast_m"].to_numpy()
    )

    log("Computing edge area ratio...")
    max_cell_area = 1_000_000
    grid["edge_area_ratio"] = grid.geometry.area / max_cell_area
    grid["edge_area_score"] = minmax_positive(grid["edge_area_ratio"].to_numpy())

    log("Computing preliminary physical potential...")
    grid["preliminary_physical_potential"] = (
        0.80 * grid["coastal_proximity_score"]
        + 0.20 * grid["edge_area_score"]
    )

    grid["preliminary_physical_potential"] = grid[
        "preliminary_physical_potential"
    ].clip(0, 1)

    return grid


def run_qa(grid: gpd.GeoDataFrame) -> None:
    log("Running QA checks...")

    required_fields = [
        "cell_id",
        "distance_to_coast_m",
        "coastal_proximity_score",
        "edge_area_ratio",
        "edge_area_score",
        "preliminary_physical_potential",
    ]

    for field in required_fields:
        if field not in grid.columns:
            raise AssertionError(f"Missing required field: {field}")

    if grid.empty:
        raise AssertionError("Grid is empty.")

    if grid["cell_id"].duplicated().any():
        raise AssertionError("Duplicate cell_id values found.")

    score_fields = [
        "coastal_proximity_score",
        "edge_area_score",
        "preliminary_physical_potential",
    ]

    for field in score_fields:
        min_score = grid[field].min()
        max_score = grid[field].max()

        if min_score < 0 or max_score > 1:
            raise AssertionError(
                f"{field} outside expected 0-1 range: {min_score}, {max_score}"
            )

    log(f"Cells: {len(grid):,}")
    log(
        "Preliminary physical potential range: "
        f"{grid['preliminary_physical_potential'].min():.3f} "
        f"to {grid['preliminary_physical_potential'].max():.3f}"
    )

    log("QA complete.")


def main() -> None:
    log("Starting Script 04: Compute initial proxy features")

    try:
        grid = load_grid()
        coast = load_coastline()

        features = compute_features(grid, coast)
        run_qa(features)

        log(f"Writing output: {OUTPUT_PATH}")
        features.to_file(
            OUTPUT_PATH,
            layer="coastal_grid_proxy_features",
            driver="GPKG",
        )

        log("Complete.")
        log(f"Created: {OUTPUT_PATH}")

    except Exception as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()