#!/usr/bin/env python3
"""
Phase 1 Script 02: Create 1 km Coastal Analysis Grid

Input:
    data/processed/study_area_25km.gpkg

Output:
    data/processed/coastal_grid_1km.gpkg

Purpose:
    Generate the master UREM analysis lattice:
    1 km x 1 km grid cells clipped to the California coastal study area.

Notes:
    - Uses EPSG:3310 California Albers.
    - Grid cells are generated as full 1 km squares.
    - Cells are retained if they intersect the study area.
    - Output geometry is clipped to the study area.
    - Each cell receives a permanent unique cell_id.
"""

from pathlib import Path
import sys
import math

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


# -----------------------------
# Project paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_STUDY_AREA = PROCESSED_DIR / "study_area_25km.gpkg"
OUTPUT_GRID = PROCESSED_DIR / "coastal_grid_1km.gpkg"


# -----------------------------
# Constants
# -----------------------------

CRS_CA_ALBERS = "EPSG:3310"
GRID_SIZE_M = 1_000


# -----------------------------
# Helper functions
# -----------------------------

def log(message: str) -> None:
    print(f"[02_create_grid] {message}")


def load_study_area() -> gpd.GeoDataFrame:
    if not INPUT_STUDY_AREA.exists():
        raise FileNotFoundError(
            f"Missing input study area: {INPUT_STUDY_AREA}. "
            "Run scripts/01_create_study_area.py first."
        )

    log(f"Reading study area: {INPUT_STUDY_AREA}")
    study = gpd.read_file(INPUT_STUDY_AREA)

    if study.empty:
        raise ValueError("Study area file is empty.")

    if study.crs is None:
        raise ValueError("Study area CRS is missing.")

    if study.crs.to_string() != CRS_CA_ALBERS:
        log(f"Reprojecting study area from {study.crs} to {CRS_CA_ALBERS}")
        study = study.to_crs(CRS_CA_ALBERS)

    study["geometry"] = study.geometry.make_valid()
    study = study[~study.geometry.is_empty].copy()

    study["__dissolve"] = 1
    study = study.dissolve(by="__dissolve").reset_index(drop=True)

    return study[["geometry"]]


def snap_bounds_to_grid(bounds, grid_size: int):
    """
    Snap bounding box outward to the nearest grid interval.
    """
    minx, miny, maxx, maxy = bounds

    minx = math.floor(minx / grid_size) * grid_size
    miny = math.floor(miny / grid_size) * grid_size
    maxx = math.ceil(maxx / grid_size) * grid_size
    maxy = math.ceil(maxy / grid_size) * grid_size

    return minx, miny, maxx, maxy


def create_candidate_grid(study: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Create full 1 km candidate grid over the study area bounding box.
    """
    minx, miny, maxx, maxy = snap_bounds_to_grid(
        study.total_bounds,
        GRID_SIZE_M,
    )

    log("Creating candidate 1 km grid...")
    log(f"Grid bounds: {minx}, {miny}, {maxx}, {maxy}")

    polygons = []
    rows = []
    cols = []

    row_id = 0
    y = miny

    while y < maxy:
        col_id = 0
        x = minx

        while x < maxx:
            polygons.append(box(x, y, x + GRID_SIZE_M, y + GRID_SIZE_M))
            rows.append(row_id)
            cols.append(col_id)

            x += GRID_SIZE_M
            col_id += 1

        y += GRID_SIZE_M
        row_id += 1

    grid = gpd.GeoDataFrame(
        {
            "grid_row": rows,
            "grid_col": cols,
        },
        geometry=polygons,
        crs=CRS_CA_ALBERS,
    )

    log(f"Candidate grid cells created: {len(grid):,}")

    return grid


def clip_grid_to_study_area(
    grid: gpd.GeoDataFrame,
    study: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Retain and clip cells that intersect the study area.
    """
    log("Selecting grid cells that intersect the study area...")

    study_geom = study.geometry.iloc[0]
    grid = grid[grid.intersects(study_geom)].copy()

    if grid.empty:
        raise ValueError("No grid cells intersect the study area.")

    log(f"Intersecting cells before clip: {len(grid):,}")

    log("Clipping grid cells to study area...")
    clipped = gpd.overlay(grid, study, how="intersection")

    clipped["geometry"] = clipped.geometry.make_valid()
    clipped = clipped[~clipped.geometry.is_empty].copy()

    if clipped.empty:
        raise ValueError("Grid is empty after clipping.")

    clipped["cell_area_m2"] = clipped.geometry.area
    clipped = clipped[clipped["cell_area_m2"] > 0].copy()

    log(f"Grid cells after clip: {len(clipped):,}")

    return clipped


def assign_cell_ids(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Assign stable unique cell IDs based on grid row/column order.
    """
    log("Assigning permanent cell IDs...")

    grid = grid.sort_values(["grid_row", "grid_col"]).reset_index(drop=True)

    grid["cell_id"] = [
        f"CA{i:08d}" for i in range(1, len(grid) + 1)
    ]

    centroids = grid.geometry.centroid
    grid["centroid_x"] = centroids.x
    grid["centroid_y"] = centroids.y

    grid = grid[
        [
            "cell_id",
            "grid_row",
            "grid_col",
            "centroid_x",
            "centroid_y",
            "cell_area_m2",
            "geometry",
        ]
    ]

    return grid


def run_qa(grid: gpd.GeoDataFrame) -> None:
    log("Running QA checks...")

    if grid.empty:
        raise AssertionError("QA failed: grid is empty.")

    if grid.crs is None:
        raise AssertionError("QA failed: grid CRS is missing.")

    if grid.crs.to_string() != CRS_CA_ALBERS:
        raise AssertionError(
            f"QA failed: expected CRS {CRS_CA_ALBERS}, got {grid.crs}"
        )

    if grid["cell_id"].duplicated().any():
        raise AssertionError("QA failed: duplicate cell_id values found.")

    invalid_count = (~grid.geometry.is_valid).sum()
    if invalid_count > 0:
        raise AssertionError(f"QA failed: {invalid_count} invalid geometries.")

    zero_area_count = (grid.geometry.area <= 0).sum()
    if zero_area_count > 0:
        raise AssertionError(f"QA failed: {zero_area_count} zero-area cells.")

    total_area_km2 = grid.geometry.area.sum() / 1_000_000
    cell_count = len(grid)

    log(f"Grid cell count: {cell_count:,}")
    log(f"Total clipped grid area: {total_area_km2:,.0f} km²")

    if not (50_000 <= cell_count <= 80_000):
        log(
            f"WARNING: Grid cell count {cell_count:,} is outside rough expected range. "
            "Inspect in QGIS."
        )

    if not (55_000 <= total_area_km2 <= 65_000):
        log(
            f"WARNING: Total clipped grid area {total_area_km2:,.0f} km² "
            "is outside rough expected range. Inspect in QGIS."
        )

    log("QA complete.")


def main() -> None:
    log("Starting Phase 1 Script 02: Create 1 km grid")

    try:
        study = load_study_area()
        candidate_grid = create_candidate_grid(study)
        clipped_grid = clip_grid_to_study_area(candidate_grid, study)
        final_grid = assign_cell_ids(clipped_grid)
        run_qa(final_grid)

        log(f"Writing output: {OUTPUT_GRID}")
        final_grid.to_file(
            OUTPUT_GRID,
            layer="coastal_grid_1km",
            driver="GPKG",
        )

        log("Complete.")
        log(f"Created: {OUTPUT_GRID}")

    except Exception as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()