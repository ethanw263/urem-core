#!/usr/bin/env python3
"""
Script 117: Build Oregon Coast Grid v01

Purpose:
Create a 1 km grid clipped to the Oregon Coast study area.

Input:
- data/processed/oregon_coast_study_area_v01.gpkg

Output:
- data/processed/oregon_coast_grid_v01.gpkg
"""

from pathlib import Path
import geopandas as gpd
from shapely.geometry import box

SCRIPT_NAME = "117_build_oregon_coast_grid_v01"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

STUDY_AREA_PATH = PROCESSED_DIR / "oregon_coast_study_area_v01.gpkg"
OUTPUT_PATH = PROCESSED_DIR / "oregon_coast_grid_v01.gpkg"

GRID_SIZE_METERS = 1_000
TARGET_CRS = "EPSG:5070"


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def main():
    log("Starting Oregon Coast grid creation")

    if not STUDY_AREA_PATH.exists():
        raise FileNotFoundError(f"Missing study area file: {STUDY_AREA_PATH}")

    log(f"Reading study area: {STUDY_AREA_PATH}")
    study_area = gpd.read_file(STUDY_AREA_PATH).to_crs(TARGET_CRS)

    study_area = study_area.dissolve()
    study_geom = study_area.geometry.iloc[0]

    minx, miny, maxx, maxy = study_geom.bounds

    log("Building 1 km candidate grid")

    cells = []
    cell_ids = []

    cell_id = 0
    x = minx

    while x < maxx:
        y = miny
        while y < maxy:
            geom = box(x, y, x + GRID_SIZE_METERS, y + GRID_SIZE_METERS)

            if geom.intersects(study_geom):
                clipped = geom.intersection(study_geom)

                if not clipped.is_empty:
                    cells.append(clipped)
                    cell_ids.append(cell_id)
                    cell_id += 1

            y += GRID_SIZE_METERS
        x += GRID_SIZE_METERS

    grid = gpd.GeoDataFrame(
        {
            "cell_id": cell_ids,
            "domain": "oregon_coast",
            "grid_m": GRID_SIZE_METERS,
        },
        geometry=cells,
        crs=TARGET_CRS,
    )

    grid["area_sq_km"] = grid.geometry.area / 1_000_000
    grid["centroid_x"] = grid.geometry.centroid.x
    grid["centroid_y"] = grid.geometry.centroid.y

    log(f"Created grid cells: {len(grid):,}")
    log(f"Total grid area: {grid['area_sq_km'].sum():,.2f} sq km")
    log(f"Writing output: {OUTPUT_PATH}")

    grid.to_file(
        OUTPUT_PATH,
        layer="oregon_coast_grid_v01",
        driver="GPKG",
    )

    log("Done")


if __name__ == "__main__":
    main()