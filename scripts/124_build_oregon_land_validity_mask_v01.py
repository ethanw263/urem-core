#!/usr/bin/env python3
"""
124_build_oregon_land_validity_mask_v01.py

Build Oregon land validity mask.

Inputs:
- data/processed/oregon_coast_grid_v01.gpkg
- data/raw/land/ne_10m_land.shp

Outputs:
- data/processed/oregon_land_validity_mask_v01.gpkg
- data/processed/oregon_land_validity_mask_v01.csv
"""

from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
from shapely.validation import make_valid

warnings.filterwarnings("ignore")


SCRIPT_NAME = "124_build_oregon_land_validity_mask_v01"

BASE_DIR = Path(__file__).resolve().parents[1]

GRID_PATH = BASE_DIR / "data/processed/oregon_coast_grid_v01.gpkg"

LAND_SHP = BASE_DIR / "data/raw/land/ne_10m_land.shp"

OUT_GPKG = BASE_DIR / "data/processed/oregon_land_validity_mask_v01.gpkg"
OUT_CSV = BASE_DIR / "data/processed/oregon_land_validity_mask_v01.csv"

MIN_LAND_AREA_SHARE = 0.25


def log(msg: str) -> None:
    print(f"[{SCRIPT_NAME}] {msg}")


def main():
    log("Starting Oregon land validity mask")

    if not GRID_PATH.exists():
        raise FileNotFoundError(f"Grid not found: {GRID_PATH}")

    if not LAND_SHP.exists():
        raise FileNotFoundError(f"Land shapefile not found: {LAND_SHP}")

    log(f"Reading grid: {GRID_PATH}")
    grid = gpd.read_file(GRID_PATH)

    if grid.crs is None:
        raise ValueError("Grid has no CRS")

    log(f"Grid rows: {len(grid):,}")
    log(f"Grid CRS: {grid.crs}")

    log(f"Reading land polygons: {LAND_SHP}")
    land = gpd.read_file(LAND_SHP)

    if land.crs is None:
        land = land.set_crs("EPSG:4326")

    land = land.to_crs(grid.crs)

    log("Cleaning land geometries")

    bbox = gpd.GeoDataFrame(
        geometry=[grid.unary_union.envelope.buffer(5000)],
        crs=grid.crs,
    )

    land = gpd.overlay(land, bbox, how="intersection")

    land["geometry"] = land.geometry.apply(
        lambda g: make_valid(g) if g is not None and not g.is_empty else g
    )

    land = land[land.geometry.notna()].copy()
    land = land[~land.geometry.is_empty].copy()
    land["geometry"] = land.geometry.buffer(0)

    log("Dissolving cleaned land polygons")
    land_union = land.geometry.unary_union

    log("Computing land area share per Oregon grid cell")

    land_areas = []
    centroid_on_land = []

    for idx, geom in enumerate(grid.geometry):
        if idx % 5000 == 0:
            log(f"Processing cell {idx:,}/{len(grid):,}")

        try:
            cell_area = geom.area
            land_intersection = geom.intersection(land_union)
            land_area = land_intersection.area if not land_intersection.is_empty else 0.0
            land_share = land_area / cell_area if cell_area > 0 else 0.0

            land_areas.append(land_area)
            centroid_on_land.append(bool(geom.centroid.within(land_union)))

        except Exception:
            land_areas.append(np.nan)
            centroid_on_land.append(False)

    grid["land_area_m2"] = land_areas
    grid["land_area_share"] = grid["land_area_m2"] / grid.geometry.area
    grid["water_area_share"] = 1 - grid["land_area_share"]
    grid["centroid_on_land"] = centroid_on_land

    grid["is_valid_land_candidate"] = (
        (grid["land_area_share"] >= MIN_LAND_AREA_SHARE)
        | (grid["centroid_on_land"])
    )

    grid["land_validity_method"] = (
        f"Natural Earth land polygon intersection; "
        f"valid if land_area_share >= {MIN_LAND_AREA_SHARE} or centroid_on_land"
    )

    if OUT_GPKG.exists():
        OUT_GPKG.unlink()

    log(f"Writing GeoPackage: {OUT_GPKG}")
    grid.to_file(
        OUT_GPKG,
        layer="oregon_land_validity_mask_v01",
        driver="GPKG",
    )

    log(f"Writing CSV: {OUT_CSV}")
    grid.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")

    print("\nLand validity summary:")
    print(grid[["land_area_share", "water_area_share"]].describe())

    print("\nValid land candidate counts:")
    print(grid["is_valid_land_candidate"].value_counts())


if __name__ == "__main__":
    main()