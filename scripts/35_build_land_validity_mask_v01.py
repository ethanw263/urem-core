#!/usr/bin/env python3
"""
Script 35: Build Land Validity Mask v01

Purpose:
- Identify grid cells that are valid land candidates.
- Remove ocean / bay / offshore water cells from UREM ranking and validation.

Inputs:
- data/processed/coastal_grid_1km.gpkg
- Natural Earth land polygons, auto-downloaded if missing

Outputs:
- data/processed/land_validity_mask_v01.gpkg
- data/processed/land_validity_mask_v01.csv
"""

from pathlib import Path
import zipfile
import requests
import warnings
from shapely.validation import make_valid

import geopandas as gpd
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parents[1]

GRID_PATH = BASE_DIR / "data/processed/coastal_grid_1km.gpkg"

LAND_DIR = BASE_DIR / "data/raw/land"
LAND_ZIP = LAND_DIR / "ne_10m_land.zip"
LAND_SHP = LAND_DIR / "ne_10m_land.shp"

OUT_GPKG = BASE_DIR / "data/processed/land_validity_mask_v01.gpkg"
OUT_CSV = BASE_DIR / "data/processed/land_validity_mask_v01.csv"

NATURAL_EARTH_LAND_URL = (
    "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip"
)

MIN_LAND_AREA_SHARE = 0.25


def log(msg: str) -> None:
    print(f"[35_build_land_validity_mask_v01] {msg}")


def download_land_if_needed():
    LAND_DIR.mkdir(parents=True, exist_ok=True)

    if LAND_SHP.exists():
        log(f"Land shapefile already exists: {LAND_SHP}")
        return

    log("Natural Earth land polygons missing. Downloading...")
    r = requests.get(NATURAL_EARTH_LAND_URL, timeout=120)
    r.raise_for_status()

    LAND_ZIP.write_bytes(r.content)

    log("Extracting Natural Earth land polygons")
    with zipfile.ZipFile(LAND_ZIP, "r") as z:
        z.extractall(LAND_DIR)


def main():
    log("Starting Script 35")

    if not GRID_PATH.exists():
        raise FileNotFoundError(f"Grid not found: {GRID_PATH}")

    download_land_if_needed()

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

    from shapely.validation import make_valid

    log("Cleaning land geometries")

    # Keep only land polygons near the study grid to reduce topology problems
    minx, miny, maxx, maxy = grid.total_bounds
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

    # buffer(0) is a second safety repair
    land["geometry"] = land.geometry.buffer(0)

    log("Dissolving cleaned land polygons")
    land_union = land.geometry.unary_union

    log("Computing land area share per grid cell")

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

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

    log(f"Writing GeoPackage: {OUT_GPKG}")
    grid.to_file(OUT_GPKG, driver="GPKG")

    log(f"Writing CSV: {OUT_CSV}")
    grid.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")

    print("\nLand validity summary:")
    print(grid[["land_area_share", "water_area_share"]].describe())

    print("\nValid land candidate counts:")
    print(grid["is_valid_land_candidate"].value_counts())


if __name__ == "__main__":
    main()
