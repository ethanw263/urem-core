#!/usr/bin/env python3
"""
05_add_elevation.py

Adds elevation metrics to the UREM coastal grid using AWS Terrain Tiles
in Terrarium PNG format.

Input:
- data/processed/coastal_grid_proxy_features.gpkg

Output:
- data/processed/coastal_grid_elevation.gpkg
"""

from pathlib import Path
import logging
import math
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data/processed/coastal_grid_proxy_features.gpkg"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/coastal_grid_elevation.gpkg"
TILE_CACHE_DIR = PROJECT_ROOT / "data/raw/dem/terrain_tiles"

TARGET_CRS = "EPSG:3310"
WGS84 = "EPSG:4326"

ZOOM = 12
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

REQUEST_SLEEP_SECONDS = 0.02
REQUEST_TIMEOUT = 30


def setup_logger():
    logger = logging.getLogger("05_add_elevation")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def lonlat_to_tile(lon, lat, zoom):
    lat_rad = math.radians(lat)
    n = 2 ** zoom

    x_float = (lon + 180.0) / 360.0 * n
    y_float = (
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
    )

    x_tile = int(math.floor(x_float))
    y_tile = int(math.floor(y_float))

    px = int((x_float - x_tile) * 256)
    py = int((y_float - y_tile) * 256)

    px = min(max(px, 0), 255)
    py = min(max(py, 0), 255)

    return x_tile, y_tile, px, py


def decode_terrarium_pixel(pixel):
    r, g, b = pixel[:3]
    return (r * 256 + g + b / 256.0) - 32768.0


def download_tile(z, x, y, logger):
    TILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tile_dir = TILE_CACHE_DIR / str(z) / str(x)
    tile_dir.mkdir(parents=True, exist_ok=True)

    tile_path = tile_dir / f"{y}.png"

    if tile_path.exists():
        return tile_path

    url = TILE_URL.format(z=z, x=x, y=y)

    response = requests.get(url, timeout=REQUEST_TIMEOUT)

    if response.status_code != 200:
        logger.warning(f"Tile request failed {response.status_code}: {url}")
        return None

    tile_path.write_bytes(response.content)
    time.sleep(REQUEST_SLEEP_SECONDS)

    return tile_path


def sample_elevation(lon, lat, tile_cache, logger):
    if pd.isna(lon) or pd.isna(lat):
        return np.nan

    try:
        x, y, px, py = lonlat_to_tile(lon, lat, ZOOM)
        key = (ZOOM, x, y)

        if key not in tile_cache:
            tile_path = download_tile(ZOOM, x, y, logger)

            if tile_path is None:
                tile_cache[key] = None
            else:
                tile_cache[key] = Image.open(tile_path).convert("RGB")

        img = tile_cache[key]

        if img is None:
            return np.nan

        pixel = img.getpixel((px, py))
        elev = decode_terrarium_pixel(pixel)

        # Basic sanity filter.
        if elev < -500 or elev > 5000:
            return np.nan

        return elev

    except Exception:
        return np.nan


def main():
    logger = setup_logger()
    logger.info("Starting Script 05: Add elevation")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input grid: {INPUT_PATH}")

    logger.info(f"Reading grid: {INPUT_PATH}")

    grid = gpd.read_file(INPUT_PATH).to_crs(TARGET_CRS)

    if grid.empty:
        raise ValueError("Input grid is empty.")

    logger.info(f"Grid cells loaded: {len(grid):,}")
    logger.info(f"Grid CRS: {grid.crs}")

    if "cell_id" not in grid.columns:
        grid["cell_id"] = range(1, len(grid) + 1)

    logger.info("Creating grid centroids")

    points = grid[["cell_id", "geometry"]].copy()
    points["geometry"] = points.geometry.centroid
    points = points.to_crs(WGS84)

    points["lon"] = points.geometry.x
    points["lat"] = points.geometry.y

    logger.info("Sampling elevation from AWS Terrain Tiles")
    logger.info(f"Tile zoom level: {ZOOM}")
    logger.info(f"Tile cache: {TILE_CACHE_DIR}")

    tile_cache = {}
    elevations = []

    for row in tqdm(points.itertuples(index=False), total=len(points)):
        elevations.append(
            sample_elevation(
                lon=row.lon,
                lat=row.lat,
                tile_cache=tile_cache,
                logger=logger,
            )
        )

    grid["elevation_m"] = elevations
    grid["elevation_source"] = "AWS Terrain Tiles Terrarium"
    grid["elevation_sample_method"] = "cell_centroid"
    grid["elevation_zoom"] = ZOOM

    valid = grid["elevation_m"].notna()

    logger.info("QA summary")
    logger.info(f"Cells processed: {len(grid):,}")
    logger.info(f"Cells with elevation: {valid.sum():,}")
    logger.info(f"Missing elevation count: {(~valid).sum():,}")
    logger.info(f"Unique tiles used/downloaded this run: {len(tile_cache):,}")
    logger.info(f"Output CRS: {grid.crs}")

    if valid.any():
        logger.info(f"Min elevation: {grid.loc[valid, 'elevation_m'].min():,.2f} m")
        logger.info(f"Mean elevation: {grid.loc[valid, 'elevation_m'].mean():,.2f} m")
        logger.info(f"Max elevation: {grid.loc[valid, 'elevation_m'].max():,.2f} m")
    else:
        raise ValueError("No valid elevations were sampled.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    grid.to_file(
        OUTPUT_PATH,
        layer="coastal_grid_elevation",
        driver="GPKG",
    )

    logger.info(f"Saved output: {OUTPUT_PATH}")
    logger.info("Script 05 completed successfully.")


if __name__ == "__main__":
    main()