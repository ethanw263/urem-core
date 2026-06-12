#!/usr/bin/env python3
"""
07_add_slope.py

Adds DEM-derived slope metrics to the UREM coastal grid.

Input:
- data/processed/coastal_grid_relief.gpkg

Uses cached AWS Terrain Terrarium tiles from:
- data/raw/dem/terrain_tiles/12/{x}/{y}.png

Output:
- data/processed/coastal_grid_slope.gpkg

Method:
- Uses each cell centroid.
- Reads a small DEM pixel window around the centroid.
- Computes local slope from DEM gradients.
- Adds:
    slope_deg
    slope_pct
    slope_score
"""

from pathlib import Path
import logging
import math
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data/processed/coastal_grid_relief.gpkg"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/coastal_grid_slope.gpkg"
TILE_CACHE_DIR = PROJECT_ROOT / "data/raw/dem/terrain_tiles"

TARGET_CRS = "EPSG:3310"
WGS84 = "EPSG:4326"

ZOOM = 12
WINDOW_RADIUS_PIXELS = 2


def setup_logger():
    logger = logging.getLogger("07_add_slope")
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

    return ((s - smin) / (smax - smin)).clip(0, 1)


def lonlat_to_tile_pixel(lon, lat, zoom):
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


def tile_pixel_size_m(lat, zoom):
    earth_circumference_m = 40075016.686
    return earth_circumference_m * math.cos(math.radians(lat)) / ((2 ** zoom) * 256)


def decode_terrarium_array(arr):
    arr = arr.astype(float)
    return (arr[:, :, 0] * 256 + arr[:, :, 1] + arr[:, :, 2] / 256.0) - 32768.0


def load_tile(z, x, y, tile_cache):
    key = (z, x, y)

    if key in tile_cache:
        return tile_cache[key]

    tile_path = TILE_CACHE_DIR / str(z) / str(x) / f"{y}.png"

    if not tile_path.exists():
        tile_cache[key] = None
        return None

    img = Image.open(tile_path).convert("RGB")
    arr = np.array(img)
    elev = decode_terrarium_array(arr)

    tile_cache[key] = elev
    return elev


def get_elevation_pixel(z, x_tile, y_tile, px, py, tile_cache):
    # Handle tile-edge rollover.
    if px < 0:
        x_tile -= 1
        px += 256
    elif px > 255:
        x_tile += 1
        px -= 256

    if py < 0:
        y_tile -= 1
        py += 256
    elif py > 255:
        y_tile += 1
        py -= 256

    tile = load_tile(z, x_tile, y_tile, tile_cache)

    if tile is None:
        return np.nan

    val = tile[py, px]

    if val < -500 or val > 5000:
        return np.nan

    return val


def sample_slope(lon, lat, tile_cache):
    if pd.isna(lon) or pd.isna(lat):
        return np.nan, np.nan

    try:
        x_tile, y_tile, px, py = lonlat_to_tile_pixel(lon, lat, ZOOM)

        window = []

        for dy in range(-WINDOW_RADIUS_PIXELS, WINDOW_RADIUS_PIXELS + 1):
            row = []
            for dx in range(-WINDOW_RADIUS_PIXELS, WINDOW_RADIUS_PIXELS + 1):
                elev = get_elevation_pixel(
                    ZOOM,
                    x_tile,
                    y_tile,
                    px + dx,
                    py + dy,
                    tile_cache,
                )
                row.append(elev)
            window.append(row)

        dem = np.array(window, dtype=float)

        if np.isnan(dem).sum() > dem.size * 0.25:
            return np.nan, np.nan

        center = WINDOW_RADIUS_PIXELS
        pixel_size = tile_pixel_size_m(lat, ZOOM)

        dz_dy, dz_dx = np.gradient(dem, pixel_size, pixel_size)

        center_slope_rise_run = math.sqrt(
            dz_dx[center, center] ** 2 + dz_dy[center, center] ** 2
        )

        slope_deg = math.degrees(math.atan(center_slope_rise_run))
        slope_pct = center_slope_rise_run * 100.0

        if slope_deg < 0 or slope_deg > 90:
            return np.nan, np.nan

        return slope_deg, slope_pct

    except Exception:
        return np.nan, np.nan


def main():
    logger = setup_logger()
    logger.info("Starting Script 07: Add DEM-derived slope")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    if not TILE_CACHE_DIR.exists():
        raise FileNotFoundError(f"Missing terrain tile cache: {TILE_CACHE_DIR}")

    logger.info(f"Reading grid: {INPUT_PATH}")
    logger.info(f"Using tile cache: {TILE_CACHE_DIR}")

    grid = gpd.read_file(INPUT_PATH).to_crs(TARGET_CRS)

    if grid.empty:
        raise ValueError("Input grid is empty.")

    if "cell_id" not in grid.columns:
        grid["cell_id"] = range(1, len(grid) + 1)

    logger.info(f"Cells loaded: {len(grid):,}")
    logger.info(f"Input CRS: {grid.crs}")
    logger.info(f"Tile zoom: {ZOOM}")
    logger.info(f"Slope window radius: {WINDOW_RADIUS_PIXELS} pixels")

    points = grid[["cell_id", "geometry"]].copy()
    points["geometry"] = points.geometry.centroid
    points = points.to_crs(WGS84)

    points["lon"] = points.geometry.x
    points["lat"] = points.geometry.y

    tile_cache = {}
    slope_deg_values = []
    slope_pct_values = []

    logger.info("Sampling DEM-derived slope")

    for row in tqdm(points.itertuples(index=False), total=len(points)):
        slope_deg, slope_pct = sample_slope(row.lon, row.lat, tile_cache)
        slope_deg_values.append(slope_deg)
        slope_pct_values.append(slope_pct)

    grid["slope_deg"] = slope_deg_values
    grid["slope_pct"] = slope_pct_values
    grid["slope_source"] = "AWS Terrain Tiles Terrarium"
    grid["slope_method"] = "DEM_gradient_centroid_window"
    grid["slope_zoom"] = ZOOM
    grid["slope_window_radius_pixels"] = WINDOW_RADIUS_PIXELS
    grid["slope_score"] = minmax_score(grid["slope_deg"])

    valid = grid["slope_deg"].notna()

    logger.info("QA summary")
    logger.info(f"Cells processed: {len(grid):,}")
    logger.info(f"Cells with slope: {valid.sum():,}")
    logger.info(f"Missing slope count: {(~valid).sum():,}")
    logger.info(f"Unique tiles loaded from cache: {sum(v is not None for v in tile_cache.values()):,}")
    logger.info(f"Missing neighbor tiles encountered: {sum(v is None for v in tile_cache.values()):,}")
    logger.info(f"Output CRS: {grid.crs}")

    if valid.any():
        logger.info(f"Min slope: {grid.loc[valid, 'slope_deg'].min():,.2f} degrees")
        logger.info(f"Mean slope: {grid.loc[valid, 'slope_deg'].mean():,.2f} degrees")
        logger.info(f"Max slope: {grid.loc[valid, 'slope_deg'].max():,.2f} degrees")
        logger.info(f"Max slope score: {grid['slope_score'].max():.4f}")
    else:
        raise ValueError("No valid slope values were computed.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    grid.to_file(
        OUTPUT_PATH,
        layer="coastal_grid_slope",
        driver="GPKG",
    )

    logger.info(f"Saved output: {OUTPUT_PATH}")
    logger.info("Script 07 completed successfully.")


if __name__ == "__main__":
    main()