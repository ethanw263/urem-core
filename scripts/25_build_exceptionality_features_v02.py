#!/usr/bin/env python3
"""
Script 25: Build Exceptionality Features v02

Uses existing Phase 2 outputs instead of raw DEM raster:
- coastal_grid_elevation.gpkg
- coastal_grid_relief.gpkg
- coastal_grid_slope.gpkg
"""

from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parents[1]

GRID_PATH = BASE_DIR / "data/processed/coastal_grid_1km.gpkg"
ELEVATION_PATH = BASE_DIR / "data/processed/coastal_grid_elevation.gpkg"
RELIEF_PATH = BASE_DIR / "data/processed/coastal_grid_relief.gpkg"
SLOPE_PATH = BASE_DIR / "data/processed/coastal_grid_slope.gpkg"

COASTLINE_PATH = BASE_DIR / "data/raw/coastline/ne_10m_coastline.shp"
BEACH_PATH = BASE_DIR / "data/processed/beaches.gpkg"
CLIFF_PATH = BASE_DIR / "data/processed/cliffs.gpkg"

OUT_GPKG = BASE_DIR / "data/processed/exceptionality_features_v02.gpkg"
OUT_CSV = BASE_DIR / "data/processed/exceptionality_features_v02.csv"


def log(msg: str) -> None:
    print(f"[25_build_exceptionality_features_v02] {msg}")


def safe_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    min_val = s.min(skipna=True)
    max_val = s.max(skipna=True)

    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(0, index=series.index)

    return (s - min_val) / (max_val - min_val)


def read_optional_vector(path: Path, target_crs):
    if not path.exists():
        log(f"Optional layer missing, skipping: {path}")
        return None

    gdf = gpd.read_file(path)

    if gdf.empty:
        log(f"Optional layer empty, skipping: {path}")
        return None

    if gdf.crs is None:
        raise ValueError(f"Layer has no CRS: {path}")

    return gdf.to_crs(target_crs)


def nearest_distance_m(points_gdf: gpd.GeoDataFrame, target_gdf: gpd.GeoDataFrame) -> pd.Series:
    if target_gdf is None or target_gdf.empty:
        return pd.Series(np.nan, index=points_gdf.index)

    target_union = target_gdf.geometry.unary_union

    distances = []
    for geom in points_gdf.geometry:
        try:
            distances.append(float(geom.distance(target_union)))
        except Exception:
            distances.append(np.nan)

    return pd.Series(distances, index=points_gdf.index)


def add_existing_phase2_features(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Adding existing Phase 2 elevation, relief, and slope features")

    for path in [ELEVATION_PATH, RELIEF_PATH, SLOPE_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required Phase 2 file missing: {path}")

    elevation = gpd.read_file(ELEVATION_PATH).to_crs(grid.crs)
    relief = gpd.read_file(RELIEF_PATH).to_crs(grid.crs)
    slope = gpd.read_file(SLOPE_PATH).to_crs(grid.crs)

    grid = grid.reset_index(drop=True)
    elevation = elevation.reset_index(drop=True)
    relief = relief.reset_index(drop=True)
    slope = slope.reset_index(drop=True)

    if len(grid) != len(elevation):
        raise ValueError("Grid and elevation file row counts do not match")
    if len(grid) != len(relief):
        raise ValueError("Grid and relief file row counts do not match")
    if len(grid) != len(slope):
        raise ValueError("Grid and slope file row counts do not match")

    for source_gdf, prefix in [
        (elevation, "elev"),
        (relief, "relief"),
        (slope, "slope"),
    ]:
        for col in source_gdf.columns:
            if col == "geometry":
                continue

            new_col = col

            if new_col in grid.columns:
                new_col = f"{prefix}_{col}"

            grid[new_col] = source_gdf[col]

    return grid


def compute_vector_distance_features(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Computing optional coastline/beach/cliff distance features")

    coastline = read_optional_vector(COASTLINE_PATH, grid.crs)
    beaches = read_optional_vector(BEACH_PATH, grid.crs)
    cliffs = read_optional_vector(CLIFF_PATH, grid.crs)

    grid["dist_coastline_m"] = nearest_distance_m(grid, coastline)
    grid["dist_beach_m"] = nearest_distance_m(grid, beaches)
    grid["dist_cliff_m"] = nearest_distance_m(grid, cliffs)

    return grid


def compute_coastline_complexity(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    coastline = read_optional_vector(COASTLINE_PATH, grid.crs)

    if coastline is None or coastline.empty:
        log("Coastline layer missing. Setting coastline complexity to NaN.")
        grid["coastline_length_3km_m"] = np.nan
        grid["coastline_complexity_3km"] = np.nan
        return grid

    log("Computing coastline complexity")

    coast_union = coastline.geometry.unary_union
    lengths = []

    for idx, geom in enumerate(grid.geometry):
        if idx % 5000 == 0:
            log(f"Coastline complexity {idx:,}/{len(grid):,}")

        try:
            buffer_geom = geom.centroid.buffer(3000)
            clipped = coast_union.intersection(buffer_geom)
            lengths.append(float(clipped.length))
        except Exception:
            lengths.append(np.nan)

    grid["coastline_length_3km_m"] = lengths
    grid["coastline_complexity_3km"] = safe_minmax(grid["coastline_length_3km_m"])

    return grid


def compute_exceptionality_component_scores(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Computing exceptionality component scores")

    numeric_cols = grid.select_dtypes(include=[np.number]).columns.tolist()

    elev_cols = [c for c in numeric_cols if "elev" in c.lower()]
    relief_cols = [c for c in numeric_cols if "relief" in c.lower()]
    slope_cols = [c for c in numeric_cols if "slope" in c.lower()]

    if relief_cols:
        grid["score_relief_existing"] = safe_minmax(grid[relief_cols].mean(axis=1))
    else:
        grid["score_relief_existing"] = np.nan

    if slope_cols:
        grid["score_slope_existing"] = safe_minmax(grid[slope_cols].mean(axis=1))
    else:
        grid["score_slope_existing"] = np.nan

    if elev_cols:
        grid["score_elevation_existing"] = safe_minmax(grid[elev_cols].mean(axis=1))
    else:
        grid["score_elevation_existing"] = np.nan

    grid["score_coastline_proximity"] = 1 - safe_minmax(grid["dist_coastline_m"])
    grid["score_beach_proximity"] = 1 - safe_minmax(grid["dist_beach_m"])
    grid["score_cliff_proximity"] = 1 - safe_minmax(grid["dist_cliff_m"])
    grid["score_coastline_complexity"] = safe_minmax(grid["coastline_length_3km_m"])

    component_cols = [
        "score_relief_existing",
        "score_slope_existing",
        "score_elevation_existing",
        "score_coastline_proximity",
        "score_beach_proximity",
        "score_cliff_proximity",
        "score_coastline_complexity",
    ]

    grid["exceptionality_feature_count"] = grid[component_cols].notna().sum(axis=1)
    grid["exceptionality_preview_score_v02"] = grid[component_cols].mean(axis=1, skipna=True)

    return grid


def main():
    log("Starting Script 25")

    if not GRID_PATH.exists():
        raise FileNotFoundError(f"Grid not found: {GRID_PATH}")

    log(f"Reading grid: {GRID_PATH}")
    grid = gpd.read_file(GRID_PATH)

    if grid.empty:
        raise ValueError("Grid is empty")

    if grid.crs is None:
        raise ValueError("Grid has no CRS")

    log(f"Grid rows: {len(grid):,}")
    log(f"Grid CRS: {grid.crs}")

    grid = add_existing_phase2_features(grid)
    grid = compute_vector_distance_features(grid)
    grid = compute_coastline_complexity(grid)
    grid = compute_exceptionality_component_scores(grid)

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

    log(f"Writing GeoPackage: {OUT_GPKG}")
    grid.to_file(OUT_GPKG, driver="GPKG")

    log(f"Writing CSV: {OUT_CSV}")
    grid.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")


if __name__ == "__main__":
    main()