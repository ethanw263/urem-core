#!/usr/bin/env python3
"""
Phase 1 Script 01: Create California Coastal Study Area

Output:
    data/processed/study_area_25km.gpkg

Purpose:
    Create the Version 1 UREM-Golf MVF study area:
    California land area within 25 km of the California coastline.

Inputs:
    - Coastline vector file placed under data/raw/coastline/
    - California boundary vector file placed under data/raw/boundaries/

Notes:
    - All buffering is performed in EPSG:3310 (California Albers).
    - This script intentionally does NOT create the 1km grid yet.
    - This script does NOT download data automatically.
"""

from pathlib import Path
import sys

import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union


# -----------------------------
# Project paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
COASTLINE_DIR = RAW_DIR / "coastline"
BOUNDARIES_DIR = RAW_DIR / "boundaries"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "phase1"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_STUDY_AREA = PROCESSED_DIR / "study_area_25km.gpkg"

for folder in [COASTLINE_DIR, BOUNDARIES_DIR, INTERIM_DIR, PROCESSED_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Constants
# -----------------------------

CRS_CA_ALBERS = "EPSG:3310"
BUFFER_DISTANCE_M = 25_000


# -----------------------------
# Helper functions
# -----------------------------

def log(message: str) -> None:
    print(f"[01_create_study_area] {message}")


def find_vector_files(folder: Path) -> list[Path]:
    extensions = [".gpkg", ".shp", ".geojson", ".json"]
    files = []

    for ext in extensions:
        files.extend(folder.rglob(f"*{ext}"))

    return files


def read_first_vector_file(folder: Path, label: str) -> gpd.GeoDataFrame:
    files = find_vector_files(folder)

    if not files:
        raise FileNotFoundError(
            f"No vector file found for {label} in {folder}. "
            "Expected one of: .gpkg, .shp, .geojson, .json. "
            "Make sure the full shapefile set is present, not only the .shp file."
        )

    priority = {
        ".gpkg": 0,
        ".shp": 1,
        ".geojson": 2,
        ".json": 3,
    }

    files = sorted(files, key=lambda p: priority.get(p.suffix.lower(), 99))
    selected = files[0]

    log(f"Reading {label}: {selected}")

    gdf = gpd.read_file(selected)

    if gdf.empty:
        raise ValueError(f"{label} file is empty: {selected}")

    if gdf.crs is None:
        raise ValueError(
            f"{label} file has no CRS: {selected}. "
            "Make sure the .prj file is present or define the CRS manually."
        )

    return gdf


def load_california_boundary() -> gpd.GeoDataFrame:
    """
    Load California boundary from local files under data/raw/boundaries/.
    """
    ca = read_first_vector_file(BOUNDARIES_DIR, "California state boundary")
    ca = ca.to_crs(CRS_CA_ALBERS)

    ca["geometry"] = ca.geometry.make_valid()
    ca = ca[~ca.geometry.is_empty].copy()

    ca["__dissolve"] = 1
    ca = ca.dissolve(by="__dissolve").reset_index(drop=True)

    return ca[["geometry"]]


def load_coastline() -> gpd.GeoDataFrame:
    """
    Load coastline data from local files under data/raw/coastline/.
    """
    coast = read_first_vector_file(COASTLINE_DIR, "coastline")
    coast = coast.to_crs(CRS_CA_ALBERS)

    coast["geometry"] = coast.geometry.make_valid()
    coast = coast[~coast.geometry.is_empty].copy()

    return coast[["geometry"]]


def clip_coastline_to_california_area(
    coast: gpd.GeoDataFrame,
    ca_boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Clip coastline to a broad California bounding area.
    This avoids accidentally using unrelated shoreline segments.
    """
    minx, miny, maxx, maxy = ca_boundary.total_bounds

    buffer_m = 100_000
    broad_box = box(
        minx - buffer_m,
        miny - buffer_m,
        maxx + buffer_m,
        maxy + buffer_m,
    )

    coast_clip = coast[coast.intersects(broad_box)].copy()

    if coast_clip.empty:
        raise ValueError(
            "No coastline features intersect the California bounding region. "
            "Check that your coastline file covers California and has the correct CRS."
        )

    return coast_clip


def create_study_area(
    ca_boundary: gpd.GeoDataFrame,
    coast: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Buffer the coastline by 25 km, then intersect with California boundary.
    """
    log("Unioning coastline geometry...")
    coast_union = unary_union(coast.geometry)

    log(f"Buffering coastline by {BUFFER_DISTANCE_M / 1000:.1f} km...")
    coast_buffer = coast_union.buffer(BUFFER_DISTANCE_M)

    coast_buffer_gdf = gpd.GeoDataFrame(
        {"buffer_km": [BUFFER_DISTANCE_M / 1000]},
        geometry=[coast_buffer],
        crs=CRS_CA_ALBERS,
    )

    log("Intersecting coastline buffer with California boundary...")
    study = gpd.overlay(ca_boundary, coast_buffer_gdf, how="intersection")

    if study.empty:
        raise ValueError("Study area is empty after clipping. Check inputs.")

    study["geometry"] = study.geometry.make_valid()
    study = study[~study.geometry.is_empty].copy()

    study["study_id"] = "ca_coastal_25km_v1"
    study["buffer_km"] = BUFFER_DISTANCE_M / 1000
    study["source"] = "local_coastline + local_ca_boundary"

    study["__dissolve"] = 1
    study = study.dissolve(by="__dissolve").reset_index(drop=True)

    study = study[["study_id", "buffer_km", "source", "geometry"]]

    return study


def run_qa(study: gpd.GeoDataFrame) -> None:
    """
    Basic QA checks for Phase 1.
    """
    log("Running QA checks...")

    if study.empty:
        raise AssertionError("QA failed: study area is empty.")

    if study.crs is None:
        raise AssertionError("QA failed: study area CRS is missing.")

    if study.crs.to_string() != CRS_CA_ALBERS:
        raise AssertionError(
            f"QA failed: expected CRS {CRS_CA_ALBERS}, got {study.crs}"
        )

    invalid_count = (~study.geometry.is_valid).sum()

    if invalid_count > 0:
        raise AssertionError(f"QA failed: {invalid_count} invalid geometries.")

    area_km2 = study.geometry.area.sum() / 1_000_000

    if area_km2 <= 0:
        raise AssertionError("QA failed: study area has non-positive area.")

    if not (20_000 <= area_km2 <= 120_000):
        log(
            f"WARNING: Study area is {area_km2:,.0f} km², outside rough sanity range. "
            "This may still be valid, but inspect the output in QGIS."
        )

    log(f"Study area area: {area_km2:,.0f} km²")
    log("QA complete.")


def main() -> None:
    log("Starting Phase 1 Script 01: Create study area")

    try:
        ca_boundary = load_california_boundary()
        coastline = load_coastline()

        coastline = clip_coastline_to_california_area(coastline, ca_boundary)

        study_area = create_study_area(ca_boundary, coastline)
        run_qa(study_area)

        log(f"Writing output: {OUTPUT_STUDY_AREA}")
        study_area.to_file(
            OUTPUT_STUDY_AREA,
            layer="study_area_25km",
            driver="GPKG",
        )

        log("Complete.")
        log(f"Created: {OUTPUT_STUDY_AREA}")

    except Exception as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()