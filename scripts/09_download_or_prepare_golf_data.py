#!/usr/bin/env python3
"""
Script 09: Download or Prepare Golf Course Data

Downloads OSM golf courses in small chunks to avoid Overpass timeout failures.

Inputs:
    data/processed/study_area_25km.gpkg

Outputs:
    data/raw/golf/osm_golf_courses_raw.gpkg
    data/processed/golf_courses_ca_coastal.gpkg
"""

from pathlib import Path
import sys
import time

import geopandas as gpd
import pandas as pd
import osmnx as ox
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_GOLF_DIR = PROJECT_ROOT / "data" / "raw" / "golf"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

STUDY_AREA_PATH = PROCESSED_DIR / "study_area_25km.gpkg"

RAW_OUTPUT = RAW_GOLF_DIR / "osm_golf_courses_raw.gpkg"
PROCESSED_OUTPUT = PROCESSED_DIR / "golf_courses_ca_coastal.gpkg"

RAW_GOLF_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CRS_WGS84 = "EPSG:4326"
CRS_CA_ALBERS = "EPSG:3310"

TAGS = {"leisure": "golf_course"}

# Smaller = safer but slower.
CHUNK_DEGREES = 1.0
SLEEP_SECONDS = 2


def log(message: str) -> None:
    print(f"[09_download_or_prepare_golf_data] {message}")


def load_study_area_wgs84() -> gpd.GeoDataFrame:
    log(f"Reading study area: {STUDY_AREA_PATH}")
    study = gpd.read_file(STUDY_AREA_PATH)

    if study.empty:
        raise ValueError("Study area is empty.")

    if study.crs is None:
        raise ValueError("Study area CRS is missing.")

    study = study.to_crs(CRS_WGS84)
    study["geometry"] = study.geometry.make_valid()
    study = study[~study.geometry.is_empty].copy()

    study["__dissolve"] = 1
    study = study.dissolve(by="__dissolve").reset_index(drop=True)

    return study[["geometry"]]


def make_query_chunks(study: gpd.GeoDataFrame) -> list:
    study_geom = study.geometry.iloc[0]
    minx, miny, maxx, maxy = study.total_bounds

    chunks = []

    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            chunk = box(
                x,
                y,
                min(x + CHUNK_DEGREES, maxx),
                min(y + CHUNK_DEGREES, maxy),
            )

            if chunk.intersects(study_geom):
                chunks.append(chunk.intersection(study_geom))

            y += CHUNK_DEGREES
        x += CHUNK_DEGREES

    return chunks


def download_chunk(chunk_geom, chunk_number: int, total_chunks: int):
    log(f"Downloading chunk {chunk_number}/{total_chunks}")

    for attempt in range(1, 4):
        try:
            gdf = ox.features.features_from_polygon(
                chunk_geom,
                tags=TAGS,
            )

            if gdf.empty:
                log(f"Chunk {chunk_number}: no golf courses found.")
                return None

            gdf = gdf.reset_index()
            log(f"Chunk {chunk_number}: found {len(gdf):,} raw features.")
            return gdf

        except Exception as exc:
            log(f"Chunk {chunk_number} attempt {attempt} failed: {exc}")
            time.sleep(SLEEP_SECONDS * attempt)

    log(f"Chunk {chunk_number}: failed after 3 attempts. Skipping.")
    return None


def download_osm_golf_courses(study: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.requests_timeout = 600

    chunks = make_query_chunks(study)
    log(f"Created {len(chunks)} download chunks.")

    all_results = []

    for i, chunk in enumerate(chunks, start=1):
        result = download_chunk(chunk, i, len(chunks))

        if result is not None and not result.empty:
            all_results.append(result)

        time.sleep(SLEEP_SECONDS)

    if not all_results:
        raise ValueError("No OSM golf course features were downloaded.")

    golf = pd.concat(all_results, ignore_index=True)

    return gpd.GeoDataFrame(golf, geometry="geometry", crs=CRS_WGS84)


def clean_golf_data(golf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Cleaning golf data...")

    if golf.crs is None:
        golf = golf.set_crs(CRS_WGS84)

    golf = golf.to_crs(CRS_CA_ALBERS)
    golf["geometry"] = golf.geometry.make_valid()
    golf = golf[~golf.geometry.is_empty].copy()

    golf = golf[golf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    if golf.empty:
        raise ValueError("No polygon golf course features remained after cleaning.")

    # Remove duplicates from overlapping chunks.
    if "osmid" in golf.columns:
        golf = golf.drop_duplicates(subset=["osmid"])
    elif "id" in golf.columns:
        golf = golf.drop_duplicates(subset=["id"])
    else:
        golf = golf.drop_duplicates(subset=["geometry"])

    golf = golf.reset_index(drop=True)
    golf["golf_id"] = [f"GOLF{i:05d}" for i in range(1, len(golf) + 1)]

    for field in ["name", "leisure", "sport", "operator", "website"]:
        if field not in golf.columns:
            golf[field] = None

    golf["area_m2"] = golf.geometry.area
    golf = golf[golf["area_m2"] > 10_000].copy()

    keep_fields = [
        "golf_id",
        "name",
        "leisure",
        "sport",
        "operator",
        "website",
        "area_m2",
        "geometry",
    ]

    return golf[keep_fields]


def run_qa(golf: gpd.GeoDataFrame) -> None:
    log("Running QA checks...")

    if golf.empty:
        raise AssertionError("QA failed: golf dataset is empty.")

    if golf.crs.to_string() != CRS_CA_ALBERS:
        raise AssertionError(f"QA failed: expected {CRS_CA_ALBERS}, got {golf.crs}")

    if golf["golf_id"].duplicated().any():
        raise AssertionError("QA failed: duplicate golf_id values.")

    invalid_count = (~golf.geometry.is_valid).sum()
    if invalid_count > 0:
        raise AssertionError(f"QA failed: {invalid_count} invalid geometries.")

    log(f"Golf course polygons: {len(golf):,}")
    log(f"Total golf polygon area km²: {golf.geometry.area.sum() / 1_000_000:,.2f}")
    log("QA complete.")


def main() -> None:
    log("Starting Script 09: Download or prepare golf data")

    try:
        study = load_study_area_wgs84()

        if RAW_OUTPUT.exists():
            log(f"Raw OSM golf file already exists. Reading: {RAW_OUTPUT}")
            raw_golf = gpd.read_file(RAW_OUTPUT)
        else:
            raw_golf = download_osm_golf_courses(study)

            log(f"Writing raw OSM golf output: {RAW_OUTPUT}")
            raw_golf.to_file(
                RAW_OUTPUT,
                layer="osm_golf_courses_raw",
                driver="GPKG",
            )

        golf = clean_golf_data(raw_golf)
        run_qa(golf)

        log(f"Writing processed golf output: {PROCESSED_OUTPUT}")
        golf.to_file(
            PROCESSED_OUTPUT,
            layer="golf_courses_ca_coastal",
            driver="GPKG",
        )

        log("Complete.")
        log(f"Created: {PROCESSED_OUTPUT}")

    except Exception as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()