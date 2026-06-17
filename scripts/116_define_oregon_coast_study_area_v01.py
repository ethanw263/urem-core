#!/usr/bin/env python3
"""
Script 116: Define Oregon Coast Study Area v01
"""

from pathlib import Path
import geopandas as gpd

SCRIPT_NAME = "116_define_oregon_coast_study_area_v01"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

STATES_PATH = RAW_DIR / "padus" / "PADUS4_1_State_CA_GDB_KMZ" / "tl_2022_us_state.shp"
COASTLINE_PATH = RAW_DIR / "coastline" / "ne_10m_coastline.shp"

OUTPUT_PATH = PROCESSED_DIR / "oregon_coast_study_area_v01.gpkg"

TARGET_CRS = "EPSG:5070"
COAST_BUFFER_METERS = 25_000


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def main():
    log("Starting Oregon Coast study area definition")

    if not STATES_PATH.exists():
        raise FileNotFoundError(f"Missing states file: {STATES_PATH}")

    if not COASTLINE_PATH.exists():
        raise FileNotFoundError(f"Missing coastline file: {COASTLINE_PATH}")

    log(f"Reading states: {STATES_PATH}")
    states = gpd.read_file(STATES_PATH)

    log(f"Reading coastline: {COASTLINE_PATH}")
    coastline = gpd.read_file(COASTLINE_PATH)

    states = states.to_crs(TARGET_CRS)
    coastline = coastline.to_crs(TARGET_CRS)

    if "NAME" not in states.columns:
        raise ValueError(f"Expected NAME column not found. Columns: {list(states.columns)}")

    oregon = states[states["NAME"].str.lower() == "oregon"].copy()

    if oregon.empty:
        raise ValueError("Could not find Oregon in state boundary file.")

    oregon = oregon.dissolve()
    oregon["state"] = "Oregon"

    log("Creating 25 km coastline buffer")
    coast_buffer = coastline.copy()
    coast_buffer["geometry"] = coast_buffer.geometry.buffer(COAST_BUFFER_METERS)
    coast_buffer = coast_buffer.dissolve()

    log("Intersecting Oregon with coastline buffer")
    study_area = gpd.overlay(oregon, coast_buffer, how="intersection")

    if study_area.empty:
        raise ValueError("Oregon coast study area is empty.")

    study_area = study_area[["state", "geometry"]].copy()
    study_area["domain"] = "oregon_coast"
    study_area["buffer_m"] = COAST_BUFFER_METERS
    study_area["crs"] = TARGET_CRS
    study_area["area_sq_km"] = study_area.geometry.area / 1_000_000

    log(f"Area: {study_area['area_sq_km'].sum():,.2f} sq km")
    log(f"Writing: {OUTPUT_PATH}")

    study_area.to_file(
        OUTPUT_PATH,
        layer="oregon_coast_study_area_v01",
        driver="GPKG",
    )

    log("Done")


if __name__ == "__main__":
    main()