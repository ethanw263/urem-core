#!/usr/bin/env python3
"""
Script 60: Download and Prepare PAD-US California Protected Areas

Purpose:
- Download PAD-US data manually (one-time).
- Extract California protected lands.
- Save lightweight California-only layer.

Input:
- data/raw/padus/PADUS4_0.gdb

Outputs:
- data/processed/california_protected_areas.gpkg
"""

from pathlib import Path
import geopandas as gpd

SCRIPT_NAME = "60_download_prepare_padus"

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_GDB = (
    BASE_DIR
    / "data/raw/padus/PADUS4_1_State_CA_GDB_KMZ"
    / "PADUS4_1_StateCA.gdb"
)

OUTPUT_GPKG = BASE_DIR / "data/processed/california_protected_areas.gpkg"

CALIFORNIA_BOUNDS = (
    -124.5,
    32.0,
    -114.0,
    42.2,
)


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def main():

    log("Starting PAD-US preparation")

    if not INPUT_GDB.exists():
        raise FileNotFoundError(
            f"""
PAD-US geodatabase not found.

Download PAD-US from:

https://www.usgs.gov/programs/gap-analysis-project/pad-us-data-download

Extract into:

data/raw/padus/

Expected:
data/raw/padus/PADUS4_0.gdb
"""
        )

    layers = gpd.list_layers(INPUT_GDB)

    print("\nAvailable layers:")
    print(layers)

    target_layer = None

    for name in layers["name"]:
        if "Combined" in name:
            target_layer = name
            break

    if target_layer is None:
        target_layer = layers.iloc[0]["name"]

    log(f"Reading layer: {target_layer}")

    gdf = gpd.read_file(
        INPUT_GDB,
        layer=target_layer,
    )

    log(f"Rows before clip: {len(gdf):,}")

    gdf = gdf.to_crs("EPSG:4326")

    xmin, ymin, xmax, ymax = CALIFORNIA_BOUNDS

    gdf = gdf.cx[xmin:xmax, ymin:ymax]

    log(f"Rows after California clip: {len(gdf):,}")

    gdf = gdf.to_crs("EPSG:3310")

    OUTPUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

    log(f"Writing: {OUTPUT_GPKG}")

    gdf.to_file(
        OUTPUT_GPKG,
        driver="GPKG",
    )

    log("Done")


if __name__ == "__main__":
    main()