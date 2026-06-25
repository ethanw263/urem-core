#!/usr/bin/env python3

from pathlib import Path
import geopandas as gpd
import pandas as pd

SCRIPT_NAME = "168_standardize_oregon_padus_protected_areas_v01"

PADUS_GDB = Path("data/raw/padus/PADUS4_1_State_OR_GDB_KMZ/PADUS4_1_StateOR.gdb")
STUDY_AREA = Path("data/processed/oregon_coast_study_area_v01.gpkg")

OUTPUT_DIR = Path("data/validation/standardized")
OUTPUT_GPKG = OUTPUT_DIR / "protected_areas_oregon_padus_v01.gpkg"
OUTPUT_CSV = OUTPUT_DIR / "protected_areas_oregon_padus_v01.csv"
OUTPUT_METADATA = OUTPUT_DIR / "protected_areas_oregon_padus_v01_metadata.csv"


PREFERRED_LAYER = "PADUS4_1Comb_DOD_Trib_NGP_Fee_Desig_Ease_State_OR"


def repair_geometries(gdf, label):
    print(f"[{SCRIPT_NAME}] Repairing geometries: {label}")

    gdf = gdf.copy()

    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        print(f"[{SCRIPT_NAME}] make_valid unavailable/failed; using buffer(0)")
        gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    return gdf


def choose_padus_layer(gdb_path: Path):
    layers = gpd.list_layers(gdb_path)

    print()
    print(f"[{SCRIPT_NAME}] Available PAD-US layers:")
    print(layers.to_string(index=False))

    names = layers["name"].tolist()

    if PREFERRED_LAYER in names:
        return PREFERRED_LAYER

    preferred_keywords = [
        "Comb",
        "Combined",
        "Fee",
        "Designation",
    ]

    for keyword in preferred_keywords:
        matches = [n for n in names if keyword.lower() in n.lower()]
        if matches:
            return matches[0]

    raise ValueError("Could not identify a usable PAD-US layer.")


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PADUS_GDB.exists():
        raise FileNotFoundError(f"Missing PAD-US GDB: {PADUS_GDB}")

    if not STUDY_AREA.exists():
        raise FileNotFoundError(f"Missing study area: {STUDY_AREA}")

    layer_name = choose_padus_layer(PADUS_GDB)

    print()
    print(f"[{SCRIPT_NAME}] Selected PAD-US layer: {layer_name}")

    padus = gpd.read_file(PADUS_GDB, layer=layer_name)
    study = gpd.read_file(STUDY_AREA)

    print(f"[{SCRIPT_NAME}] PAD-US rows before clip: {len(padus):,}")
    print(f"[{SCRIPT_NAME}] Study area rows: {len(study):,}")
    print(f"[{SCRIPT_NAME}] PAD-US CRS: {padus.crs}")
    print(f"[{SCRIPT_NAME}] Study CRS: {study.crs}")

    if padus.crs != study.crs:
        print(f"[{SCRIPT_NAME}] Reprojecting PAD-US to study CRS")
        padus = padus.to_crs(study.crs)

    padus = repair_geometries(padus, "PAD-US")
    study = repair_geometries(study, "study area")

    print(f"[{SCRIPT_NAME}] PAD-US rows after geometry repair: {len(padus):,}")

    print(f"[{SCRIPT_NAME}] Spatial prefiltering to study area bounds")

    minx, miny, maxx, maxy = study.total_bounds

    padus_prefilter = padus.cx[minx:maxx, miny:maxy].copy()

    print(f"[{SCRIPT_NAME}] PAD-US rows after bbox prefilter: {len(padus_prefilter):,}")

    print(f"[{SCRIPT_NAME}] Clipping PAD-US to study area")

    try:
        clipped = gpd.clip(
            padus_prefilter,
            study,
            keep_geom_type=False,
        )
    except Exception as e:
        print(f"[{SCRIPT_NAME}] gpd.clip failed: {e}")
        print(f"[{SCRIPT_NAME}] Retrying after buffer(0) repair")

        padus_prefilter["geometry"] = padus_prefilter.geometry.buffer(0)
        study["geometry"] = study.geometry.buffer(0)

        clipped = gpd.clip(
            padus_prefilter,
            study,
            keep_geom_type=False,
        )

    clipped = repair_geometries(clipped, "clipped PAD-US")

    clipped["validation_dataset_key"] = "protected_areas"
    clipped["validation_dataset_name"] = "Protected Areas"
    clipped["validation_source"] = "USGS PAD-US 4.1"
    clipped["validation_region"] = "Oregon Coast"
    clipped["independent_of_model"] = True
    clipped["standardized_validation_layer_v01"] = True
    clipped["source_padus_layer"] = layer_name
    clipped["area_km2_v01"] = clipped.geometry.area / 1_000_000

    print(f"[{SCRIPT_NAME}] PAD-US rows after clip: {len(clipped):,}")
    print(f"[{SCRIPT_NAME}] Total clipped area km2: {clipped['area_km2_v01'].sum():,.2f}")

    metadata = pd.DataFrame(
        [
            {
                "dataset_key": "protected_areas",
                "dataset_name": "Protected Areas",
                "source": "USGS PAD-US 4.1",
                "region": "Oregon Coast",
                "raw_path": str(PADUS_GDB),
                "selected_layer": layer_name,
                "study_area": str(STUDY_AREA),
                "output_gpkg": str(OUTPUT_GPKG),
                "feature_count": len(clipped),
                "total_area_km2": clipped["area_km2_v01"].sum(),
                "independent_of_model": True,
                "recommended_metrics": "intersects;overlap;distance",
                "notes": "First standardized external validation dataset for Phase II RDE Validation Framework.",
            }
        ]
    )

    print()
    print(f"[{SCRIPT_NAME}] Writing GPKG: {OUTPUT_GPKG}")
    clipped.to_file(OUTPUT_GPKG, driver="GPKG")

    print(f"[{SCRIPT_NAME}] Writing CSV: {OUTPUT_CSV}")
    clipped.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Writing metadata: {OUTPUT_METADATA}")
    metadata.to_csv(OUTPUT_METADATA, index=False)

    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()