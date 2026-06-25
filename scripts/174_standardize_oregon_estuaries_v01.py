#!/usr/bin/env python3

from pathlib import Path
import geopandas as gpd
import pandas as pd

SCRIPT_NAME = "174_standardize_oregon_estuaries_v01"

ESTUARY_GDB = Path("data/validation/estuaries/PMEP_West_Coast_USA_Estuary_Extent_V1.gdb")
ESTUARY_LAYER = "PMEP_West_Coast_USA_Estuary_Extent_V1"

STUDY_AREA = Path("data/processed/oregon_coast_study_area_v01.gpkg")

OUTPUT_DIR = Path("data/validation/standardized")
OUTPUT_GPKG = OUTPUT_DIR / "estuaries_oregon_pmep_v01.gpkg"
OUTPUT_CSV = OUTPUT_DIR / "estuaries_oregon_pmep_v01.csv"
OUTPUT_METADATA = OUTPUT_DIR / "estuaries_oregon_pmep_v01_metadata.csv"


def repair_geometries(gdf, label):
    print(f"[{SCRIPT_NAME}] Repairing geometries: {label}")

    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    return gdf


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not ESTUARY_GDB.exists():
        raise FileNotFoundError(f"Missing estuary GDB: {ESTUARY_GDB}")

    if not STUDY_AREA.exists():
        raise FileNotFoundError(f"Missing study area: {STUDY_AREA}")

    estuaries = gpd.read_file(ESTUARY_GDB, layer=ESTUARY_LAYER)
    study = gpd.read_file(STUDY_AREA)

    print(f"[{SCRIPT_NAME}] Estuary rows before clip: {len(estuaries):,}")
    print(f"[{SCRIPT_NAME}] Study area rows: {len(study):,}")
    print(f"[{SCRIPT_NAME}] Estuary CRS: {estuaries.crs}")
    print(f"[{SCRIPT_NAME}] Study CRS: {study.crs}")

    if estuaries.crs != study.crs:
        print(f"[{SCRIPT_NAME}] Reprojecting estuaries to study CRS")
        estuaries = estuaries.to_crs(study.crs)

    estuaries = repair_geometries(estuaries, "PMEP estuaries")
    study = repair_geometries(study, "study area")

    minx, miny, maxx, maxy = study.total_bounds
    estuary_prefilter = estuaries.cx[minx:maxx, miny:maxy].copy()

    print(f"[{SCRIPT_NAME}] Estuary rows after bbox prefilter: {len(estuary_prefilter):,}")

    print(f"[{SCRIPT_NAME}] Clipping estuaries to study area")

    clipped = gpd.clip(
        estuary_prefilter,
        study,
        keep_geom_type=False,
    )

    clipped = repair_geometries(clipped, "clipped estuaries")

    clipped["validation_dataset_key"] = "estuaries"
    clipped["validation_dataset_name"] = "Estuaries"
    clipped["validation_source"] = "PMEP West Coast USA Estuary Extent V1"
    clipped["validation_region"] = "Oregon Coast"
    clipped["independent_of_model"] = True
    clipped["standardized_validation_layer_v01"] = True
    clipped["source_estuary_layer"] = ESTUARY_LAYER
    clipped["area_km2_v01"] = clipped.geometry.area / 1_000_000

    print(f"[{SCRIPT_NAME}] Estuary rows after clip: {len(clipped):,}")
    print(f"[{SCRIPT_NAME}] Total clipped estuary area km2: {clipped['area_km2_v01'].sum():,.2f}")

    metadata = pd.DataFrame(
        [
            {
                "dataset_key": "estuaries",
                "dataset_name": "Estuaries",
                "source": "PMEP West Coast USA Estuary Extent V1",
                "region": "Oregon Coast",
                "raw_path": str(ESTUARY_GDB),
                "selected_layer": ESTUARY_LAYER,
                "study_area": str(STUDY_AREA),
                "output_gpkg": str(OUTPUT_GPKG),
                "feature_count": len(clipped),
                "total_area_km2": clipped["area_km2_v01"].sum(),
                "independent_of_model": True,
                "recommended_metrics": "intersects;overlap;distance",
                "notes": "Standardized PMEP estuary layer for Oregon external validation.",
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