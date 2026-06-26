#!/usr/bin/env python3

from pathlib import Path
import geopandas as gpd
import pandas as pd

SCRIPT_NAME = "181_standardize_oregon_gnis_named_natural_features_v01"

GNIS_GPKG = Path("data/raw/validation/gnis/Gazetteer_OR_GPKG/Gazetteer_OR_GPKG.gpkg")
GNIS_LAYER = "Gaz_Features"
STUDY_AREA = Path("data/processed/oregon_coast_study_area_v01.gpkg")

OUTPUT_DIR = Path("data/validation/standardized")
OUTPUT_GPKG = OUTPUT_DIR / "gnis_named_natural_features_oregon_v01.gpkg"
OUTPUT_CSV = OUTPUT_DIR / "gnis_named_natural_features_oregon_v01.csv"
OUTPUT_METADATA = OUTPUT_DIR / "gnis_named_natural_features_oregon_v01_metadata.csv"

NATURAL_CLASSES = {
    "Stream", "Spring", "Summit", "Reservoir", "Valley", "Lake", "Flat",
    "Ridge", "Island", "Gap", "Basin", "Falls", "Cape", "Pillar",
    "Cliff", "Swamp", "Rapids", "Bar", "Bay", "Channel", "Bend",
    "Range", "Beach", "Gut", "Slope", "Glacier", "Bench", "Crater",
    "Plain", "Lava", "Arch", "Woods"
}


def classify_group(cls):
    if cls in {"Beach", "Cape", "Bay", "Island", "Bar", "Channel", "Gut"}:
        return "coastal_landform"
    if cls in {"Stream", "Spring", "Reservoir", "Lake", "Falls", "Swamp", "Rapids"}:
        return "hydrology"
    if cls in {"Summit", "Valley", "Flat", "Ridge", "Gap", "Basin", "Range", "Slope", "Bench", "Plain"}:
        return "terrain"
    if cls in {"Cliff", "Pillar", "Crater", "Lava", "Arch", "Glacier", "Woods"}:
        return "distinct_natural_feature"
    return "other_natural_feature"


def main():
    print(f"[{SCRIPT_NAME}] Starting")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gnis = gpd.read_file(GNIS_GPKG, layer=GNIS_LAYER)
    study = gpd.read_file(STUDY_AREA)

    print(f"[{SCRIPT_NAME}] GNIS rows before filtering: {len(gnis):,}")
    print(f"[{SCRIPT_NAME}] Study CRS: {study.crs}")
    print(f"[{SCRIPT_NAME}] GNIS CRS: {gnis.crs}")

    if "feature_class" not in gnis.columns:
        raise ValueError("Expected field not found: feature_class")

    natural = gnis[gnis["feature_class"].isin(NATURAL_CLASSES)].copy()
    print(f"[{SCRIPT_NAME}] Natural-feature rows before coastal clip: {len(natural):,}")

    if "prim_long_dec" in natural.columns and "prim_lat_dec" in natural.columns:
        natural = gpd.GeoDataFrame(
            natural,
            geometry=gpd.points_from_xy(natural["prim_long_dec"], natural["prim_lat_dec"]),
            crs="EPSG:4326",
        )
    else:
        print(f"[{SCRIPT_NAME}] Using existing geometry because primary coordinate fields were not found")

    if natural.crs != study.crs:
        natural = natural.to_crs(study.crs)

    study_union = study.geometry.union_all()
    clipped = natural[natural.intersects(study_union)].copy()

    clipped["validation_dataset_key"] = "named_natural_features"
    clipped["validation_dataset_name"] = "Named Natural Features"
    clipped["validation_source"] = "USGS GNIS FullModel / Gazetteer OR"
    clipped["validation_region"] = "Oregon Coast"
    clipped["independent_of_model"] = True
    clipped["standardized_validation_layer_v01"] = True
    clipped["gnis_feature_group_v01"] = clipped["feature_class"].apply(classify_group)

    print(f"[{SCRIPT_NAME}] GNIS rows after coastal clip: {len(clipped):,}")

    print()
    print(f"[{SCRIPT_NAME}] Feature class counts:")
    print(clipped["feature_class"].value_counts().head(40))

    print()
    print(f"[{SCRIPT_NAME}] Feature group counts:")
    print(clipped["gnis_feature_group_v01"].value_counts())

    metadata = pd.DataFrame([{
        "dataset_key": "named_natural_features",
        "dataset_name": "Named Natural Features",
        "source": "USGS GNIS FullModel / Gazetteer OR",
        "region": "Oregon Coast",
        "raw_path": str(GNIS_GPKG),
        "selected_layer": GNIS_LAYER,
        "study_area": str(STUDY_AREA),
        "output_gpkg": str(OUTPUT_GPKG),
        "feature_count": len(clipped),
        "independent_of_model": True,
        "recommended_metrics": "distance;density",
        "notes": "Standardized GNIS named natural features for Oregon external validation."
    }])

    print(f"[{SCRIPT_NAME}] Writing GPKG: {OUTPUT_GPKG}")
    clipped.to_file(OUTPUT_GPKG, driver="GPKG")

    print(f"[{SCRIPT_NAME}] Writing CSV: {OUTPUT_CSV}")
    clipped.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Writing metadata: {OUTPUT_METADATA}")
    metadata.to_csv(OUTPUT_METADATA, index=False)

    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()