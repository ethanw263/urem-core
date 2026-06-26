#!/usr/bin/env python3

from pathlib import Path
import geopandas as gpd
import pandas as pd

SCRIPT_NAME = "185_build_gnis_subgroup_validation_layers_v01"

GNIS_STANDARDIZED = Path("data/validation/standardized/gnis_named_natural_features_oregon_v01.gpkg")

OUTPUT_DIR = Path("data/validation/standardized/gnis_subgroups")
OUTPUT_METADATA = OUTPUT_DIR / "gnis_subgroup_validation_layers_metadata_v01.csv"

GROUP_FIELD = "gnis_feature_group_v01"

GROUPS = [
    "coastal_landform",
    "hydrology",
    "terrain",
    "distinct_natural_feature",
    "other_natural_feature",
]


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not GNIS_STANDARDIZED.exists():
        raise FileNotFoundError(f"Missing standardized GNIS layer: {GNIS_STANDARDIZED}")

    gnis = gpd.read_file(GNIS_STANDARDIZED)

    if GROUP_FIELD not in gnis.columns:
        raise ValueError(f"Missing required field: {GROUP_FIELD}")

    print(f"[{SCRIPT_NAME}] Input GNIS rows: {len(gnis):,}")

    metadata_rows = []

    for group in GROUPS:
        sub = gnis[gnis[GROUP_FIELD] == group].copy()

        if sub.empty:
            print(f"[{SCRIPT_NAME}] Skipping empty group: {group}")
            continue

        output_gpkg = OUTPUT_DIR / f"gnis_{group}_oregon_v01.gpkg"
        output_csv = OUTPUT_DIR / f"gnis_{group}_oregon_v01.csv"

        print()
        print(f"[{SCRIPT_NAME}] Group: {group}")
        print(f"[{SCRIPT_NAME}] Rows: {len(sub):,}")
        print(f"[{SCRIPT_NAME}] Top feature classes:")
        print(sub["feature_class"].value_counts().head(20).to_string())

        sub["validation_dataset_key"] = f"gnis_{group}"
        sub["validation_dataset_name"] = f"GNIS {group.replace('_', ' ').title()}"
        sub["validation_source"] = "USGS GNIS FullModel / Gazetteer OR"
        sub["validation_region"] = "Oregon Coast"
        sub["independent_of_model"] = True
        sub["standardized_validation_layer_v01"] = True

        print(f"[{SCRIPT_NAME}] Writing GPKG: {output_gpkg}")
        sub.to_file(output_gpkg, driver="GPKG")

        print(f"[{SCRIPT_NAME}] Writing CSV: {output_csv}")
        sub.drop(columns="geometry").to_csv(output_csv, index=False)

        metadata_rows.append({
            "dataset_key": f"gnis_{group}",
            "dataset_name": f"GNIS {group.replace('_', ' ').title()}",
            "feature_group": group,
            "feature_count": len(sub),
            "source": "USGS GNIS FullModel / Gazetteer OR",
            "output_gpkg": str(output_gpkg),
            "output_csv": str(output_csv),
            "recommended_metrics": "distance;density",
            "independent_of_model": True,
        })

    metadata = pd.DataFrame(metadata_rows)

    print()
    print(f"[{SCRIPT_NAME}] Writing metadata: {OUTPUT_METADATA}")
    metadata.to_csv(OUTPUT_METADATA, index=False)

    print()
    print(f"[{SCRIPT_NAME}] Subgroup summary:")
    print(metadata[["dataset_key", "feature_count", "output_gpkg"]].to_string(index=False))

    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()