#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

SCRIPT_NAME = "193_update_rde_validation_registry_after_gnis_v01"

REGISTRY = Path("data/validation/rde_experiment_framework_registry_v01.csv")
OUTPUT_MD = Path("data/validation/rde_validation_experiment_registry_v02.md")

UPDATES = {
    "oregon_gnis_named_features_external_validation": {
        "status": "complete",
        "result_file": "data/validation/results/oregon_gnis_named_features_external_validation_summary_v01.csv",
        "notes": "Aggregate GNIS was not supported; dataset is heterogeneous and should be interpreted through subgroups.",
    },
    "oregon_validation_evidence_synthesis": {
        "status": "complete",
        "result_file": "data/validation/synthesis/oregon_experiment_synthesis_v01_experiment_level.csv",
        "notes": "Updated synthesis includes PAD-US, estuaries, aggregate GNIS, and GNIS subgroup experiments.",
    },
}

NEW_EXPERIMENTS = [
    {
        "experiment_key": "oregon_gnis_coastal_landforms_validation",
        "study_region": "Oregon Coast",
        "dataset_key": "gnis_coastal_landforms",
        "dataset_name": "GNIS Coastal Landforms",
        "validation_type": "spatial_null_model_validation",
        "validation_layer": "data/validation/standardized/gnis_subgroups/gnis_coastal_landform_oregon_v01.gpkg",
        "result_file": "data/validation/results/oregon_coast_gnis_coastal_landforms_null_model_comparison_v01.csv",
        "status": "complete",
        "phase": "phase_ii_statistical_validation",
        "scientific_question": "Are Oregon RDE transition hotspots closer to named coastal landforms than random hotspot placements?",
        "notes": "Strong support. Coastal landforms reverse the aggregate GNIS non-supportive result.",
    },
    {
        "experiment_key": "oregon_gnis_hydrology_validation",
        "study_region": "Oregon Coast",
        "dataset_key": "gnis_hydrology",
        "dataset_name": "GNIS Hydrology",
        "validation_type": "spatial_null_model_validation",
        "validation_layer": "data/validation/standardized/gnis_subgroups/gnis_hydrology_oregon_v01.gpkg",
        "result_file": "data/validation/results/oregon_coast_gnis_hydrology_null_model_comparison_v01.csv",
        "status": "complete",
        "phase": "phase_ii_statistical_validation",
        "scientific_question": "Are Oregon RDE transition hotspots closer to named hydrologic features than random hotspot placements?",
        "notes": "Not supported / inverse. This helps show RDE is not merely proximity to arbitrary named features.",
    },
    {
        "experiment_key": "oregon_gnis_terrain_validation",
        "study_region": "Oregon Coast",
        "dataset_key": "gnis_terrain",
        "dataset_name": "GNIS Terrain",
        "validation_type": "spatial_null_model_validation",
        "validation_layer": "data/validation/standardized/gnis_subgroups/gnis_terrain_oregon_v01.gpkg",
        "result_file": "data/validation/results/oregon_coast_gnis_terrain_null_model_comparison_v01.csv",
        "status": "complete",
        "phase": "phase_ii_statistical_validation",
        "scientific_question": "Are Oregon RDE transition hotspots closer to named terrain features than random hotspot placements?",
        "notes": "Not supported / inverse. Supports the refined hypothesis that recognition-bearing coastal features matter more than arbitrary terrain names.",
    },
    {
        "experiment_key": "oregon_gnis_distinct_natural_features_validation",
        "study_region": "Oregon Coast",
        "dataset_key": "gnis_distinct_natural_features",
        "dataset_name": "GNIS Distinct Natural Features",
        "validation_type": "spatial_null_model_validation",
        "validation_layer": "data/validation/standardized/gnis_subgroups/gnis_distinct_natural_feature_oregon_v01.gpkg",
        "result_file": "data/validation/results/oregon_coast_gnis_distinct_natural_features_null_model_comparison_v01.csv",
        "status": "complete",
        "phase": "phase_ii_statistical_validation",
        "scientific_question": "Are Oregon RDE transition hotspots closer to named distinct natural features than random hotspot placements?",
        "notes": "Mixed / heterogeneous. Small feature count; interpret cautiously.",
    },
]


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    df = pd.read_csv(REGISTRY)

    for key, updates in UPDATES.items():
        mask = df["experiment_key"] == key
        for col, value in updates.items():
            df.loc[mask, col] = value

    new = pd.DataFrame(NEW_EXPERIMENTS)
    df = df[~df["experiment_key"].isin(new["experiment_key"])]
    df = pd.concat([df, new], ignore_index=True)

    df["result_exists"] = df["result_file"].apply(lambda p: Path(p).exists() if isinstance(p, str) and p else False)
    df["validation_layer_exists"] = df["validation_layer"].apply(lambda p: Path(p).exists() if isinstance(p, str) and p else False)

    df.to_csv(REGISTRY, index=False)

    lines = [
        "# RDE Validation Experiment Registry v02",
        "",
        "Updated after GNIS aggregate and subgroup validation.",
        "",
        "## Status Counts",
        "",
        df["status"].value_counts().reset_index().to_markdown(index=False),
        "",
        "## Experiments",
        "",
        df.to_markdown(index=False),
    ]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[{SCRIPT_NAME}] Registry rows: {len(df):,}")
    print(df[["experiment_key", "dataset_name", "status", "result_exists"]].to_string(index=False))
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()