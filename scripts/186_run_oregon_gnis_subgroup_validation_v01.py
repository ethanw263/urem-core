#!/usr/bin/env python3

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.datasets import ValidationDataset
from src.validation.pipeline import ValidationPipelineRunner


SCRIPT_NAME = "186_run_oregon_gnis_subgroup_validation_v01"

HOTSPOTS = "data/processed/oregon_transition_hotspot_validation_summary_v01.gpkg"
STUDY_AREA = "data/processed/oregon_coast_study_area_v01.gpkg"

GNIS_COASTAL = "data/validation/standardized/gnis_subgroups/gnis_coastal_landform_oregon_v01.gpkg"

N_SIMULATIONS = 500


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    runner = ValidationPipelineRunner(
        study_name="Oregon Coast",
        hotspots_path=HOTSPOTS,
        study_area_path=STUDY_AREA,
        hotspot_id_column="transition_hotspot_id_v01",
        hotspot_score_column="mean_transition_score_v01",
        output_dir="data/validation/results",
    )

    dataset = ValidationDataset(
        name="GNIS Coastal Landforms",
        path=GNIS_COASTAL,
        geometry_type="point",
        source="USGS GNIS FullModel / Gazetteer OR",
        metrics=[
            "distance",
            "density",
        ],
        weight=0.15,
        id_column="feature_id",
        category="geographic_recognition",
        independent_of_model=True,
        notes="GNIS subgroup validation for coastal landform features: Island, Cape, Bay, Channel, Bar, Beach, Gut.",
    )

    print(f"[{SCRIPT_NAME}] Running full validation pipeline")
    print(f"[{SCRIPT_NAME}] Dataset: {dataset.name}")
    print(f"[{SCRIPT_NAME}] Simulations: {N_SIMULATIONS:,}")

    outputs = runner.run_full_validation(
        dataset=dataset,
        n_simulations=N_SIMULATIONS,
        random_seed=42,
        progress_interval=50,
    )

    print()
    print(f"[{SCRIPT_NAME}] Outputs:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")

    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()