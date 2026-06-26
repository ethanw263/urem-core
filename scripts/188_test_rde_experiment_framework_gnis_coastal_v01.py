#!/usr/bin/env python3

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.experiments.experiment import ValidationExperiment
from src.validation.experiments.runner import RDEExperimentRunner
from src.validation.experiments.registry import append_experiment_registry
from src.validation.experiments.reporting import write_experiment_summary


SCRIPT_NAME = "188_test_rde_experiment_framework_gnis_coastal_v01"

SUMMARY_MD = Path(
    "data/validation/experiments/oregon_gnis_coastal_landforms_experiment_summary_v01.md"
)


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    experiment = ValidationExperiment(
        experiment_key="oregon_gnis_coastal_landforms",
        experiment_name="Oregon GNIS Coastal Landforms Validation",
        study_name="Oregon Coast",

        hotspots_path="data/processed/oregon_transition_hotspot_validation_summary_v01.gpkg",
        study_area_path="data/processed/oregon_coast_study_area_v01.gpkg",
        dataset_path="data/validation/standardized/gnis_subgroups/gnis_coastal_landform_oregon_v01.gpkg",

        dataset_name="GNIS Coastal Landforms",
        dataset_source="USGS GNIS FullModel / Gazetteer OR",
        dataset_category="geographic_recognition",
        geometry_type="point",
        metrics=["distance", "density"],

        independent_of_model=True,
        dataset_weight=0.15,

        run_external_validation=True,
        run_null_model=True,

        n_simulations=500,
        random_seed=42,
        progress_interval=50,

        notes=(
            "Tests whether Oregon RDE transition hotspots are closer to "
            "officially named coastal landforms than expected under randomized "
            "hotspot placement. Coastal landforms include Island, Cape, Bay, "
            "Channel, Bar, Beach, and Gut."
        ),
    )

    runner = RDEExperimentRunner(
        output_dir="data/validation/results",
    )

    outputs = runner.run(experiment)

    append_experiment_registry(
        experiment=experiment,
        outputs=outputs,
    )

    write_experiment_summary(
        experiment=experiment,
        outputs=outputs,
        output_path=str(SUMMARY_MD),
    )

    print()
    print(f"[{SCRIPT_NAME}] Outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")

    print()
    print(f"[{SCRIPT_NAME}] Registry updated:")
    print("  data/validation/rde_experiment_framework_registry_v01.csv")

    print()
    print(f"[{SCRIPT_NAME}] Summary written:")
    print(f"  {SUMMARY_MD}")

    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()