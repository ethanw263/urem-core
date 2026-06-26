#!/usr/bin/env python3

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.experiments.experiment import ValidationExperiment
from src.validation.experiments.runner import RDEExperimentRunner
from src.validation.experiments.registry import append_experiment_registry
from src.validation.experiments.reporting import write_experiment_summary


SCRIPT_NAME = "189_run_oregon_gnis_remaining_subgroup_experiments_v01"

HOTSPOTS = "data/processed/oregon_transition_hotspot_validation_summary_v01.gpkg"
STUDY_AREA = "data/processed/oregon_coast_study_area_v01.gpkg"

OUTPUT_DIR = "data/validation/results"
SUMMARY_DIR = Path("data/validation/experiments")

N_SIMULATIONS = 500


EXPERIMENTS = [
    {
        "experiment_key": "oregon_gnis_hydrology",
        "experiment_name": "Oregon GNIS Hydrology Validation",
        "dataset_path": "data/validation/standardized/gnis_subgroups/gnis_hydrology_oregon_v01.gpkg",
        "dataset_name": "GNIS Hydrology",
        "dataset_category": "geographic_recognition_hydrology",
        "notes": (
            "Tests whether Oregon RDE transition hotspots are closer to GNIS "
            "hydrologic named features than expected under randomized hotspot "
            "placement. Includes Stream, Lake, Reservoir, Rapids, Spring, Falls, and Swamp."
        ),
    },
    {
        "experiment_key": "oregon_gnis_terrain",
        "experiment_name": "Oregon GNIS Terrain Validation",
        "dataset_path": "data/validation/standardized/gnis_subgroups/gnis_terrain_oregon_v01.gpkg",
        "dataset_name": "GNIS Terrain",
        "dataset_category": "geographic_recognition_terrain",
        "notes": (
            "Tests whether Oregon RDE transition hotspots are closer to GNIS "
            "terrain named features than expected under randomized hotspot "
            "placement. Includes Summit, Ridge, Valley, Flat, Gap, Basin, Slope, and Bench."
        ),
    },
    {
        "experiment_key": "oregon_gnis_distinct_natural_features",
        "experiment_name": "Oregon GNIS Distinct Natural Features Validation",
        "dataset_path": "data/validation/standardized/gnis_subgroups/gnis_distinct_natural_feature_oregon_v01.gpkg",
        "dataset_name": "GNIS Distinct Natural Features",
        "dataset_category": "geographic_recognition_distinct_natural_feature",
        "notes": (
            "Tests whether Oregon RDE transition hotspots are closer to GNIS "
            "distinct natural features than expected under randomized hotspot "
            "placement. Includes Pillar, Cliff, and Arch."
        ),
    },
]


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    runner = RDEExperimentRunner(
        output_dir=OUTPUT_DIR,
    )

    for spec in EXPERIMENTS:
        print()
        print(f"[{SCRIPT_NAME}] Running: {spec['experiment_name']}")

        experiment = ValidationExperiment(
            experiment_key=spec["experiment_key"],
            experiment_name=spec["experiment_name"],
            study_name="Oregon Coast",

            hotspots_path=HOTSPOTS,
            study_area_path=STUDY_AREA,
            dataset_path=spec["dataset_path"],

            dataset_name=spec["dataset_name"],
            dataset_source="USGS GNIS FullModel / Gazetteer OR",
            dataset_category=spec["dataset_category"],
            geometry_type="point",
            metrics=["distance", "density"],

            independent_of_model=True,
            dataset_weight=0.15,

            run_external_validation=True,
            run_null_model=True,

            n_simulations=N_SIMULATIONS,
            random_seed=42,
            progress_interval=50,

            notes=spec["notes"],
        )

        outputs = runner.run(experiment)

        append_experiment_registry(
            experiment=experiment,
            outputs=outputs,
        )

        summary_md = SUMMARY_DIR / f"{spec['experiment_key']}_experiment_summary_v01.md"

        write_experiment_summary(
            experiment=experiment,
            outputs=outputs,
            output_path=str(summary_md),
        )

        print(f"[{SCRIPT_NAME}] Outputs:")
        for key, value in outputs.items():
            print(f"  {key}: {value}")

        print(f"[{SCRIPT_NAME}] Summary: {summary_md}")

    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()