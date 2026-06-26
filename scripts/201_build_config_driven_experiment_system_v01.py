#!/usr/bin/env python3

from pathlib import Path

SCRIPT_NAME = "201_build_config_driven_experiment_system_v01"

CONFIG_DIR = Path("configs/experiments")
EXPERIMENT_CONFIG_PY = Path("src/validation/experiments/config.py")

CONFIG_CONTENT = r'''from pathlib import Path
import json

try:
    import yaml
except ImportError:
    yaml = None

from .experiment import ValidationExperiment


def load_experiment_config(path: str) -> ValidationExperiment:
    """
    Load a ValidationExperiment from YAML or JSON.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")

    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError(
                "PyYAML is required for YAML experiment configs. "
                "Install with: pip install pyyaml"
            )
        data = yaml.safe_load(text)

    elif path.suffix.lower() == ".json":
        data = json.loads(text)

    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    return ValidationExperiment(**data)
'''

EXAMPLE_YAML = """experiment_key: oregon_gnis_coastal_landforms_config_test
experiment_name: Oregon GNIS Coastal Landforms Config Test
study_name: Oregon Coast

hotspots_path: data/processed/oregon_transition_hotspot_validation_summary_v01.gpkg
study_area_path: data/processed/oregon_coast_study_area_v01.gpkg
dataset_path: data/validation/standardized/gnis_subgroups/gnis_coastal_landform_oregon_v01.gpkg

dataset_name: GNIS Coastal Landforms
dataset_source: USGS GNIS FullModel / Gazetteer OR
dataset_category: geographic_recognition
geometry_type: point
metrics:
  - distance
  - density

hotspot_id_column: transition_hotspot_id_v01
hotspot_score_column: mean_transition_score_v01

independent_of_model: true
dataset_weight: 0.15

run_external_validation: true
run_null_model: true

n_simulations: 500
random_seed: 42
progress_interval: 50

notes: >
  Config-driven test experiment for GNIS coastal landforms.
  This proves experiments can be defined outside Python.
"""


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_CONFIG_PY.parent.mkdir(parents=True, exist_ok=True)

    EXPERIMENT_CONFIG_PY.write_text(CONFIG_CONTENT, encoding="utf-8")

    example_path = CONFIG_DIR / "oregon_gnis_coastal_landforms_v01.yaml"
    example_path.write_text(EXAMPLE_YAML, encoding="utf-8")

    print(f"[{SCRIPT_NAME}] Wrote: {EXPERIMENT_CONFIG_PY}")
    print(f"[{SCRIPT_NAME}] Wrote: {example_path}")
    print()
    print("Next:")
    print("  Add CLI support for running an experiment config.")
    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()