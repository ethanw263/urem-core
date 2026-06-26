#!/usr/bin/env python3

from pathlib import Path

SCRIPT_NAME = "187_build_rde_experiment_framework_v01"

BASE = Path("src/validation/experiments")
BASE.mkdir(parents=True, exist_ok=True)

FILES = {
"__init__.py": "",
"experiment.py": r'''from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ValidationExperiment:
    """
    Reusable experiment definition for RDE validation work.
    """

    experiment_key: str
    experiment_name: str
    study_name: str

    hotspots_path: str
    study_area_path: str
    dataset_path: str

    dataset_name: str
    dataset_source: str
    dataset_category: str
    geometry_type: str
    metrics: List[str]

    hotspot_id_column: str = "transition_hotspot_id_v01"
    hotspot_score_column: str = "mean_transition_score_v01"

    independent_of_model: bool = True
    dataset_weight: float = 1.0

    run_external_validation: bool = True
    run_null_model: bool = True

    n_simulations: int = 500
    random_seed: int = 42
    progress_interval: int = 50

    notes: str = ""

    def validate(self) -> None:
        if not self.experiment_key:
            raise ValueError("experiment_key is required")

        if not self.dataset_path:
            raise ValueError("dataset_path is required")

        if not self.metrics:
            raise ValueError("At least one validation metric is required")
''',

"runner.py": r'''from pathlib import Path
from typing import Dict

import pandas as pd

from src.validation.datasets import ValidationDataset
from src.validation.pipeline import ValidationPipelineRunner

from .experiment import ValidationExperiment


class RDEExperimentRunner:
    """
    Runs ValidationExperiment definitions through the reusable validation pipeline.
    """

    def __init__(self, output_dir: str = "data/validation/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, experiment: ValidationExperiment) -> Dict[str, Path]:
        experiment.validate()

        dataset = ValidationDataset(
            name=experiment.dataset_name,
            path=experiment.dataset_path,
            geometry_type=experiment.geometry_type,
            source=experiment.dataset_source,
            metrics=experiment.metrics,
            weight=experiment.dataset_weight,
            category=experiment.dataset_category,
            independent_of_model=experiment.independent_of_model,
            notes=experiment.notes,
        )

        runner = ValidationPipelineRunner(
            study_name=experiment.study_name,
            hotspots_path=experiment.hotspots_path,
            study_area_path=experiment.study_area_path,
            hotspot_id_column=experiment.hotspot_id_column,
            hotspot_score_column=experiment.hotspot_score_column,
            output_dir=str(self.output_dir),
        )

        outputs = {}

        if experiment.run_external_validation and experiment.run_null_model:
            outputs.update(
                runner.run_full_validation(
                    dataset=dataset,
                    n_simulations=experiment.n_simulations,
                    random_seed=experiment.random_seed,
                    progress_interval=experiment.progress_interval,
                )
            )

        elif experiment.run_external_validation:
            outputs.update(runner.run_external_validation(dataset))

        elif experiment.run_null_model:
            outputs.update(
                runner.run_null_model_validation(
                    dataset=dataset,
                    n_simulations=experiment.n_simulations,
                    random_seed=experiment.random_seed,
                    progress_interval=experiment.progress_interval,
                )
            )

        return outputs
''',

"registry.py": r'''from pathlib import Path
import pandas as pd

from .experiment import ValidationExperiment


def append_experiment_registry(
    experiment: ValidationExperiment,
    outputs: dict,
    registry_path: str = "data/validation/rde_experiment_framework_registry_v01.csv",
) -> None:
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "experiment_key": experiment.experiment_key,
        "experiment_name": experiment.experiment_name,
        "study_name": experiment.study_name,
        "dataset_name": experiment.dataset_name,
        "dataset_source": experiment.dataset_source,
        "dataset_category": experiment.dataset_category,
        "metrics": ";".join(experiment.metrics),
        "run_external_validation": experiment.run_external_validation,
        "run_null_model": experiment.run_null_model,
        "n_simulations": experiment.n_simulations,
        "status": "complete",
        "notes": experiment.notes,
    }

    for k, v in outputs.items():
        row[k] = str(v)

    if path.exists():
        df = pd.read_csv(path)
        df = df[df["experiment_key"] != experiment.experiment_key]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(path, index=False)
''',

"reporting.py": r'''from pathlib import Path


def write_experiment_summary(
    experiment,
    outputs,
    output_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    lines.append(f"# {experiment.experiment_name}")
    lines.append("")
    lines.append(f"- Experiment key: `{experiment.experiment_key}`")
    lines.append(f"- Study: {experiment.study_name}")
    lines.append(f"- Dataset: {experiment.dataset_name}")
    lines.append(f"- Source: {experiment.dataset_source}")
    lines.append(f"- Category: {experiment.dataset_category}")
    lines.append(f"- Metrics: {', '.join(experiment.metrics)}")
    lines.append(f"- Null model simulations: {experiment.n_simulations}")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")

    for key, value in outputs.items():
        lines.append(f"- `{key}`: `{value}`")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(experiment.notes or "No notes provided.")

    path.write_text("\n".join(lines), encoding="utf-8")
'''
}


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    for name, content in FILES.items():
        path = BASE / name
        path.write_text(content, encoding="utf-8")
        print(f"[{SCRIPT_NAME}] Wrote: {path}")

    print()
    print(f"[{SCRIPT_NAME}] Experiment framework created.")
    print("Next: create a driver script that runs GNIS Coastal Landforms through this framework.")
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()