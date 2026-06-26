from pathlib import Path
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
