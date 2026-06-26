from dataclasses import dataclass, field
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
