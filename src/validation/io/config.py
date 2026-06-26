from dataclasses import dataclass
from pathlib import Path
from typing import List

from .paths import (
    OREGON_STUDY_AREA,
    OREGON_TRANSITION_HOTSPOTS,
    VALIDATION_RESULTS_DIR,
)


@dataclass(frozen=True)
class StudyConfig:
    study_name: str
    study_area_path: Path
    hotspots_path: Path
    hotspot_id_column: str
    hotspot_score_column: str
    output_dir: Path


OREGON_VALIDATION_CONFIG = StudyConfig(
    study_name="Oregon Coast",
    study_area_path=OREGON_STUDY_AREA,
    hotspots_path=OREGON_TRANSITION_HOTSPOTS,
    hotspot_id_column="transition_hotspot_id_v01",
    hotspot_score_column="mean_transition_score_v01",
    output_dir=VALIDATION_RESULTS_DIR,
)


@dataclass(frozen=True)
class ExperimentDefaults:
    n_simulations: int = 500
    random_seed: int = 42
    progress_interval: int = 50
    null_model_type: str = "random_translation"


DEFAULT_EXPERIMENTS = ExperimentDefaults()
