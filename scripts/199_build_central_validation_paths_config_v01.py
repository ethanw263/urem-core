#!/usr/bin/env python3

from pathlib import Path

SCRIPT_NAME = "199_build_central_validation_paths_config_v01"

IO_DIR = Path("src/validation/io")
INIT_PY = IO_DIR / "__init__.py"
PATHS_PY = IO_DIR / "paths.py"
CONFIG_PY = IO_DIR / "config.py"

PATHS_CONTENT = r'''from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

VALIDATION_DIR = DATA_DIR / "validation"
VALIDATION_STANDARDIZED_DIR = VALIDATION_DIR / "standardized"
VALIDATION_RESULTS_DIR = VALIDATION_DIR / "results"
VALIDATION_SYNTHESIS_DIR = VALIDATION_DIR / "synthesis"
VALIDATION_EXPERIMENTS_DIR = VALIDATION_DIR / "experiments"
VALIDATION_AUDIT_DIR = VALIDATION_DIR / "framework_audit"

RAW_DIR = DATA_DIR / "raw"
RAW_VALIDATION_DIR = RAW_DIR / "validation"

OREGON_STUDY_AREA = PROCESSED_DIR / "oregon_coast_study_area_v01.gpkg"
OREGON_TRANSITION_HOTSPOTS = PROCESSED_DIR / "oregon_transition_hotspot_validation_summary_v01.gpkg"

CANONICAL_EXPERIMENT_REGISTRY_V01 = VALIDATION_DIR / "rde_experiment_registry_canonical_v01.csv"
CANONICAL_EXPERIMENT_REGISTRY_V02 = VALIDATION_DIR / "rde_experiment_registry_canonical_v02.csv"
EXPERIMENT_FRAMEWORK_REGISTRY = VALIDATION_DIR / "rde_experiment_framework_registry_v01.csv"

OREGON_EXPERIMENT_SYNTHESIS_METRIC = VALIDATION_SYNTHESIS_DIR / "oregon_experiment_synthesis_v01_metric_level.csv"
OREGON_EXPERIMENT_SYNTHESIS_SUMMARY = VALIDATION_SYNTHESIS_DIR / "oregon_experiment_synthesis_v01_experiment_level.csv"
OREGON_EXPERIMENT_SYNTHESIS_REPORT = VALIDATION_SYNTHESIS_DIR / "oregon_experiment_synthesis_v01.md"


def ensure_validation_dirs() -> None:
    for path in [
        VALIDATION_DIR,
        VALIDATION_STANDARDIZED_DIR,
        VALIDATION_RESULTS_DIR,
        VALIDATION_SYNTHESIS_DIR,
        VALIDATION_EXPERIMENTS_DIR,
        VALIDATION_AUDIT_DIR,
        RAW_VALIDATION_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def as_posix(path: Path) -> str:
    return path.as_posix()
'''

CONFIG_CONTENT = r'''from dataclasses import dataclass
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
'''


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    IO_DIR.mkdir(parents=True, exist_ok=True)

    INIT_PY.write_text("", encoding="utf-8")
    PATHS_PY.write_text(PATHS_CONTENT, encoding="utf-8")
    CONFIG_PY.write_text(CONFIG_CONTENT, encoding="utf-8")

    print(f"[{SCRIPT_NAME}] Wrote: {INIT_PY}")
    print(f"[{SCRIPT_NAME}] Wrote: {PATHS_PY}")
    print(f"[{SCRIPT_NAME}] Wrote: {CONFIG_PY}")
    print()
    print(f"[{SCRIPT_NAME}] Central validation path/config modules created.")
    print()
    print("Next step:")
    print("  Update future scripts/runners to import paths from src.validation.io.paths")
    print("  and study defaults from src.validation.io.config.")
    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()