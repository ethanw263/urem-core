from pathlib import Path


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
