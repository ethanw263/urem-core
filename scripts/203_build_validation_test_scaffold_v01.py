#!/usr/bin/env python3

from pathlib import Path

SCRIPT_NAME = "203_build_validation_test_scaffold_v01"

TESTS_DIR = Path("tests")
TESTS_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "tests/__init__.py": "",

    "tests/test_validation_registry.py": r'''from pathlib import Path
import pandas as pd


REGISTRY = Path("data/validation/rde_experiment_registry_canonical_v02.csv")


def test_canonical_registry_exists():
    assert REGISTRY.exists()


def test_registry_has_rows():
    df = pd.read_csv(REGISTRY)
    assert len(df) >= 1


def test_completed_experiments_have_outputs():
    df = pd.read_csv(REGISTRY)
    completed = df[df["status"] == "complete"]

    assert len(completed) >= 1

    for _, row in completed.iterrows():
        assert Path(row["external_summary_csv"]).exists()
        assert Path(row["null_comparison_csv"]).exists()
''',

    "tests/test_experiment_config.py": r'''from pathlib import Path

from src.validation.experiments.config import load_experiment_config


CONFIG = Path("configs/experiments/oregon_gnis_coastal_landforms_v01.yaml")


def test_experiment_config_exists():
    assert CONFIG.exists()


def test_experiment_config_loads():
    experiment = load_experiment_config(str(CONFIG))

    assert experiment.experiment_key
    assert experiment.dataset_name == "GNIS Coastal Landforms"
    assert "distance" in experiment.metrics
    assert "density" in experiment.metrics
''',

    "tests/test_experiment_discovery.py": r'''from src.validation.experiments.discovery import get_synthesis_ready_experiments


def test_discovery_finds_ready_experiments():
    df = get_synthesis_ready_experiments(
        "data/validation/rde_experiment_registry_canonical_v02.csv"
    )

    assert len(df) >= 1
    assert df["experiment_ready_for_synthesis"].all()
''',

    "tests/test_synthesis_outputs.py": r'''from pathlib import Path
import pandas as pd


SUMMARY = Path("data/validation/synthesis/oregon_experiment_synthesis_v01_experiment_level.csv")
METRIC = Path("data/validation/synthesis/oregon_experiment_synthesis_v01_metric_level.csv")


def test_synthesis_outputs_exist():
    assert SUMMARY.exists()
    assert METRIC.exists()


def test_synthesis_has_expected_experiments():
    df = pd.read_csv(SUMMARY)

    assert "oregon_pmep_estuaries" in set(df["experiment_key"])
    assert "oregon_padus_protected_areas" in set(df["experiment_key"])
    assert "oregon_gnis_coastal_landforms" in set(df["experiment_key"])
''',

    "tests/test_validation_paths.py": r'''from src.validation.io.paths import (
    OREGON_STUDY_AREA,
    OREGON_TRANSITION_HOTSPOTS,
    VALIDATION_RESULTS_DIR,
    VALIDATION_SYNTHESIS_DIR,
)


def test_core_paths_exist():
    assert OREGON_STUDY_AREA.exists()
    assert OREGON_TRANSITION_HOTSPOTS.exists()
    assert VALIDATION_RESULTS_DIR.exists()
    assert VALIDATION_SYNTHESIS_DIR.exists()
''',
}


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    for path_str, content in FILES.items():
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"[{SCRIPT_NAME}] Wrote: {path}")

    print()
    print(f"[{SCRIPT_NAME}] Test scaffold created.")
    print()
    print("Run with:")
    print("  python -m pytest tests")
    print()
    print("If pytest is not installed:")
    print("  pip install pytest")
    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()