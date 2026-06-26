from pathlib import Path

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
