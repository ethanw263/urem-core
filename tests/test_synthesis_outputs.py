from pathlib import Path
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
