from pathlib import Path
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
