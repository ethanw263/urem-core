from src.validation.experiments.discovery import get_synthesis_ready_experiments


def test_discovery_finds_ready_experiments():
    df = get_synthesis_ready_experiments(
        "data/validation/rde_experiment_registry_canonical_v02.csv"
    )

    assert len(df) >= 1
    assert df["experiment_ready_for_synthesis"].all()
