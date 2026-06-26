from pathlib import Path
import pandas as pd

from .experiment import ValidationExperiment


def append_experiment_registry(
    experiment: ValidationExperiment,
    outputs: dict,
    registry_path: str = "data/validation/rde_experiment_framework_registry_v01.csv",
) -> None:
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "experiment_key": experiment.experiment_key,
        "experiment_name": experiment.experiment_name,
        "study_name": experiment.study_name,
        "dataset_name": experiment.dataset_name,
        "dataset_source": experiment.dataset_source,
        "dataset_category": experiment.dataset_category,
        "metrics": ";".join(experiment.metrics),
        "run_external_validation": experiment.run_external_validation,
        "run_null_model": experiment.run_null_model,
        "n_simulations": experiment.n_simulations,
        "status": "complete",
        "notes": experiment.notes,
    }

    for k, v in outputs.items():
        row[k] = str(v)

    if path.exists():
        df = pd.read_csv(path)
        df = df[df["experiment_key"] != experiment.experiment_key]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(path, index=False)
