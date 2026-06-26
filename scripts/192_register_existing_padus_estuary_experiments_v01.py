#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

SCRIPT_NAME = "192_register_existing_padus_estuary_experiments_v01"

REGISTRY = Path("data/validation/rde_experiment_framework_registry_v01.csv")

EXPERIMENTS = [
    {
        "experiment_key": "oregon_padus_protected_areas",
        "experiment_name": "Oregon PAD-US Protected Areas Validation",
        "study_name": "Oregon Coast",
        "dataset_name": "Protected Areas",
        "dataset_source": "USGS PAD-US 4.1",
        "dataset_category": "conservation_landscape",
        "metrics": "intersects;overlap;distance",
        "run_external_validation": True,
        "run_null_model": True,
        "n_simulations": 500,
        "status": "complete",
        "notes": "Tests whether Oregon RDE transition hotspots are embedded in independently mapped protected areas.",
        "external_results_csv": "data/validation/results/oregon_padus_external_validation_results_v01.csv",
        "external_summary_csv": "data/validation/results/oregon_padus_external_validation_summary_v01.csv",
        "external_report_md": "data/validation/results/oregon_padus_external_validation_report_v01.md",
        "observed_csv": "data/validation/results/oregon_padus_observed_validation_metrics_v01.csv",
        "null_results_csv": "data/validation/results/oregon_padus_null_model_results_v01.csv",
        "null_comparison_csv": "data/validation/results/oregon_padus_null_model_comparison_v01.csv",
        "null_report_md": "data/validation/results/oregon_padus_null_model_report_v01.md",
    },
    {
        "experiment_key": "oregon_pmep_estuaries",
        "experiment_name": "Oregon PMEP Estuaries Validation",
        "study_name": "Oregon Coast",
        "dataset_name": "Estuaries",
        "dataset_source": "PMEP West Coast USA Estuary Extent V1",
        "dataset_category": "coastal_transition_environment",
        "metrics": "intersects;overlap;distance",
        "run_external_validation": True,
        "run_null_model": True,
        "n_simulations": 500,
        "status": "complete",
        "notes": "Tests whether Oregon RDE transition hotspots are closer to and more embedded in estuarine environments than random hotspot placements.",
        "external_results_csv": "data/validation/results/oregon_estuary_external_validation_results_v01.csv",
        "external_summary_csv": "data/validation/results/oregon_estuary_external_validation_summary_v01.csv",
        "external_report_md": "data/validation/results/oregon_estuary_external_validation_report_v01.md",
        "observed_csv": "data/validation/results/oregon_estuary_observed_validation_metrics_v01.csv",
        "null_results_csv": "data/validation/results/oregon_estuary_null_model_results_v01.csv",
        "null_comparison_csv": "data/validation/results/oregon_estuary_null_model_comparison_v01.csv",
        "null_report_md": "data/validation/results/oregon_estuary_null_model_report_v01.md",
    },
]


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    REGISTRY.parent.mkdir(parents=True, exist_ok=True)

    new = pd.DataFrame(EXPERIMENTS)

    if REGISTRY.exists():
        old = pd.read_csv(REGISTRY)
        old = old[~old["experiment_key"].isin(new["experiment_key"])]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new

    out.to_csv(REGISTRY, index=False)

    print(f"[{SCRIPT_NAME}] Registry rows: {len(out):,}")
    print(f"[{SCRIPT_NAME}] Writing: {REGISTRY}")
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()