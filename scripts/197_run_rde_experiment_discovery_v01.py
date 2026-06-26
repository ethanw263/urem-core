#!/usr/bin/env python3

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.experiments.discovery import write_discovery_report


SCRIPT_NAME = "197_run_rde_experiment_discovery_v01"


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    outputs = write_discovery_report(
        registry_path="data/validation/rde_experiment_registry_canonical_v02.csv",
        output_csv="data/validation/synthesis/rde_experiment_discovery_report_v01.csv",
        output_md="data/validation/synthesis/rde_experiment_discovery_report_v01.md",
    )

    discovered = outputs["discovered_df"]

    print()
    print(f"[{SCRIPT_NAME}] Discovery rows: {len(discovered):,}")
    print()
    print(discovered[
        [
            "experiment_key",
            "dataset_name",
            "experiment_ready_for_synthesis",
            "external_summary_csv_exists",
            "null_comparison_csv_exists",
        ]
    ].to_string(index=False))

    print()
    print(f"[{SCRIPT_NAME}] Wrote: {outputs['discovery_csv']}")
    print(f"[{SCRIPT_NAME}] Wrote: {outputs['discovery_md']}")

    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()