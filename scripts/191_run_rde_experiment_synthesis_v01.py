#!/usr/bin/env python3

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.experiments.synthesis import build_experiment_synthesis


SCRIPT_NAME = "191_run_rde_experiment_synthesis_v01"


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    outputs = build_experiment_synthesis(
        registry_path="data/validation/rde_experiment_framework_registry_v01.csv",
        output_dir="data/validation/synthesis",
        output_prefix="oregon_experiment_synthesis_v01",
    )

    print()
    print(f"[{SCRIPT_NAME}] Outputs:")
    print(f"  metric_csv: {outputs['metric_csv']}")
    print(f"  summary_csv: {outputs['summary_csv']}")
    print(f"  report_md: {outputs['report_md']}")

    print()
    print(f"[{SCRIPT_NAME}] Experiment summary:")
    print(outputs["summary_df"].to_string(index=False))

    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()