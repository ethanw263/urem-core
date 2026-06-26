#!/usr/bin/env python3

from pathlib import Path

SCRIPT_NAME = "202_update_validation_cli_with_run_config_v01"
CLI_PATH = Path("scripts/200_validation_cli_v01.py")

CONTENT = r'''#!/usr/bin/env python3

from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.io.paths import ensure_validation_dirs
from src.validation.experiments.discovery import write_discovery_report
from src.validation.experiments.synthesis import build_experiment_synthesis
from src.validation.experiments.config import load_experiment_config
from src.validation.experiments.runner import RDEExperimentRunner
from src.validation.experiments.registry import append_experiment_registry
from src.validation.experiments.reporting import write_experiment_summary


def run_discovery(args):
    ensure_validation_dirs()

    outputs = write_discovery_report(
        registry_path=args.registry,
        output_csv=args.output_csv,
        output_md=args.output_md,
    )

    discovered = outputs["discovered_df"]

    print()
    print("[validation_cli] Discovery complete")
    print(f"Rows: {len(discovered):,}")
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


def run_synthesis(args):
    ensure_validation_dirs()

    outputs = build_experiment_synthesis(
        registry_path=args.registry,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
    )

    print()
    print("[validation_cli] Synthesis complete")
    print(f"Metric CSV:  {outputs['metric_csv']}")
    print(f"Summary CSV: {outputs['summary_csv']}")
    print(f"Report MD:   {outputs['report_md']}")
    print()
    print(outputs["summary_df"].to_string(index=False))


def run_experiment_config(args):
    ensure_validation_dirs()

    experiment = load_experiment_config(args.config)

    runner = RDEExperimentRunner(
        output_dir=args.output_dir,
    )

    outputs = runner.run(experiment)

    if not args.no_registry:
        append_experiment_registry(
            experiment=experiment,
            outputs=outputs,
            registry_path=args.registry,
        )

    if args.summary_md:
        summary_path = args.summary_md
    else:
        summary_path = (
            f"data/validation/experiments/"
            f"{experiment.experiment_key}_experiment_summary_v01.md"
        )

    write_experiment_summary(
        experiment=experiment,
        outputs=outputs,
        output_path=summary_path,
    )

    print()
    print("[validation_cli] Experiment complete")
    print(f"Experiment: {experiment.experiment_name}")
    print()
    print("Outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")

    print()
    print(f"Summary: {summary_path}")

    if not args.no_registry:
        print(f"Registry updated: {args.registry}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified RDE validation framework CLI"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    discovery = sub.add_parser("discovery", help="Run experiment discovery report")
    discovery.add_argument(
        "--registry",
        default="data/validation/rde_experiment_registry_canonical_v02.csv",
    )
    discovery.add_argument(
        "--output-csv",
        default="data/validation/synthesis/rde_experiment_discovery_report_v01.csv",
    )
    discovery.add_argument(
        "--output-md",
        default="data/validation/synthesis/rde_experiment_discovery_report_v01.md",
    )
    discovery.set_defaults(func=run_discovery)

    synthesis = sub.add_parser("synthesis", help="Run experiment evidence synthesis")
    synthesis.add_argument(
        "--registry",
        default="data/validation/rde_experiment_registry_canonical_v02.csv",
    )
    synthesis.add_argument(
        "--output-dir",
        default="data/validation/synthesis",
    )
    synthesis.add_argument(
        "--output-prefix",
        default="oregon_experiment_synthesis_v01",
    )
    synthesis.set_defaults(func=run_synthesis)

    run = sub.add_parser("run", help="Run a validation experiment config")
    run.add_argument("config", help="Path to YAML or JSON experiment config")
    run.add_argument(
        "--output-dir",
        default="data/validation/results",
    )
    run.add_argument(
        "--registry",
        default="data/validation/rde_experiment_framework_registry_v01.csv",
    )
    run.add_argument(
        "--summary-md",
        default=None,
    )
    run.add_argument(
        "--no-registry",
        action="store_true",
        help="Run experiment without updating registry",
    )
    run.set_defaults(func=run_experiment_config)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
'''


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    CLI_PATH.write_text(CONTENT, encoding="utf-8")

    print(f"[{SCRIPT_NAME}] Updated: {CLI_PATH}")
    print()
    print("Test with:")
    print("  python scripts/200_validation_cli_v01.py run configs/experiments/oregon_gnis_coastal_landforms_v01.yaml")
    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()