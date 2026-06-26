#!/usr/bin/env python3

from pathlib import Path

SCRIPT_NAME = "200_build_unified_validation_cli_v01"
OUTPUT = Path("scripts/200_validation_cli_v01.py")

CONTENT = r'''#!/usr/bin/env python3

from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.io.paths import ensure_validation_dirs
from src.validation.experiments.discovery import write_discovery_report
from src.validation.experiments.synthesis import build_experiment_synthesis


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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
'''

def main():
    print(f"[{SCRIPT_NAME}] Starting")

    OUTPUT.write_text(CONTENT, encoding="utf-8")

    print(f"[{SCRIPT_NAME}] Wrote: {OUTPUT}")
    print()
    print("Test with:")
    print("  python scripts/200_validation_cli_v01.py discovery")
    print("  python scripts/200_validation_cli_v01.py synthesis")
    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()