#!/usr/bin/env python3

from pathlib import Path

SCRIPT_NAME = "196_build_rde_experiment_discovery_engine_v01"

OUTPUT_PATH = Path("src/validation/experiments/discovery.py")

CONTENT = r'''from pathlib import Path
import pandas as pd


REQUIRED_OUTPUT_COLUMNS = [
    "external_summary_csv",
    "null_comparison_csv",
]


def load_registry(
    registry_path: str = "data/validation/rde_experiment_registry_canonical_v02.csv",
) -> pd.DataFrame:
    path = Path(registry_path)

    if not path.exists():
        raise FileNotFoundError(f"Missing experiment registry: {path}")

    return pd.read_csv(path)


def discover_completed_experiments(
    registry_path: str = "data/validation/rde_experiment_registry_canonical_v02.csv",
) -> pd.DataFrame:
    """
    Discover completed experiments with required output files present.
    """

    df = load_registry(registry_path)

    if "status" not in df.columns:
        raise ValueError("Registry is missing required column: status")

    completed = df[df["status"] == "complete"].copy()

    for col in REQUIRED_OUTPUT_COLUMNS:
        if col not in completed.columns:
            completed[col] = ""

        completed[f"{col}_exists"] = completed[col].apply(
            lambda p: Path(p).exists() if isinstance(p, str) and p else False
        )

    completed["experiment_ready_for_synthesis"] = completed[
        [f"{col}_exists" for col in REQUIRED_OUTPUT_COLUMNS]
    ].all(axis=1)

    return completed


def get_synthesis_ready_experiments(
    registry_path: str = "data/validation/rde_experiment_registry_canonical_v02.csv",
) -> pd.DataFrame:
    discovered = discover_completed_experiments(registry_path)

    return discovered[
        discovered["experiment_ready_for_synthesis"]
    ].copy().reset_index(drop=True)


def write_discovery_report(
    registry_path: str = "data/validation/rde_experiment_registry_canonical_v02.csv",
    output_csv: str = "data/validation/synthesis/rde_experiment_discovery_report_v01.csv",
    output_md: str = "data/validation/synthesis/rde_experiment_discovery_report_v01.md",
) -> dict:
    discovered = discover_completed_experiments(registry_path)

    output_csv = Path(output_csv)
    output_md = Path(output_md)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    discovered.to_csv(output_csv, index=False)

    lines = []
    lines.append("# RDE Experiment Discovery Report v01")
    lines.append("")
    lines.append("This report identifies completed experiments that are ready for synthesis.")
    lines.append("")
    lines.append("## Readiness Counts")
    lines.append("")
    lines.append(
        discovered["experiment_ready_for_synthesis"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "ready", "experiment_ready_for_synthesis": "count"})
        .to_markdown(index=False)
    )
    lines.append("")
    lines.append("## Experiments")
    lines.append("")
    display_cols = [
        "experiment_key",
        "experiment_name",
        "dataset_name",
        "status",
        "external_summary_csv_exists",
        "null_comparison_csv_exists",
        "experiment_ready_for_synthesis",
    ]
    display_cols = [c for c in display_cols if c in discovered.columns]
    lines.append(discovered[display_cols].to_markdown(index=False))

    output_md.write_text("\n".join(lines), encoding="utf-8")

    return {
        "discovery_csv": output_csv,
        "discovery_md": output_md,
        "discovered_df": discovered,
    }
'''


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(CONTENT, encoding="utf-8")

    print(f"[{SCRIPT_NAME}] Wrote: {OUTPUT_PATH}")
    print()
    print(f"[{SCRIPT_NAME}] Experiment discovery engine created.")
    print("Next: run a discovery report, then update synthesis to use discovered experiments.")
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()