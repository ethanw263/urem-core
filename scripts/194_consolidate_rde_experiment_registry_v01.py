#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

SCRIPT_NAME = "194_consolidate_rde_experiment_registry_v01"

REGISTRY = Path("data/validation/rde_experiment_framework_registry_v01.csv")
OUTPUT = Path("data/validation/rde_experiment_registry_canonical_v01.csv")
OUTPUT_MD = Path("data/validation/rde_experiment_registry_canonical_v01.md")

CANONICAL_KEEP = [
    "oregon_padus_protected_areas",
    "oregon_pmep_estuaries",
    "oregon_gnis_coastal_landforms",
    "oregon_gnis_hydrology",
    "oregon_gnis_terrain",
    "oregon_gnis_distinct_natural_features",
]


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    if not REGISTRY.exists():
        raise FileNotFoundError(f"Missing registry: {REGISTRY}")

    df = pd.read_csv(REGISTRY)

    print(f"[{SCRIPT_NAME}] Input rows: {len(df):,}")

    df = df[df["experiment_key"].isin(CANONICAL_KEEP)].copy()

    df["result_exists"] = df["null_comparison_csv"].apply(
        lambda p: Path(p).exists() if isinstance(p, str) and p else False
    )

    df["external_summary_exists"] = df["external_summary_csv"].apply(
        lambda p: Path(p).exists() if isinstance(p, str) and p else False
    )

    df = df.sort_values("experiment_key").reset_index(drop=True)

    print(f"[{SCRIPT_NAME}] Canonical rows: {len(df):,}")
    print()
    print(df[[
        "experiment_key",
        "dataset_name",
        "status",
        "result_exists",
        "external_summary_exists",
    ]].to_string(index=False))

    df.to_csv(OUTPUT, index=False)

    lines = [
        "# Canonical RDE Experiment Registry v01",
        "",
        "This is the cleaned single-source experiment registry after consolidating older validation registry formats.",
        "",
        "## Experiments",
        "",
        df.to_markdown(index=False),
    ]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"[{SCRIPT_NAME}] Wrote: {OUTPUT}")
    print(f"[{SCRIPT_NAME}] Wrote: {OUTPUT_MD}")
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()