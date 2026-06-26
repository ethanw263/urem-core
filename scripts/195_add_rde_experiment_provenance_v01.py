#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

SCRIPT_NAME = "195_add_rde_experiment_provenance_v01"

REGISTRY = Path("data/validation/rde_experiment_registry_canonical_v01.csv")
OUTPUT = Path("data/validation/rde_experiment_registry_canonical_v02.csv")
OUTPUT_MD = Path("data/validation/rde_experiment_registry_canonical_v02.md")

FRAMEWORK_VERSION = "rde_experiment_framework_v01"
VALIDATION_ENGINE_VERSION = "validation_engine_v02"
NULL_MODEL_VERSION = "spatial_random_translation_v02"
SYNTHESIS_ENGINE_VERSION = "experiment_synthesis_v01"


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    df = pd.read_csv(REGISTRY)

    now = datetime.now(timezone.utc).isoformat()

    df["framework_version"] = FRAMEWORK_VERSION
    df["validation_engine_version"] = VALIDATION_ENGINE_VERSION
    df["null_model_version"] = NULL_MODEL_VERSION
    df["synthesis_engine_version"] = SYNTHESIS_ENGINE_VERSION
    df["registry_version"] = "canonical_v02"
    df["provenance_updated_utc"] = now

    if "random_seed" not in df.columns:
        df["random_seed"] = 42

    if "n_simulations" not in df.columns:
        df["n_simulations"] = 500

    df["null_model_type"] = "random_translation"
    df["study_area_path"] = "data/processed/oregon_coast_study_area_v01.gpkg"
    df["hotspots_path"] = "data/processed/oregon_transition_hotspot_validation_summary_v01.gpkg"

    df["null_comparison_exists"] = df["null_comparison_csv"].apply(
        lambda p: Path(p).exists() if isinstance(p, str) and p else False
    )

    df["external_summary_exists"] = df["external_summary_csv"].apply(
        lambda p: Path(p).exists() if isinstance(p, str) and p else False
    )

    print(f"[{SCRIPT_NAME}] Rows: {len(df):,}")
    print()
    print(df[[
        "experiment_key",
        "dataset_name",
        "framework_version",
        "null_model_version",
        "n_simulations",
        "random_seed",
        "null_comparison_exists",
        "external_summary_exists",
    ]].to_string(index=False))

    df.to_csv(OUTPUT, index=False)

    lines = [
        "# Canonical RDE Experiment Registry v02",
        "",
        "This registry adds provenance metadata for reproducibility.",
        "",
        f"Updated UTC: `{now}`",
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