#!/usr/bin/env python3

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import pandas as pd

from src.validation.null_models import (
    NullModelConfig,
    validation_metrics_for_layer,
    run_spatial_null_model,
    compare_observed_to_null,
)


SCRIPT_NAME = "171_run_oregon_padus_null_model_validation_v01"

HOTSPOTS = Path("data/processed/oregon_transition_hotspot_validation_summary_v01.gpkg")
STUDY_AREA = Path("data/processed/oregon_coast_study_area_v01.gpkg")
PADUS = Path("data/validation/standardized/protected_areas_oregon_padus_v01.gpkg")

OUTPUT_DIR = Path("data/validation/results")

OUTPUT_OBSERVED_CSV = OUTPUT_DIR / "oregon_padus_observed_validation_metrics_v01.csv"
OUTPUT_NULL_CSV = OUTPUT_DIR / "oregon_padus_null_model_results_v01.csv"
OUTPUT_COMPARISON_CSV = OUTPUT_DIR / "oregon_padus_null_model_comparison_v01.csv"
OUTPUT_REPORT_MD = OUTPUT_DIR / "oregon_padus_null_model_report_v01.md"

N_SIMULATIONS = 500


def write_report(observed, comparison, output_path):
    lines = []

    lines.append("# Oregon PAD-US Null Model Validation Report v01")
    lines.append("")
    lines.append("This report compares observed Oregon RDE transition hotspots against randomized hotspot placements.")
    lines.append("")
    lines.append("## Observed Metrics")
    lines.append("")
    lines.append(pd.DataFrame([observed]).to_markdown(index=False))
    lines.append("")
    lines.append("## Observed vs Null")
    lines.append("")
    lines.append(comparison.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Low Monte Carlo p-values indicate that observed hotspots align with PAD-US protected areas "
        "more strongly than expected under the spatial null model."
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hotspots = gpd.read_file(HOTSPOTS)
    study = gpd.read_file(STUDY_AREA)
    padus = gpd.read_file(PADUS)

    if study.crs != hotspots.crs:
        study = study.to_crs(hotspots.crs)

    if padus.crs != hotspots.crs:
        padus = padus.to_crs(hotspots.crs)

    print(f"[{SCRIPT_NAME}] Hotspots: {len(hotspots):,}")
    print(f"[{SCRIPT_NAME}] Study area rows: {len(study):,}")
    print(f"[{SCRIPT_NAME}] PAD-US features: {len(padus):,}")

    observed = validation_metrics_for_layer(
        hotspots=hotspots,
        validation_layer=padus,
    )

    print()
    print(f"[{SCRIPT_NAME}] Observed metrics:")
    for k, v in observed.items():
        print(f"  {k}: {v}")

    config = NullModelConfig(
        n_simulations=N_SIMULATIONS,
        random_seed=42,
        null_model_type="random_translation",
        max_attempts_per_feature=250,
    )

    print()
    print(f"[{SCRIPT_NAME}] Running null model simulations: {N_SIMULATIONS:,}")

    null_results = run_spatial_null_model(
        hotspots=hotspots,
        study_area=study,
        validation_layer=padus,
        config=config,
    )

    comparison = compare_observed_to_null(
        observed_metrics=observed,
        null_results=null_results,
    )

    print()
    print(f"[{SCRIPT_NAME}] Observed vs null:")
    print(comparison.to_string(index=False))

    print()
    print(f"[{SCRIPT_NAME}] Writing observed metrics: {OUTPUT_OBSERVED_CSV}")
    pd.DataFrame([observed]).to_csv(OUTPUT_OBSERVED_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Writing null results: {OUTPUT_NULL_CSV}")
    null_results.to_csv(OUTPUT_NULL_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Writing comparison: {OUTPUT_COMPARISON_CSV}")
    comparison.to_csv(OUTPUT_COMPARISON_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Writing markdown report: {OUTPUT_REPORT_MD}")
    write_report(observed, comparison, OUTPUT_REPORT_MD)

    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()