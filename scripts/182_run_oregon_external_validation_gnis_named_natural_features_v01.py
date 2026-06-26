#!/usr/bin/env python3

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.datasets import ValidationDataset
from src.validation.engine import ValidationEngine
from src.validation.reporting import write_markdown_report


SCRIPT_NAME = "182_run_oregon_external_validation_gnis_named_natural_features_v01"

HOTSPOTS = "data/processed/oregon_transition_hotspot_validation_summary_v01.gpkg"
GNIS = "data/validation/standardized/gnis_named_natural_features_oregon_v01.gpkg"

OUTPUT_DIR = Path("data/validation/results")
OUTPUT_RESULTS_CSV = OUTPUT_DIR / "oregon_gnis_named_features_external_validation_results_v01.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "oregon_gnis_named_features_external_validation_summary_v01.csv"
OUTPUT_REPORT_MD = OUTPUT_DIR / "oregon_gnis_named_features_external_validation_report_v01.md"


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    engine = ValidationEngine(
        hotspots_path=HOTSPOTS,
        hotspot_id_column="transition_hotspot_id_v01",
        hotspot_score_column="mean_transition_score_v01",
    )

    gnis_dataset = ValidationDataset(
        name="GNIS Named Natural Features",
        path=GNIS,
        geometry_type="point",
        source="USGS GNIS FullModel / Gazetteer OR",
        metrics=[
            "distance",
            "density",
        ],
        weight=0.15,
        id_column="feature_id",
        category="geographic_recognition",
        independent_of_model=True,
        notes="External validation layer for named natural geographic features.",
    )

    engine.register_dataset(gnis_dataset)

    results = engine.run()
    summary = engine.summarize(results)

    print()
    print(f"[{SCRIPT_NAME}] Results rows: {len(results):,}")
    print()
    print(f"[{SCRIPT_NAME}] Summary:")
    print(summary.to_string(index=False))

    print()
    print(f"[{SCRIPT_NAME}] Writing results: {OUTPUT_RESULTS_CSV}")
    results.to_csv(OUTPUT_RESULTS_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Writing summary: {OUTPUT_SUMMARY_CSV}")
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Writing markdown report: {OUTPUT_REPORT_MD}")
    write_markdown_report(
        summary=summary,
        output_path=OUTPUT_REPORT_MD,
        title="Oregon GNIS Named Natural Features External Validation Report v01",
    )

    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()