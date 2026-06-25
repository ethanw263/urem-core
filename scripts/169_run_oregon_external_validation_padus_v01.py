#!/usr/bin/env python3

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.validation.datasets import ValidationDataset
from src.validation.engine import ValidationEngine
from src.validation.reporting import write_markdown_report


SCRIPT_NAME = "169_run_oregon_external_validation_padus_v01"

HOTSPOTS = "data/processed/oregon_transition_hotspot_validation_summary_v01.gpkg"
PADUS = "data/validation/standardized/protected_areas_oregon_padus_v01.gpkg"

OUTPUT_DIR = Path("data/validation/results")
OUTPUT_RESULTS_CSV = OUTPUT_DIR / "oregon_padus_external_validation_results_v01.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "oregon_padus_external_validation_summary_v01.csv"
OUTPUT_REPORT_MD = OUTPUT_DIR / "oregon_padus_external_validation_report_v01.md"


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    engine = ValidationEngine(
        hotspots_path=HOTSPOTS,
        hotspot_id_column="transition_hotspot_id_v01",
        hotspot_score_column="mean_transition_score_v01",
    )

    padus_dataset = ValidationDataset(
        name="Protected Areas",
        path=PADUS,
        geometry_type="polygon",
        source="USGS PAD-US 4.1",
        metrics=[
            "intersects",
            "overlap",
            "distance",
        ],
        weight=0.25,
        id_column=None,
        category="conservation",
        independent_of_model=True,
        notes="External validation layer for Oregon transition hotspots.",
    )

    engine.register_dataset(padus_dataset)

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
        title="Oregon PAD-US External Validation Report v01",
    )

    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()
