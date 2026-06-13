#!/usr/bin/env python3
"""
53_generate_validation_review_package_v04_v05b.py

Create final manual validation review package.

Purpose:
- Use v04 as the main model.
- Attach v05b comparison as diagnostic evidence only.
- Create one clean CSV for human hotspot review.

Inputs:
- data/processed/urem_hotspot_centroids_v04.csv
- data/processed/urem_hotspot_comparison_v04_v05b.csv

Outputs:
- data/processed/urem_validation_review_package_v04_v05b.csv
"""

from pathlib import Path
import pandas as pd


SCRIPT_NAME = "53_generate_validation_review_package_v04_v05b"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

V04_CENTROIDS = PROCESSED_DIR / "urem_hotspot_centroids_v04.csv"
COMPARISON = PROCESSED_DIR / "urem_hotspot_comparison_v04_v05b.csv"

OUT_CSV = PROCESSED_DIR / "urem_validation_review_package_v04_v05b.csv"


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def main():
    log("Starting validation review package")

    require_file(V04_CENTROIDS)
    require_file(COMPARISON)

    v04 = pd.read_csv(V04_CENTROIDS)
    comp = pd.read_csv(COMPARISON)

    keep_comp = [
        "v04_rank",
        "v04_hotspot_id",
        "nearest_v05b_rank",
        "nearest_v05b_hotspot_id",
        "distance_to_nearest_v05b_km",
        "comparison_class",
    ]
    keep_comp = [c for c in keep_comp if c in comp.columns]

    comp = comp[keep_comp].copy()

    out = v04.merge(
        comp,
        left_on=["hotspot_rank_v04", "hotspot_id"],
        right_on=["v04_rank", "v04_hotspot_id"],
        how="left",
    )

    out["v05b_agreement_flag"] = out["comparison_class"].isin(
        ["survived", "near_survived"]
    )

    manual_cols = {
        "manual_region_quality_1_5": "",
        "manual_point_quality_1_5": "",
        "manual_scenic_quality_1_5": "",
        "manual_geographic_uniqueness_1_5": "",
        "manual_recreation_potential_1_5": "",
        "manual_landscape_drama_1_5": "",
        "manual_water_coast_relationship_1_5": "",
        "manual_accessibility_1_5": "",
        "manual_existing_recognition_1_5": "",
        "manual_under_recognized_exceptionality_1_5": "",
        "dominant_landscape_type": "",
        "is_success": "",
        "is_false_positive": "",
        "failure_mode": "",
        "reviewer_notes": "",
    }

    for col, default in manual_cols.items():
        out[col] = default

    preferred_cols = [
        "hotspot_rank_v04",
        "hotspot_id",
        "longitude",
        "latitude",
        "cell_count",
        "hotspot_area_km2",
        "hotspot_score_v04",
        "mean_urem_score_v04",
        "max_urem_score_v04",
        "mean_exceptionality_v03",
        "mean_observed_recognition_v04",
        "mean_expected_recognition_v05",
        "mean_under_recognition_residual_v05",
        "mean_expected_recognition_confidence_v05",
        "mean_recognition_confidence_v04",
        "mean_land_area_share",
        "best_cell_id",
        "nearest_v05b_rank",
        "distance_to_nearest_v05b_km",
        "comparison_class",
        "v05b_agreement_flag",
    ]

    preferred_cols = [c for c in preferred_cols if c in out.columns]

    out = out[preferred_cols + list(manual_cols.keys())].copy()
    out = out.sort_values("hotspot_rank_v04")

    log(f"Writing CSV: {OUT_CSV}")
    out.to_csv(OUT_CSV, index=False)

    log("Done")

    print("\nReview package summary:")
    print(f"Rows: {len(out):,}")
    print(out["comparison_class"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()