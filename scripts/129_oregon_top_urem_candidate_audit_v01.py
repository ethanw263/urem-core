#!/usr/bin/env python3
"""
129_oregon_top_urem_candidate_audit_v01.py

Create Oregon top UREM candidate audit package.

Inputs:
- data/processed/oregon_ranked_urem_candidates_v05.gpkg
- data/processed/oregon_urem_score_v05.gpkg

Outputs:
- data/processed/oregon_top_urem_candidates_v05.gpkg
- data/processed/oregon_top_urem_candidates_v05.csv
- data/processed/oregon_top_100_urem_candidates_v05.gpkg
- data/processed/oregon_top_500_urem_candidates_v05.gpkg
- data/processed/oregon_urem_candidate_audit_summary_v01.csv
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd


SCRIPT_NAME = "129_oregon_top_urem_candidate_audit_v01"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RANKED_PATH = PROCESSED_DIR / "oregon_ranked_urem_candidates_v05.gpkg"
SCORE_PATH = PROCESSED_DIR / "oregon_urem_score_v05.gpkg"

OUT_ALL_GPKG = PROCESSED_DIR / "oregon_top_urem_candidates_v05.gpkg"
OUT_ALL_CSV = PROCESSED_DIR / "oregon_top_urem_candidates_v05.csv"
OUT_TOP100_GPKG = PROCESSED_DIR / "oregon_top_100_urem_candidates_v05.gpkg"
OUT_TOP500_GPKG = PROCESSED_DIR / "oregon_top_500_urem_candidates_v05.gpkg"
OUT_SUMMARY_CSV = PROCESSED_DIR / "oregon_urem_candidate_audit_summary_v01.csv"


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def main():
    log("Starting Oregon top UREM candidate audit")

    if not RANKED_PATH.exists():
        raise FileNotFoundError(f"Missing ranked candidates: {RANKED_PATH}")

    ranked = gpd.read_file(RANKED_PATH)

    log(f"Ranked candidates loaded: {len(ranked):,}")
    log(f"CRS: {ranked.crs}")

    keep_cols = [
        "cell_id",
        "candidate_rank_v05",
        "urem_rank_v05",
        "urem_percentile_v05",
        "urem_tier_v05",
        "urem_score_v05",
        "urem_score_v05_raw",
        "physical_exceptionality_v03",
        "observed_recognition_v04",
        "expected_recognition_v05",
        "recognition_residual_v05",
        "positive_under_recognition_residual_v05",
        "terrain_drama_v03",
        "scenic_coast_v03",
        "relief_norm_v03",
        "slope_norm_v03",
        "coast_proximity_v03",
        "coast_complexity_v03",
        "recognition_total_count_3km_v04",
        "recognition_category_coverage_v04",
        "recognition_cell_confidence_v04",
        "expected_recognition_confidence_v05",
        "land_area_share",
        "water_area_share",
        "is_valid_land_candidate",
        "geometry",
    ]

    keep_cols = [c for c in keep_cols if c in ranked.columns]

    audit = ranked[keep_cols].copy()
    audit = audit.sort_values("candidate_rank_v05").reset_index(drop=True)

    log("Writing full candidate audit package")

    if OUT_ALL_GPKG.exists():
        OUT_ALL_GPKG.unlink()

    audit.to_file(
        OUT_ALL_GPKG,
        layer="oregon_top_urem_candidates_v05",
        driver="GPKG",
    )

    audit.drop(columns="geometry").to_csv(OUT_ALL_CSV, index=False)

    top100 = audit.head(100).copy()
    top500 = audit.head(500).copy()

    if OUT_TOP100_GPKG.exists():
        OUT_TOP100_GPKG.unlink()

    if OUT_TOP500_GPKG.exists():
        OUT_TOP500_GPKG.unlink()

    top100.to_file(
        OUT_TOP100_GPKG,
        layer="oregon_top_100_urem_candidates_v05",
        driver="GPKG",
    )

    top500.to_file(
        OUT_TOP500_GPKG,
        layer="oregon_top_500_urem_candidates_v05",
        driver="GPKG",
    )

    summary_rows = []

    for label, subset in [
        ("all_candidates", audit),
        ("top_100", top100),
        ("top_500", top500),
    ]:
        row = {
            "group": label,
            "n": len(subset),
            "mean_urem_score_v05": subset["urem_score_v05"].mean(),
            "mean_physical_exceptionality_v03": subset["physical_exceptionality_v03"].mean(),
            "mean_observed_recognition_v04": subset["observed_recognition_v04"].mean(),
            "mean_expected_recognition_v05": subset["expected_recognition_v05"].mean(),
            "mean_positive_under_recognition_residual_v05": subset[
                "positive_under_recognition_residual_v05"
            ].mean(),
            "mean_terrain_drama_v03": subset["terrain_drama_v03"].mean(),
            "mean_scenic_coast_v03": subset["scenic_coast_v03"].mean(),
            "mean_land_area_share": subset["land_area_share"].mean(),
            "mean_recognition_total_count_3km_v04": subset[
                "recognition_total_count_3km_v04"
            ].mean(),
        }
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    log("Summary:")
    print(summary)

    log(f"Wrote: {OUT_ALL_GPKG}")
    log(f"Wrote: {OUT_ALL_CSV}")
    log(f"Wrote: {OUT_TOP100_GPKG}")
    log(f"Wrote: {OUT_TOP500_GPKG}")
    log(f"Wrote: {OUT_SUMMARY_CSV}")
    log("Done")


if __name__ == "__main__":
    main()