#!/usr/bin/env python3
"""
133_compare_oregon_v05_v06.py

Compare Oregon UREM v05 and v06.

Purpose:
Evaluate how the KNN expected-recognition upgrade changed:
- expected recognition
- under-recognition residuals
- UREM scores
- candidate geography

Inputs:
- data/processed/oregon_urem_score_v05.gpkg
- data/processed/oregon_urem_score_v06.gpkg

Outputs:
- data/processed/oregon_v05_v06_comparison.gpkg
- data/processed/oregon_v05_v06_comparison.csv
- data/processed/oregon_v05_v06_comparison_summary.csv
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd


SCRIPT_NAME = "133_compare_oregon_v05_v06"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

V05_PATH = PROCESSED_DIR / "oregon_urem_score_v05.gpkg"
V06_PATH = PROCESSED_DIR / "oregon_urem_score_v06.gpkg"

OUT_GPKG = PROCESSED_DIR / "oregon_v05_v06_comparison.gpkg"
OUT_CSV = PROCESSED_DIR / "oregon_v05_v06_comparison.csv"
OUT_SUMMARY_CSV = PROCESSED_DIR / "oregon_v05_v06_comparison_summary.csv"


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def safe_num(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def classify_change(x):
    if x >= 0.15:
        return "strong_increase"
    if x >= 0.05:
        return "moderate_increase"
    if x <= -0.15:
        return "strong_decrease"
    if x <= -0.05:
        return "moderate_decrease"
    return "stable"


def main():
    log("Starting Oregon v05-v06 comparison")

    if not V05_PATH.exists():
        raise FileNotFoundError(f"Missing v05 file: {V05_PATH}")

    if not V06_PATH.exists():
        raise FileNotFoundError(f"Missing v06 file: {V06_PATH}")

    log(f"Reading v05: {V05_PATH}")
    v05 = gpd.read_file(V05_PATH)

    log(f"Reading v06: {V06_PATH}")
    v06 = gpd.read_file(V06_PATH)

    log(f"v05 rows: {len(v05):,}")
    log(f"v06 rows: {len(v06):,}")

    keep_v05 = [
        "cell_id",
        "expected_recognition_v05",
        "recognition_residual_v05",
        "positive_under_recognition_residual_v05",
        "urem_score_v05_raw",
        "urem_score_v05",
        "passes_urem_candidate_filter_v05",
    ]

    keep_v06 = [
        "cell_id",
        "expected_recognition_v06",
        "recognition_residual_v06",
        "positive_under_recognition_residual_v06",
        "urem_score_v06_raw",
        "urem_score_v06",
        "urem_score_v06_fullrange",
        "passes_urem_candidate_filter_v06",
        "physical_exceptionality_v03",
        "observed_recognition_v04",
        "recognition_total_count_3km_v04",
        "recognition_category_coverage_v04",
        "land_area_share",
        "is_valid_land_candidate",
        "geometry",
    ]

    keep_v05 = [c for c in keep_v05 if c in v05.columns]
    keep_v06 = [c for c in keep_v06 if c in v06.columns]

    df = v06[keep_v06].copy()

    df = df.merge(
        pd.DataFrame(v05[keep_v05]),
        on="cell_id",
        how="left",
    )

    numeric_cols = [
        "expected_recognition_v05",
        "expected_recognition_v06",
        "recognition_residual_v05",
        "recognition_residual_v06",
        "positive_under_recognition_residual_v05",
        "positive_under_recognition_residual_v06",
        "urem_score_v05_raw",
        "urem_score_v06_raw",
        "urem_score_v05",
        "urem_score_v06",
        "urem_score_v06_fullrange",
        "physical_exceptionality_v03",
        "observed_recognition_v04",
        "recognition_total_count_3km_v04",
        "recognition_category_coverage_v04",
        "land_area_share",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_num(df, col)

    df["expected_recognition_change_v06_minus_v05"] = (
        df["expected_recognition_v06"] - df["expected_recognition_v05"]
    )

    df["recognition_residual_change_v06_minus_v05"] = (
        df["recognition_residual_v06"] - df["recognition_residual_v05"]
    )

    df["positive_under_recognition_change_v06_minus_v05"] = (
        df["positive_under_recognition_residual_v06"]
        - df["positive_under_recognition_residual_v05"]
    )

    df["urem_score_raw_change_v06_minus_v05"] = (
        df["urem_score_v06_raw"] - df["urem_score_v05_raw"]
    )

    df["expected_recognition_change_class"] = (
        df["expected_recognition_change_v06_minus_v05"].apply(classify_change)
    )

    df["residual_change_class"] = (
        df["recognition_residual_change_v06_minus_v05"].apply(classify_change)
    )

    df["urem_score_raw_change_class"] = (
        df["urem_score_raw_change_v06_minus_v05"].apply(classify_change)
    )

    df["candidate_status_change_v05_to_v06"] = "non_candidate_both"

    v05_candidate = (
        df["passes_urem_candidate_filter_v05"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )

    v06_candidate = (
        df["passes_urem_candidate_filter_v06"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )

    df.loc[v05_candidate & v06_candidate, "candidate_status_change_v05_to_v06"] = (
        "candidate_both"
    )

    df.loc[v05_candidate & ~v06_candidate, "candidate_status_change_v05_to_v06"] = (
        "lost_candidate_in_v06"
    )

    df.loc[~v05_candidate & v06_candidate, "candidate_status_change_v05_to_v06"] = (
        "new_candidate_in_v06"
    )

    summary_rows = []

    for label, subset in [
        ("all_cells", df),
        ("v05_candidates", df[v05_candidate]),
        ("v06_candidates", df[v06_candidate]),
        ("candidate_both", df[df["candidate_status_change_v05_to_v06"] == "candidate_both"]),
        ("lost_candidate_in_v06", df[df["candidate_status_change_v05_to_v06"] == "lost_candidate_in_v06"]),
        ("new_candidate_in_v06", df[df["candidate_status_change_v05_to_v06"] == "new_candidate_in_v06"]),
    ]:
        summary_rows.append(
            {
                "group": label,
                "n": len(subset),
                "mean_expected_recognition_v05": subset["expected_recognition_v05"].mean(),
                "mean_expected_recognition_v06": subset["expected_recognition_v06"].mean(),
                "mean_expected_recognition_change": subset[
                    "expected_recognition_change_v06_minus_v05"
                ].mean(),
                "mean_positive_under_recognition_v05": subset[
                    "positive_under_recognition_residual_v05"
                ].mean(),
                "mean_positive_under_recognition_v06": subset[
                    "positive_under_recognition_residual_v06"
                ].mean(),
                "mean_urem_score_v05_raw": subset["urem_score_v05_raw"].mean(),
                "mean_urem_score_v06_raw": subset["urem_score_v06_raw"].mean(),
                "mean_urem_score_raw_change": subset[
                    "urem_score_raw_change_v06_minus_v05"
                ].mean(),
                "mean_physical_exceptionality_v03": subset[
                    "physical_exceptionality_v03"
                ].mean(),
                "mean_observed_recognition_v04": subset[
                    "observed_recognition_v04"
                ].mean(),
            }
        )

    summary = pd.DataFrame(summary_rows)

    log("Candidate status change counts:")
    print(df["candidate_status_change_v05_to_v06"].value_counts())

    log("UREM raw score change class counts:")
    print(df["urem_score_raw_change_class"].value_counts())

    log("Summary:")
    print(summary)

    if OUT_GPKG.exists():
        OUT_GPKG.unlink()

    log(f"Writing GPKG: {OUT_GPKG}")
    df.to_file(
        OUT_GPKG,
        layer="oregon_v05_v06_comparison",
        driver="GPKG",
    )

    log(f"Writing CSV: {OUT_CSV}")
    df.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log(f"Writing summary CSV: {OUT_SUMMARY_CSV}")
    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    log("Done")


if __name__ == "__main__":
    main()