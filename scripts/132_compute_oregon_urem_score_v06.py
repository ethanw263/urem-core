#!/usr/bin/env python3
"""
132_compute_oregon_urem_score_v06.py

Compute Oregon UREM Score v06 using:
- Exceptionality v03
- Recognition v04
- Expected Recognition v06 KNN
- Land-validity filter

Inputs:
- data/processed/oregon_expected_recognition_v06_knn.gpkg

Outputs:
- data/processed/oregon_urem_score_v06.gpkg
- data/processed/oregon_urem_score_v06.csv
- data/processed/oregon_ranked_urem_candidates_v06.gpkg
- data/processed/oregon_ranked_urem_candidates_v06.csv
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd


SCRIPT_NAME = "132_compute_oregon_urem_score_v06"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = PROCESSED_DIR / "oregon_expected_recognition_v06_knn.gpkg"

OUT_SCORE_GPKG = PROCESSED_DIR / "oregon_urem_score_v06.gpkg"
OUT_SCORE_CSV = PROCESSED_DIR / "oregon_urem_score_v06.csv"

OUT_RANKED_GPKG = PROCESSED_DIR / "oregon_ranked_urem_candidates_v06.gpkg"
OUT_RANKED_CSV = PROCESSED_DIR / "oregon_ranked_urem_candidates_v06.csv"


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def norm01(s):
    s = pd.to_numeric(s, errors="coerce")
    lo = s.quantile(0.02)
    hi = s.quantile(0.98)

    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=s.index)

    return ((s - lo) / (hi - lo)).clip(0, 1)


def norm01_full_range(s):
    s = pd.to_numeric(s, errors="coerce")
    mn = s.min()
    mx = s.max()

    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.0, index=s.index)

    return ((s - mn) / (mx - mn)).clip(0, 1)


def safe_num(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def main():
    log("Starting Oregon UREM Score v06")

    if not INPUT_GPKG.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_GPKG}")

    gdf = gpd.read_file(INPUT_GPKG)

    log(f"Rows: {len(gdf):,}")
    log(f"CRS: {gdf.crs}")

    gdf["physical_exceptionality_v03"] = safe_num(
        gdf,
        "physical_exceptionality_v03",
    )

    gdf["positive_under_recognition_residual_v06"] = safe_num(
        gdf,
        "positive_under_recognition_residual_v06",
    )

    gdf["expected_recognition_confidence_v06"] = safe_num(
        gdf,
        "expected_recognition_confidence_v06",
        0.5,
    )

    gdf["recognition_cell_confidence_v04"] = safe_num(
        gdf,
        "recognition_cell_confidence_v04",
        0.5,
    )

    gdf["land_area_share"] = safe_num(gdf, "land_area_share", 1.0)
    gdf["is_valid_land_candidate"] = safe_num(
        gdf,
        "is_valid_land_candidate",
        1.0,
    )

    gdf["urem_exceptionality_component_v06"] = (
        gdf["physical_exceptionality_v03"].clip(0, 1)
    )

    gdf["urem_under_recognition_component_v06"] = norm01(
        gdf["positive_under_recognition_residual_v06"]
    )

    gdf["urem_confidence_component_v06"] = (
        0.50 * gdf["expected_recognition_confidence_v06"]
        + 0.50 * gdf["recognition_cell_confidence_v04"]
    ).clip(0, 1)

    gdf["urem_land_validity_component_v06"] = (
        gdf["land_area_share"].clip(0, 1)
        * gdf["is_valid_land_candidate"].clip(0, 1)
    )

    gdf["urem_score_v06_raw"] = (
        gdf["urem_exceptionality_component_v06"]
        * gdf["urem_under_recognition_component_v06"]
        * (0.65 + 0.35 * gdf["urem_confidence_component_v06"])
        * gdf["urem_land_validity_component_v06"]
    )

    # Keep two versions:
    # - percentile-normalized score for continuity with v05
    # - full-range score for less top-tail saturation in maps/review
    gdf["urem_score_v06"] = norm01(gdf["urem_score_v06_raw"])
    gdf["urem_score_v06_fullrange"] = norm01_full_range(gdf["urem_score_v06_raw"])

    gdf["urem_score_v06_additive_diagnostic"] = (
        0.45 * gdf["urem_exceptionality_component_v06"]
        + 0.35 * gdf["urem_under_recognition_component_v06"]
        + 0.10 * gdf["urem_confidence_component_v06"]
        + 0.10 * gdf["urem_land_validity_component_v06"]
    ).clip(0, 1)

    gdf["passes_land_filter_v06"] = (
        (gdf["is_valid_land_candidate"].astype(str).str.lower().isin(["true", "1"]))
        & (gdf["land_area_share"] >= 0.50)
    )

    gdf["passes_confidence_filter_v06"] = (
        gdf["urem_confidence_component_v06"] >= 0.35
    )

    valid_pool = (
        gdf["passes_land_filter_v06"]
        & gdf["passes_confidence_filter_v06"]
    )

    score_cutoff = gdf.loc[valid_pool, "urem_score_v06_raw"].quantile(0.90)

    gdf["passes_score_filter_v06"] = (
        gdf["urem_score_v06_raw"] >= score_cutoff
    )

    gdf["passes_exceptionality_filter_v06"] = (
        gdf["physical_exceptionality_v03"]
        >= gdf["physical_exceptionality_v03"].quantile(0.60)
    )

    gdf["passes_under_recognition_filter_v06"] = (
        gdf["positive_under_recognition_residual_v06"] > 0
    )

    gdf["passes_urem_candidate_filter_v06"] = (
        gdf["passes_land_filter_v06"]
        & gdf["passes_confidence_filter_v06"]
        & gdf["passes_score_filter_v06"]
        & gdf["passes_exceptionality_filter_v06"]
        & gdf["passes_under_recognition_filter_v06"]
    )

    ranked = gdf[gdf["passes_urem_candidate_filter_v06"]].copy()

    # Important: rank by raw score to preserve top-tail differences.
    ranked = ranked.sort_values(
        "urem_score_v06_raw",
        ascending=False,
    ).reset_index(drop=True)

    ranked["candidate_rank_v06"] = ranked.index + 1
    ranked["urem_rank_v06"] = ranked["candidate_rank_v06"]

    ranked["urem_percentile_v06"] = (
        1 - ((ranked["candidate_rank_v06"] - 1) / max(len(ranked), 1))
    )

    def tier(rank):
        if rank <= 100:
            return "top_100"
        if rank <= 500:
            return "top_500"
        if rank <= 1000:
            return "top_1000"
        return "candidate"

    ranked["urem_tier_v06"] = ranked["candidate_rank_v06"].apply(tier)

    log("Summary:")
    log(f"Mean urem_score_v06: {gdf['urem_score_v06'].mean():.4f}")
    log(f"Mean urem_score_v06_fullrange: {gdf['urem_score_v06_fullrange'].mean():.4f}")
    log(f"Max urem_score_v06: {gdf['urem_score_v06'].max():.4f}")
    log(f"Max urem_score_v06_raw: {gdf['urem_score_v06_raw'].max():.4f}")
    log(f"Ranked UREM v06 candidates: {len(ranked):,}")

    if OUT_SCORE_GPKG.exists():
        OUT_SCORE_GPKG.unlink()

    if OUT_RANKED_GPKG.exists():
        OUT_RANKED_GPKG.unlink()

    log(f"Writing score GPKG: {OUT_SCORE_GPKG}")
    gdf.to_file(
        OUT_SCORE_GPKG,
        layer="oregon_urem_score_v06",
        driver="GPKG",
    )

    log(f"Writing score CSV: {OUT_SCORE_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_SCORE_CSV, index=False)

    log(f"Writing ranked candidates GPKG: {OUT_RANKED_GPKG}")
    ranked.to_file(
        OUT_RANKED_GPKG,
        layer="oregon_ranked_urem_candidates_v06",
        driver="GPKG",
    )

    log(f"Writing ranked candidates CSV: {OUT_RANKED_CSV}")
    ranked.drop(columns="geometry").to_csv(OUT_RANKED_CSV, index=False)

    log("Done")


if __name__ == "__main__":
    main()