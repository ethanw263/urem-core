#!/usr/bin/env python3
"""
47_compute_urem_score_v05.py

Compute UREM Score v05 using:
- Exceptionality v03
- Recognition v04
- Expected Recognition v05
- Land-validity filter from v04

Inputs:
- data/processed/expected_recognition_v05.gpkg

Outputs:
- data/processed/urem_score_v05.gpkg
- data/processed/urem_score_v05.csv
- data/processed/ranked_urem_candidates_v05.gpkg
- data/processed/ranked_urem_candidates_v05.csv
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd


SCRIPT_NAME = "47_compute_urem_score_v05"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = PROCESSED_DIR / "expected_recognition_v05.gpkg"

OUT_SCORE_GPKG = PROCESSED_DIR / "urem_score_v05.gpkg"
OUT_SCORE_CSV = PROCESSED_DIR / "urem_score_v05.csv"
OUT_RANKED_GPKG = PROCESSED_DIR / "ranked_urem_candidates_v05.gpkg"
OUT_RANKED_CSV = PROCESSED_DIR / "ranked_urem_candidates_v05.csv"


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def norm01(s):
    s = pd.to_numeric(s, errors="coerce")
    lo = s.quantile(0.02)
    hi = s.quantile(0.98)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=s.index)
    return ((s - lo) / (hi - lo)).clip(0, 1)


def safe_num(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def main():
    log("Starting UREM Score v05")

    if not INPUT_GPKG.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_GPKG}")

    gdf = gpd.read_file(INPUT_GPKG)

    log(f"Rows: {len(gdf):,}")
    log(f"CRS: {gdf.crs}")

    gdf["physical_exceptionality_v03"] = safe_num(gdf, "physical_exceptionality_v03")
    gdf["positive_under_recognition_residual_v05"] = safe_num(
        gdf, "positive_under_recognition_residual_v05"
    )
    gdf["expected_recognition_confidence_v05"] = safe_num(
        gdf, "expected_recognition_confidence_v05", 0.5
    )
    gdf["recognition_cell_confidence_v04"] = safe_num(
        gdf, "recognition_cell_confidence_v04", 0.5
    )
    gdf["land_area_share"] = safe_num(gdf, "land_area_share", 1.0)
    gdf["is_valid_land_candidate"] = safe_num(gdf, "is_valid_land_candidate", 1.0)

    gdf["urem_exceptionality_component_v05"] = gdf["physical_exceptionality_v03"].clip(0, 1)
    gdf["urem_under_recognition_component_v05"] = norm01(
        gdf["positive_under_recognition_residual_v05"]
    )

    gdf["urem_confidence_component_v05"] = (
        0.50 * gdf["expected_recognition_confidence_v05"]
        + 0.50 * gdf["recognition_cell_confidence_v04"]
    ).clip(0, 1)

    gdf["urem_land_validity_component_v05"] = (
        gdf["land_area_share"].clip(0, 1) * gdf["is_valid_land_candidate"].clip(0, 1)
    )

    # Multiplicative score: must be physically exceptional AND under-recognized.
    gdf["urem_score_v05_raw"] = (
        gdf["urem_exceptionality_component_v05"]
        * gdf["urem_under_recognition_component_v05"]
        * (0.65 + 0.35 * gdf["urem_confidence_component_v05"])
        * gdf["urem_land_validity_component_v05"]
    )

    gdf["urem_score_v05"] = norm01(gdf["urem_score_v05_raw"])

    gdf["urem_score_v05_additive_diagnostic"] = (
        0.45 * gdf["urem_exceptionality_component_v05"]
        + 0.35 * gdf["urem_under_recognition_component_v05"]
        + 0.10 * gdf["urem_confidence_component_v05"]
        + 0.10 * gdf["urem_land_validity_component_v05"]
    ).clip(0, 1)

    # Candidate filters
    gdf["passes_land_filter_v05"] = (
        (gdf["is_valid_land_candidate"].astype(str).str.lower().isin(["true", "1"]))
        & (gdf["land_area_share"] >= 0.50)
    )

    gdf["passes_confidence_filter_v05"] = (
        gdf["urem_confidence_component_v05"] >= 0.35
    )

    # Instead of requiring top exceptionality AND top under-recognition separately,
    # use the combined UREM score. This avoids eliminating valid tradeoff cases.
    valid_pool = (
        gdf["passes_land_filter_v05"]
        & gdf["passes_confidence_filter_v05"]
    )

    score_cutoff = gdf.loc[valid_pool, "urem_score_v05"].quantile(0.90)

    gdf["passes_score_filter_v05"] = (
        gdf["urem_score_v05"] >= score_cutoff
    )

    gdf["passes_exceptionality_filter_v05"] = (
        gdf["physical_exceptionality_v03"] >= gdf["physical_exceptionality_v03"].quantile(0.60)
    )

    gdf["passes_under_recognition_filter_v05"] = (
        gdf["positive_under_recognition_residual_v05"] > 0
    )

    gdf["passes_urem_candidate_filter_v05"] = (
        gdf["passes_land_filter_v05"]
        & gdf["passes_confidence_filter_v05"]
        & gdf["passes_score_filter_v05"]
        & gdf["passes_exceptionality_filter_v05"]
        & gdf["passes_under_recognition_filter_v05"]
    )

    ranked = gdf[gdf["passes_urem_candidate_filter_v05"]].copy()
    ranked = ranked.sort_values("urem_score_v05", ascending=False).reset_index(drop=True)
    ranked["candidate_rank_v05"] = ranked.index + 1
    ranked["urem_rank_v05"] = ranked["candidate_rank_v05"]
    ranked["urem_percentile_v05"] = 1 - ((ranked["candidate_rank_v05"] - 1) / max(len(ranked), 1))

    def tier(rank):
        if rank <= 100:
            return "top_100"
        if rank <= 500:
            return "top_500"
        if rank <= 1000:
            return "top_1000"
        return "candidate"

    ranked["urem_tier_v05"] = ranked["candidate_rank_v05"].apply(tier)

    log("Summary:")
    log(f"Mean urem_score_v05: {gdf['urem_score_v05'].mean():.4f}")
    log(f"Max urem_score_v05: {gdf['urem_score_v05'].max():.4f}")
    log(f"Ranked UREM v05 candidates: {len(ranked):,}")

    log(f"Writing score GPKG: {OUT_SCORE_GPKG}")
    gdf.to_file(OUT_SCORE_GPKG, layer="urem_score_v05", driver="GPKG")

    log(f"Writing score CSV: {OUT_SCORE_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_SCORE_CSV, index=False)

    log(f"Writing ranked candidates GPKG: {OUT_RANKED_GPKG}")
    ranked.to_file(OUT_RANKED_GPKG, layer="ranked_urem_candidates_v05", driver="GPKG")

    log(f"Writing ranked candidates CSV: {OUT_RANKED_CSV}")
    ranked.drop(columns="geometry").to_csv(OUT_RANKED_CSV, index=False)

    log("Done")


if __name__ == "__main__":
    main()