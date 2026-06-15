#!/usr/bin/env python3
"""
65_compute_urem_score_v07_no_coast.py

UREM v07 No-Coast Branch Test

Purpose:
- Test whether UREM is truly coast-dominated.
- Remove direct coastal/scenic-coast variables from exceptionality.
- Use terrain-only exceptionality:
  relief + slope + terrain drama + elevation.
- Keep recognition/residual from v06.
- Generate ranked candidates for comparison against coastal-heavy v06.

Input:
- data/processed/expected_recognition_v06.gpkg

Outputs:
- data/processed/urem_score_v07_no_coast.gpkg
- data/processed/urem_score_v07_no_coast.csv
- data/processed/ranked_urem_candidates_v07_no_coast.gpkg
- data/processed/ranked_urem_candidates_v07_no_coast.csv
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd


SCRIPT_NAME = "65_compute_urem_score_v07_no_coast"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = PROCESSED_DIR / "expected_recognition_v06.gpkg"

OUT_SCORE_GPKG = PROCESSED_DIR / "urem_score_v07_no_coast.gpkg"
OUT_SCORE_CSV = PROCESSED_DIR / "urem_score_v07_no_coast.csv"
OUT_RANKED_GPKG = PROCESSED_DIR / "ranked_urem_candidates_v07_no_coast.gpkg"
OUT_RANKED_CSV = PROCESSED_DIR / "ranked_urem_candidates_v07_no_coast.csv"


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

    log(f"Missing column: {col}. Using default {default}.")
    return pd.Series(default, index=df.index)


def bool_series(df, col):
    if col not in df.columns:
        return pd.Series(True, index=df.index)

    return df[col].astype(str).str.lower().isin(["true", "1", "yes"])


def main():
    log("Starting UREM v07 No-Coast branch test")

    if not INPUT_GPKG.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_GPKG}")

    gdf = gpd.read_file(INPUT_GPKG)

    log(f"Rows: {len(gdf):,}")
    log(f"CRS: {gdf.crs}")

    # -----------------------------
    # Core variables
    # -----------------------------

    gdf["terrain_drama_v03"] = safe_num(gdf, "terrain_drama_v03")
    gdf["local_relief_m"] = safe_num(gdf, "local_relief_m")
    gdf["slope_deg"] = safe_num(gdf, "slope_deg")
    gdf["elevation_m"] = safe_num(gdf, "elevation_m")

    gdf["positive_under_recognition_residual_v06"] = safe_num(
        gdf,
        "positive_under_recognition_residual_v06",
    )

    gdf["recognition_cell_confidence_v04"] = safe_num(
        gdf,
        "recognition_cell_confidence_v04",
        0.5,
    )

    gdf["expected_recognition_confidence_v06"] = safe_num(
        gdf,
        "expected_recognition_confidence_v06",
        0.5,
    )

    gdf["land_area_share"] = safe_num(gdf, "land_area_share", 1.0)

    # -----------------------------
    # Valid land
    # -----------------------------

    gdf["passes_land_filter_v07"] = (
        bool_series(gdf, "valid_land_v06")
        & bool_series(gdf, "is_valid_land_candidate")
        & (gdf["land_area_share"] >= 0.50)
    )

    # -----------------------------
    # Terrain-only exceptionality
    # No coast variables used here.
    # -----------------------------

    gdf["local_relief_norm_v07"] = norm01(gdf["local_relief_m"])
    gdf["slope_norm_v07"] = norm01(gdf["slope_deg"])
    gdf["elevation_norm_v07"] = norm01(gdf["elevation_m"])
    gdf["terrain_drama_norm_v07"] = norm01(gdf["terrain_drama_v03"])

    gdf["terrain_only_exceptionality_v07"] = (
        0.35 * gdf["terrain_drama_norm_v07"]
        + 0.30 * gdf["local_relief_norm_v07"]
        + 0.25 * gdf["slope_norm_v07"]
        + 0.10 * gdf["elevation_norm_v07"]
    ).clip(0, 1)

    # -----------------------------
    # Under-recognition component
    # -----------------------------

    gdf["under_recognition_component_v07"] = norm01(
        gdf["positive_under_recognition_residual_v06"]
    )

    # -----------------------------
    # Confidence component
    # -----------------------------

    gdf["confidence_component_v07"] = (
        0.50 * gdf["recognition_cell_confidence_v04"]
        + 0.50 * gdf["expected_recognition_confidence_v06"]
    ).clip(0, 1)

    # -----------------------------
    # UREM v07 no-coast score
    # Must be terrain exceptional AND under-recognized.
    # -----------------------------

    gdf["urem_score_v07_no_coast_raw"] = (
        gdf["terrain_only_exceptionality_v07"]
        * gdf["under_recognition_component_v07"]
        * (0.70 + 0.30 * gdf["confidence_component_v07"])
        * gdf["passes_land_filter_v07"].astype(float)
    )

    gdf["urem_score_v07_no_coast"] = norm01(
        gdf["urem_score_v07_no_coast_raw"]
    )

    # -----------------------------
    # Candidate filters
    # -----------------------------

    valid_pool = gdf[gdf["passes_land_filter_v07"]].copy()

    terrain_cutoff = valid_pool["terrain_only_exceptionality_v07"].quantile(0.80)
    residual_cutoff = valid_pool["positive_under_recognition_residual_v06"].quantile(0.70)
    score_cutoff = valid_pool["urem_score_v07_no_coast"].quantile(0.90)

    log(f"Terrain exceptionality cutoff: {terrain_cutoff:.4f}")
    log(f"Residual cutoff: {residual_cutoff:.4f}")
    log(f"Score cutoff: {score_cutoff:.4f}")

    gdf["passes_terrain_filter_v07"] = (
        gdf["terrain_only_exceptionality_v07"] >= terrain_cutoff
    )

    gdf["passes_residual_filter_v07"] = (
        gdf["positive_under_recognition_residual_v06"] >= residual_cutoff
    )

    gdf["passes_score_filter_v07"] = (
        gdf["urem_score_v07_no_coast"] >= score_cutoff
    )

    gdf["passes_confidence_filter_v07"] = (
        gdf["confidence_component_v07"] >= 0.25
    )

    gdf["passes_urem_candidate_filter_v07"] = (
        gdf["passes_land_filter_v07"]
        & gdf["passes_terrain_filter_v07"]
        & gdf["passes_residual_filter_v07"]
        & gdf["passes_score_filter_v07"]
        & gdf["passes_confidence_filter_v07"]
    )

    ranked = gdf[gdf["passes_urem_candidate_filter_v07"]].copy()

    ranked = ranked.sort_values(
        "urem_score_v07_no_coast",
        ascending=False,
    ).reset_index(drop=True)

    ranked["candidate_rank_v07_no_coast"] = ranked.index + 1
    ranked["urem_rank_v07_no_coast"] = ranked["candidate_rank_v07_no_coast"]
    ranked["urem_percentile_v07_no_coast"] = 1 - (
        (ranked["candidate_rank_v07_no_coast"] - 1) / max(len(ranked), 1)
    )

    # -----------------------------
    # Summary
    # -----------------------------

    log("Summary:")
    log(f"Mean terrain_only_exceptionality_v07: {gdf['terrain_only_exceptionality_v07'].mean():.4f}")
    log(f"Mean urem_score_v07_no_coast: {gdf['urem_score_v07_no_coast'].mean():.4f}")
    log(f"Max urem_score_v07_no_coast: {gdf['urem_score_v07_no_coast'].max():.4f}")
    log(f"Ranked candidates v07 no-coast: {len(ranked):,}")

    if "distance_to_coast_m" in ranked.columns and len(ranked) > 0:
        d = pd.to_numeric(ranked["distance_to_coast_m"], errors="coerce")
        log(f"Ranked mean distance to coast: {d.mean():.2f} m")
        log(f"Ranked median distance to coast: {d.median():.2f} m")
        log(f"Ranked pct >10km inland: {(d > 10000).mean():.4f}")

    # -----------------------------
    # Write outputs
    # -----------------------------

    log(f"Writing score GPKG: {OUT_SCORE_GPKG}")
    gdf.to_file(
        OUT_SCORE_GPKG,
        layer="urem_score_v07_no_coast",
        driver="GPKG",
    )

    log(f"Writing score CSV: {OUT_SCORE_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_SCORE_CSV, index=False)

    log(f"Writing ranked GPKG: {OUT_RANKED_GPKG}")
    ranked.to_file(
        OUT_RANKED_GPKG,
        layer="ranked_urem_candidates_v07_no_coast",
        driver="GPKG",
    )

    log(f"Writing ranked CSV: {OUT_RANKED_CSV}")
    ranked.drop(columns="geometry").to_csv(OUT_RANKED_CSV, index=False)

    log("Done")


if __name__ == "__main__":
    main()