#!/usr/bin/env python3
"""
50_compute_urem_score_v05b.py

Compute UREM v05b as a calibrated hybrid model.

Purpose:
- Preserve v04 diversity.
- Add Exceptionality v03 as a refinement, not a replacement.
- Avoid v05 collapse onto only rugged northern coast.

Inputs:
- data/processed/urem_score_v04.gpkg
- data/processed/exceptionality_score_v03.gpkg

Outputs:
- data/processed/urem_score_v05b.gpkg
- data/processed/urem_score_v05b.csv
- data/processed/ranked_urem_candidates_v05b.gpkg
- data/processed/ranked_urem_candidates_v05b.csv
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd


SCRIPT_NAME = "50_compute_urem_score_v05b"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

UREM_V04 = PROCESSED_DIR / "urem_score_v04.gpkg"
EXC_V03 = PROCESSED_DIR / "exceptionality_score_v03.gpkg"

OUT_SCORE_GPKG = PROCESSED_DIR / "urem_score_v05b.gpkg"
OUT_SCORE_CSV = PROCESSED_DIR / "urem_score_v05b.csv"
OUT_RANKED_GPKG = PROCESSED_DIR / "ranked_urem_candidates_v05b.gpkg"
OUT_RANKED_CSV = PROCESSED_DIR / "ranked_urem_candidates_v05b.csv"


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


def bool_series(df, col):
    if col not in df.columns:
        return pd.Series(True, index=df.index)
    return df[col].astype(str).str.lower().isin(["true", "1", "yes"])


def main():
    log("Starting UREM Score v05b calibrated hybrid")

    if not UREM_V04.exists():
        raise FileNotFoundError(f"Missing {UREM_V04}")
    if not EXC_V03.exists():
        raise FileNotFoundError(f"Missing {EXC_V03}")

    base = gpd.read_file(UREM_V04)
    exc = gpd.read_file(EXC_V03)

    keep_exc = [
        "cell_id",
        "physical_exceptionality_v03",
        "terrain_drama_v03",
        "scenic_coast_v03",
        "flat_coastal_edge_penalty_v03",
        "complex_flat_shoreline_penalty_v03",
    ]
    keep_exc = [c for c in keep_exc if c in exc.columns]

    gdf = base.merge(
        pd.DataFrame(exc[keep_exc]),
        on="cell_id",
        how="left",
        suffixes=("", "_from_v03"),
    )

    gdf["urem_score_v04"] = safe_num(gdf, "urem_score_v04")
    gdf["physical_exceptionality_score_v02"] = safe_num(gdf, "physical_exceptionality_score_v02")
    gdf["positive_under_recognition_residual_v04"] = safe_num(
        gdf, "positive_under_recognition_residual_v04"
    )

    gdf["physical_exceptionality_v03"] = safe_num(gdf, "physical_exceptionality_v03")
    gdf["terrain_drama_v03"] = safe_num(gdf, "terrain_drama_v03")
    gdf["scenic_coast_v03"] = safe_num(gdf, "scenic_coast_v03")
    gdf["flat_coastal_edge_penalty_v03"] = safe_num(gdf, "flat_coastal_edge_penalty_v03")
    gdf["complex_flat_shoreline_penalty_v03"] = safe_num(
        gdf, "complex_flat_shoreline_penalty_v03"
    )

    gdf["land_area_share"] = safe_num(gdf, "land_area_share", 1.0)
    gdf["recognition_cell_confidence_v04"] = safe_num(
        gdf, "recognition_cell_confidence_v04", 0.5
    )

    gdf["passes_land_filter_v05b"] = (
        bool_series(gdf, "is_valid_land_candidate")
        & (gdf["land_area_share"] >= 0.50)
    )

    # v03 should refine, not dominate.
    gdf["v03_refinement_bonus_v05b"] = (
        0.40 * gdf["physical_exceptionality_v03"]
        + 0.25 * gdf["terrain_drama_v03"]
        + 0.25 * gdf["scenic_coast_v03"]
        - 0.10 * gdf["flat_coastal_edge_penalty_v03"]
    ).clip(0, 1)

    gdf["v03_flat_edge_penalty_v05b"] = (
        0.65 * gdf["flat_coastal_edge_penalty_v03"]
        + 0.35 * gdf["complex_flat_shoreline_penalty_v03"]
    ).clip(0, 1)

    raw = (
        0.70 * gdf["urem_score_v04"]
        + 0.20 * gdf["v03_refinement_bonus_v05b"]
        + 0.10 * gdf["positive_under_recognition_residual_v04"]
        - 0.08 * gdf["v03_flat_edge_penalty_v05b"]
    )

    raw = raw * gdf["passes_land_filter_v05b"].astype(float)

    gdf["urem_score_v05b_raw"] = raw
    gdf["urem_score_v05b"] = norm01(raw)

    valid_pool = gdf[gdf["passes_land_filter_v05b"]].copy()

    score_cutoff = valid_pool["urem_score_v05b"].quantile(0.90)
    exceptionality_floor = gdf["physical_exceptionality_score_v02"].quantile(0.60)
    residual_floor = gdf["positive_under_recognition_residual_v04"].quantile(0.50)

    gdf["passes_score_filter_v05b"] = gdf["urem_score_v05b"] >= score_cutoff
    gdf["passes_exceptionality_filter_v05b"] = (
        gdf["physical_exceptionality_score_v02"] >= exceptionality_floor
    )
    gdf["passes_under_recognition_filter_v05b"] = (
        gdf["positive_under_recognition_residual_v04"] >= residual_floor
    )
    gdf["passes_confidence_filter_v05b"] = (
        gdf["recognition_cell_confidence_v04"] >= 0.25
    )

    gdf["passes_urem_candidate_filter_v05b"] = (
        gdf["passes_land_filter_v05b"]
        & gdf["passes_score_filter_v05b"]
        & gdf["passes_exceptionality_filter_v05b"]
        & gdf["passes_under_recognition_filter_v05b"]
        & gdf["passes_confidence_filter_v05b"]
    )

    ranked = gdf[gdf["passes_urem_candidate_filter_v05b"]].copy()
    ranked = ranked.sort_values("urem_score_v05b", ascending=False).reset_index(drop=True)
    ranked["candidate_rank_v05b"] = ranked.index + 1
    ranked["urem_rank_v05b"] = ranked["candidate_rank_v05b"]
    ranked["urem_percentile_v05b"] = 1 - (
        (ranked["candidate_rank_v05b"] - 1) / max(len(ranked), 1)
    )

    log("Summary:")
    log(f"Mean urem_score_v05b: {gdf['urem_score_v05b'].mean():.4f}")
    log(f"Max urem_score_v05b: {gdf['urem_score_v05b'].max():.4f}")
    log(f"Ranked UREM v05b candidates: {len(ranked):,}")

    log(f"Writing score GPKG: {OUT_SCORE_GPKG}")
    gdf.to_file(OUT_SCORE_GPKG, layer="urem_score_v05b", driver="GPKG")

    log(f"Writing score CSV: {OUT_SCORE_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_SCORE_CSV, index=False)

    log(f"Writing ranked candidates GPKG: {OUT_RANKED_GPKG}")
    ranked.to_file(OUT_RANKED_GPKG, layer="ranked_urem_candidates_v05b", driver="GPKG")

    log(f"Writing ranked candidates CSV: {OUT_RANKED_CSV}")
    ranked.drop(columns="geometry").to_csv(OUT_RANKED_CSV, index=False)

    log("Done")


if __name__ == "__main__":
    main()