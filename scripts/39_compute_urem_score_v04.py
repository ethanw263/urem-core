#!/usr/bin/env python3
"""
Script 39: Compute UREM Score v04

Purpose:
- Compute final UREM v04 score.
- Uses improved Recognition v04.
- Excludes invalid water/ocean cells from final candidates.

Input:
- data/processed/expected_recognition_v04.gpkg

Outputs:
- data/processed/urem_score_v04.gpkg
- data/processed/urem_score_v04.csv
- data/processed/ranked_urem_candidates_v04.gpkg
- data/processed/ranked_urem_candidates_v04.csv
"""

from pathlib import Path
import warnings

import geopandas as gpd
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data/processed/expected_recognition_v04.gpkg"

OUT_GPKG = BASE_DIR / "data/processed/urem_score_v04.gpkg"
OUT_CSV = BASE_DIR / "data/processed/urem_score_v04.csv"

RANKED_GPKG = BASE_DIR / "data/processed/ranked_urem_candidates_v04.gpkg"
RANKED_CSV = BASE_DIR / "data/processed/ranked_urem_candidates_v04.csv"


def log(msg: str) -> None:
    print(f"[39_compute_urem_score_v04] {msg}")


def safe_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    mn = s.min()
    mx = s.max()
    if mx == mn:
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)


def require_columns(gdf, cols):
    missing = [c for c in cols if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_score(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Computing UREM v04 score")

    required = [
        "physical_exceptionality_score_v02",
        "positive_under_recognition_residual_v04",
        "comparable_confidence_v04",
        "recognition_cell_confidence_v04",
        "observed_recognition_v04",
        "expected_recognition_v04",
        "is_valid_land_candidate",
    ]

    require_columns(gdf, required)

    gdf["urem_exceptionality_component_v04"] = safe_minmax(
        gdf["physical_exceptionality_score_v02"]
    )

    gdf["urem_under_recognition_component_v04"] = safe_minmax(
        gdf["positive_under_recognition_residual_v04"]
    )

    gdf["urem_confidence_component_v04"] = (
        0.65 * gdf["comparable_confidence_v04"]
        + 0.35 * gdf["recognition_cell_confidence_v04"]
    ).clip(0, 1)

    gdf["urem_land_validity_component_v04"] = (
        gdf["is_valid_land_candidate"].astype(bool).astype(float)
    )

    # Main v04 score.
    # Invalid land cells get forced to zero.
    gdf["urem_score_v04_raw"] = (
        gdf["urem_exceptionality_component_v04"]
        * gdf["urem_under_recognition_component_v04"]
        * gdf["urem_confidence_component_v04"]
        * gdf["urem_land_validity_component_v04"]
    )

    gdf["urem_score_v04"] = safe_minmax(gdf["urem_score_v04_raw"])

    # Diagnostic additive version.
    gdf["urem_score_v04_additive_diagnostic"] = safe_minmax(
        (
            0.45 * gdf["urem_exceptionality_component_v04"]
            + 0.40 * gdf["urem_under_recognition_component_v04"]
            + 0.15 * gdf["urem_confidence_component_v04"]
        )
        * gdf["urem_land_validity_component_v04"]
    )

    return gdf


def apply_candidate_filters(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Applying v04 candidate filters")

    valid = gdf[gdf["is_valid_land_candidate"].astype(bool)].copy()

    exceptionality_cutoff = valid["physical_exceptionality_score_v02"].quantile(0.60)
    residual_cutoff = valid["positive_under_recognition_residual_v04"].quantile(0.60)
    confidence_cutoff = valid["comparable_confidence_v04"].quantile(0.25)

    gdf["passes_land_filter_v04"] = gdf["is_valid_land_candidate"].astype(bool)

    gdf["passes_exceptionality_filter_v04"] = (
        gdf["physical_exceptionality_score_v02"] >= exceptionality_cutoff
    )

    gdf["passes_under_recognition_filter_v04"] = (
        gdf["positive_under_recognition_residual_v04"] >= residual_cutoff
    )

    gdf["passes_confidence_filter_v04"] = (
        gdf["comparable_confidence_v04"] >= confidence_cutoff
    )

    gdf["passes_urem_candidate_filter_v04"] = (
        gdf["passes_land_filter_v04"]
        & gdf["passes_exceptionality_filter_v04"]
        & gdf["passes_under_recognition_filter_v04"]
        & gdf["passes_confidence_filter_v04"]
    )

    gdf["urem_rank_v04"] = (
        gdf["urem_score_v04"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    gdf["urem_percentile_v04"] = gdf["urem_score_v04"].rank(pct=True)

    gdf["urem_tier_v04"] = pd.cut(
        gdf["urem_percentile_v04"],
        bins=[-0.001, 0.80, 0.95, 0.99, 1.001],
        labels=[
            "background",
            "candidate",
            "strong_candidate",
            "top_candidate",
        ],
    ).astype(str)

    return gdf


def create_ranked_candidates(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Creating ranked UREM v04 candidates")

    ranked = gdf[gdf["passes_urem_candidate_filter_v04"]].copy()

    ranked = ranked.sort_values(
        "urem_score_v04",
        ascending=False,
    ).reset_index(drop=True)

    ranked["candidate_rank_v04"] = range(1, len(ranked) + 1)

    return ranked


def main():
    log("Starting Script 39")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    log(f"Reading input: {INPUT_PATH}")
    gdf = gpd.read_file(INPUT_PATH)

    if gdf.empty:
        raise ValueError("Input is empty")

    if gdf.crs is None:
        raise ValueError("Input has no CRS")

    log(f"Rows: {len(gdf):,}")
    log(f"CRS: {gdf.crs}")

    gdf = compute_score(gdf)
    gdf = apply_candidate_filters(gdf)
    ranked = create_ranked_candidates(gdf)

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

    log(f"Writing full UREM v04 GeoPackage: {OUT_GPKG}")
    gdf.to_file(OUT_GPKG, driver="GPKG")

    log(f"Writing full UREM v04 CSV: {OUT_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log(f"Writing ranked candidates GeoPackage: {RANKED_GPKG}")
    ranked.to_file(RANKED_GPKG, driver="GPKG")

    log(f"Writing ranked candidates CSV: {RANKED_CSV}")
    ranked.drop(columns="geometry").to_csv(RANKED_CSV, index=False)

    log("Done")

    print("\nUREM v04 summary:")
    print(
        gdf[
            [
                "physical_exceptionality_score_v02",
                "observed_recognition_v04",
                "expected_recognition_v04",
                "positive_under_recognition_residual_v04",
                "comparable_confidence_v04",
                "recognition_cell_confidence_v04",
                "is_valid_land_candidate",
                "urem_score_v04",
                "urem_score_v04_additive_diagnostic",
            ]
        ].describe(include="all")
    )

    print("\nCandidate filter counts:")
    print(
        gdf[
            [
                "passes_land_filter_v04",
                "passes_exceptionality_filter_v04",
                "passes_under_recognition_filter_v04",
                "passes_confidence_filter_v04",
                "passes_urem_candidate_filter_v04",
            ]
        ].sum()
    )

    print("\nUREM tier counts:")
    print(gdf["urem_tier_v04"].value_counts())

    print(f"\nRanked candidates: {len(ranked):,}")


if __name__ == "__main__":
    main()