#!/usr/bin/env python3
"""
Script 31: Compute UREM Score v03

Purpose:
- Combine physical exceptionality, under-recognition residual, and confidence.
- Produce final UREM v03 candidate ranking.

Inputs:
- data/processed/expected_recognition_v03.gpkg

Outputs:
- data/processed/urem_score_v03.gpkg
- data/processed/urem_score_v03.csv
- data/processed/ranked_urem_candidates_v03.gpkg
- data/processed/ranked_urem_candidates_v03.csv
"""

from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data/processed/expected_recognition_v03.gpkg"

OUT_GPKG = BASE_DIR / "data/processed/urem_score_v03.gpkg"
OUT_CSV = BASE_DIR / "data/processed/urem_score_v03.csv"

RANKED_GPKG = BASE_DIR / "data/processed/ranked_urem_candidates_v03.gpkg"
RANKED_CSV = BASE_DIR / "data/processed/ranked_urem_candidates_v03.csv"


def log(msg: str) -> None:
    print(f"[31_compute_urem_score_v03] {msg}")


def safe_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    mn = s.min()
    mx = s.max()

    if mx == mn:
        return pd.Series(0.0, index=s.index)

    return (s - mn) / (mx - mn)


def require_columns(gdf: gpd.GeoDataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_urem_v03(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Computing UREM v03 score")

    required = [
        "physical_exceptionality_score_v02",
        "positive_under_recognition_residual_v03",
        "comparable_confidence_v03",
        "recognition_cell_confidence_v03",
        "observed_recognition_v03",
        "expected_recognition_v03",
    ]

    require_columns(gdf, required)

    gdf["urem_exceptionality_component_v03"] = safe_minmax(
        gdf["physical_exceptionality_score_v02"]
    )

    gdf["urem_under_recognition_component_v03"] = safe_minmax(
        gdf["positive_under_recognition_residual_v03"]
    )

    gdf["urem_comparable_confidence_component_v03"] = safe_minmax(
        gdf["comparable_confidence_v03"]
    )

    gdf["urem_recognition_confidence_component_v03"] = safe_minmax(
        gdf["recognition_cell_confidence_v03"]
    )

    # Main multiplicative score.
    # This forces a good candidate to have:
    # - physical exceptionality
    # - under-recognition
    # - confidence
    gdf["urem_score_v03_raw"] = (
        gdf["urem_exceptionality_component_v03"]
        * gdf["urem_under_recognition_component_v03"]
        * (
            0.70 * gdf["comparable_confidence_v03"]
            + 0.30 * gdf["recognition_cell_confidence_v03"]
        )
    )

    gdf["urem_score_v03"] = safe_minmax(gdf["urem_score_v03_raw"])

    # Secondary additive diagnostic score.
    # Useful for checking whether multiplicative score is too harsh.
    gdf["urem_score_v03_additive_diagnostic"] = safe_minmax(
        0.45 * gdf["urem_exceptionality_component_v03"]
        + 0.40 * gdf["urem_under_recognition_component_v03"]
        + 0.15 * (
            0.70 * gdf["comparable_confidence_v03"]
            + 0.30 * gdf["recognition_cell_confidence_v03"]
        )
    )

    return gdf


def apply_candidate_filters(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Applying candidate filters and tiers")

    exceptionality_q60 = gdf["physical_exceptionality_score_v02"].quantile(0.60)
    residual_q60 = gdf["positive_under_recognition_residual_v03"].quantile(0.60)
    confidence_q25 = gdf["comparable_confidence_v03"].quantile(0.25)

    gdf["passes_exceptionality_filter_v03"] = (
        gdf["physical_exceptionality_score_v02"] >= exceptionality_q60
    )

    gdf["passes_under_recognition_filter_v03"] = (
        gdf["positive_under_recognition_residual_v03"] >= residual_q60
    )

    gdf["passes_confidence_filter_v03"] = (
        gdf["comparable_confidence_v03"] >= confidence_q25
    )

    gdf["passes_urem_candidate_filter_v03"] = (
        gdf["passes_exceptionality_filter_v03"]
        & gdf["passes_under_recognition_filter_v03"]
        & gdf["passes_confidence_filter_v03"]
    )

    gdf["urem_rank_v03"] = (
        gdf["urem_score_v03"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    gdf["urem_percentile_v03"] = (
        gdf["urem_score_v03"]
        .rank(pct=True)
    )

    gdf["urem_tier_v03"] = pd.cut(
        gdf["urem_percentile_v03"],
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
    log("Creating ranked UREM v03 candidate output")

    ranked = gdf[gdf["passes_urem_candidate_filter_v03"]].copy()

    ranked = ranked.sort_values(
        by="urem_score_v03",
        ascending=False,
    )

    ranked["candidate_rank_v03"] = range(1, len(ranked) + 1)

    return ranked


def main():
    log("Starting Script 31")

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

    gdf = compute_urem_v03(gdf)
    gdf = apply_candidate_filters(gdf)

    ranked = create_ranked_candidates(gdf)

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

    log(f"Writing full UREM score GeoPackage: {OUT_GPKG}")
    gdf.to_file(OUT_GPKG, driver="GPKG")

    log(f"Writing full UREM score CSV: {OUT_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log(f"Writing ranked candidates GeoPackage: {RANKED_GPKG}")
    ranked.to_file(RANKED_GPKG, driver="GPKG")

    log(f"Writing ranked candidates CSV: {RANKED_CSV}")
    ranked.drop(columns="geometry").to_csv(RANKED_CSV, index=False)

    log("Done")

    print("\nUREM v03 summary:")
    print(
        gdf[
            [
                "physical_exceptionality_score_v02",
                "observed_recognition_v03",
                "expected_recognition_v03",
                "positive_under_recognition_residual_v03",
                "comparable_confidence_v03",
                "recognition_cell_confidence_v03",
                "urem_score_v03",
                "urem_score_v03_additive_diagnostic",
            ]
        ].describe()
    )

    print("\nCandidate filter counts:")
    print(
        gdf[
            [
                "passes_exceptionality_filter_v03",
                "passes_under_recognition_filter_v03",
                "passes_confidence_filter_v03",
                "passes_urem_candidate_filter_v03",
            ]
        ].sum()
    )

    print("\nUREM tier counts:")
    print(gdf["urem_tier_v03"].value_counts())

    print(f"\nRanked candidates: {len(ranked):,}")


if __name__ == "__main__":
    main()