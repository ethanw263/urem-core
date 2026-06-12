#!/usr/bin/env python3
"""
Script 26: Compute Exceptionality Score v02

Purpose:
- Convert Script 25 features into a cleaner physical exceptionality score.
- This score is independent of recognition.
- Later UREM v03 should use this as the "exceptionality" part of:
  UREM v03 = Exceptionality × Under-recognition × Confidence

Inputs:
- data/processed/exceptionality_features_v02.gpkg

Outputs:
- data/processed/exceptionality_score_v02.gpkg
- data/processed/exceptionality_score_v02.csv
"""

from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data/processed/exceptionality_features_v02.gpkg"

OUT_GPKG = BASE_DIR / "data/processed/exceptionality_score_v02.gpkg"
OUT_CSV = BASE_DIR / "data/processed/exceptionality_score_v02.csv"


def log(msg: str) -> None:
    print(f"[26_compute_exceptionality_score_v02] {msg}")


def safe_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    min_val = s.min(skipna=True)
    max_val = s.max(skipna=True)

    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(0.0, index=series.index)

    return (s - min_val) / (max_val - min_val)


def safe_percentile_rank(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.rank(pct=True).fillna(0.0)


def require_columns(gdf: gpd.GeoDataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_group_scores(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Computing terrain, coastal, and complexity group scores")

    required = [
        "score_relief_existing",
        "score_slope_existing",
        "score_elevation_existing",
        "score_coastline_proximity",
        "score_coastline_complexity",
    ]

    require_columns(gdf, required)

    # Terrain score:
    # Relief and slope matter more than elevation alone.
    gdf["terrain_exceptionality_score"] = (
        0.45 * gdf["score_relief_existing"]
        + 0.40 * gdf["score_slope_existing"]
        + 0.15 * gdf["score_elevation_existing"]
    )

    # Coastal relationship score:
    # Nearer coastline is good, but coastline complexity matters too.
    gdf["coastal_exceptionality_score"] = (
        0.65 * gdf["score_coastline_proximity"]
        + 0.35 * gdf["score_coastline_complexity"]
    )

    # Complexity score:
    # A separate score to reward irregular coastal geometry.
    gdf["landscape_complexity_score"] = gdf["score_coastline_complexity"]

    return gdf


def compute_rarity_score(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Rarity score asks:
    How unusual is this cell relative to the full coastal grid?

    This is a simple v02 rarity proxy:
    - High relief percentile
    - High slope percentile
    - High coastline complexity percentile

    Later versions can use multivariate distance / density.
    """

    log("Computing physical rarity score")

    rarity_components = []

    for col in [
        "score_relief_existing",
        "score_slope_existing",
        "score_coastline_complexity",
    ]:
        if col in gdf.columns:
            rank_col = f"{col}_percentile"
            gdf[rank_col] = safe_percentile_rank(gdf[col])
            rarity_components.append(rank_col)

    if not rarity_components:
        gdf["physical_rarity_score"] = 0.0
    else:
        gdf["physical_rarity_score"] = gdf[rarity_components].mean(axis=1)

    return gdf


def compute_exceptionality_score(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Computing final physical exceptionality score v02")

    # Core score:
    # Terrain is currently strongest because beaches/cliffs are not available yet.
    # Coastal score matters, but coastline complexity is sparse.
    gdf["physical_exceptionality_score_v02_raw"] = (
        0.50 * gdf["terrain_exceptionality_score"]
        + 0.25 * gdf["coastal_exceptionality_score"]
        + 0.15 * gdf["landscape_complexity_score"]
        + 0.10 * gdf["physical_rarity_score"]
    )

    gdf["physical_exceptionality_score_v02"] = safe_minmax(
        gdf["physical_exceptionality_score_v02_raw"]
    )

    return gdf


def compute_confidence(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Computing exceptionality confidence score")

    # Data availability confidence.
    feature_cols = [
        "score_relief_existing",
        "score_slope_existing",
        "score_elevation_existing",
        "score_coastline_proximity",
        "score_coastline_complexity",
    ]

    available = [c for c in feature_cols if c in gdf.columns]
    gdf["exceptionality_data_coverage"] = gdf[available].notna().sum(axis=1) / len(feature_cols)

    # Coastline complexity confidence:
    # If coastline length is zero, that does not mean bad data.
    # It may simply mean the grid cell is not directly adjacent to complex coastline.
    # So we do not penalize zero coastline complexity heavily.
    gdf["exceptionality_confidence_v02"] = gdf["exceptionality_data_coverage"].clip(0, 1)

    return gdf


def assign_tiers(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Assigning exceptionality tiers")

    score = gdf["physical_exceptionality_score_v02"]

    gdf["exceptionality_tier_v02"] = pd.cut(
        score,
        bins=[-0.001, 0.50, 0.70, 0.85, 1.001],
        labels=[
            "low",
            "moderate",
            "high",
            "very_high",
        ],
    ).astype(str)

    return gdf


def main():
    log("Starting Script 26")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    log(f"Reading input: {INPUT_PATH}")
    gdf = gpd.read_file(INPUT_PATH)

    if gdf.empty:
        raise ValueError("Input file is empty")

    if gdf.crs is None:
        raise ValueError("Input file has no CRS")

    log(f"Rows: {len(gdf):,}")
    log(f"CRS: {gdf.crs}")

    gdf = compute_group_scores(gdf)
    gdf = compute_rarity_score(gdf)
    gdf = compute_exceptionality_score(gdf)
    gdf = compute_confidence(gdf)
    gdf = assign_tiers(gdf)

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

    log(f"Writing GeoPackage: {OUT_GPKG}")
    gdf.to_file(OUT_GPKG, driver="GPKG")

    log(f"Writing CSV: {OUT_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")

    print("\nTop-level score summary:")
    print(
        gdf[
            [
                "terrain_exceptionality_score",
                "coastal_exceptionality_score",
                "landscape_complexity_score",
                "physical_rarity_score",
                "physical_exceptionality_score_v02",
                "exceptionality_confidence_v02",
            ]
        ].describe()
    )

    print("\nTier counts:")
    print(gdf["exceptionality_tier_v02"].value_counts())


if __name__ == "__main__":
    main()