#!/usr/bin/env python3
"""
Script 29: Build Fingerprint v03

Purpose:
- Build improved comparable-place fingerprint.
- Fingerprint is used ONLY for comparable-place matching.
- Recognition variables are intentionally excluded.

Inputs:
- exceptionality_score_v02.gpkg

Outputs:
- fingerprint_v03.gpkg
- fingerprint_v03.csv
"""

from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data/processed/exceptionality_score_v02.gpkg"

OUT_GPKG = BASE_DIR / "data/processed/fingerprint_v03.gpkg"
OUT_CSV = BASE_DIR / "data/processed/fingerprint_v03.csv"


def log(msg):
    print(f"[29_build_fingerprint_v03] {msg}")


def safe_minmax(series):
    s = pd.to_numeric(series, errors="coerce")

    mn = s.min(skipna=True)
    mx = s.max(skipna=True)

    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.0, index=s.index)

    return (s - mn) / (mx - mn)


def build_fingerprint(gdf):

    log("Building fingerprint variables")

    gdf["fp_coastal_proximity_v03"] = safe_minmax(
        gdf["score_coastline_proximity"]
    )

    gdf["fp_elevation_v03"] = safe_minmax(
        gdf["score_elevation_existing"]
    )

    gdf["fp_relief_v03"] = safe_minmax(
        gdf["score_relief_existing"]
    )

    gdf["fp_slope_v03"] = safe_minmax(
        gdf["score_slope_existing"]
    )

    gdf["fp_coastline_complexity_v03"] = safe_minmax(
        gdf["score_coastline_complexity"]
    )

    gdf["fp_exceptionality_v03"] = safe_minmax(
        gdf["physical_exceptionality_score_v02"]
    )

    gdf["fp_rarity_v03"] = safe_minmax(
        gdf["physical_rarity_score"]
    )

    fp_cols = [
        "fp_coastal_proximity_v03",
        "fp_elevation_v03",
        "fp_relief_v03",
        "fp_slope_v03",
        "fp_coastline_complexity_v03",
        "fp_exceptionality_v03",
        "fp_rarity_v03",
    ]

    gdf["fingerprint_valid_feature_count_v03"] = (
        gdf[fp_cols]
        .notna()
        .sum(axis=1)
    )

    gdf["fingerprint_completeness_v03"] = (
        gdf["fingerprint_valid_feature_count_v03"]
        / len(fp_cols)
    )

    gdf["fingerprint_version"] = "v03"

    return gdf


def main():

    log("Starting Script 29")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    log(f"Reading: {INPUT_PATH}")

    gdf = gpd.read_file(INPUT_PATH)

    log(f"Rows: {len(gdf):,}")
    log(f"CRS: {gdf.crs}")

    gdf = build_fingerprint(gdf)

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

    log(f"Writing: {OUT_GPKG}")
    gdf.to_file(OUT_GPKG, driver="GPKG")

    log(f"Writing: {OUT_CSV}")
    gdf.drop(columns="geometry").to_csv(
        OUT_CSV,
        index=False,
    )

    print("\nFingerprint summary:")

    print(
        gdf[
            [
                "fp_coastal_proximity_v03",
                "fp_elevation_v03",
                "fp_relief_v03",
                "fp_slope_v03",
                "fp_coastline_complexity_v03",
                "fp_exceptionality_v03",
                "fp_rarity_v03",
                "fingerprint_completeness_v03",
            ]
        ].describe()
    )

    log("Done")


if __name__ == "__main__":
    main()