#!/usr/bin/env python3
"""
122_build_oregon_exceptionality_v03.py

Build Oregon Exceptionality v03.

Mirrors California Script 45.

Inputs:
- data/processed/oregon_exceptionality_features_v02.gpkg

Outputs:
- data/processed/oregon_exceptionality_features_v03.gpkg
- data/processed/oregon_exceptionality_features_v03.csv
- data/processed/oregon_exceptionality_score_v03.gpkg
- data/processed/oregon_exceptionality_score_v03.csv
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd


SCRIPT_NAME = "122_build_oregon_exceptionality_v03"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = PROCESSED_DIR / "oregon_exceptionality_features_v02.gpkg"

OUT_FEATURES_GPKG = PROCESSED_DIR / "oregon_exceptionality_features_v03.gpkg"
OUT_FEATURES_CSV = PROCESSED_DIR / "oregon_exceptionality_features_v03.csv"

OUT_SCORE_GPKG = PROCESSED_DIR / "oregon_exceptionality_score_v03.gpkg"
OUT_SCORE_CSV = PROCESSED_DIR / "oregon_exceptionality_score_v03.csv"


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def norm01(series):
    s = pd.to_numeric(series, errors="coerce")

    lo = s.quantile(0.02)
    hi = s.quantile(0.98)

    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=s.index)

    return ((s - lo) / (hi - lo)).clip(0, 1)


def col(gdf, name, default=0.0):
    if name in gdf.columns:
        return pd.to_numeric(gdf[name], errors="coerce").fillna(0)

    log(f"Missing column: {name}. Using {default}.")
    return pd.Series(default, index=gdf.index)


def main():
    log("Starting Oregon Exceptionality v03")

    if not INPUT_GPKG.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_GPKG}")

    gdf = gpd.read_file(INPUT_GPKG)

    log(f"Rows: {len(gdf):,}")
    log(f"CRS: {gdf.crs}")

    # --------------------------------------------------
    # Base exceptionality
    # --------------------------------------------------

    gdf["base_exceptionality_v02_norm"] = norm01(
        col(gdf, "exceptionality_preview_score_v02")
    )

    # --------------------------------------------------
    # Terrain
    # --------------------------------------------------

    gdf["relief_norm_v03"] = norm01(col(gdf, "local_relief_m"))
    gdf["relief_score_v03"] = norm01(col(gdf, "relief_score"))

    gdf["slope_norm_v03"] = norm01(col(gdf, "slope_deg"))
    gdf["slope_score_v03"] = norm01(col(gdf, "slope_score"))

    gdf["elevation_norm_v03"] = norm01(col(gdf, "elevation_m"))

    gdf["terrain_drama_v03"] = (
        0.35 * gdf["relief_norm_v03"]
        + 0.25 * gdf["relief_score_v03"]
        + 0.25 * gdf["slope_norm_v03"]
        + 0.10 * gdf["slope_score_v03"]
        + 0.05 * gdf["elevation_norm_v03"]
    ).clip(0, 1)

    # --------------------------------------------------
    # Coast
    # --------------------------------------------------

    gdf["coast_proximity_v03"] = norm01(
        col(gdf, "score_coastline_proximity")
    )

    gdf["beach_proximity_v03"] = norm01(
        col(gdf, "score_beach_proximity")
    )

    gdf["cliff_proximity_v03"] = norm01(
        col(gdf, "score_cliff_proximity")
    )

    gdf["coast_complexity_v03"] = norm01(
        col(gdf, "score_coastline_complexity")
    )

    gdf["coastline_length_norm_v03"] = norm01(
        col(gdf, "coastline_length_3km_m")
    )

    gdf["coastline_complexity_norm_v03"] = norm01(
        col(gdf, "coastline_complexity_3km")
    )

    # --------------------------------------------------
    # Scenic Coast
    # --------------------------------------------------

    gdf["scenic_coast_v03"] = (
        gdf["coast_proximity_v03"]
        * (
            0.35 * gdf["terrain_drama_v03"]
            + 0.25 * gdf["cliff_proximity_v03"]
            + 0.20 * gdf["beach_proximity_v03"]
            + 0.20 * gdf["coast_complexity_v03"]
        )
    ).clip(0, 1)

    # --------------------------------------------------
    # Penalties
    # --------------------------------------------------

    gdf["flat_coastal_edge_penalty_v03"] = (
        gdf["coast_proximity_v03"]
        * (1 - gdf["terrain_drama_v03"])
        * (1 - gdf["cliff_proximity_v03"])
        * 0.85
    ).clip(0, 1)

    gdf["complex_flat_shoreline_penalty_v03"] = (
        gdf["coast_complexity_v03"]
        * (1 - gdf["terrain_drama_v03"])
        * 0.65
    ).clip(0, 1)

    # --------------------------------------------------
    # Final Exceptionality
    # --------------------------------------------------

    raw = (
        0.20 * gdf["base_exceptionality_v02_norm"]
        + 0.30 * gdf["terrain_drama_v03"]
        + 0.25 * gdf["scenic_coast_v03"]
        + 0.10 * gdf["cliff_proximity_v03"]
        + 0.07 * gdf["beach_proximity_v03"]
        + 0.08 * gdf["coast_complexity_v03"]
        - 0.18 * gdf["flat_coastal_edge_penalty_v03"]
        - 0.12 * gdf["complex_flat_shoreline_penalty_v03"]
    )

    gdf["physical_exceptionality_raw_v03"] = raw
    gdf["physical_exceptionality_v03"] = norm01(raw)

    gdf["exceptionality_v03_rank"] = (
        gdf["physical_exceptionality_v03"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # --------------------------------------------------
    # QA
    # --------------------------------------------------

    log("Summary")
    log(
        f"Mean physical_exceptionality_v03: "
        f"{gdf['physical_exceptionality_v03'].mean():.4f}"
    )

    log(
        f"Max physical_exceptionality_v03: "
        f"{gdf['physical_exceptionality_v03'].max():.4f}"
    )

    log(
        f"Top rank score: "
        f"{gdf['physical_exceptionality_v03'].max():.4f}"
    )

    # --------------------------------------------------
    # Outputs
    # --------------------------------------------------

    log(f"Writing features GPKG: {OUT_FEATURES_GPKG}")

    gdf.to_file(
        OUT_FEATURES_GPKG,
        layer="oregon_exceptionality_features_v03",
        driver="GPKG",
    )

    log(f"Writing features CSV: {OUT_FEATURES_CSV}")

    gdf.drop(columns="geometry").to_csv(
        OUT_FEATURES_CSV,
        index=False,
    )

    keep_cols = [
        "cell_id",
        "physical_exceptionality_v03",
        "physical_exceptionality_raw_v03",
        "exceptionality_v03_rank",
        "base_exceptionality_v02_norm",
        "terrain_drama_v03",
        "scenic_coast_v03",
        "relief_norm_v03",
        "slope_norm_v03",
        "coast_proximity_v03",
        "beach_proximity_v03",
        "cliff_proximity_v03",
        "coast_complexity_v03",
        "flat_coastal_edge_penalty_v03",
        "complex_flat_shoreline_penalty_v03",
        "geometry",
    ]

    score_gdf = gdf[
        [c for c in keep_cols if c in gdf.columns]
    ].copy()

    log(f"Writing score GPKG: {OUT_SCORE_GPKG}")

    score_gdf.to_file(
        OUT_SCORE_GPKG,
        layer="oregon_exceptionality_score_v03",
        driver="GPKG",
    )

    log(f"Writing score CSV: {OUT_SCORE_CSV}")

    score_gdf.drop(columns="geometry").to_csv(
        OUT_SCORE_CSV,
        index=False,
    )

    log("Done")


if __name__ == "__main__":
    main()