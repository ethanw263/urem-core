#!/usr/bin/env python3
"""
Script 28: Compute Recognition Score v03

Purpose:
- Convert Recognition v03 features into final observed recognition scores.
- Keep category-level recognition scores for interpretability.
- Add recognition confidence / coverage flags.
- Identify known missing-recognition limitations.

Inputs:
- data/processed/recognition_features_v03.gpkg

Outputs:
- data/processed/recognition_score_v03.gpkg
- data/processed/recognition_score_v03.csv
"""

from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data/processed/recognition_features_v03.gpkg"

OUT_GPKG = BASE_DIR / "data/processed/recognition_score_v03.gpkg"
OUT_CSV = BASE_DIR / "data/processed/recognition_score_v03.csv"


def log(msg: str) -> None:
    print(f"[28_compute_recognition_score_v03] {msg}")


def safe_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    mn = s.min()
    mx = s.max()

    if mx == mn:
        return pd.Series(0.0, index=s.index)

    return (s - mn) / (mx - mn)


def safe_log_score(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    return safe_minmax(np.log1p(s))


def require_columns(gdf: gpd.GeoDataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def rebuild_category_scores(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Rebuilding category scores from counts")

    count_cols = [
        "viewpoints_count_3km",
        "peaks_count_3km",
        "attractions_count_3km",
        "campgrounds_count_3km",
        "picnic_sites_count_3km",
        "information_count_3km",
        "parks_recreation_count_3km",
        "trails_paths_count_3km",
        "beaches_count_3km",
        "named_natural_features_count_3km",
        "wiki_features_count_3km",
        "named_features_count_3km",
    ]

    require_columns(gdf, count_cols)

    for col in count_cols:
        score_col = col.replace("_count_3km", "_score_v03")
        gdf[score_col] = safe_log_score(gdf[col])

    return gdf


def compute_group_scores(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Computing final Recognition v03 group scores")

    gdf["recognition_recreation_v03"] = (
        0.25 * gdf["trails_paths_score_v03"]
        + 0.20 * gdf["parks_recreation_score_v03"]
        + 0.15 * gdf["beaches_score_v03"]
        + 0.15 * gdf["campgrounds_score_v03"]
        + 0.10 * gdf["picnic_sites_score_v03"]
        + 0.15 * gdf["named_natural_features_score_v03"]
    )

    gdf["recognition_tourism_v03"] = (
        0.35 * gdf["attractions_score_v03"]
        + 0.30 * gdf["viewpoints_score_v03"]
        + 0.20 * gdf["information_score_v03"]
        + 0.15 * gdf["wiki_features_score_v03"]
    )

    gdf["recognition_natural_landmark_v03"] = (
        0.40 * gdf["peaks_score_v03"]
        + 0.35 * gdf["named_natural_features_score_v03"]
        + 0.25 * gdf["beaches_score_v03"]
    )

    gdf["recognition_general_named_v03"] = (
        0.60 * gdf["named_features_score_v03"]
        + 0.40 * gdf["wiki_features_score_v03"]
    )

    return gdf


def compute_final_score(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Computing final observed recognition score v03")

    gdf["observed_recognition_v03_raw"] = (
        0.35 * gdf["recognition_recreation_v03"]
        + 0.30 * gdf["recognition_tourism_v03"]
        + 0.20 * gdf["recognition_natural_landmark_v03"]
        + 0.15 * gdf["recognition_general_named_v03"]
    )

    gdf["observed_recognition_v03"] = safe_minmax(
        gdf["observed_recognition_v03_raw"]
    )

    count_cols = [c for c in gdf.columns if c.endswith("_count_3km")]
    gdf["recognition_total_count_3km_v03"] = gdf[count_cols].sum(axis=1)

    gdf["recognition_total_count_score_v03"] = safe_log_score(
        gdf["recognition_total_count_3km_v03"]
    )

    return gdf


def compute_confidence_and_flags(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Computing recognition confidence and limitation flags")

    count_cols = [
        "viewpoints_count_3km",
        "peaks_count_3km",
        "attractions_count_3km",
        "campgrounds_count_3km",
        "picnic_sites_count_3km",
        "information_count_3km",
        "parks_recreation_count_3km",
        "trails_paths_count_3km",
        "beaches_count_3km",
        "named_natural_features_count_3km",
        "wiki_features_count_3km",
        "named_features_count_3km",
    ]

    gdf["recognition_category_coverage_v03"] = (
        (gdf[count_cols] > 0).sum(axis=1) / len(count_cols)
    )

    gdf["has_any_recognition_v03"] = (
        gdf["recognition_total_count_3km_v03"] > 0
    )

    gdf["has_strong_recognition_v03"] = (
        gdf["observed_recognition_v03"] >= gdf["observed_recognition_v03"].quantile(0.75)
    )

    # Dataset-level limitation flags.
    # These are constant for all rows but useful metadata.
    gdf["recognition_v03_trails_missing_flag"] = int(gdf["trails_paths_count_3km"].sum() == 0)
    gdf["recognition_v03_beaches_missing_flag"] = int(gdf["beaches_count_3km"].sum() == 0)
    gdf["recognition_v03_parks_sparse_flag"] = int(gdf["parks_recreation_count_3km"].sum() < 100)

    # Confidence:
    # This is NOT saying the model is scientifically validated.
    # It only measures internal feature availability.
    base_confidence = 0.75

    if gdf["recognition_v03_trails_missing_flag"].iloc[0] == 1:
        base_confidence -= 0.15

    if gdf["recognition_v03_beaches_missing_flag"].iloc[0] == 1:
        base_confidence -= 0.10

    if gdf["recognition_v03_parks_sparse_flag"].iloc[0] == 1:
        base_confidence -= 0.10

    base_confidence = max(0.30, base_confidence)

    gdf["recognition_dataset_confidence_v03"] = base_confidence

    # Cell-level confidence: category coverage helps a little, but sparse places can still be valid.
    gdf["recognition_cell_confidence_v03"] = (
        0.70 * gdf["recognition_dataset_confidence_v03"]
        + 0.30 * gdf["recognition_category_coverage_v03"]
    ).clip(0, 1)

    return gdf


def assign_tiers(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Assigning observed recognition tiers")

    gdf["observed_recognition_tier_v03"] = pd.cut(
        gdf["observed_recognition_v03"],
        bins=[-0.001, 0.05, 0.20, 0.50, 1.001],
        labels=[
            "minimal",
            "low",
            "moderate",
            "high",
        ],
    ).astype(str)

    return gdf


def main():
    log("Starting Script 28")

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

    gdf = rebuild_category_scores(gdf)
    gdf = compute_group_scores(gdf)
    gdf = compute_final_score(gdf)
    gdf = compute_confidence_and_flags(gdf)
    gdf = assign_tiers(gdf)

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

    log(f"Writing GeoPackage: {OUT_GPKG}")
    gdf.to_file(OUT_GPKG, driver="GPKG")

    log(f"Writing CSV: {OUT_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")

    print("\nRecognition v03 score summary:")
    print(
        gdf[
            [
                "observed_recognition_v03",
                "recognition_total_count_3km_v03",
                "recognition_category_coverage_v03",
                "recognition_dataset_confidence_v03",
                "recognition_cell_confidence_v03",
            ]
        ].describe()
    )

    print("\nRecognition tier counts:")
    print(gdf["observed_recognition_tier_v03"].value_counts())

    print("\nDataset limitation flags:")
    print(
        gdf[
            [
                "recognition_v03_trails_missing_flag",
                "recognition_v03_beaches_missing_flag",
                "recognition_v03_parks_sparse_flag",
            ]
        ].iloc[0]
    )


if __name__ == "__main__":
    main()