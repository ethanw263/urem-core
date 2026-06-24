#!/usr/bin/env python3
"""
126_compute_oregon_recognition_score_v04.py

Compute Oregon Recognition Score v04.
"""

from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_NAME = "126_compute_oregon_recognition_score_v04"

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data/processed/oregon_recognition_features_v04.gpkg"

OUT_GPKG = BASE_DIR / "data/processed/oregon_recognition_score_v04.gpkg"
OUT_CSV = BASE_DIR / "data/processed/oregon_recognition_score_v04.csv"


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def safe_minmax(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    mn, mx = s.min(), s.max()

    if mx == mn:
        return pd.Series(0.0, index=s.index)

    return (s - mn) / (mx - mn)


def capped_log_score(series, q=0.99):
    s = pd.to_numeric(series, errors="coerce").fillna(0)

    cap = s.quantile(q)
    s = s.clip(upper=cap)

    return safe_minmax(np.log1p(s))


def main():
    log("Starting Oregon Recognition Score v04")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    gdf = gpd.read_file(INPUT_PATH)

    log(f"Rows: {len(gdf):,}")
    log(f"CRS: {gdf.crs}")

    count_cols = [
        "trail_path_count_3km",
        "beach_count_3km",
        "park_recreation_count_3km",
        "protected_area_count_3km",
        "parking_count_3km",
        "trailhead_count_3km",
        "visitor_information_count_3km",
        "viewpoint_count_3km",
        "named_natural_feature_count_3km",
        "tourism_recreation_count_3km",
    ]

    missing = [c for c in count_cols if c not in gdf.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    log("Computing capped log-normalized category scores")

    for col in count_cols:
        score_col = col.replace("_count_3km", "_score_v04")
        gdf[score_col] = capped_log_score(gdf[col])

    log("Computing recognition group scores")

    gdf["recognition_trails_access_v04"] = (
        0.55 * gdf["trail_path_score_v04"]
        + 0.30 * gdf["parking_score_v04"]
        + 0.15 * gdf["trailhead_score_v04"]
    )

    gdf["recognition_recreation_v04"] = (
        0.25 * gdf["trail_path_score_v04"]
        + 0.20 * gdf["park_recreation_score_v04"]
        + 0.15 * gdf["beach_score_v04"]
        + 0.15 * gdf["tourism_recreation_score_v04"]
        + 0.15 * gdf["visitor_information_score_v04"]
        + 0.10 * gdf["viewpoint_score_v04"]
    )

    gdf["recognition_natural_protected_v04"] = (
        0.30 * gdf["protected_area_score_v04"]
        + 0.25 * gdf["named_natural_feature_score_v04"]
        + 0.20 * gdf["beach_score_v04"]
        + 0.15 * gdf["viewpoint_score_v04"]
        + 0.10 * gdf["park_recreation_score_v04"]
    )

    gdf["observed_recognition_v04_raw"] = (
        0.40 * gdf["recognition_recreation_v04"]
        + 0.35 * gdf["recognition_natural_protected_v04"]
        + 0.25 * gdf["recognition_trails_access_v04"]
    )

    gdf["observed_recognition_v04"] = safe_minmax(
        gdf["observed_recognition_v04_raw"]
    )

    gdf["recognition_total_count_3km_v04"] = gdf[count_cols].sum(axis=1)

    gdf["recognition_category_coverage_v04"] = (
        (gdf[count_cols] > 0).sum(axis=1) / len(count_cols)
    )

    gdf["recognition_dataset_confidence_v04"] = 0.80

    gdf["recognition_cell_confidence_v04"] = (
        0.65 * gdf["recognition_dataset_confidence_v04"]
        + 0.35 * gdf["recognition_category_coverage_v04"]
    ).clip(0, 1)

    if "is_valid_land_candidate" in gdf.columns:
        gdf["recognition_valid_land_flag_v04"] = (
            gdf["is_valid_land_candidate"].astype(bool)
        )
    else:
        gdf["recognition_valid_land_flag_v04"] = True

    gdf["observed_recognition_tier_v04"] = pd.cut(
        gdf["observed_recognition_v04"],
        bins=[-0.001, 0.05, 0.20, 0.50, 1.001],
        labels=["minimal", "low", "moderate", "high"],
    ).astype(str)

    if OUT_GPKG.exists():
        OUT_GPKG.unlink()

    log(f"Writing GeoPackage: {OUT_GPKG}")

    gdf.to_file(
        OUT_GPKG,
        layer="oregon_recognition_score_v04",
        driver="GPKG",
    )

    log(f"Writing CSV: {OUT_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")

    print("\nRecognition v04 summary:")
    print(
        gdf[
            [
                "observed_recognition_v04",
                "recognition_total_count_3km_v04",
                "recognition_category_coverage_v04",
                "recognition_cell_confidence_v04",
            ]
        ].describe()
    )

    print("\nRecognition tier counts:")
    print(gdf["observed_recognition_tier_v04"].value_counts())

    print("\nLand-valid counts:")
    print(gdf["recognition_valid_land_flag_v04"].value_counts())


if __name__ == "__main__":
    main()