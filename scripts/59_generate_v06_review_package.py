#!/usr/bin/env python3
"""
59_generate_v06_review_package.py

Reusable review exporter for v06 diagnostics.

Important:
UREM review candidates must be physically exceptional first.
This prevents low-recognition but mediocre cells from dominating.

Modes:
- residual_only
- exceptionality_only
- exceptional_residual
- inland_exceptional_residual

Input:
- data/processed/expected_recognition_v06.gpkg

Outputs:
- data/processed/v06_review_package_<MODE>.csv
- data/processed/v06_review_package_<MODE>.gpkg
- data/processed/v06_review_package_<MODE>.kml
"""

from pathlib import Path
import warnings

import geopandas as gpd
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_NAME = "59_generate_v06_review_package"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = PROCESSED_DIR / "expected_recognition_v06.gpkg"

# Change this when needed
REVIEW_MODE = "exceptional_residual"

TOP_N = 100

# Hard floor: must be genuinely physically exceptional
MIN_EXCEPTIONALITY_SCORE = 0.70

# Additional percentile filters
MIN_EXCEPTIONALITY_QUANTILE = 0.80
MIN_RESIDUAL_QUANTILE = 0.80

INLAND_DISTANCE_M = 10_000

OUT_CSV = PROCESSED_DIR / f"v06_review_package_{REVIEW_MODE}.csv"
OUT_GPKG = PROCESSED_DIR / f"v06_review_package_{REVIEW_MODE}.gpkg"
OUT_KML = PROCESSED_DIR / f"v06_review_package_{REVIEW_MODE}.kml"


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def bool_series(df, col):
    if col not in df.columns:
        return pd.Series(True, index=df.index)
    return df[col].astype(str).str.lower().isin(["true", "1", "yes"])


def prepare_base(gdf):
    required = [
        "cell_id",
        "positive_under_recognition_residual_v06",
        "expected_recognition_v06_raw",
        "expected_recognition_v06",
        "observed_recognition_v04",
        "physical_exceptionality_v03",
        "distance_to_coast_m",
        "coastal_band_v06",
    ]

    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = [
        "positive_under_recognition_residual_v06",
        "recognition_residual_v06",
        "expected_recognition_v06_raw",
        "expected_recognition_v06",
        "observed_recognition_v04",
        "physical_exceptionality_v03",
        "physical_exceptionality_score_v02",
        "distance_to_coast_m",
        "terrain_drama_v03",
        "scenic_coast_v03",
        "flat_coastal_edge_penalty_v03",
        "complex_flat_shoreline_penalty_v03",
        "elevation_m",
        "local_relief_m",
        "slope_deg",
        "land_area_share",
        "recognition_cell_confidence_v04",
        "recognition_total_count_3km_v04",
        "recognition_category_coverage_v04",
    ]

    for c in numeric_cols:
        if c in gdf.columns:
            gdf[c] = pd.to_numeric(gdf[c], errors="coerce")

    valid = bool_series(gdf, "valid_land_v06")

    if "land_area_share" in gdf.columns:
        valid = valid & (gdf["land_area_share"].fillna(0) >= 0.50)

    return gdf[valid].copy()


def select_review_rows(gdf):
    exc_quantile_cutoff = gdf["physical_exceptionality_v03"].quantile(
        MIN_EXCEPTIONALITY_QUANTILE
    )
    residual_cutoff = gdf["positive_under_recognition_residual_v06"].quantile(
        MIN_RESIDUAL_QUANTILE
    )

    # Use the stricter of hard score floor and percentile floor
    exceptionality_cutoff = max(
        MIN_EXCEPTIONALITY_SCORE,
        exc_quantile_cutoff,
    )

    log(f"Exceptionality hard floor: {MIN_EXCEPTIONALITY_SCORE:.3f}")
    log(f"Exceptionality quantile floor: {exc_quantile_cutoff:.3f}")
    log(f"Final exceptionality cutoff: {exceptionality_cutoff:.3f}")
    log(f"Residual quantile cutoff: {residual_cutoff:.3f}")

    if REVIEW_MODE == "residual_only":
        selected = gdf[
            gdf["physical_exceptionality_v03"] >= exceptionality_cutoff
        ].copy()

        selected = selected.sort_values(
            "positive_under_recognition_residual_v06",
            ascending=False,
        )

    elif REVIEW_MODE == "exceptionality_only":
        selected = gdf.sort_values(
            "physical_exceptionality_v03",
            ascending=False,
        )

    elif REVIEW_MODE == "exceptional_residual":
        selected = gdf[
            (gdf["physical_exceptionality_v03"] >= exceptionality_cutoff)
            & (gdf["positive_under_recognition_residual_v06"] >= residual_cutoff)
        ].copy()

        selected = selected.sort_values(
            [
                "positive_under_recognition_residual_v06",
                "physical_exceptionality_v03",
            ],
            ascending=[False, False],
        )

    elif REVIEW_MODE == "inland_exceptional_residual":
        selected = gdf[
            (gdf["distance_to_coast_m"] > INLAND_DISTANCE_M)
            & (gdf["physical_exceptionality_v03"] >= exceptionality_cutoff)
            & (gdf["positive_under_recognition_residual_v06"] >= residual_cutoff)
        ].copy()

        selected = selected.sort_values(
            [
                "positive_under_recognition_residual_v06",
                "physical_exceptionality_v03",
            ],
            ascending=[False, False],
        )

    else:
        raise ValueError(f"Unknown REVIEW_MODE: {REVIEW_MODE}")

    return selected.head(TOP_N).copy()


def main():
    log(f"Starting v06 review package | mode={REVIEW_MODE}")

    require_file(INPUT_GPKG)

    gdf = gpd.read_file(INPUT_GPKG)

    if gdf.empty:
        raise ValueError("Input file is empty.")

    if gdf.crs is None:
        raise ValueError("Input file has no CRS.")

    base = prepare_base(gdf)
    review = select_review_rows(base)

    if review.empty:
        raise ValueError(f"No review rows selected for mode: {REVIEW_MODE}")

    review["review_rank"] = range(1, len(review) + 1)

    points = review.copy()
    points["geometry"] = points.geometry.centroid
    points = points.to_crs("EPSG:4326")

    points["longitude"] = points.geometry.x
    points["latitude"] = points.geometry.y

    manual_cols = [
        "manual_place_name",
        "manual_point_quality_1_5",
        "manual_region_quality_1_5",
        "manual_scenic_quality_1_5",
        "manual_existing_recognition_1_5",
        "manual_under_recognized_exceptionality_1_5",
        "dominant_landscape_type",
        "is_success",
        "is_false_positive",
        "failure_mode",
        "reviewer_notes",
    ]

    for c in manual_cols:
        points[c] = ""

    preferred_cols = [
        "review_rank",
        "cell_id",
        "longitude",
        "latitude",
        "distance_to_coast_m",
        "coastal_band_v06",
        "positive_under_recognition_residual_v06",
        "recognition_residual_v06",
        "expected_recognition_v06_raw",
        "expected_recognition_v06",
        "observed_recognition_v04",
        "physical_exceptionality_v03",
        "physical_exceptionality_score_v02",
        "terrain_drama_v03",
        "scenic_coast_v03",
        "flat_coastal_edge_penalty_v03",
        "complex_flat_shoreline_penalty_v03",
        "elevation_m",
        "local_relief_m",
        "slope_deg",
        "land_area_share",
        "recognition_cell_confidence_v04",
        "recognition_total_count_3km_v04",
        "recognition_category_coverage_v04",
    ]

    preferred_cols = [c for c in preferred_cols if c in points.columns]

    out = points[preferred_cols + manual_cols + ["geometry"]].copy()

    log(f"Writing CSV: {OUT_CSV}")
    out.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log(f"Writing GeoPackage: {OUT_GPKG}")
    out.to_file(OUT_GPKG, layer=f"v06_review_package_{REVIEW_MODE}", driver="GPKG")

    log(f"Writing KML: {OUT_KML}")
    kml = out.copy()
    kml["Name"] = (
        "v06 "
        + REVIEW_MODE
        + " "
        + kml["review_rank"].astype(str)
        + " | "
        + kml["cell_id"].astype(str)
    )

    kml["Description"] = (
        "Rank: " + kml["review_rank"].astype(str)
        + " | Residual v06: "
        + kml["positive_under_recognition_residual_v06"].round(4).astype(str)
        + " | Expected raw: "
        + kml["expected_recognition_v06_raw"].round(4).astype(str)
        + " | Observed: "
        + kml["observed_recognition_v04"].round(4).astype(str)
        + " | Exceptionality v03: "
        + kml["physical_exceptionality_v03"].round(4).astype(str)
        + " | Band: "
        + kml["coastal_band_v06"].astype(str)
    )

    try:
        kml[["Name", "Description", "geometry"]].to_file(OUT_KML, driver="KML")
    except Exception as exc:
        log(f"KML export failed: {exc}")

    log("Done")

    print("\nReview package summary:")
    print(f"Mode: {REVIEW_MODE}")
    print(f"Rows: {len(out):,}")

    print("\nTop 25:")
    print(
        out[
            [
                "review_rank",
                "cell_id",
                "longitude",
                "latitude",
                "distance_to_coast_m",
                "coastal_band_v06",
                "positive_under_recognition_residual_v06",
                "expected_recognition_v06_raw",
                "observed_recognition_v04",
                "physical_exceptionality_v03",
            ]
        ]
        .head(25)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()