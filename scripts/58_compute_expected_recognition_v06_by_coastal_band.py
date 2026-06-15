#!/usr/bin/env python3
"""
58_compute_expected_recognition_v06_by_coastal_band.py

Compute Expected Recognition v06 using coastal-distance bands.

Purpose:
- Fix coastal lock-in.
- Estimate expected recognition within comparable coastal/inland bands.
- Prevent coastal cells from dominating only because coast has higher expected recognition.

Inputs:
- data/processed/urem_score_v05b.gpkg

Outputs:
- data/processed/expected_recognition_v06.gpkg
- data/processed/expected_recognition_v06.csv
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd


SCRIPT_NAME = "58_compute_expected_recognition_v06_by_coastal_band"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = PROCESSED_DIR / "urem_score_v05b.gpkg"

OUT_GPKG = PROCESSED_DIR / "expected_recognition_v06.gpkg"
OUT_CSV = PROCESSED_DIR / "expected_recognition_v06.csv"


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def norm01(s):
    s = pd.to_numeric(s, errors="coerce")
    lo = s.quantile(0.02)
    hi = s.quantile(0.98)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=s.index)
    return ((s - lo) / (hi - lo)).clip(0, 1)


def bool_series(df, col):
    if col not in df.columns:
        return pd.Series(True, index=df.index)
    return df[col].astype(str).str.lower().isin(["true", "1", "yes"])


def coastal_band(distance_m):
    if pd.isna(distance_m):
        return "unknown"
    if distance_m <= 2_000:
        return "coast_0_2km"
    if distance_m <= 5_000:
        return "coast_2_5km"
    if distance_m <= 10_000:
        return "coast_5_10km"
    if distance_m <= 25_000:
        return "inland_10_25km"
    return "inland_25km_plus"


def main():
    log("Starting Expected Recognition v06 by coastal band")

    if not INPUT_GPKG.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_GPKG}")

    gdf = gpd.read_file(INPUT_GPKG)

    required = [
        "cell_id",
        "distance_to_coast_m",
        "physical_exceptionality_v03",
        "observed_recognition_v04",
    ]

    for c in required:
        if c not in gdf.columns:
            raise ValueError(f"Missing required column: {c}")

    gdf["distance_to_coast_m"] = pd.to_numeric(
        gdf["distance_to_coast_m"], errors="coerce"
    )

    gdf["coastal_band_v06"] = gdf["distance_to_coast_m"].apply(coastal_band)

    gdf["valid_land_v06"] = (
        bool_series(gdf, "is_valid_land_candidate")
        & (pd.to_numeric(gdf.get("land_area_share", 1), errors="coerce").fillna(1) >= 0.50)
    )

    gdf["physical_exceptionality_v03"] = pd.to_numeric(
        gdf["physical_exceptionality_v03"], errors="coerce"
    )
    gdf["observed_recognition_v04"] = pd.to_numeric(
        gdf["observed_recognition_v04"], errors="coerce"
    )

    pool = gdf[
        gdf["valid_land_v06"]
        & gdf["physical_exceptionality_v03"].notna()
        & gdf["observed_recognition_v04"].notna()
        & (gdf["coastal_band_v06"] != "unknown")
    ].copy()

    log(f"Comparable pool rows: {len(pool):,}")

    if len(pool) < 100:
        raise ValueError("Comparable pool too small.")

    n_bins = 10

    pool["exceptionality_bin_v06"] = (
        pool.groupby("coastal_band_v06")["physical_exceptionality_v03"]
        .transform(
            lambda s: pd.qcut(
                s.rank(method="first"),
                q=min(n_bins, max(2, len(s) // 50)),
                labels=False,
                duplicates="drop",
            )
        )
    )

    bin_stats = (
        pool.groupby(["coastal_band_v06", "exceptionality_bin_v06"])
        .agg(
            expected_recognition_v06_raw=("observed_recognition_v04", "mean"),
            expected_recognition_v06_median=("observed_recognition_v04", "median"),
            expected_recognition_v06_neighbors=("observed_recognition_v04", "count"),
            mean_exceptionality_v03_in_band_bin=("physical_exceptionality_v03", "mean"),
            mean_observed_recognition_v04_in_band_bin=("observed_recognition_v04", "mean"),
        )
        .reset_index()
    )

    # Assign all rows to same band/bin logic.
    gdf["exceptionality_bin_v06"] = (
        gdf.groupby("coastal_band_v06")["physical_exceptionality_v03"]
        .transform(
            lambda s: pd.qcut(
                s.rank(method="first"),
                q=min(n_bins, max(2, s.notna().sum() // 50)),
                labels=False,
                duplicates="drop",
            )
            if s.notna().sum() >= 100
            else pd.Series([pd.NA] * len(s), index=s.index)
        )
    )

    gdf = gdf.merge(
        bin_stats,
        on=["coastal_band_v06", "exceptionality_bin_v06"],
        how="left",
    )

    band_means = (
        pool.groupby("coastal_band_v06")["observed_recognition_v04"]
        .mean()
        .rename("band_mean_observed_recognition_v04")
        .reset_index()
    )

    gdf = gdf.merge(band_means, on="coastal_band_v06", how="left")

    global_mean = pool["observed_recognition_v04"].mean()

    gdf["expected_recognition_v06_raw"] = (
        gdf["expected_recognition_v06_raw"]
        .fillna(gdf["band_mean_observed_recognition_v04"])
        .fillna(global_mean)
    )

    gdf["expected_recognition_v06"] = norm01(gdf["expected_recognition_v06_raw"])

    gdf["recognition_residual_v06"] = (
        gdf["expected_recognition_v06"] - gdf["observed_recognition_v04"]
    )

    gdf["positive_under_recognition_residual_v06"] = (
        gdf["recognition_residual_v06"].clip(lower=0)
    )

    gdf["over_recognition_residual_v06"] = (
        (-gdf["recognition_residual_v06"]).clip(lower=0)
    )

    gdf["expected_recognition_confidence_v06"] = norm01(
        gdf["expected_recognition_v06_neighbors"]
    ).fillna(0)

    log("Summary by coastal band:")
    summary = (
        gdf[gdf["valid_land_v06"]]
        .groupby("coastal_band_v06")
        .agg(
            cells=("cell_id", "count"),
            mean_observed_recognition_v04=("observed_recognition_v04", "mean"),
            mean_expected_recognition_v06=("expected_recognition_v06", "mean"),
            mean_positive_residual_v06=("positive_under_recognition_residual_v06", "mean"),
            mean_exceptionality_v03=("physical_exceptionality_v03", "mean"),
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

    log(f"Writing GPKG: {OUT_GPKG}")
    gdf.to_file(OUT_GPKG, layer="expected_recognition_v06", driver="GPKG")

    log(f"Writing CSV: {OUT_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")


if __name__ == "__main__":
    main()