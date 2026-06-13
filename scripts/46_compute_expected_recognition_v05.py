#!/usr/bin/env python3
"""
46_compute_expected_recognition_v05.py

Compute Expected Recognition v05 using Exceptionality v03 and Recognition v04.

Purpose:
- Replace expected_recognition_v04 with a new expected recognition estimate
  driven by improved physical_exceptionality_v03.
- Keep recognition v04 unchanged.
- Keep land validity filtering from v04.
- Produce new recognition residuals for UREM v05.

Inputs:
- data/processed/exceptionality_score_v03.gpkg
- data/processed/recognition_score_v04.gpkg
- data/processed/urem_score_v04.gpkg

Outputs:
- data/processed/expected_recognition_v05.gpkg
- data/processed/expected_recognition_v05.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd


SCRIPT_NAME = "46_compute_expected_recognition_v05"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

EXCEPTIONALITY_V03 = PROCESSED_DIR / "exceptionality_score_v03.gpkg"
RECOGNITION_V04 = PROCESSED_DIR / "recognition_score_v04.gpkg"
UREM_V04 = PROCESSED_DIR / "urem_score_v04.gpkg"

OUT_GPKG = PROCESSED_DIR / "expected_recognition_v05.gpkg"
OUT_CSV = PROCESSED_DIR / "expected_recognition_v05.csv"


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


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


def main():
    log("Starting Expected Recognition v05")

    require_file(EXCEPTIONALITY_V03)
    require_file(RECOGNITION_V04)
    require_file(UREM_V04)

    log(f"Reading exceptionality v03: {EXCEPTIONALITY_V03}")
    exc = gpd.read_file(EXCEPTIONALITY_V03)

    log(f"Reading recognition v04: {RECOGNITION_V04}")
    rec = gpd.read_file(RECOGNITION_V04)

    log(f"Reading UREM v04 base: {UREM_V04}")
    base = gpd.read_file(UREM_V04)

    log(f"Base rows: {len(base):,}")

    keep_exc = [
        "cell_id",
        "physical_exceptionality_v03",
        "terrain_drama_v03",
        "scenic_coast_v03",
        "cliff_proximity_v03",
        "beach_proximity_v03",
        "flat_coastal_edge_penalty_v03",
        "complex_flat_shoreline_penalty_v03",
    ]
    keep_exc = [c for c in keep_exc if c in exc.columns]

    keep_rec = [
        "cell_id",
        "observed_recognition_v04",
        "recognition_cell_confidence_v04",
        "recognition_dataset_confidence_v04",
        "recognition_total_count_3km_v04",
        "recognition_category_coverage_v04",
        "recognition_valid_land_flag_v04",
        "land_area_share",
        "water_area_share",
        "is_valid_land_candidate",
    ]
    keep_rec = [c for c in keep_rec if c in rec.columns]

    df = base.drop(columns=["geometry"], errors="ignore").copy()

    df = df.merge(
        pd.DataFrame(exc[keep_exc]),
        on="cell_id",
        how="left",
        suffixes=("", "_from_exc_v03"),
    )

    df = df.merge(
        pd.DataFrame(rec[keep_rec]),
        on="cell_id",
        how="left",
        suffixes=("", "_from_rec_v04"),
    )

    # Resolve any duplicate recognition columns from merge.
    for c in keep_rec:
        alt = f"{c}_from_rec_v04"
        if alt in df.columns:
            df[c] = df[c].combine_first(df[alt]) if c in df.columns else df[alt]

    df["physical_exceptionality_v03"] = safe_num(df, "physical_exceptionality_v03")
    df["observed_recognition_v04"] = safe_num(df, "observed_recognition_v04")
    df["recognition_cell_confidence_v04"] = safe_num(df, "recognition_cell_confidence_v04", 0.5)
    df["land_area_share"] = safe_num(df, "land_area_share", 1.0)
    df["is_valid_land_candidate"] = safe_num(df, "is_valid_land_candidate", 1.0)

    # Valid comparison pool:
    # Use land-valid cells with reasonable recognition confidence.
    valid_pool = (
        (df["is_valid_land_candidate"] == 1)
        & (df["land_area_share"] >= 0.50)
        & (df["recognition_cell_confidence_v04"] >= 0.25)
        & df["physical_exceptionality_v03"].notna()
        & df["observed_recognition_v04"].notna()
    )

    pool = df.loc[valid_pool, [
        "cell_id",
        "physical_exceptionality_v03",
        "observed_recognition_v04",
        "recognition_cell_confidence_v04",
    ]].copy()

    log(f"Comparable pool rows: {len(pool):,}")

    if len(pool) < 100:
        raise ValueError("Comparable pool too small for expected recognition v05.")

    # Quantile bins by physical exceptionality.
    # This estimates: expected recognition given physical exceptionality.
    n_bins = 25
    pool["exceptionality_bin_v05"] = pd.qcut(
        pool["physical_exceptionality_v03"],
        q=n_bins,
        labels=False,
        duplicates="drop",
    )

    bin_stats = (
        pool.groupby("exceptionality_bin_v05")
        .agg(
            expected_recognition_v05_raw=("observed_recognition_v04", "mean"),
            expected_recognition_v05_median=("observed_recognition_v04", "median"),
            expected_recognition_v05_neighbors=("observed_recognition_v04", "count"),
            mean_exceptionality_v03_in_bin=("physical_exceptionality_v03", "mean"),
            mean_recognition_confidence_v04_in_bin=("recognition_cell_confidence_v04", "mean"),
        )
        .reset_index()
    )

    df["exceptionality_bin_v05"] = pd.qcut(
        df["physical_exceptionality_v03"],
        q=n_bins,
        labels=False,
        duplicates="drop",
    )

    df = df.merge(bin_stats, on="exceptionality_bin_v05", how="left")

    global_expected = pool["observed_recognition_v04"].mean()

    df["expected_recognition_v05_raw"] = df["expected_recognition_v05_raw"].fillna(global_expected)
    df["expected_recognition_v05"] = norm01(df["expected_recognition_v05_raw"])

    # Residual:
    # Positive means expected recognition is higher than observed recognition.
    df["recognition_residual_v05"] = (
        df["expected_recognition_v05"] - df["observed_recognition_v04"]
    )

    df["positive_under_recognition_residual_v05"] = (
        df["recognition_residual_v05"].clip(lower=0)
    )

    df["over_recognition_residual_v05"] = (
        (-df["recognition_residual_v05"]).clip(lower=0)
    )

    # Confidence for expected recognition based on bin size and recognition confidence.
    df["expected_recognition_neighbor_confidence_v05"] = norm01(
        df["expected_recognition_v05_neighbors"]
    )

    df["expected_recognition_confidence_v05"] = (
        0.60 * df["expected_recognition_neighbor_confidence_v05"].fillna(0)
        + 0.40 * df["mean_recognition_confidence_v04_in_bin"].fillna(0)
    ).clip(0, 1)

    # Attach geometry from base.
    out = gpd.GeoDataFrame(
        df,
        geometry=base.geometry,
        crs=base.crs,
    )

    log("Summary:")
    log(f"Mean physical_exceptionality_v03: {out['physical_exceptionality_v03'].mean():.4f}")
    log(f"Mean observed_recognition_v04: {out['observed_recognition_v04'].mean():.4f}")
    log(f"Mean expected_recognition_v05: {out['expected_recognition_v05'].mean():.4f}")
    log(f"Mean recognition_residual_v05: {out['recognition_residual_v05'].mean():.4f}")
    log(f"Mean positive_under_recognition_residual_v05: {out['positive_under_recognition_residual_v05'].mean():.4f}")

    log(f"Writing GPKG: {OUT_GPKG}")
    out.to_file(OUT_GPKG, layer="expected_recognition_v05", driver="GPKG")

    log(f"Writing CSV: {OUT_CSV}")
    out.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")


if __name__ == "__main__":
    main()