#!/usr/bin/env python3
"""
68_accessibility_friction_analysis_v03.py

Fast accessibility / infrastructure scarcity diagnostic.

This version does NOT parse OSM.

It uses existing UREM v07 / Recognition v04 aggregate columns to test whether
top UREM candidates have lower recognition/access infrastructure density than
the coastal baseline, while also having high terrain friction.

Scientific question:
Are top UREM candidates high-exceptionality places with lower-than-baseline
recognition/access infrastructure and higher terrain friction?
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

UNIVERSE_PATH = DATA / "urem_score_v07_no_coast.gpkg"
CANDIDATE_PATH = DATA / "ranked_urem_candidates_v07_no_coast.gpkg"
REGION_PATH = DATA / "v07_no_coast_discovery_regions.gpkg"

OUT_GPKG = DATA / "accessibility_friction_v03.gpkg"
OUT_CSV = DATA / "accessibility_friction_v03.csv"
OUT_SUMMARY_CSV = DATA / "accessibility_friction_summary_v03.csv"
OUT_REGION_CSV = DATA / "accessibility_friction_region_summary_v03.csv"
OUT_REGION_GPKG = DATA / "accessibility_friction_region_summary_v03.gpkg"

TOP_N = 300

logging.basicConfig(
    level=logging.INFO,
    format="[68_accessibility_friction_analysis_v03] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


def pct_rank(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").rank(pct=True)


def safe_ratio(a, b):
    if b == 0 or pd.isna(b):
        return np.nan
    return a / b


def main():
    log.info(f"Reading universe: {UNIVERSE_PATH}")
    universe = gpd.read_file(UNIVERSE_PATH)

    log.info(f"Reading candidates: {CANDIDATE_PATH}")
    candidates = gpd.read_file(CANDIDATE_PATH)

    log.info(f"Universe rows: {len(universe):,}")
    log.info(f"Candidate rows: {len(candidates):,}")

    score_col = "urem_score_v07_no_coast"
    if score_col not in candidates.columns:
        raise KeyError(f"Missing score column in candidates: {score_col}")

    required = [
        "cell_id",
        "observed_recognition_v04",
        "recognition_total_count_3km_v04",
        "recognition_category_coverage_v04",
        "recognition_cell_confidence_v04",
        "local_relief_m",
        "slope_deg",
        "terrain_drama_v03",
        "physical_exceptionality_v03",
        "expected_recognition_v06",
        "positive_under_recognition_residual_v06",
        "urem_score_v07_no_coast",
    ]

    missing = [c for c in required if c not in universe.columns]
    if missing:
        raise KeyError(f"Missing required columns in universe: {missing}")

    top_candidates = candidates.sort_values(score_col, ascending=False).head(TOP_N)

    universe["is_top_urem_candidate"] = universe["cell_id"].isin(set(top_candidates["cell_id"]))

    log.info(f"Top UREM cells marked: {universe['is_top_urem_candidate'].sum():,}")

    # Terrain friction: harder physical landscape.
    universe["slope_friction_pct_v03"] = pct_rank(universe["slope_deg"])
    universe["relief_friction_pct_v03"] = pct_rank(universe["local_relief_m"])
    universe["terrain_drama_pct_v03"] = pct_rank(universe["terrain_drama_v03"])

    universe["terrain_friction_v03"] = universe[
        [
            "slope_friction_pct_v03",
            "relief_friction_pct_v03",
            "terrain_drama_pct_v03",
        ]
    ].mean(axis=1)

    # Infrastructure / recognition scarcity.
    # Lower observed recognition and lower recognition count = higher access/visibility deficit.
    universe["recognition_count_pct_v03"] = pct_rank(universe["recognition_total_count_3km_v04"])
    universe["observed_recognition_pct_v03"] = pct_rank(universe["observed_recognition_v04"])
    universe["category_coverage_pct_v03"] = pct_rank(universe["recognition_category_coverage_v04"])
    universe["recognition_confidence_pct_v03"] = pct_rank(universe["recognition_cell_confidence_v04"])

    universe["recognition_infrastructure_scarcity_v03"] = 1 - universe[
        [
            "recognition_count_pct_v03",
            "observed_recognition_pct_v03",
            "category_coverage_pct_v03",
            "recognition_confidence_pct_v03",
        ]
    ].mean(axis=1)

    # Combined explanatory friction:
    # high terrain difficulty + low recognition/access infrastructure.
    universe["accessibility_friction_v03"] = universe[
        [
            "terrain_friction_v03",
            "recognition_infrastructure_scarcity_v03",
        ]
    ].mean(axis=1)

    # Recognition disequilibrium mechanism diagnostic:
    # high exceptionality + high scarcity + high under-recognition.
    universe["physical_exceptionality_pct_v03"] = pct_rank(
        universe["physical_exceptionality_v03"]
    )

    universe["under_recognition_residual_pct_v03"] = pct_rank(
        universe["positive_under_recognition_residual_v06"]
    )

    universe["recognition_disequilibrium_mechanism_v03"] = universe[
        [
            "physical_exceptionality_pct_v03",
            "recognition_infrastructure_scarcity_v03",
            "under_recognition_residual_pct_v03",
        ]
    ].mean(axis=1)

    top = universe[universe["is_top_urem_candidate"]].copy()
    baseline = universe.copy()

    variables = [
        "observed_recognition_v04",
        "recognition_total_count_3km_v04",
        "recognition_category_coverage_v04",
        "recognition_cell_confidence_v04",
        "local_relief_m",
        "slope_deg",
        "terrain_drama_v03",
        "physical_exceptionality_v03",
        "positive_under_recognition_residual_v06",
        "terrain_friction_v03",
        "recognition_infrastructure_scarcity_v03",
        "accessibility_friction_v03",
        "recognition_disequilibrium_mechanism_v03",
    ]

    summary_rows = []

    for v in variables:
        base_mean = baseline[v].mean()
        top_mean = top[v].mean()

        summary_rows.append(
            {
                "variable": v,
                "baseline_mean": base_mean,
                "top_urem_mean": top_mean,
                "top_minus_baseline": top_mean - base_mean,
                "ratio_top_to_baseline": safe_ratio(top_mean, base_mean),
                "baseline_median": baseline[v].median(),
                "top_urem_median": top[v].median(),
                "top_mean_percentile_vs_baseline": (baseline[v] <= top_mean).mean(),
            }
        )

    summary = pd.DataFrame(summary_rows)

    log.info(f"Writing GPKG: {OUT_GPKG}")
    universe.to_file(OUT_GPKG, driver="GPKG")

    log.info(f"Writing CSV: {OUT_CSV}")
    universe.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log.info(f"Writing summary CSV: {OUT_SUMMARY_CSV}")
    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    if REGION_PATH.exists():
        log.info(f"Reading regions: {REGION_PATH}")
        regions = gpd.read_file(REGION_PATH)

        if regions.crs != universe.crs:
            regions = regions.to_crs(universe.crs)

        joined = gpd.sjoin(universe, regions, how="left", predicate="within")

        region_col = None
        for c in ["region_id", "discovery_region_id", "cluster_id", "id"]:
            if c in joined.columns:
                region_col = c
                break

        if region_col:
            region_summary = (
                joined.dropna(subset=[region_col])
                .groupby(region_col)
                .agg(
                    cells=("cell_id", "count"),
                    top_urem_cells=("is_top_urem_candidate", "sum"),
                    mean_observed_recognition=("observed_recognition_v04", "mean"),
                    mean_recognition_count_3km=("recognition_total_count_3km_v04", "mean"),
                    mean_category_coverage=("recognition_category_coverage_v04", "mean"),
                    mean_local_relief_m=("local_relief_m", "mean"),
                    mean_slope_deg=("slope_deg", "mean"),
                    mean_terrain_drama=("terrain_drama_v03", "mean"),
                    mean_physical_exceptionality=("physical_exceptionality_v03", "mean"),
                    mean_under_recognition_residual=("positive_under_recognition_residual_v06", "mean"),
                    mean_terrain_friction=("terrain_friction_v03", "mean"),
                    mean_recognition_infrastructure_scarcity=(
                        "recognition_infrastructure_scarcity_v03",
                        "mean",
                    ),
                    mean_accessibility_friction=("accessibility_friction_v03", "mean"),
                    mean_recognition_disequilibrium_mechanism=(
                        "recognition_disequilibrium_mechanism_v03",
                        "mean",
                    ),
                )
                .reset_index()
            )

            regions_out = regions.merge(region_summary, on=region_col, how="left")

            log.info(f"Writing region summary CSV: {OUT_REGION_CSV}")
            region_summary.to_csv(OUT_REGION_CSV, index=False)

            log.info(f"Writing region summary GPKG: {OUT_REGION_GPKG}")
            regions_out.to_file(OUT_REGION_GPKG, driver="GPKG")

    print("\nAccessibility / Recognition Friction v03 Summary")
    print("------------------------------------------------")
    print(summary.to_string(index=False))

    print("\nInterpretation")
    print("--------------")
    print("High terrain_friction_v03 means top candidates are physically harder landscapes.")
    print("High recognition_infrastructure_scarcity_v03 means they have lower recognition/access signal.")
    print("High accessibility_friction_v03 means both are happening together.")
    print("High recognition_disequilibrium_mechanism_v03 supports the explanation that")
    print("UREM discoveries may represent high-exceptionality places where recognition")
    print("has not accumulated in proportion to physical potential.")


if __name__ == "__main__":
    main()