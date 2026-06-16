#!/usr/bin/env python3
"""
76_opportunity_structure_index_v01.py

Purpose
-------
Create the first formal Opportunity Structure Index (OSI) for UREM.

This separates:
1. physical potential
2. recognition opportunity
3. observed recognition

Scientific question
-------------------
How much opportunity has each place had to accumulate recognition?

Outputs
-------
data/processed/opportunity_structure_index_v01.csv
data/processed/opportunity_structure_index_v01.gpkg
data/processed/opportunity_structure_summary_v01.csv
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = DATA / "accessibility_friction_v03.gpkg"

OUT_CSV = DATA / "opportunity_structure_index_v01.csv"
OUT_GPKG = DATA / "opportunity_structure_index_v01.gpkg"
OUT_SUMMARY = DATA / "opportunity_structure_summary_v01.csv"

logging.basicConfig(
    level=logging.INFO,
    format="[76_opportunity_structure_index_v01] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


def pct_rank(s):
    return pd.to_numeric(s, errors="coerce").rank(pct=True)


def minmax(s):
    s = pd.to_numeric(s, errors="coerce")
    lo = s.min()
    hi = s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def main():
    log.info(f"Reading input: {INPUT_PATH}")
    gdf = gpd.read_file(INPUT_PATH)

    log.info(f"Rows: {len(gdf):,}")

    required = [
        "cell_id",
        "observed_recognition_v04",
        "recognition_total_count_3km_v04",
        "recognition_category_coverage_v04",
        "recognition_cell_confidence_v04",
        "recognition_infrastructure_scarcity_v03",
        "accessibility_friction_v03",
        "terrain_friction_v03",
        "physical_exceptionality_v03",
        "positive_under_recognition_residual_v06",
        "expected_recognition_v06",
        "distance_to_coast_m",
        "is_valid_land_candidate",
    ]

    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    gdf = gdf.copy()

    # ------------------------------------------------------------
    # Opportunity subcomponents
    # ------------------------------------------------------------

    # 1. Recognition infrastructure opportunity
    # Higher recognition count/coverage/confidence means more opportunity
    # for recognition to accumulate.
    gdf["recognition_infrastructure_opportunity_v01"] = minmax(
        (
            pct_rank(gdf["recognition_total_count_3km_v04"])
            + pct_rank(gdf["recognition_category_coverage_v04"])
            + pct_rank(gdf["recognition_cell_confidence_v04"])
        )
        / 3
    )

    # 2. Accessibility opportunity
    # Existing friction proxy is inverted.
    gdf["accessibility_opportunity_v01"] = minmax(
        1 - gdf["accessibility_friction_v03"]
    )

    # 3. Terrain ease opportunity
    # Terrain friction is inverted.
    gdf["terrain_ease_opportunity_v01"] = minmax(
        1 - gdf["terrain_friction_v03"]
    )

    # 4. Coastal exposure opportunity
    # Since current study area is coastal, distance-to-coast is not treated
    # as exceptionality here, but as exposure/visibility opportunity.
    gdf["distance_to_coast_km"] = gdf["distance_to_coast_m"] / 1000
    gdf["coastal_exposure_opportunity_v01"] = minmax(
        1 - pct_rank(gdf["distance_to_coast_km"])
    )

    # ------------------------------------------------------------
    # Opportunity Structure Index
    # ------------------------------------------------------------

    opportunity_components = [
        "recognition_infrastructure_opportunity_v01",
        "accessibility_opportunity_v01",
        "terrain_ease_opportunity_v01",
        "coastal_exposure_opportunity_v01",
    ]

    gdf["opportunity_structure_index_v01"] = gdf[opportunity_components].mean(axis=1)

    # ------------------------------------------------------------
    # Recognition opportunity gap
    # ------------------------------------------------------------
    # This is not the final UREM score.
    # It asks whether recognition is lower than opportunity would suggest.

    gdf["observed_recognition_pct_v01"] = pct_rank(gdf["observed_recognition_v04"])

    gdf["recognition_opportunity_gap_v01_raw"] = (
        gdf["opportunity_structure_index_v01"]
        - gdf["observed_recognition_pct_v01"]
    )

    gdf["positive_recognition_opportunity_gap_v01"] = (
        gdf["recognition_opportunity_gap_v01_raw"].clip(lower=0)
    )

    gdf["recognition_opportunity_gap_component_v01"] = minmax(
        gdf["positive_recognition_opportunity_gap_v01"]
    )

    # ------------------------------------------------------------
    # First Recognition Disequilibrium prototype
    # ------------------------------------------------------------
    # This asks:
    # Is this place physically exceptional AND did recognition fail relative
    # to opportunity?
    #
    # This is closer to the future method than v1/v2.

    gdf["physical_exceptionality_component_for_osi_v01"] = minmax(
        gdf["physical_exceptionality_v03"]
    )

    gdf["recognition_disequilibrium_index_v01"] = (
        gdf["physical_exceptionality_component_for_osi_v01"]
        * gdf["recognition_opportunity_gap_component_v01"]
    )

    gdf["recognition_disequilibrium_index_v01"] = minmax(
        gdf["recognition_disequilibrium_index_v01"]
    )

    valid = gdf["is_valid_land_candidate"].astype(bool)
    gdf["recognition_disequilibrium_index_v01"] = (
        gdf["recognition_disequilibrium_index_v01"].where(valid, 0)
    )

    gdf["recognition_disequilibrium_rank_v01"] = gdf[
        "recognition_disequilibrium_index_v01"
    ].rank(ascending=False, method="min")

    # ------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------

    variables = [
        "recognition_infrastructure_opportunity_v01",
        "accessibility_opportunity_v01",
        "terrain_ease_opportunity_v01",
        "coastal_exposure_opportunity_v01",
        "opportunity_structure_index_v01",
        "observed_recognition_pct_v01",
        "recognition_opportunity_gap_component_v01",
        "physical_exceptionality_component_for_osi_v01",
        "recognition_disequilibrium_index_v01",
    ]

    rows = []

    top = gdf.sort_values("recognition_disequilibrium_index_v01", ascending=False).head(300)

    for v in variables:
        rows.append(
            {
                "variable": v,
                "baseline_mean": gdf[v].mean(),
                "baseline_median": gdf[v].median(),
                "top300_mean": top[v].mean(),
                "top300_median": top[v].median(),
                "top300_to_baseline_ratio": (
                    top[v].mean() / gdf[v].mean()
                    if gdf[v].mean() != 0
                    else np.nan
                ),
                "top300_mean_percentile_vs_baseline": (gdf[v] <= top[v].mean()).mean(),
            }
        )

    summary = pd.DataFrame(rows)

    log.info(f"Writing CSV: {OUT_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log.info(f"Writing GPKG: {OUT_GPKG}")
    gdf.to_file(OUT_GPKG, driver="GPKG")

    log.info(f"Writing summary CSV: {OUT_SUMMARY}")
    summary.to_csv(OUT_SUMMARY, index=False)

    print("\nOpportunity Structure Index v01 Summary")
    print("---------------------------------------")
    print(summary.to_string(index=False))

    print("\nTop 10 Recognition Disequilibrium Cells")
    print("---------------------------------------")
    display_cols = [
        "cell_id",
        "recognition_disequilibrium_rank_v01",
        "recognition_disequilibrium_index_v01",
        "physical_exceptionality_v03",
        "opportunity_structure_index_v01",
        "observed_recognition_v04",
        "recognition_opportunity_gap_component_v01",
        "distance_to_coast_km",
    ]
    print(top[display_cols].head(10).to_string(index=False))

    print("\nInterpretation")
    print("--------------")
    print("This is the first explicit Opportunity Structure Index.")
    print("It asks whether recognition is low relative to opportunity, then combines")
    print("that gap with physical exceptionality.")
    print()
    print("This is not a replacement for Coastal UREM v1.0 yet.")
    print("It is the first prototype of the deeper Recognition Disequilibrium framework.")


if __name__ == "__main__":
    main()