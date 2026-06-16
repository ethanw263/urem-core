#!/usr/bin/env python3
"""
74_opportunity_adjusted_urem_v2_prototype.py

Purpose
-------
Prototype UREM v2: Opportunity-Adjusted Recognition Disequilibrium.

This does NOT replace Coastal UREM v1.0.

Scientific question
-------------------
Which places remain under-recognized after accounting for both:
1. physical exceptionality
2. recognition opportunity / friction context

Core idea
---------
v1:
    expected recognition = function(physical comparability)

v2 prototype:
    expected recognition = function(physical comparability + opportunity structure)

Outputs
-------
data/processed/urem_v2_opportunity_adjusted_score.csv
data/processed/urem_v2_opportunity_adjusted_score.gpkg
data/processed/ranked_urem_v2_opportunity_adjusted_candidates.csv
data/processed/ranked_urem_v2_opportunity_adjusted_candidates.gpkg
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = DATA / "accessibility_friction_v03.gpkg"

OUT_SCORE_CSV = DATA / "urem_v2_opportunity_adjusted_score.csv"
OUT_SCORE_GPKG = DATA / "urem_v2_opportunity_adjusted_score.gpkg"
OUT_RANKED_CSV = DATA / "ranked_urem_v2_opportunity_adjusted_candidates.csv"
OUT_RANKED_GPKG = DATA / "ranked_urem_v2_opportunity_adjusted_candidates.gpkg"

TOP_N = 1000

logging.basicConfig(
    level=logging.INFO,
    format="[74_opportunity_adjusted_urem_v2_prototype] %(levelname)s: %(message)s",
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
        "expected_recognition_v06",
        "physical_exceptionality_v03",
        "terrain_only_exceptionality_v07",
        "positive_under_recognition_residual_v06",
        "recognition_infrastructure_scarcity_v03",
        "accessibility_friction_v03",
        "terrain_friction_v03",
        "passes_land_filter_v07",
        "passes_confidence_filter_v07",
        "is_valid_land_candidate",
    ]

    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    gdf = gdf.copy()

    # ------------------------------------------------------------------
    # 1. Opportunity structure proxy
    # ------------------------------------------------------------------
    # Higher opportunity means recognition had more chance to accumulate.
    # Higher friction/scarcity means lower opportunity.
    # This is a prototype, not the final theoretical model.

    gdf["recognition_opportunity_v2"] = 1 - gdf[
        [
            "recognition_infrastructure_scarcity_v03",
            "accessibility_friction_v03",
        ]
    ].mean(axis=1)

    gdf["recognition_opportunity_v2"] = minmax(gdf["recognition_opportunity_v2"])

    # ------------------------------------------------------------------
    # 2. Opportunity-adjusted expected recognition
    # ------------------------------------------------------------------
    # If opportunity is low, expected recognition should be discounted.
    # This prevents remote/inaccessible places from being unfairly labeled
    # under-recognized purely because access is poor.

    gdf["expected_recognition_opportunity_adjusted_v2"] = (
        gdf["expected_recognition_v06"]
        * (0.50 + 0.50 * gdf["recognition_opportunity_v2"])
    )

    # ------------------------------------------------------------------
    # 3. Opportunity-adjusted disequilibrium
    # ------------------------------------------------------------------

    gdf["recognition_disequilibrium_v2_raw"] = (
        gdf["expected_recognition_opportunity_adjusted_v2"]
        - gdf["observed_recognition_v04"]
    )

    gdf["positive_recognition_disequilibrium_v2"] = (
        gdf["recognition_disequilibrium_v2_raw"].clip(lower=0)
    )

    gdf["recognition_disequilibrium_component_v2"] = minmax(
        gdf["positive_recognition_disequilibrium_v2"]
    )

    # ------------------------------------------------------------------
    # 4. UREM v2 prototype score
    # ------------------------------------------------------------------
    # This separates:
    # - physical exceptionality
    # - opportunity-adjusted recognition disequilibrium
    # - data/land validity
    #
    # It does not reward low opportunity directly.
    # It asks: after adjusting expectation downward for low opportunity,
    # is the place STILL under-recognized?

    gdf["physical_potential_component_v2"] = minmax(
        gdf["physical_exceptionality_v03"]
    )

    gdf["urem_v2_opportunity_adjusted_raw"] = (
        gdf["physical_potential_component_v2"]
        * gdf["recognition_disequilibrium_component_v2"]
    )

    validity = (
        gdf["passes_land_filter_v07"].astype(bool)
        & gdf["passes_confidence_filter_v07"].astype(bool)
        & gdf["is_valid_land_candidate"].astype(bool)
    )

    gdf["passes_v2_validity_filter"] = validity

    gdf["urem_v2_opportunity_adjusted_score"] = minmax(
        gdf["urem_v2_opportunity_adjusted_raw"].where(validity, 0)
    )

    # ------------------------------------------------------------------
    # 5. Candidate filters
    # ------------------------------------------------------------------

    gdf["passes_v2_exceptionality_filter"] = (
        gdf["physical_potential_component_v2"] >= gdf["physical_potential_component_v2"].quantile(0.75)
    )

    gdf["passes_v2_disequilibrium_filter"] = (
        gdf["recognition_disequilibrium_component_v2"] >= gdf["recognition_disequilibrium_component_v2"].quantile(0.75)
    )

    gdf["passes_v2_score_filter"] = (
        gdf["urem_v2_opportunity_adjusted_score"] >= gdf["urem_v2_opportunity_adjusted_score"].quantile(0.95)
    )

    gdf["passes_urem_v2_candidate_filter"] = (
        gdf["passes_v2_validity_filter"]
        & gdf["passes_v2_exceptionality_filter"]
        & gdf["passes_v2_disequilibrium_filter"]
        & gdf["passes_v2_score_filter"]
    )

    # Rank the top valid cells regardless of strict filter intersection.
    # The strict filter is kept as a diagnostic, not a hard gate.
    ranked = (
        gdf[gdf["passes_v2_validity_filter"]]
        .sort_values("urem_v2_opportunity_adjusted_score", ascending=False)
        .head(TOP_N)
        .copy()
    )

    ranked["passes_urem_v2_candidate_filter"] = ranked[
        "passes_urem_v2_candidate_filter"
    ].astype(bool)

    ranked["urem_v2_rank"] = range(1, len(ranked) + 1)

    log.info(f"Ranked v2 candidates: {len(ranked):,}")

    # ------------------------------------------------------------------
    # 6. Outputs
    # ------------------------------------------------------------------

    log.info(f"Writing score CSV: {OUT_SCORE_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_SCORE_CSV, index=False)

    log.info(f"Writing score GPKG: {OUT_SCORE_GPKG}")
    gdf.to_file(OUT_SCORE_GPKG, driver="GPKG")

    log.info(f"Writing ranked CSV: {OUT_RANKED_CSV}")
    ranked.drop(columns="geometry").to_csv(OUT_RANKED_CSV, index=False)

    log.info(f"Writing ranked GPKG: {OUT_RANKED_GPKG}")
    ranked.to_file(OUT_RANKED_GPKG, driver="GPKG")

    print("\nUREM v2 Opportunity-Adjusted Prototype Summary")
    print("----------------------------------------------")
    print(f"Input cells: {len(gdf):,}")
    print(f"Valid cells: {validity.sum():,}")
    print(f"Ranked v2 candidates: {len(ranked):,}")
    print(
        "Strict v2 candidate filter count: "
        f"{gdf['passes_urem_v2_candidate_filter'].sum():,}"
    )
    print("")
    print("Mean values:")
    print(f"Observed recognition: {gdf['observed_recognition_v04'].mean():.4f}")
    print(f"Expected recognition v1/v6: {gdf['expected_recognition_v06'].mean():.4f}")
    print(
        "Opportunity-adjusted expected recognition: "
        f"{gdf['expected_recognition_opportunity_adjusted_v2'].mean():.4f}"
    )
    print(f"Recognition opportunity v2: {gdf['recognition_opportunity_v2'].mean():.4f}")
    print("")
    print("Top candidate means:")
    print(f"Physical potential: {ranked['physical_potential_component_v2'].mean():.4f}")
    print(
        "Opportunity-adjusted disequilibrium: "
        f"{ranked['recognition_disequilibrium_component_v2'].mean():.4f}"
    )
    print(f"Opportunity-adjusted UREM v2 score: {ranked['urem_v2_opportunity_adjusted_score'].mean():.4f}")
    print("")
    print("Interpretation:")
    print("If v2 candidates overlap strongly with Coastal v1 regions, then v1 discoveries")
    print("remain strong even after opportunity adjustment.")
    print("If v2 shifts to new areas, then opportunity structure is changing the theory.")


if __name__ == "__main__":
    main()