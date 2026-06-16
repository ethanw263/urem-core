#!/usr/bin/env python3
"""
78_recognition_transmission_framework_v01.py

Purpose
-------
Create the first Recognition Transmission Framework for UREM.

This script separates:
1. Physical Potential
2. Opportunity Structure
3. Recognition Transmission
4. Observed Recognition
5. Recognition Disequilibrium

Scientific question
-------------------
Through what channels does recognition spread across geography?

This is not a final model. It is a first formal framework and proxy index.

Inputs
------
data/processed/opportunity_structure_index_v01.gpkg
data/processed/urem_variable_framework_v01.csv

Outputs
-------
data/processed/recognition_transmission_framework_v01.csv
data/processed/recognition_transmission_index_v01.csv
data/processed/recognition_transmission_index_v01.gpkg
data/processed/recognition_transmission_framework_v01.md
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

INPUT_GPKG = DATA / "opportunity_structure_index_v01.gpkg"
ONTOLOGY_CSV = DATA / "urem_variable_framework_v01.csv"

OUT_FRAMEWORK_CSV = DATA / "recognition_transmission_framework_v01.csv"
OUT_INDEX_CSV = DATA / "recognition_transmission_index_v01.csv"
OUT_INDEX_GPKG = DATA / "recognition_transmission_index_v01.gpkg"
OUT_MD = DATA / "recognition_transmission_framework_v01.md"

logging.basicConfig(
    level=logging.INFO,
    format="[78_recognition_transmission_framework_v01] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


def pct_rank(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").rank(pct=True)


def minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo = s.min()
    hi = s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                v = ""
            vals.append(str(v).replace("|", "/").replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    log.info(f"Reading input layer: {INPUT_GPKG}")
    gdf = gpd.read_file(INPUT_GPKG)

    log.info(f"Rows: {len(gdf):,}")

    if ONTOLOGY_CSV.exists():
        ontology = pd.read_csv(ONTOLOGY_CSV)
    else:
        ontology = pd.DataFrame()

    required = [
        "cell_id",
        "observed_recognition_v04",
        "recognition_total_count_3km_v04",
        "recognition_category_coverage_v04",
        "recognition_cell_confidence_v04",
        "recognition_infrastructure_opportunity_v01",
        "accessibility_opportunity_v01",
        "terrain_ease_opportunity_v01",
        "coastal_exposure_opportunity_v01",
        "opportunity_structure_index_v01",
        "physical_exceptionality_v03",
        "recognition_disequilibrium_index_v01",
        "is_valid_land_candidate",
    ]

    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    gdf = gdf.copy()

    # ------------------------------------------------------------------
    # Recognition Transmission ontology
    # ------------------------------------------------------------------
    # These are conceptual channels. Some are currently proxied. Others are
    # future data needs.
    # ------------------------------------------------------------------

    framework_rows = [
        {
            "transmission_channel": "Infrastructure Transmission",
            "definition": "Recognition spreads through mapped, built, or recreational infrastructure.",
            "current_proxy_variables": "recognition_total_count_3km_v04; recognition_category_coverage_v04; recognition_infrastructure_opportunity_v01",
            "future_data": "roads, trailheads, parking, visitor centers, campgrounds, scenic pullouts",
            "status": "partially_available",
        },
        {
            "transmission_channel": "Accessibility Transmission",
            "definition": "Recognition spreads more easily where people can physically reach or pass near a place.",
            "current_proxy_variables": "accessibility_opportunity_v01; terrain_ease_opportunity_v01",
            "future_data": "road-network travel time, hiking time, route impedance, trailhead access",
            "status": "partially_available",
        },
        {
            "transmission_channel": "Exposure Transmission",
            "definition": "Recognition spreads through repeated exposure from nearby populations, travelers, and passersby.",
            "current_proxy_variables": "coastal_exposure_opportunity_v01",
            "future_data": "population within drive-time bands, tourism flows, highway traffic, lodging density",
            "status": "weak_proxy",
        },
        {
            "transmission_channel": "Digital / Media Transmission",
            "definition": "Recognition spreads through digital traces, media, search, photos, and online attention.",
            "current_proxy_variables": "observed_recognition_v04",
            "future_data": "Wikipedia, Flickr, Instagram, Google Places, AllTrails, search trends, travel blogs",
            "status": "future_needed",
        },
        {
            "transmission_channel": "Institutional Transmission",
            "definition": "Recognition spreads when institutions name, protect, promote, or manage a place.",
            "current_proxy_variables": "recognition_category_coverage_v04",
            "future_data": "park status, protected area agency, visitor bureaus, official maps, designations",
            "status": "weak_proxy",
        },
        {
            "transmission_channel": "Network Diffusion Transmission",
            "definition": "Recognition spreads across connected spatial networks rather than uniformly across space.",
            "current_proxy_variables": "",
            "future_data": "road graph, trail graph, tourism corridor graph, social-media co-visitation graph",
            "status": "future_needed",
        },
    ]

    framework = pd.DataFrame(framework_rows)

    # ------------------------------------------------------------------
    # First proxy Recognition Transmission Index
    # ------------------------------------------------------------------
    # This is intentionally simple and based only on current available columns.
    #
    # Higher RTI = stronger channels for recognition to spread.
    # ------------------------------------------------------------------

    gdf["infrastructure_transmission_proxy_v01"] = minmax(
        (
            pct_rank(gdf["recognition_total_count_3km_v04"])
            + pct_rank(gdf["recognition_category_coverage_v04"])
            + pct_rank(gdf["recognition_cell_confidence_v04"])
        )
        / 3
    )

    gdf["accessibility_transmission_proxy_v01"] = minmax(
        (
            gdf["accessibility_opportunity_v01"]
            + gdf["terrain_ease_opportunity_v01"]
        )
        / 2
    )

    gdf["exposure_transmission_proxy_v01"] = minmax(
        gdf["coastal_exposure_opportunity_v01"]
    )

    gdf["recognition_transmission_index_v01"] = gdf[
        [
            "infrastructure_transmission_proxy_v01",
            "accessibility_transmission_proxy_v01",
            "exposure_transmission_proxy_v01",
        ]
    ].mean(axis=1)

    # Transmission deficit:
    # areas where transmission channels are weak.
    gdf["recognition_transmission_deficit_v01"] = (
        1 - gdf["recognition_transmission_index_v01"]
    )

    # Transmission-adjusted disequilibrium diagnostic:
    # physically exceptional + recognition opportunity gap + transmission deficit.
    gdf["physical_exceptionality_pct_v01"] = pct_rank(gdf["physical_exceptionality_v03"])
    gdf["recognition_disequilibrium_pct_v01"] = pct_rank(
        gdf["recognition_disequilibrium_index_v01"]
    )

    gdf["transmission_limited_disequilibrium_v01"] = minmax(
        gdf[
            [
                "physical_exceptionality_pct_v01",
                "recognition_disequilibrium_pct_v01",
                "recognition_transmission_deficit_v01",
            ]
        ].mean(axis=1)
    )

    valid = gdf["is_valid_land_candidate"].astype(bool)
    gdf["transmission_limited_disequilibrium_v01"] = (
        gdf["transmission_limited_disequilibrium_v01"].where(valid, 0)
    )

    gdf["transmission_limited_rank_v01"] = gdf[
        "transmission_limited_disequilibrium_v01"
    ].rank(ascending=False, method="min")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    top = gdf.sort_values(
        "transmission_limited_disequilibrium_v01",
        ascending=False,
    ).head(300)

    summary_rows = []
    variables = [
        "infrastructure_transmission_proxy_v01",
        "accessibility_transmission_proxy_v01",
        "exposure_transmission_proxy_v01",
        "recognition_transmission_index_v01",
        "recognition_transmission_deficit_v01",
        "recognition_disequilibrium_index_v01",
        "transmission_limited_disequilibrium_v01",
    ]

    for v in variables:
        summary_rows.append(
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

    summary = pd.DataFrame(summary_rows)

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    log.info(f"Writing framework CSV: {OUT_FRAMEWORK_CSV}")
    framework.to_csv(OUT_FRAMEWORK_CSV, index=False)

    log.info(f"Writing index CSV: {OUT_INDEX_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_INDEX_CSV, index=False)

    log.info(f"Writing index GPKG: {OUT_INDEX_GPKG}")
    gdf.to_file(OUT_INDEX_GPKG, driver="GPKG")

    md = []
    md.append("# Recognition Transmission Framework v01")
    md.append("")
    md.append("## Purpose")
    md.append("")
    md.append("This document defines Recognition Transmission as a separate UREM methodology layer.")
    md.append("")
    md.append("Recognition Transmission asks:")
    md.append("")
    md.append("> Through what channels does recognition spread across geography?")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Transmission Channels")
    md.append("")
    md.append(dataframe_to_markdown(framework))
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Why this matters")
    md.append("")
    md.append("Physical potential explains why a place may deserve recognition.")
    md.append("")
    md.append("Opportunity structure explains whether recognition had a chance to accumulate.")
    md.append("")
    md.append("Recognition transmission explains how recognition actually spreads.")
    md.append("")
    md.append("This separates UREM from a simple suitability or residual model.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Current proxy index")
    md.append("")
    md.append("The first proxy Recognition Transmission Index combines:")
    md.append("")
    md.append("- infrastructure transmission")
    md.append("- accessibility transmission")
    md.append("- exposure transmission")
    md.append("")
    md.append("This is not final. It is a first executable prototype.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Summary Statistics")
    md.append("")
    md.append(dataframe_to_markdown(summary))
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Future improvement")
    md.append("")
    md.append("Future UREM versions should replace proxies with explicit network models:")
    md.append("")
    md.append("- road graph exposure")
    md.append("- trail graph exposure")
    md.append("- tourism corridor exposure")
    md.append("- population travel-time exposure")
    md.append("- digital/media diffusion")
    md.append("- institutional designation pathways")

    log.info(f"Writing MD: {OUT_MD}")
    OUT_MD.write_text("\n".join(md))

    print("\nRecognition Transmission Framework v01")
    print("--------------------------------------")
    print(framework.to_string(index=False))

    print("\nRecognition Transmission Proxy Summary")
    print("--------------------------------------")
    print(summary.to_string(index=False))

    print("\nTop 10 Transmission-Limited Disequilibrium Cells")
    print("-----------------------------------------------")
    display_cols = [
        "cell_id",
        "transmission_limited_rank_v01",
        "transmission_limited_disequilibrium_v01",
        "recognition_transmission_index_v01",
        "recognition_transmission_deficit_v01",
        "recognition_disequilibrium_index_v01",
        "physical_exceptionality_v03",
        "observed_recognition_v04",
    ]
    print(top[display_cols].head(10).to_string(index=False))

    print("\nWrote:")
    print(f"- {OUT_FRAMEWORK_CSV}")
    print(f"- {OUT_INDEX_CSV}")
    print(f"- {OUT_INDEX_GPKG}")
    print(f"- {OUT_MD}")


if __name__ == "__main__":
    main()