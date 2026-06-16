#!/usr/bin/env python3
"""
84_recognition_disequilibrium_taxonomy_v02.py

Purpose
-------
Create the first reliable taxonomy of Recognition Disequilibrium landscapes.

Fix from v01
------------
v01 relied too heavily on intermediate region summary fields, which caused
some regions to have NaN RDE metrics.

v02 rebuilds region summaries directly from:
    recognition_disequilibrium_equation_v01.gpkg
and:
    consensus_rde_candidate_regions_v01.gpkg

This avoids missing RDE metrics.

Scientific question
-------------------
Are all recognition-disequilibrium landscapes the same, or do they represent
different mechanisms of recognition failure?

Inputs
------
data/processed/consensus_rde_candidate_regions_v01.gpkg
data/processed/recognition_disequilibrium_equation_v01.gpkg

Outputs
-------
data/processed/rde_taxonomy_regions_v02.gpkg
data/processed/rde_taxonomy_region_summary_v02.csv
data/processed/rde_taxonomy_class_summary_v02.csv
data/processed/rde_taxonomy_framework_v02.md
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

REGIONS_GPKG = DATA / "consensus_rde_candidate_regions_v01.gpkg"
RDE_GPKG = DATA / "recognition_disequilibrium_equation_v01.gpkg"

OUT_GPKG = DATA / "rde_taxonomy_regions_v02.gpkg"
OUT_REGION_CSV = DATA / "rde_taxonomy_region_summary_v02.csv"
OUT_CLASS_CSV = DATA / "rde_taxonomy_class_summary_v02.csv"
OUT_MD = DATA / "rde_taxonomy_framework_v02.md"

logging.basicConfig(
    level=logging.INFO,
    format="[84_recognition_disequilibrium_taxonomy_v02] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


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


def classify_taxonomy(row):
    p = row.get("mean_P_physical_potential_v01", np.nan)
    o = row.get("mean_O_opportunity_structure_v01", np.nan)
    t = row.get("mean_T_recognition_transmission_v01", np.nan)
    r = row.get("mean_R_observed_recognition_v01", np.nan)

    ineff = row.get("mean_rde_recognition_inefficiency_subtype_v01", np.nan)
    trans_lim = row.get("mean_rde_transmission_limited_subtype_v01", np.nan)
    opp_fail = row.get("mean_rde_opportunity_failure_subtype_v01", np.nan)

    model_share = row.get("mean_model_presence_share", np.nan)

    high_p = p >= 0.75
    high_o = o >= 0.50
    low_o = o < 0.45
    high_t = t >= 0.50
    low_t = t < 0.45
    low_r = r < 0.20

    high_ineff = ineff >= 0.75
    high_trans_lim = trans_lim >= 0.75
    high_opp_fail = opp_fail >= 0.75

    strong_multimodel = model_share >= 0.50

    if high_p and high_o and high_t and low_r and high_ineff:
        return "Recognition Inefficiency Landscape"

    if high_p and low_t and low_r and high_trans_lim:
        return "Transmission-Limited Landscape"

    if high_p and low_o and low_r and high_opp_fail:
        return "Opportunity-Failure Landscape"

    if high_p and low_r and strong_multimodel:
        return "Multi-Model Recognition Deficit"

    if high_p and low_r:
        return "High-Potential Low-Recognition Landscape"

    if high_ineff or high_trans_lim or high_opp_fail:
        return "Partial Disequilibrium Landscape"

    return "Ambiguous / Secondary RDE Landscape"


def mechanism_score(row):
    vals = []

    for col in [
        "mean_rde_recognition_inefficiency_subtype_v01",
        "mean_rde_transmission_limited_subtype_v01",
        "mean_rde_opportunity_failure_subtype_v01",
        "mean_rde_v01_composite_score",
    ]:
        v = row.get(col, np.nan)
        if not pd.isna(v):
            vals.append(v)

    if not vals:
        return np.nan

    return float(np.mean(vals))


def main():
    log.info(f"Reading candidate regions: {REGIONS_GPKG}")
    regions = gpd.read_file(REGIONS_GPKG)

    log.info(f"Reading RDE layer: {RDE_GPKG}")
    rde = gpd.read_file(RDE_GPKG)

    if rde.crs != regions.crs:
        rde = rde.to_crs(regions.crs)

    required_region_cols = [
        "candidate_class",
        "candidate_region_id",
        "region_area_km2",
    ]

    missing_region = [c for c in required_region_cols if c not in regions.columns]
    if missing_region:
        raise KeyError(f"Missing region columns: {missing_region}")

    required_rde_cols = [
        "cell_id",
        "rde_v01_composite_score",
        "rde_v01_score",
        "rde_recognition_inefficiency_subtype_v01",
        "rde_transmission_limited_subtype_v01",
        "rde_opportunity_failure_subtype_v01",
        "P_physical_potential_v01",
        "O_opportunity_structure_v01",
        "T_recognition_transmission_v01",
        "R_observed_recognition_v01",
        "physical_exceptionality_v03",
        "observed_recognition_v04",
        "expected_recognition_v06",
        "recognition_transmission_index_v01",
        "opportunity_structure_index_v01",
    ]

    missing_rde = [c for c in required_rde_cols if c not in rde.columns]
    if missing_rde:
        raise KeyError(f"Missing RDE columns: {missing_rde}")

    log.info(f"Regions: {len(regions):,}")
    log.info(f"RDE cells: {len(rde):,}")

    # ------------------------------------------------------------
    # Spatially join RDE cells directly to candidate regions.
    # ------------------------------------------------------------

    log.info("Joining RDE cells to candidate regions...")

    joined = gpd.sjoin(
        rde[required_rde_cols + ["geometry"]],
        regions[
            [
                "candidate_class",
                "candidate_region_id",
                "region_area_km2",
                "geometry",
            ]
        ],
        how="inner",
        predicate="within",
    )

    log.info(f"Joined cell-region records: {len(joined):,}")

    if joined.empty:
        raise ValueError("No RDE cells joined to candidate regions.")

    # ------------------------------------------------------------
    # Summarize RDE metrics by region.
    # ------------------------------------------------------------

    agg_dict = {
        "cell_count": ("cell_id", "count"),
        "mean_rde_v01_composite_score": ("rde_v01_composite_score", "mean"),
        "median_rde_v01_composite_score": ("rde_v01_composite_score", "median"),
        "mean_rde_v01_score": ("rde_v01_score", "mean"),
        "mean_rde_recognition_inefficiency_subtype_v01": (
            "rde_recognition_inefficiency_subtype_v01",
            "mean",
        ),
        "mean_rde_transmission_limited_subtype_v01": (
            "rde_transmission_limited_subtype_v01",
            "mean",
        ),
        "mean_rde_opportunity_failure_subtype_v01": (
            "rde_opportunity_failure_subtype_v01",
            "mean",
        ),
        "mean_P_physical_potential_v01": ("P_physical_potential_v01", "mean"),
        "mean_O_opportunity_structure_v01": ("O_opportunity_structure_v01", "mean"),
        "mean_T_recognition_transmission_v01": (
            "T_recognition_transmission_v01",
            "mean",
        ),
        "mean_R_observed_recognition_v01": ("R_observed_recognition_v01", "mean"),
        "mean_physical_exceptionality_v03": ("physical_exceptionality_v03", "mean"),
        "mean_observed_recognition_v04": ("observed_recognition_v04", "mean"),
        "mean_expected_recognition_v06": ("expected_recognition_v06", "mean"),
        "mean_recognition_transmission_index_v01": (
            "recognition_transmission_index_v01",
            "mean",
        ),
        "mean_opportunity_structure_index_v01": (
            "opportunity_structure_index_v01",
            "mean",
        ),
    }

    region_summary = (
        joined.groupby(["candidate_class", "candidate_region_id", "region_area_km2"])
        .agg(**agg_dict)
        .reset_index()
    )

    # Recover model presence info from candidate_class as approximation.
    # This avoids relying on prior NaN-prone summaries.
    def class_presence_share(candidate_class):
        if candidate_class == "Universal consensus candidate":
            return 1.00
        if candidate_class == "RDE-supported multi-model candidate":
            return 0.50
        if candidate_class == "RDE-specific candidate":
            return 0.25
        return np.nan

    region_summary["mean_model_presence_share"] = region_summary[
        "candidate_class"
    ].apply(class_presence_share)

    # ------------------------------------------------------------
    # Taxonomy classification
    # ------------------------------------------------------------

    region_summary["rde_taxonomy_class_v02"] = region_summary.apply(
        classify_taxonomy,
        axis=1,
    )

    region_summary["rde_mechanism_clarity_score_v02"] = region_summary.apply(
        mechanism_score,
        axis=1,
    )

    region_summary["rde_taxonomy_priority_score_v02"] = region_summary[
        [
            "mean_rde_v01_composite_score",
            "rde_mechanism_clarity_score_v02",
            "mean_model_presence_share",
        ]
    ].mean(axis=1)

    def priority_class(score):
        if pd.isna(score):
            return "unclassified"
        if score >= 0.80:
            return "core taxonomy example"
        if score >= 0.70:
            return "strong taxonomy example"
        if score >= 0.60:
            return "promising taxonomy example"
        return "secondary taxonomy example"

    region_summary["rde_taxonomy_priority_class_v02"] = region_summary[
        "rde_taxonomy_priority_score_v02"
    ].apply(priority_class)

    # ------------------------------------------------------------
    # Class summary.
    # ------------------------------------------------------------

    class_summary = (
        region_summary.groupby("rde_taxonomy_class_v02")
        .agg(
            region_count=("candidate_region_id", "count"),
            total_cells=("cell_count", "sum"),
            mean_region_area_km2=("region_area_km2", "mean"),
            mean_priority_score=("rde_taxonomy_priority_score_v02", "mean"),
            mean_mechanism_clarity=("rde_mechanism_clarity_score_v02", "mean"),
            class_mean_rde_v01_composite_score=("mean_rde_v01_composite_score", "mean"),
            class_mean_rde_recognition_inefficiency_subtype_v01=(
                "mean_rde_recognition_inefficiency_subtype_v01",
                "mean",
            ),
            class_mean_rde_transmission_limited_subtype_v01=(
                "mean_rde_transmission_limited_subtype_v01",
                "mean",
            ),
            class_mean_rde_opportunity_failure_subtype_v01=(
                "mean_rde_opportunity_failure_subtype_v01",
                "mean",
            ),
            class_mean_model_presence_share=("mean_model_presence_share", "mean"),
            class_mean_physical_exceptionality_v03=(
                "mean_physical_exceptionality_v03",
                "mean",
            ),
            class_mean_observed_recognition_v04=(
                "mean_observed_recognition_v04",
                "mean",
            ),
            class_mean_O_opportunity_structure_v01=(
                "mean_O_opportunity_structure_v01",
                "mean",
            ),
            class_mean_T_recognition_transmission_v01=(
                "mean_T_recognition_transmission_v01",
                "mean",
            ),
        )
        .reset_index()
        .sort_values("mean_priority_score", ascending=False)
    )

    # ------------------------------------------------------------
    # Merge taxonomy back to geometries.
    # ------------------------------------------------------------

    out_gdf = regions.merge(
        region_summary,
        on=["candidate_class", "candidate_region_id", "region_area_km2"],
        how="left",
    )

    out_gdf = out_gdf.sort_values(
        "rde_taxonomy_priority_score_v02",
        ascending=False,
    )

    out_table = region_summary.sort_values(
        "rde_taxonomy_priority_score_v02",
        ascending=False,
    )

    # ------------------------------------------------------------
    # Outputs.
    # ------------------------------------------------------------

    log.info(f"Writing taxonomy GPKG: {OUT_GPKG}")
    out_gdf.to_file(OUT_GPKG, driver="GPKG")

    log.info(f"Writing region CSV: {OUT_REGION_CSV}")
    out_table.to_csv(OUT_REGION_CSV, index=False)

    log.info(f"Writing class summary CSV: {OUT_CLASS_CSV}")
    class_summary.to_csv(OUT_CLASS_CSV, index=False)

    md = []

    md.append("# Recognition Disequilibrium Taxonomy v02")
    md.append("")
    md.append("## Purpose")
    md.append("")
    md.append("This document defines a mechanism-based taxonomy of RDE landscapes.")
    md.append("")
    md.append("v02 fixes the v01 issue by rebuilding all region metrics directly from the RDE cell layer.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Taxonomy Classes")
    md.append("")
    md.append("### Recognition Inefficiency Landscape")
    md.append("High physical potential, sufficient opportunity/transmission, but low observed recognition.")
    md.append("")
    md.append("### Transmission-Limited Landscape")
    md.append("High physical potential and low recognition, with weak transmission pathways.")
    md.append("")
    md.append("### Opportunity-Failure Landscape")
    md.append("High physical potential and low recognition, with weak opportunity structure.")
    md.append("")
    md.append("### Multi-Model Recognition Deficit")
    md.append("High-potential low-recognition regions supported across multiple model generations.")
    md.append("")
    md.append("### High-Potential Low-Recognition Landscape")
    md.append("High-potential low-recognition regions where the specific mechanism is less clear.")
    md.append("")
    md.append("### Partial / Ambiguous Disequilibrium Landscape")
    md.append("Regions with partial or less clean disequilibrium evidence.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Class Summary")
    md.append("")
    md.append(dataframe_to_markdown(class_summary))
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Top Taxonomy Regions")
    md.append("")
    display_cols = [
        "candidate_class",
        "candidate_region_id",
        "rde_taxonomy_class_v02",
        "rde_taxonomy_priority_class_v02",
        "rde_taxonomy_priority_score_v02",
        "rde_mechanism_clarity_score_v02",
        "cell_count",
        "region_area_km2",
        "mean_rde_v01_composite_score",
        "mean_model_presence_share",
        "mean_P_physical_potential_v01",
        "mean_O_opportunity_structure_v01",
        "mean_T_recognition_transmission_v01",
        "mean_R_observed_recognition_v01",
        "mean_observed_recognition_v04",
    ]
    display_cols = [c for c in display_cols if c in out_table.columns]
    md.append(dataframe_to_markdown(out_table[display_cols].head(30)))
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Methodological Interpretation")
    md.append("")
    md.append("The taxonomy converts RDE outputs into recognition-failure mechanisms.")
    md.append("")
    md.append("The highest-value future class is likely Recognition Inefficiency Landscape:")
    md.append("places where physical potential, opportunity, and transmission are high enough that")
    md.append("recognition should plausibly have accumulated, but observed recognition remains low.")
    md.append("")
    md.append("This is the strongest bridge from UREM as a mapping workflow to RDE as a methodology.")

    log.info(f"Writing taxonomy framework MD: {OUT_MD}")
    OUT_MD.write_text("\n".join(md))

    # ------------------------------------------------------------
    # Console output.
    # ------------------------------------------------------------

    print("\nRecognition Disequilibrium Taxonomy v02")
    print("---------------------------------------")

    print("\nClass summary:")
    print(class_summary.to_string(index=False))

    print("\nTop taxonomy regions:")
    console_cols = [
        "candidate_class",
        "candidate_region_id",
        "rde_taxonomy_class_v02",
        "rde_taxonomy_priority_class_v02",
        "rde_taxonomy_priority_score_v02",
        "rde_mechanism_clarity_score_v02",
        "cell_count",
        "region_area_km2",
        "mean_rde_v01_composite_score",
        "mean_model_presence_share",
        "mean_P_physical_potential_v01",
        "mean_O_opportunity_structure_v01",
        "mean_T_recognition_transmission_v01",
        "mean_R_observed_recognition_v01",
        "mean_observed_recognition_v04",
    ]

    console_cols = [c for c in console_cols if c in out_table.columns]

    print(out_table[console_cols].head(25).to_string(index=False))

    print("\nWrote:")
    print(f"- {OUT_GPKG}")
    print(f"- {OUT_REGION_CSV}")
    print(f"- {OUT_CLASS_CSV}")
    print(f"- {OUT_MD}")

    print("\nInterpretation")
    print("--------------")
    print("This fixed taxonomy is based directly on RDE cell values.")
    print("If Recognition Inefficiency regions appear cleanly here,")
    print("that is the strongest candidate for UREM/RDE's future novelty.")


if __name__ == "__main__":
    main()