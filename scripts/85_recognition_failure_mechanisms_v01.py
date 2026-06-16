#!/usr/bin/env python3
"""
85_recognition_failure_mechanisms_v01.py

Purpose
-------
Estimate the dominant recognition-failure mechanism for each RDE taxonomy region.

This script moves UREM/RDE from:

    where is recognition disequilibrium?

toward:

    why does recognition disequilibrium occur?

Scientific question
-------------------
Can RDE regions be classified by likely recognition-failure mechanism?

Mechanism classes
-----------------
1. Transmission Failure
2. Opportunity Failure
3. Recognition Inefficiency
4. Comparative Shadowing Proxy
5. Latent Discovery
6. Multi-Mechanism Disequilibrium
7. Ambiguous / Secondary

Inputs
------
data/processed/rde_taxonomy_regions_v02.gpkg
data/processed/rde_taxonomy_region_summary_v02.csv

Outputs
-------
data/processed/recognition_failure_mechanisms_regions_v01.gpkg
data/processed/recognition_failure_mechanisms_summary_v01.csv
data/processed/recognition_failure_mechanism_class_summary_v01.csv
data/processed/recognition_failure_mechanisms_framework_v01.md
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

REGIONS_GPKG = DATA / "rde_taxonomy_regions_v02.gpkg"
REGIONS_CSV = DATA / "rde_taxonomy_region_summary_v02.csv"

OUT_GPKG = DATA / "recognition_failure_mechanisms_regions_v01.gpkg"
OUT_SUMMARY_CSV = DATA / "recognition_failure_mechanisms_summary_v01.csv"
OUT_CLASS_CSV = DATA / "recognition_failure_mechanism_class_summary_v01.csv"
OUT_MD = DATA / "recognition_failure_mechanisms_framework_v01.md"

logging.basicConfig(
    level=logging.INFO,
    format="[85_recognition_failure_mechanisms_v01] %(levelname)s: %(message)s",
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


def clamp01(x):
    if pd.isna(x):
        return np.nan
    return max(0.0, min(1.0, float(x)))


def compute_mechanism_scores(row):
    """
    Create mechanism scores from RDE regional components.

    Higher score = stronger evidence for that mechanism.
    """

    p = row.get("mean_P_physical_potential_v01", np.nan)
    o = row.get("mean_O_opportunity_structure_v01", np.nan)
    t = row.get("mean_T_recognition_transmission_v01", np.nan)
    r = row.get("mean_R_observed_recognition_v01", np.nan)

    rde = row.get("mean_rde_v01_composite_score", np.nan)
    ineff = row.get("mean_rde_recognition_inefficiency_subtype_v01", np.nan)
    trans_lim = row.get("mean_rde_transmission_limited_subtype_v01", np.nan)
    opp_fail = row.get("mean_rde_opportunity_failure_subtype_v01", np.nan)
    model_share = row.get("mean_model_presence_share", np.nan)

    if pd.isna(p):
        p = row.get("mean_physical_exceptionality_v03", np.nan)

    if pd.isna(r):
        r = row.get("mean_observed_recognition_v04", np.nan)

    recognition_deficit = 1 - r if not pd.isna(r) else np.nan
    opportunity_deficit = 1 - o if not pd.isna(o) else np.nan
    transmission_deficit = 1 - t if not pd.isna(t) else np.nan

    # Mechanism A:
    # Recognition cannot spread well.
    transmission_failure = np.nanmean(
        [
            p,
            recognition_deficit,
            transmission_deficit,
            trans_lim,
        ]
    )

    # Mechanism B:
    # Landscape had weak opportunity to be recognized.
    opportunity_failure = np.nanmean(
        [
            p,
            recognition_deficit,
            opportunity_deficit,
            opp_fail,
        ]
    )

    # Mechanism C:
    # Opportunity and transmission exist, recognition still low.
    recognition_inefficiency = np.nanmean(
        [
            p,
            o,
            t,
            recognition_deficit,
            ineff,
        ]
    )

    # Mechanism D:
    # Comparative shadowing proxy.
    #
    # Since we do not yet have nearby-famous-place data, this is a proxy:
    # moderately/high opportunity + moderately/high transmission +
    # low recognition + high RDE, but not necessarily extreme remoteness.
    comparative_shadowing_proxy = np.nanmean(
        [
            p,
            o,
            t,
            recognition_deficit,
            rde,
        ]
    )

    # Mechanism E:
    # Latent discovery.
    #
    # High physical potential, low recognition, strong RDE,
    # but not clearly explained by low opportunity or low transmission.
    unexplained_factor = np.nanmean(
        [
            1 - abs((o if not pd.isna(o) else 0.5) - 0.5),
            1 - abs((t if not pd.isna(t) else 0.5) - 0.5),
        ]
    )

    latent_discovery = np.nanmean(
        [
            p,
            recognition_deficit,
            rde,
            unexplained_factor,
        ]
    )

    # Multi-model support is not a cause, but it increases confidence.
    confidence_support = model_share if not pd.isna(model_share) else 0.25

    return {
        "mechanism_score_transmission_failure": clamp01(transmission_failure),
        "mechanism_score_opportunity_failure": clamp01(opportunity_failure),
        "mechanism_score_recognition_inefficiency": clamp01(recognition_inefficiency),
        "mechanism_score_comparative_shadowing_proxy": clamp01(comparative_shadowing_proxy),
        "mechanism_score_latent_discovery": clamp01(latent_discovery),
        "mechanism_confidence_support": clamp01(confidence_support),
    }


def assign_primary_mechanism(row):
    scores = {
        "Transmission Failure": row.get("mechanism_score_transmission_failure", np.nan),
        "Opportunity Failure": row.get("mechanism_score_opportunity_failure", np.nan),
        "Recognition Inefficiency": row.get("mechanism_score_recognition_inefficiency", np.nan),
        "Comparative Shadowing Proxy": row.get("mechanism_score_comparative_shadowing_proxy", np.nan),
        "Latent Discovery": row.get("mechanism_score_latent_discovery", np.nan),
    }

    scores = {k: v for k, v in scores.items() if not pd.isna(v)}

    if not scores:
        return "Ambiguous / Secondary"

    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    best_name, best_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0

    if best_score < 0.55:
        return "Ambiguous / Secondary"

    if best_score - second_score < 0.035 and best_score >= 0.65:
        return "Multi-Mechanism Disequilibrium"

    return best_name


def assign_mechanism_confidence(row):
    scores = [
        row.get("mechanism_score_transmission_failure", np.nan),
        row.get("mechanism_score_opportunity_failure", np.nan),
        row.get("mechanism_score_recognition_inefficiency", np.nan),
        row.get("mechanism_score_comparative_shadowing_proxy", np.nan),
        row.get("mechanism_score_latent_discovery", np.nan),
    ]

    scores = [s for s in scores if not pd.isna(s)]

    if not scores:
        return "low"

    scores = sorted(scores, reverse=True)
    best = scores[0]
    second = scores[1] if len(scores) > 1 else 0
    separation = best - second

    support = row.get("mechanism_confidence_support", 0.25)

    if best >= 0.75 and separation >= 0.05 and support >= 0.50:
        return "high"

    if best >= 0.65 and support >= 0.25:
        return "moderate"

    return "low"


def main():
    log.info(f"Reading taxonomy regions: {REGIONS_GPKG}")
    regions = gpd.read_file(REGIONS_GPKG)

    log.info(f"Reading taxonomy summary: {REGIONS_CSV}")
    summary = pd.read_csv(REGIONS_CSV)

    required = [
        "candidate_class",
        "candidate_region_id",
        "region_area_km2",
    ]

    for col in required:
        if col not in regions.columns:
            raise KeyError(f"Missing {col} in regions GPKG.")
        if col not in summary.columns:
            raise KeyError(f"Missing {col} in summary CSV.")

    # Merge full non-geometry summary into region geometries.
    keep_cols = [
        c
        for c in summary.columns
        if c not in regions.columns or c in required
    ]

    merged = regions.merge(
        summary[keep_cols],
        on=required,
        how="left",
        suffixes=("", "_summary"),
    )

    log.info(f"Regions: {len(merged):,}")

    # ------------------------------------------------------------
    # Mechanism scoring.
    # ------------------------------------------------------------

    score_rows = merged.apply(compute_mechanism_scores, axis=1)
    score_df = pd.DataFrame(list(score_rows))

    for col in score_df.columns:
        merged[col] = score_df[col].values

    merged["primary_recognition_failure_mechanism_v01"] = merged.apply(
        assign_primary_mechanism,
        axis=1,
    )

    merged["mechanism_confidence_v01"] = merged.apply(
        assign_mechanism_confidence,
        axis=1,
    )

    # Reporting priority.
    base_priority_cols = [
        "mean_rde_v01_composite_score",
        "rde_taxonomy_priority_score_v02",
        "mechanism_confidence_support",
    ]

    existing = [c for c in base_priority_cols if c in merged.columns]

    merged["recognition_failure_priority_score_v01"] = merged[existing].mean(axis=1)

    def priority_class(score):
        if pd.isna(score):
            return "unclassified"
        if score >= 0.80:
            return "core failure-mechanism example"
        if score >= 0.70:
            return "strong failure-mechanism example"
        if score >= 0.60:
            return "promising failure-mechanism example"
        return "secondary failure-mechanism example"

    merged["recognition_failure_priority_class_v01"] = merged[
        "recognition_failure_priority_score_v01"
    ].apply(priority_class)

    # ------------------------------------------------------------
    # Class summary.
    # ------------------------------------------------------------

    class_summary = (
        merged.groupby("primary_recognition_failure_mechanism_v01")
        .agg(
            region_count=("candidate_region_id", "count"),
            total_cells=("cell_count", "sum"),
            mean_area_km2=("region_area_km2", "mean"),
            mean_priority_score=("recognition_failure_priority_score_v01", "mean"),
            mean_mechanism_confidence_support=("mechanism_confidence_support", "mean"),
            mean_transmission_failure_score=("mechanism_score_transmission_failure", "mean"),
            mean_opportunity_failure_score=("mechanism_score_opportunity_failure", "mean"),
            mean_recognition_inefficiency_score=(
                "mechanism_score_recognition_inefficiency",
                "mean",
            ),
            mean_comparative_shadowing_proxy_score=(
                "mechanism_score_comparative_shadowing_proxy",
                "mean",
            ),
            mean_latent_discovery_score=("mechanism_score_latent_discovery", "mean"),
            mean_rde_score=("mean_rde_v01_composite_score", "mean"),
            mean_physical_potential=("mean_P_physical_potential_v01", "mean"),
            mean_opportunity=("mean_O_opportunity_structure_v01", "mean"),
            mean_transmission=("mean_T_recognition_transmission_v01", "mean"),
            mean_observed_recognition=("mean_R_observed_recognition_v01", "mean"),
        )
        .reset_index()
        .sort_values("mean_priority_score", ascending=False)
    )

    # ------------------------------------------------------------
    # Output tables.
    # ------------------------------------------------------------

    region_table = merged.drop(columns="geometry").sort_values(
        "recognition_failure_priority_score_v01",
        ascending=False,
    )

    log.info(f"Writing GPKG: {OUT_GPKG}")
    merged.to_file(OUT_GPKG, driver="GPKG")

    log.info(f"Writing summary CSV: {OUT_SUMMARY_CSV}")
    region_table.to_csv(OUT_SUMMARY_CSV, index=False)

    log.info(f"Writing class CSV: {OUT_CLASS_CSV}")
    class_summary.to_csv(OUT_CLASS_CSV, index=False)

    # ------------------------------------------------------------
    # Markdown framework.
    # ------------------------------------------------------------

    md = []

    md.append("# Recognition Failure Mechanisms v01")
    md.append("")
    md.append("## Purpose")
    md.append("")
    md.append("This document describes the first mechanism-level interpretation layer for RDE.")
    md.append("")
    md.append("The goal is to move beyond detecting recognition disequilibrium toward explaining why it may occur.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Mechanism Classes")
    md.append("")
    md.append("### 1. Transmission Failure")
    md.append("")
    md.append("Recognition does not spread because transmission pathways are weak.")
    md.append("")
    md.append("Typical signature:")
    md.append("")
    md.append("- high physical potential")
    md.append("- low observed recognition")
    md.append("- low recognition transmission")
    md.append("")
    md.append("### 2. Opportunity Failure")
    md.append("")
    md.append("Recognition does not accumulate because the place had weak opportunity to be encountered.")
    md.append("")
    md.append("Typical signature:")
    md.append("")
    md.append("- high physical potential")
    md.append("- low observed recognition")
    md.append("- low opportunity structure")
    md.append("")
    md.append("### 3. Recognition Inefficiency")
    md.append("")
    md.append("Recognition should plausibly have accumulated, but did not.")
    md.append("")
    md.append("Typical signature:")
    md.append("")
    md.append("- high physical potential")
    md.append("- moderate/high opportunity")
    md.append("- moderate/high transmission")
    md.append("- low observed recognition")
    md.append("")
    md.append("This is likely the most theoretically novel mechanism.")
    md.append("")
    md.append("### 4. Comparative Shadowing Proxy")
    md.append("")
    md.append("A proxy for places potentially overshadowed by nearby better-known regions.")
    md.append("")
    md.append("This currently lacks a true nearby-famous-place dataset.")
    md.append("")
    md.append("### 5. Latent Discovery")
    md.append("")
    md.append("High-potential places with low recognition and no clear failure mechanism.")
    md.append("")
    md.append("These may be true undiscovered or under-diffused landscapes.")
    md.append("")
    md.append("### 6. Multi-Mechanism Disequilibrium")
    md.append("")
    md.append("Regions where multiple mechanisms are similarly plausible.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Mechanism Class Summary")
    md.append("")
    md.append(dataframe_to_markdown(class_summary))
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Top Failure-Mechanism Regions")
    md.append("")

    display_cols = [
        "candidate_class",
        "candidate_region_id",
        "rde_taxonomy_class_v02",
        "primary_recognition_failure_mechanism_v01",
        "mechanism_confidence_v01",
        "recognition_failure_priority_class_v01",
        "recognition_failure_priority_score_v01",
        "cell_count",
        "region_area_km2",
        "mean_rde_v01_composite_score",
        "mean_P_physical_potential_v01",
        "mean_O_opportunity_structure_v01",
        "mean_T_recognition_transmission_v01",
        "mean_R_observed_recognition_v01",
        "mechanism_score_transmission_failure",
        "mechanism_score_opportunity_failure",
        "mechanism_score_recognition_inefficiency",
        "mechanism_score_comparative_shadowing_proxy",
        "mechanism_score_latent_discovery",
    ]

    display_cols = [c for c in display_cols if c in region_table.columns]

    md.append(dataframe_to_markdown(region_table[display_cols].head(30)))
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Methodological Interpretation")
    md.append("")
    md.append("RDE now supports mechanism-level interpretation rather than only place ranking.")
    md.append("")
    md.append("The most important future work is to replace proxy mechanisms with direct data:")
    md.append("")
    md.append("- true road/trail network exposure")
    md.append("- population travel-time exposure")
    md.append("- nearby famous-place shadowing")
    md.append("- institutional designation status")
    md.append("- digital/media recognition history")
    md.append("")
    md.append("This is a major step from UREM as a hotspot method toward RDE as a theory of geographic recognition disequilibrium.")

    log.info(f"Writing MD: {OUT_MD}")
    OUT_MD.write_text("\n".join(md))

    # ------------------------------------------------------------
    # Console output.
    # ------------------------------------------------------------

    print("\nRecognition Failure Mechanisms v01")
    print("----------------------------------")

    print("\nMechanism class summary:")
    print(class_summary.to_string(index=False))

    print("\nTop regions:")
    console_cols = [
        "candidate_class",
        "candidate_region_id",
        "rde_taxonomy_class_v02",
        "primary_recognition_failure_mechanism_v01",
        "mechanism_confidence_v01",
        "recognition_failure_priority_class_v01",
        "recognition_failure_priority_score_v01",
        "cell_count",
        "region_area_km2",
        "mean_rde_v01_composite_score",
        "mean_P_physical_potential_v01",
        "mean_O_opportunity_structure_v01",
        "mean_T_recognition_transmission_v01",
        "mean_R_observed_recognition_v01",
    ]

    console_cols = [c for c in console_cols if c in region_table.columns]

    print(region_table[console_cols].head(25).to_string(index=False))

    print("\nWrote:")
    print(f"- {OUT_GPKG}")
    print(f"- {OUT_SUMMARY_CSV}")
    print(f"- {OUT_CLASS_CSV}")
    print(f"- {OUT_MD}")

    print("\nInterpretation")
    print("--------------")
    print("This script estimates why recognition may have failed.")
    print("The strongest future novelty likely lies in Recognition Inefficiency,")
    print("Latent Discovery, and eventually Comparative Shadowing once better data exists.")


if __name__ == "__main__":
    main()