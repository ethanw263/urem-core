#!/usr/bin/env python3
"""
72_coastal_urem_v1_synthesis.py

Purpose
-------
Create final Coastal UREM v1.0 synthesis tables.

This script does not create a new model.

It consolidates:
- discovery region typology
- matched counterfactual validation
- accessibility/friction mechanism
- region robustness
- final priority ranking

Outputs
-------
data/processed/coastal_urem_v1_region_synthesis.csv
data/processed/coastal_urem_v1_region_synthesis.gpkg
data/processed/coastal_urem_v1_methodology_summary.txt
"""

from pathlib import Path
import logging
import pandas as pd
import geopandas as gpd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

TYPOLOGY_CSV = DATA / "urem_discovery_region_typology_v01.csv"
TYPOLOGY_GPKG = DATA / "urem_discovery_region_typology_v01.gpkg"
COUNTERFACTUAL_CSV = DATA / "matched_counterfactual_region_summary_v01.csv"
FRICTION_CSV = DATA / "accessibility_friction_region_summary_v03.csv"
ROBUSTNESS_CSV = DATA / "urem_region_robustness_summary_v01.csv"

OUT_CSV = DATA / "coastal_urem_v1_region_synthesis.csv"
OUT_GPKG = DATA / "coastal_urem_v1_region_synthesis.gpkg"
OUT_TXT = DATA / "coastal_urem_v1_methodology_summary.txt"

logging.basicConfig(
    level=logging.INFO,
    format="[72_coastal_urem_v1_synthesis] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


def pct_rank(s):
    return pd.to_numeric(s, errors="coerce").rank(pct=True)


def classify_final_priority(row):
    score = row.get("coastal_urem_v1_final_priority_score", np.nan)
    archetype = str(row.get("discovery_archetype", ""))

    if score >= 0.80:
        return "Core v1 Discovery"
    if score >= 0.65:
        return "Strong v1 Discovery"
    if score >= 0.50:
        return "Promising v1 Discovery"
    if "Secondary" in archetype:
        return "Secondary Discovery"
    return "Exploratory Discovery"


def main():
    log.info(f"Reading typology CSV: {TYPOLOGY_CSV}")
    typology = pd.read_csv(TYPOLOGY_CSV)

    log.info(f"Reading typology GPKG: {TYPOLOGY_GPKG}")
    typology_gpkg = gpd.read_file(TYPOLOGY_GPKG)

    region_col = None
    for c in ["region_id", "discovery_region_id", "cluster_id", "id"]:
        if c in typology.columns:
            region_col = c
            break

    if region_col is None:
        raise KeyError("Could not identify region ID column.")

    merged = typology.copy()

    for path, label in [
        (COUNTERFACTUAL_CSV, "counterfactual"),
        (FRICTION_CSV, "friction"),
        (ROBUSTNESS_CSV, "robustness"),
    ]:
        if path.exists():
            log.info(f"Reading {label}: {path}")
            df = pd.read_csv(path)

            if region_col not in df.columns:
                log.warning(f"{label} file missing {region_col}; skipping.")
                continue

            new_cols = [
                c for c in df.columns
                if c == region_col or c not in merged.columns
            ]

            merged = merged.merge(df[new_cols], on=region_col, how="left")
        else:
            log.warning(f"Missing file: {path}")

    # Final synthesis score.
    # This is NOT a new UREM model. It is a reporting priority score for the
    # coastal v1 discovery regions.
    score_components = []

    if "mean_physical_exceptionality" in merged.columns:
        merged["synth_exceptionality_pct"] = pct_rank(merged["mean_physical_exceptionality"])
        score_components.append("synth_exceptionality_pct")

    if "mean_matched_gap_percentile" in merged.columns:
        merged["synth_counterfactual_gap"] = merged["mean_matched_gap_percentile"]
        score_components.append("synth_counterfactual_gap")

    if "mean_accessibility_friction" in merged.columns:
        merged["synth_friction_pct"] = pct_rank(merged["mean_accessibility_friction"])
        score_components.append("synth_friction_pct")

    if "scenario_presence_share" in merged.columns:
        merged["synth_robustness"] = merged["scenario_presence_share"]
        score_components.append("synth_robustness")

    if "typology_confidence_score" in merged.columns:
        merged["synth_typology_confidence"] = merged["typology_confidence_score"]
        score_components.append("synth_typology_confidence")

    merged["coastal_urem_v1_final_priority_score"] = merged[score_components].mean(axis=1)

    merged["coastal_urem_v1_priority_class"] = merged.apply(classify_final_priority, axis=1)

    merged = merged.sort_values(
        "coastal_urem_v1_final_priority_score",
        ascending=False,
    ).reset_index(drop=True)

    merged["coastal_urem_v1_rank"] = merged.index + 1

    # Merge final table back into geometry.
    keep_geom_cols = [region_col, "geometry"]
    geom = typology_gpkg[keep_geom_cols].copy()

    out_gdf = geom.merge(merged, on=region_col, how="left")

    log.info(f"Writing synthesis CSV: {OUT_CSV}")
    merged.to_csv(OUT_CSV, index=False)

    log.info(f"Writing synthesis GPKG: {OUT_GPKG}")
    out_gdf.to_file(OUT_GPKG, driver="GPKG")

    # Methodology summary text.
    n_regions = len(merged)
    core_count = (merged["coastal_urem_v1_priority_class"] == "Core v1 Discovery").sum()
    strong_count = (merged["coastal_urem_v1_priority_class"] == "Strong v1 Discovery").sum()

    mean_cf = merged["mean_matched_gap_percentile"].mean() if "mean_matched_gap_percentile" in merged.columns else np.nan
    mean_friction = merged["mean_accessibility_friction"].mean() if "mean_accessibility_friction" in merged.columns else np.nan
    mean_robustness = merged["scenario_presence_share"].mean() if "scenario_presence_share" in merged.columns else np.nan

    top_rows = merged.head(10)

    lines = []
    lines.append("Coastal UREM v1.0 Methodology Synthesis")
    lines.append("=======================================")
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append("Coastal UREM v1.0 identifies recognition-disequilibrium landscapes:")
    lines.append("places with high physical exceptionality, lower observed recognition than")
    lines.append("expected among comparable places, and region-scale persistence under")
    lines.append("methodological perturbation.")
    lines.append("")
    lines.append("Current interpretation")
    lines.append("----------------------")
    lines.append("UREM should now be interpreted less as a hotspot mapping workflow and more")
    lines.append("as an early framework for detecting geographic recognition disequilibrium.")
    lines.append("")
    lines.append("Region count")
    lines.append("------------")
    lines.append(f"Total discovery regions: {n_regions}")
    lines.append(f"Core v1 discoveries: {core_count}")
    lines.append(f"Strong v1 discoveries: {strong_count}")
    lines.append("")
    lines.append("Mean validation metrics")
    lines.append("-----------------------")
    lines.append(f"Mean matched counterfactual gap percentile: {mean_cf:.3f}")
    lines.append(f"Mean accessibility/friction score: {mean_friction:.3f}")
    lines.append(f"Mean region robustness scenario presence share: {mean_robustness:.3f}")
    lines.append("")
    lines.append("Top Coastal UREM v1.0 regions")
    lines.append("-----------------------------")

    for _, row in top_rows.iterrows():
        lines.append(
            f"Rank {int(row['coastal_urem_v1_rank'])}: "
            f"Region {row[region_col]} | "
            f"{row.get('coastal_urem_v1_priority_class', '')} | "
            f"{row.get('discovery_archetype', '')} | "
            f"Score {row.get('coastal_urem_v1_final_priority_score', np.nan):.3f}"
        )

    lines.append("")
    lines.append("Defensible Coastal v1.0 claims")
    lines.append("------------------------------")
    lines.append("1. UREM discovery regions exhibit high physical exceptionality.")
    lines.append("2. Top candidates are under-recognized relative to physically comparable places.")
    lines.append("3. Recognition deficits are associated with terrain/accessibility friction.")
    lines.append("4. Exact cell rankings are sensitive, but several discovery regions persist")
    lines.append("   across alternative score formulations.")
    lines.append("5. The appropriate unit of interpretation is the discovery landscape/region,")
    lines.append("   not the individual 1 km cell.")
    lines.append("")
    lines.append("Recommended freeze decision")
    lines.append("---------------------------")
    lines.append("Freeze Coastal UREM v1.0 after reviewing the synthesis outputs.")
    lines.append("Do not continue changing the coastal score formula unless a major data error")
    lines.append("is discovered.")
    lines.append("")
    lines.append("Next phase")
    lines.append("----------")
    lines.append("Move to UREM Methodology Phase II:")
    lines.append("- Geographic Recognition Disequilibrium")
    lines.append("- Recognition Opportunity Structures")
    lines.append("- Recognition Accumulation")
    lines.append("- Persistence of Recognition Deficits")
    lines.append("- Counterfactual Geography")
    lines.append("- Geographic Recognition Inefficiency")

    log.info(f"Writing methodology summary TXT: {OUT_TXT}")
    OUT_TXT.write_text("\n".join(lines))

    print("\nCoastal UREM v1.0 Region Synthesis")
    print("----------------------------------")

    display_cols = [
        "coastal_urem_v1_rank",
        region_col,
        "coastal_urem_v1_priority_class",
        "discovery_archetype",
        "coastal_urem_v1_final_priority_score",
        "mean_matched_gap_percentile",
        "mean_accessibility_friction",
        "scenario_presence_share",
        "mean_physical_exceptionality",
        "mean_observed_recognition",
        "mean_distance_to_coast_km",
    ]

    display_cols = [c for c in display_cols if c in merged.columns]
    print(merged[display_cols].to_string(index=False))

    print("\nWrote:")
    print(f"- {OUT_CSV}")
    print(f"- {OUT_GPKG}")
    print(f"- {OUT_TXT}")

    print("\nRecommended next move:")
    print("Review these outputs, then freeze Coastal UREM v1.0 and move into")
    print("UREM Methodology Phase II.")


if __name__ == "__main__":
    main()