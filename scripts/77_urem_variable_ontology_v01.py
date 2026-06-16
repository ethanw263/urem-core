#!/usr/bin/env python3
"""
77_urem_variable_ontology_v01.py

Purpose
-------
Create the first formal UREM variable ontology.

This script does not create a new score.

It classifies current UREM variables into methodology layers:

1. Physical Potential
2. Observed Recognition
3. Expected Recognition
4. Recognition Disequilibrium
5. Opportunity Structure
6. Recognition Transmission
7. Context / Constraints
8. Confidence / QA
9. Candidate / Ranking Outputs

Outputs
-------
data/processed/urem_variable_framework_v01.csv
data/processed/urem_methodology_ontology_v01.md
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = DATA / "opportunity_structure_index_v01.gpkg"

OUT_CSV = DATA / "urem_variable_framework_v01.csv"
OUT_MD = DATA / "urem_methodology_ontology_v01.md"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """
    Lightweight markdown table writer.
    Avoids requiring the optional 'tabulate' dependency.
    """
    cols = list(df.columns)

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in df.iterrows():
        values = []
        for c in cols:
            value = row[c]
            if pd.isna(value):
                value = ""
            value = str(value).replace("\n", " ").replace("|", "/")
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def classify_variable(col: str):
    c = col.lower()

    if c == "geometry":
        return ("Geometry", "Spatial geometry", "Keep", "Spatial unit geometry.")

    if any(x in c for x in ["cell_id", "grid_row", "grid_col", "centroid", "area"]):
        return (
            "Context / Spatial Index",
            "Grid identity",
            "Keep",
            "Spatial indexing and geometry metadata.",
        )

    if any(
        x in c
        for x in [
            "elevation",
            "relief",
            "slope",
            "terrain",
            "physical_exceptionality",
            "exceptionality",
            "coastline",
            "cliff",
            "beach",
            "edge_area",
        ]
    ):
        return (
            "Physical Potential",
            "Landscape form",
            "Keep / refine",
            "Describes latent geographic or landform potential.",
        )

    if any(
        x in c
        for x in [
            "observed_recognition",
            "recognition_total_count",
            "recognition_category_coverage",
            "recognition_score",
        ]
    ):
        return (
            "Observed Recognition",
            "Recognition proxy",
            "Keep / improve data",
            "Measures accumulated recognition or recognition evidence.",
        )

    if "expected_recognition" in c:
        return (
            "Expected Recognition",
            "Counterfactual expectation",
            "Keep / upgrade",
            "Estimates recognition expected under comparable conditions.",
        )

    if any(
        x in c
        for x in [
            "residual",
            "under_recognition",
            "over_recognition",
            "disequilibrium",
            "recognition_gap",
        ]
    ):
        return (
            "Recognition Disequilibrium",
            "Gap / residual",
            "Core future concept",
            "Measures mismatch between expected/opportunity recognition and observed recognition.",
        )

    if any(
        x in c
        for x in [
            "opportunity_structure",
            "opportunity",
            "accessibility",
            "friction",
            "scarcity",
            "terrain_ease",
            "coastal_exposure",
        ]
    ):
        return (
            "Opportunity Structure",
            "Recognition opportunity",
            "Core future concept",
            "Represents chance or difficulty for recognition to accumulate.",
        )

    if any(
        x in c
        for x in [
            "trail",
            "road",
            "parking",
            "population",
            "tourism",
            "network",
            "transmission",
        ]
    ):
        return (
            "Recognition Transmission",
            "Exposure / diffusion",
            "Add in future",
            "Represents channels through which recognition spreads.",
        )

    if any(x in c for x in ["confidence", "coverage", "valid", "land", "water", "passes"]):
        return (
            "Confidence / QA / Constraints",
            "Validity and filtering",
            "Keep",
            "Controls data quality, validity, and candidate eligibility.",
        )

    if any(x in c for x in ["rank", "score", "tier", "candidate", "urem"]):
        return (
            "Candidate / Ranking Outputs",
            "Model output",
            "Keep but separate",
            "Outputs or derived rankings, not primitive explanatory variables.",
        )

    if any(x in c for x in ["fingerprint", "fp_"]):
        return (
            "Comparable Geography",
            "Feature fingerprint",
            "Keep / refine",
            "Used to define comparable places for counterfactual analysis.",
        )

    return (
        "Unclassified / Review Needed",
        "Unknown",
        "Review",
        "Needs manual methodological classification.",
    )


def main():
    gdf = gpd.read_file(INPUT_PATH)

    rows = []

    for col in gdf.columns:
        category, subcategory, recommendation, description = classify_variable(col)

        rows.append(
            {
                "variable": col,
                "methodology_layer": category,
                "subcategory": subcategory,
                "recommendation": recommendation,
                "description": description,
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(OUT_CSV, index=False)

    layer_counts = (
        df.groupby("methodology_layer")
        .size()
        .reset_index(name="variable_count")
        .sort_values("variable_count", ascending=False)
    )

    md = []
    md.append("# UREM Variable Ontology v01")
    md.append("")
    md.append("## Purpose")
    md.append("")
    md.append("This document formally separates UREM variables into methodology layers.")
    md.append("")
    md.append("The goal is to shift UREM from a score-building workflow into a formal")
    md.append("Recognition Disequilibrium methodology.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Core Methodology Layers")
    md.append("")
    md.append("1. **Physical Potential**")
    md.append("   - Latent geographic or landform quality.")
    md.append("")
    md.append("2. **Observed Recognition**")
    md.append("   - Measured recognition, attention, infrastructure, or mapped salience.")
    md.append("")
    md.append("3. **Expected Recognition**")
    md.append("   - Recognition expected from comparable geography.")
    md.append("")
    md.append("4. **Recognition Disequilibrium**")
    md.append("   - Gap between expected/opportunity recognition and observed recognition.")
    md.append("")
    md.append("5. **Opportunity Structure**")
    md.append("   - Conditions affecting whether recognition had a chance to accumulate.")
    md.append("")
    md.append("6. **Recognition Transmission**")
    md.append("   - Networks or pathways through which recognition spreads.")
    md.append("")
    md.append("7. **Context / Constraints**")
    md.append("   - Spatial identity, land/water validity, and geographic context.")
    md.append("")
    md.append("8. **Confidence / QA**")
    md.append("   - Data quality, coverage, confidence, and filters.")
    md.append("")
    md.append("9. **Candidate / Ranking Outputs**")
    md.append("   - Model outputs, ranks, tiers, and candidate flags.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Variable Counts by Layer")
    md.append("")
    md.append(dataframe_to_markdown(layer_counts))
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Full Variable Framework")
    md.append("")
    md.append(dataframe_to_markdown(df))
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Key Methodological Interpretation")
    md.append("")
    md.append("UREM should no longer be treated as a single score composed of mixed variables.")
    md.append("")
    md.append("Instead, UREM should be treated as a layered recognition-disequilibrium framework:")
    md.append("")
    md.append("```text")
    md.append("Physical Potential")
    md.append("        +")
    md.append("Opportunity Structure")
    md.append("        +")
    md.append("Recognition Transmission")
    md.append("        ->")
    md.append("Expected / Opportunity-Adjusted Recognition")
    md.append("        -")
    md.append("Observed Recognition")
    md.append("        =")
    md.append("Recognition Disequilibrium")
    md.append("```")
    md.append("")
    md.append("This ontology is the first formal step toward UREM Methodology Phase II.")

    OUT_MD.write_text("\n".join(md))

    print("\nUREM Variable Ontology v01")
    print("--------------------------")
    print(layer_counts.to_string(index=False))
    print("")
    print(f"Wrote CSV: {OUT_CSV}")
    print(f"Wrote MD: {OUT_MD}")


if __name__ == "__main__":
    main()