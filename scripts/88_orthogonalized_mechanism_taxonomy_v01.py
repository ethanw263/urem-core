#!/usr/bin/env python3
"""
88_orthogonalized_mechanism_taxonomy_v01.py

Purpose
-------
Create mechanism classes using orthogonalized RDE dimensions.

This fixes the failure from Script 85, where mechanism classes collapsed
because opportunity, transmission, and recognition were too correlated.

Scientific question
-------------------
Once RDE dimensions are separated, can we identify distinct mechanisms of
recognition disequilibrium?

Inputs
------
data/processed/orthogonalized_rde_dimensions_v01.gpkg

Outputs
-------
data/processed/orthogonalized_mechanism_taxonomy_v01.csv
data/processed/orthogonalized_mechanism_taxonomy_v01.gpkg
data/processed/orthogonalized_mechanism_taxonomy_summary_v01.csv
data/processed/orthogonalized_mechanism_taxonomy_framework_v01.md
"""

from pathlib import Path
import logging
import pandas as pd
import geopandas as gpd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

INPUT_GPKG = DATA / "orthogonalized_rde_dimensions_v01.gpkg"

OUT_CSV = DATA / "orthogonalized_mechanism_taxonomy_v01.csv"
OUT_GPKG = DATA / "orthogonalized_mechanism_taxonomy_v01.gpkg"
OUT_SUMMARY = DATA / "orthogonalized_mechanism_taxonomy_summary_v01.csv"
OUT_MD = DATA / "orthogonalized_mechanism_taxonomy_framework_v01.md"

logging.basicConfig(
    level=logging.INFO,
    format="[88_orthogonalized_mechanism_taxonomy_v01] %(levelname)s: %(message)s",
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


def class3(x):
    if pd.isna(x):
        return "unknown"
    if x >= 0.67:
        return "high"
    if x >= 0.33:
        return "medium"
    return "low"


def classify_mechanism(row):
    p = row["P_class"]
    o = row["O_class"]
    t = row["Tnet_class"]
    u = row["Unrec_class"]

    # U = net under-recognition.
    if p == "high" and o == "high" and t == "high" and u == "high":
        return "Comparative Shadowing / Recognition Diversion Candidate"

    if p == "high" and o in {"medium", "high"} and t in {"medium", "high"} and u == "high":
        return "Recognition Inefficiency"

    if p == "high" and o == "low" and u in {"medium", "high"}:
        return "Opportunity Failure"

    if p == "high" and t == "low" and u in {"medium", "high"}:
        return "Transmission Failure"

    if p == "high" and o == "low" and t == "low" and u in {"medium", "high"}:
        return "Deep Isolation / Expected Obscurity"

    if p == "high" and u == "high":
        return "General Under-Recognized Exceptionality"

    if p == "high" and u in {"low", "medium"}:
        return "Recognized / Explained Exceptional Landscape"

    if p in {"low", "medium"} and u == "high":
        return "Recognition Residual Without Strong Physical Potential"

    return "Background / Mixed"


def main():
    log.info(f"Reading orthogonalized RDE layer: {INPUT_GPKG}")
    gdf = gpd.read_file(INPUT_GPKG)

    required = [
        "cell_id",
        "P_orthogonal_v01",
        "O_base_opportunity_v01",
        "T_net_transmission_v01",
        "R_net_under_recognition_v01",
        "orthogonalized_rde_v01",
        "rde_v01_composite_score",
        "is_valid_land_candidate",
    ]

    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    gdf = gdf.copy()

    valid = gdf["is_valid_land_candidate"].astype(bool)

    # ------------------------------------------------------------
    # Dimension classes.
    # ------------------------------------------------------------

    gdf["P_class"] = gdf["P_orthogonal_v01"].apply(class3)
    gdf["O_class"] = gdf["O_base_opportunity_v01"].apply(class3)
    gdf["Tnet_class"] = gdf["T_net_transmission_v01"].apply(class3)
    gdf["Unrec_class"] = gdf["R_net_under_recognition_v01"].apply(class3)

    gdf["orthogonalized_mechanism_class_v01"] = gdf.apply(
        classify_mechanism,
        axis=1,
    )

    gdf["orthogonalized_mechanism_class_v01"] = gdf[
        "orthogonalized_mechanism_class_v01"
    ].where(valid, "Invalid / Noncandidate")

    # ------------------------------------------------------------
    # Mechanism confidence.
    # ------------------------------------------------------------

    # Confidence here means the cell strongly expresses the assigned mechanism.
    gdf["mechanism_expression_score_v01"] = (
        0.40 * gdf["P_orthogonal_v01"]
        + 0.20 * gdf["O_base_opportunity_v01"]
        + 0.20 * gdf["T_net_transmission_v01"]
        + 0.20 * gdf["R_net_under_recognition_v01"]
    )

    gdf["mechanism_expression_score_v01"] = gdf[
        "mechanism_expression_score_v01"
    ].where(valid, 0)

    def confidence(score):
        if score >= 0.80:
            return "high"
        if score >= 0.65:
            return "moderate"
        return "low"

    gdf["mechanism_expression_confidence_v01"] = gdf[
        "mechanism_expression_score_v01"
    ].apply(confidence)

    # ------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------

    summary = (
        gdf[valid]
        .groupby("orthogonalized_mechanism_class_v01")
        .agg(
            cell_count=("cell_id", "count"),
            mean_P=("P_orthogonal_v01", "mean"),
            mean_O=("O_base_opportunity_v01", "mean"),
            mean_Tnet=("T_net_transmission_v01", "mean"),
            mean_net_under_recognition=("R_net_under_recognition_v01", "mean"),
            mean_orthogonalized_rde=("orthogonalized_rde_v01", "mean"),
            mean_original_rde=("rde_v01_composite_score", "mean"),
            mean_expression_score=("mechanism_expression_score_v01", "mean"),
        )
        .reset_index()
        .sort_values("mean_orthogonalized_rde", ascending=False)
    )

    # ------------------------------------------------------------
    # Outputs.
    # ------------------------------------------------------------

    log.info(f"Writing CSV: {OUT_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log.info(f"Writing GPKG: {OUT_GPKG}")
    gdf.to_file(OUT_GPKG, driver="GPKG")

    log.info(f"Writing summary CSV: {OUT_SUMMARY}")
    summary.to_csv(OUT_SUMMARY, index=False)

    md = []
    md.append("# Orthogonalized Mechanism Taxonomy v01")
    md.append("")
    md.append("## Purpose")
    md.append("")
    md.append("This taxonomy classifies recognition-disequilibrium mechanisms using orthogonalized RDE dimensions.")
    md.append("")
    md.append("It follows Script 87, which showed that raw opportunity, transmission, and recognition proxies were too correlated.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Orthogonalized Dimensions")
    md.append("")
    md.append("- **P** = Physical potential")
    md.append("- **O** = Base opportunity")
    md.append("- **Tnet** = Transmission not explained by opportunity")
    md.append("- **Unrec** = Under-recognition not explained by opportunity/transmission")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Mechanism Classes")
    md.append("")
    md.append("### Recognition Inefficiency")
    md.append("High physical potential, sufficient opportunity/transmission, and high net under-recognition.")
    md.append("")
    md.append("### Comparative Shadowing / Recognition Diversion Candidate")
    md.append("High physical potential, high opportunity, high transmission, and high net under-recognition.")
    md.append("")
    md.append("### Opportunity Failure")
    md.append("High physical potential and low opportunity, with net under-recognition.")
    md.append("")
    md.append("### Transmission Failure")
    md.append("High physical potential and low net transmission, with net under-recognition.")
    md.append("")
    md.append("### Deep Isolation / Expected Obscurity")
    md.append("High physical potential, low opportunity, low transmission, and under-recognition.")
    md.append("")
    md.append("### General Under-Recognized Exceptionality")
    md.append("High physical potential and under-recognition, but mechanism is less specific.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Class Summary")
    md.append("")
    md.append(dataframe_to_markdown(summary))
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Methodological Interpretation")
    md.append("")
    md.append("This is the first mechanism taxonomy built on separable dimensions.")
    md.append("")
    md.append("If Recognition Inefficiency and Comparative Shadowing classes appear with meaningful counts,")
    md.append("they are likely the most novel future research targets.")
    md.append("")
    md.append("If most high-RDE cells fall into Opportunity or Transmission Failure,")
    md.append("the method is still mostly explaining under-recognition as isolation/access limitation.")

    log.info(f"Writing MD: {OUT_MD}")
    OUT_MD.write_text("\n".join(md))

    print("\nOrthogonalized Mechanism Taxonomy v01")
    print("-------------------------------------")
    print(summary.to_string(index=False))

    print("\nTop 20 cells by orthogonalized RDE:")
    display_cols = [
        "cell_id",
        "orthogonalized_rde_v01",
        "orthogonalized_mechanism_class_v01",
        "mechanism_expression_confidence_v01",
        "P_orthogonal_v01",
        "O_base_opportunity_v01",
        "T_net_transmission_v01",
        "R_net_under_recognition_v01",
        "rde_v01_composite_score",
    ]

    print(
        gdf[valid]
        .sort_values("orthogonalized_rde_v01", ascending=False)
        [display_cols]
        .head(20)
        .to_string(index=False)
    )

    print("\nWrote:")
    print(f"- {OUT_CSV}")
    print(f"- {OUT_GPKG}")
    print(f"- {OUT_SUMMARY}")
    print(f"- {OUT_MD}")

    print("\nInterpretation")
    print("--------------")
    print("This is the first mechanism taxonomy using separable RDE dimensions.")
    print("The key result is whether high-ranked cells fall into Recognition Inefficiency,")
    print("Comparative Shadowing, Opportunity Failure, or Transmission Failure.")


if __name__ == "__main__":
    main()