#!/usr/bin/env python3
"""
86_mechanism_separability_analysis_v01.py

Purpose
-------
Test whether the core RDE dimensions are actually separable.

Scientific question
-------------------
Are Physical Potential, Opportunity Structure, Recognition Transmission,
and Observed Recognition distinct dimensions, or are they mostly measuring
the same thing?

Why this matters
----------------
If these dimensions are highly correlated, mechanism classification will collapse.

If they are partially independent, RDE can support a real mechanism theory.

Inputs
------
data/processed/recognition_disequilibrium_equation_v01.gpkg

Outputs
-------
data/processed/mechanism_separability_correlation_v01.csv
data/processed/mechanism_separability_dimension_summary_v01.csv
data/processed/mechanism_separability_region_classes_v01.csv
data/processed/mechanism_separability_region_classes_v01.gpkg
data/processed/mechanism_separability_framework_v01.md
"""

from pathlib import Path
import logging
import itertools
import numpy as np
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

INPUT_GPKG = DATA / "recognition_disequilibrium_equation_v01.gpkg"

OUT_CORR = DATA / "mechanism_separability_correlation_v01.csv"
OUT_DIM_SUMMARY = DATA / "mechanism_separability_dimension_summary_v01.csv"
OUT_CLASS_CSV = DATA / "mechanism_separability_region_classes_v01.csv"
OUT_CLASS_GPKG = DATA / "mechanism_separability_region_classes_v01.gpkg"
OUT_MD = DATA / "mechanism_separability_framework_v01.md"

logging.basicConfig(
    level=logging.INFO,
    format="[86_mechanism_separability_analysis_v01] %(levelname)s: %(message)s",
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


def classify_dimension_value(x):
    if pd.isna(x):
        return "unknown"
    if x >= 0.67:
        return "high"
    if x >= 0.33:
        return "medium"
    return "low"


def classify_mechanism_space(row):
    p = row["P_class"]
    o = row["O_class"]
    t = row["T_class"]
    r = row["R_class"]

    # R = observed recognition. Low R means under-recognized.
    if p == "high" and o in {"medium", "high"} and t in {"medium", "high"} and r == "low":
        return "Recognition Inefficiency Space"

    if p == "high" and o == "low" and r == "low":
        return "Opportunity Failure Space"

    if p == "high" and t == "low" and r == "low":
        return "Transmission Failure Space"

    if p == "high" and o == "low" and t == "low" and r == "low":
        return "Deep Isolation / Expected Obscurity Space"

    if p == "high" and r == "low":
        return "General High-Potential Under-Recognition Space"

    if p == "high" and r in {"medium", "high"}:
        return "Recognized Exceptional Space"

    if p in {"low", "medium"} and r == "high":
        return "Over-Recognized / Attention Concentration Space"

    return "Background / Mixed Space"


def main():
    log.info(f"Reading RDE layer: {INPUT_GPKG}")
    gdf = gpd.read_file(INPUT_GPKG)

    log.info(f"Rows: {len(gdf):,}")

    required = [
        "cell_id",
        "P_physical_potential_v01",
        "O_opportunity_structure_v01",
        "T_recognition_transmission_v01",
        "R_observed_recognition_v01",
        "rde_v01_composite_score",
        "is_valid_land_candidate",
    ]

    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    gdf = gdf[gdf["is_valid_land_candidate"].astype(bool)].copy()

    log.info(f"Valid rows: {len(gdf):,}")

    dims = {
        "P_physical_potential": "P_physical_potential_v01",
        "O_opportunity_structure": "O_opportunity_structure_v01",
        "T_recognition_transmission": "T_recognition_transmission_v01",
        "R_observed_recognition": "R_observed_recognition_v01",
    }

    dim_cols = list(dims.values())

    # ------------------------------------------------------------
    # Pairwise correlations
    # ------------------------------------------------------------

    corr_rows = []

    for (name_a, col_a), (name_b, col_b) in itertools.combinations(dims.items(), 2):
        pearson = gdf[col_a].corr(gdf[col_b], method="pearson")
        spearman = gdf[col_a].corr(gdf[col_b], method="spearman")

        corr_rows.append(
            {
                "dimension_a": name_a,
                "dimension_b": name_b,
                "pearson_corr": pearson,
                "spearman_corr": spearman,
                "abs_pearson_corr": abs(pearson),
                "abs_spearman_corr": abs(spearman),
                "interpretation": (
                    "high_overlap"
                    if abs(spearman) >= 0.75
                    else "moderate_overlap"
                    if abs(spearman) >= 0.40
                    else "low_overlap"
                ),
            }
        )

    corr_df = pd.DataFrame(corr_rows).sort_values(
        "abs_spearman_corr",
        ascending=False,
    )

    # ------------------------------------------------------------
    # Dimension summary
    # ------------------------------------------------------------

    dim_summary_rows = []

    for name, col in dims.items():
        dim_summary_rows.append(
            {
                "dimension": name,
                "mean": gdf[col].mean(),
                "median": gdf[col].median(),
                "std": gdf[col].std(),
                "p10": gdf[col].quantile(0.10),
                "p25": gdf[col].quantile(0.25),
                "p75": gdf[col].quantile(0.75),
                "p90": gdf[col].quantile(0.90),
            }
        )

    dim_summary = pd.DataFrame(dim_summary_rows)

    # ------------------------------------------------------------
    # Mechanism space classification
    # ------------------------------------------------------------

    gdf["P_class"] = gdf["P_physical_potential_v01"].apply(classify_dimension_value)
    gdf["O_class"] = gdf["O_opportunity_structure_v01"].apply(classify_dimension_value)
    gdf["T_class"] = gdf["T_recognition_transmission_v01"].apply(classify_dimension_value)
    gdf["R_class"] = gdf["R_observed_recognition_v01"].apply(classify_dimension_value)

    gdf["mechanism_space_class_v01"] = gdf.apply(classify_mechanism_space, axis=1)

    class_summary = (
        gdf.groupby("mechanism_space_class_v01")
        .agg(
            cell_count=("cell_id", "count"),
            mean_P=("P_physical_potential_v01", "mean"),
            mean_O=("O_opportunity_structure_v01", "mean"),
            mean_T=("T_recognition_transmission_v01", "mean"),
            mean_R=("R_observed_recognition_v01", "mean"),
            mean_RDE=("rde_v01_composite_score", "mean"),
        )
        .reset_index()
        .sort_values("mean_RDE", ascending=False)
    )

    # ------------------------------------------------------------
    # Separability score
    # ------------------------------------------------------------

    mean_abs_spearman = corr_df["abs_spearman_corr"].mean()
    max_abs_spearman = corr_df["abs_spearman_corr"].max()

    separability_score = 1 - mean_abs_spearman

    if max_abs_spearman >= 0.85:
        separability_interpretation = "weak_separability_one_or_more_dimensions_nearly_redundant"
    elif mean_abs_spearman >= 0.60:
        separability_interpretation = "moderate_to_weak_separability"
    elif mean_abs_spearman >= 0.35:
        separability_interpretation = "moderate_separability"
    else:
        separability_interpretation = "strong_separability"

    # ------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------

    log.info(f"Writing correlation CSV: {OUT_CORR}")
    corr_df.to_csv(OUT_CORR, index=False)

    log.info(f"Writing dimension summary CSV: {OUT_DIM_SUMMARY}")
    dim_summary.to_csv(OUT_DIM_SUMMARY, index=False)

    log.info(f"Writing class CSV: {OUT_CLASS_CSV}")
    class_summary.to_csv(OUT_CLASS_CSV, index=False)

    log.info(f"Writing class GPKG: {OUT_CLASS_GPKG}")
    gdf.to_file(OUT_CLASS_GPKG, driver="GPKG")

    md = []

    md.append("# Mechanism Separability Analysis v01")
    md.append("")
    md.append("## Purpose")
    md.append("")
    md.append("This analysis tests whether the core RDE dimensions are separable:")
    md.append("")
    md.append("- Physical Potential")
    md.append("- Opportunity Structure")
    md.append("- Recognition Transmission")
    md.append("- Observed Recognition")
    md.append("")
    md.append("If these dimensions are not separable, mechanism classification collapses.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Separability Result")
    md.append("")
    md.append(f"Mean absolute Spearman correlation: {mean_abs_spearman:.3f}")
    md.append(f"Max absolute Spearman correlation: {max_abs_spearman:.3f}")
    md.append(f"Separability score: {separability_score:.3f}")
    md.append(f"Interpretation: {separability_interpretation}")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Pairwise Correlations")
    md.append("")
    md.append(dataframe_to_markdown(corr_df))
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Dimension Summary")
    md.append("")
    md.append(dataframe_to_markdown(dim_summary))
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Mechanism Space Classes")
    md.append("")
    md.append(dataframe_to_markdown(class_summary))
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Methodological Interpretation")
    md.append("")
    md.append("If separability is strong or moderate, RDE can support a mechanism theory.")
    md.append("")
    md.append("If separability is weak, future work must replace proxy variables with")
    md.append("more independent measures of opportunity and transmission.")
    md.append("")
    md.append("This analysis determines whether the RDE framework is ready for mechanism")
    md.append("classification or whether the dimensions need better data first.")

    log.info(f"Writing MD: {OUT_MD}")
    OUT_MD.write_text("\n".join(md))

    # ------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------

    print("\nMechanism Separability Analysis v01")
    print("-----------------------------------")

    print(f"Mean absolute Spearman correlation: {mean_abs_spearman:.3f}")
    print(f"Max absolute Spearman correlation: {max_abs_spearman:.3f}")
    print(f"Separability score: {separability_score:.3f}")
    print(f"Interpretation: {separability_interpretation}")

    print("\nPairwise correlations:")
    print(corr_df.to_string(index=False))

    print("\nMechanism space classes:")
    print(class_summary.to_string(index=False))

    print("\nWrote:")
    print(f"- {OUT_CORR}")
    print(f"- {OUT_DIM_SUMMARY}")
    print(f"- {OUT_CLASS_CSV}")
    print(f"- {OUT_CLASS_GPKG}")
    print(f"- {OUT_MD}")

    print("\nInterpretation")
    print("--------------")
    print("This tells us whether RDE's dimensions are genuinely distinct.")
    print("If correlations are too high, the current mechanism theory needs better")
    print("opportunity/transmission data before we can claim strong mechanism separation.")


if __name__ == "__main__":
    main()