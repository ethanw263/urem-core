#!/usr/bin/env python3
"""
82_compare_model_generations_v01.py

Purpose
-------
Compare candidate universes across the major UREM model generations:

1. Coastal UREM v1
2. Opportunity-adjusted UREM v2
3. Transmission-limited disequilibrium
4. Recognition Disequilibrium Equation v01

Scientific question
-------------------
Is RDE v01 merely repackaging earlier UREM scores, or does it identify a
meaningfully different recognition-disequilibrium candidate universe?

Outputs
-------
data/processed/urem_model_generation_overlap_v01.csv
data/processed/urem_model_generation_cell_membership_v01.csv
data/processed/urem_model_generation_cell_membership_v01.gpkg
data/processed/urem_model_generation_summary_v01.txt
"""

from pathlib import Path
import logging
import itertools
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

V1_PATH = DATA / "ranked_urem_candidates_v07_no_coast.gpkg"
V2_PATH = DATA / "ranked_urem_v2_opportunity_adjusted_candidates.gpkg"
TRANSMISSION_PATH = DATA / "recognition_transmission_index_v01.gpkg"
RDE_PATH = DATA / "ranked_rde_v01_candidates.gpkg"

OUT_OVERLAP_CSV = DATA / "urem_model_generation_overlap_v01.csv"
OUT_MEMBERSHIP_CSV = DATA / "urem_model_generation_cell_membership_v01.csv"
OUT_MEMBERSHIP_GPKG = DATA / "urem_model_generation_cell_membership_v01.gpkg"
OUT_TXT = DATA / "urem_model_generation_summary_v01.txt"

TOP_N = 300

logging.basicConfig(
    level=logging.INFO,
    format="[82_compare_model_generations_v01] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


def top_ids_from_ranked(path, score_col=None, top_n=TOP_N):
    gdf = gpd.read_file(path)

    if score_col and score_col in gdf.columns:
        gdf = gdf.sort_values(score_col, ascending=False)

    top = gdf.head(top_n).copy()
    return top, set(top["cell_id"])


def jaccard(a, b):
    return len(a & b) / len(a | b) if len(a | b) else 1.0


def main():
    log.info("Reading model generations...")

    v1_top, v1_ids = top_ids_from_ranked(
        V1_PATH,
        score_col="urem_score_v07_no_coast",
    )

    v2_top, v2_ids = top_ids_from_ranked(
        V2_PATH,
        score_col="urem_v2_opportunity_adjusted_score",
    )

    transmission_full = gpd.read_file(TRANSMISSION_PATH)
    transmission_top = (
        transmission_full.sort_values(
            "transmission_limited_disequilibrium_v01",
            ascending=False,
        )
        .head(TOP_N)
        .copy()
    )
    transmission_ids = set(transmission_top["cell_id"])

    rde_top, rde_ids = top_ids_from_ranked(
        RDE_PATH,
        score_col="rde_v01_composite_score",
    )

    models = {
        "coastal_v1": v1_ids,
        "opportunity_v2": v2_ids,
        "transmission_limited": transmission_ids,
        "rde_v01": rde_ids,
    }

    # ------------------------------------------------------------
    # Pairwise overlaps
    # ------------------------------------------------------------

    overlap_rows = []

    for a_name, b_name in itertools.combinations(models.keys(), 2):
        a = models[a_name]
        b = models[b_name]

        overlap = len(a & b)
        union = len(a | b)

        overlap_rows.append(
            {
                "model_a": a_name,
                "model_b": b_name,
                "top_n": TOP_N,
                "overlap_count": overlap,
                "overlap_share_of_top_n": overlap / TOP_N,
                "jaccard": jaccard(a, b),
                "union_count": union,
            }
        )

    overlap_df = pd.DataFrame(overlap_rows)

    # ------------------------------------------------------------
    # Cell membership table
    # ------------------------------------------------------------

    all_ids = set().union(*models.values())

    base = pd.concat(
        [
            v1_top,
            v2_top,
            transmission_top,
            rde_top,
        ],
        ignore_index=True,
    )

    base = base.drop_duplicates(subset=["cell_id"]).copy()

    base = base[base["cell_id"].isin(all_ids)].copy()

    for model_name, ids in models.items():
        base[f"in_{model_name}_top_{TOP_N}"] = base["cell_id"].isin(ids)

    membership_cols = [f"in_{m}_top_{TOP_N}" for m in models.keys()]
    base["model_presence_count"] = base[membership_cols].sum(axis=1)
    base["model_presence_share"] = base["model_presence_count"] / len(models)

    def classify(row):
        count = row["model_presence_count"]

        if count == 4:
            return "Universal consensus candidate"
        if row[f"in_rde_v01_top_{TOP_N}"] and count >= 2:
            return "RDE-supported multi-model candidate"
        if row[f"in_rde_v01_top_{TOP_N}"] and count == 1:
            return "RDE-specific candidate"
        if row[f"in_coastal_v1_top_{TOP_N}"] and count == 1:
            return "Coastal v1-specific candidate"
        if row[f"in_opportunity_v2_top_{TOP_N}"] and count == 1:
            return "Opportunity v2-specific candidate"
        if row[f"in_transmission_limited_top_{TOP_N}"] and count == 1:
            return "Transmission-specific candidate"
        return "Other multi-model candidate"

    base["model_generation_class"] = base.apply(classify, axis=1)

    # ------------------------------------------------------------
    # Class summary
    # ------------------------------------------------------------

    class_summary = (
        base.groupby("model_generation_class")
        .size()
        .reset_index(name="cell_count")
        .sort_values("cell_count", ascending=False)
    )

    # ------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------

    log.info(f"Writing overlap CSV: {OUT_OVERLAP_CSV}")
    overlap_df.to_csv(OUT_OVERLAP_CSV, index=False)

    log.info(f"Writing membership CSV: {OUT_MEMBERSHIP_CSV}")
    base.drop(columns="geometry").to_csv(OUT_MEMBERSHIP_CSV, index=False)

    log.info(f"Writing membership GPKG: {OUT_MEMBERSHIP_GPKG}")
    base.to_file(OUT_MEMBERSHIP_GPKG, driver="GPKG")

    # ------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------

    lines = []
    lines.append("UREM Model Generation Comparison v01")
    lines.append("====================================")
    lines.append("")
    lines.append(f"Top N per model: {TOP_N}")
    lines.append("")
    lines.append("Models compared:")
    lines.append("- Coastal UREM v1")
    lines.append("- Opportunity-adjusted UREM v2")
    lines.append("- Transmission-limited disequilibrium")
    lines.append("- Recognition Disequilibrium Equation v01")
    lines.append("")
    lines.append("Pairwise overlap:")
    lines.append(overlap_df.to_string(index=False))
    lines.append("")
    lines.append("Candidate class summary:")
    lines.append(class_summary.to_string(index=False))
    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("- High RDE overlap with earlier models means RDE consolidates prior logic.")
    lines.append("- Low RDE overlap means RDE creates a new candidate universe.")
    lines.append("- Consensus candidates are especially strong empirical examples.")
    lines.append("- RDE-specific candidates may represent the new methodology's unique contribution.")

    OUT_TXT.write_text("\n".join(lines))

    # ------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------

    print("\nUREM Model Generation Comparison v01")
    print("------------------------------------")

    print("\nPairwise overlap:")
    print(overlap_df.to_string(index=False))

    print("\nCandidate class summary:")
    print(class_summary.to_string(index=False))

    print("\nRDE overlap highlights:")
    for _, row in overlap_df.iterrows():
        if row["model_a"] == "rde_v01" or row["model_b"] == "rde_v01":
            other = row["model_b"] if row["model_a"] == "rde_v01" else row["model_a"]
            print(
                f"RDE vs {other}: "
                f"{row['overlap_count']} overlap "
                f"({row['overlap_share_of_top_n']:.2%}), "
                f"Jaccard {row['jaccard']:.3f}"
            )

    print("\nWrote:")
    print(f"- {OUT_OVERLAP_CSV}")
    print(f"- {OUT_MEMBERSHIP_CSV}")
    print(f"- {OUT_MEMBERSHIP_GPKG}")
    print(f"- {OUT_TXT}")

    print("\nInterpretation")
    print("--------------")
    print("If RDE has moderate overlap with all models, it may be consolidating")
    print("the prior UREM logic into a cleaner mathematical core.")
    print("")
    print("If RDE has very low overlap, it may be creating a new theory but")
    print("needs more validation.")
    print("")
    print("Consensus candidates are the safest examples.")
    print("RDE-specific candidates are the most interesting future research targets.")


if __name__ == "__main__":
    main()