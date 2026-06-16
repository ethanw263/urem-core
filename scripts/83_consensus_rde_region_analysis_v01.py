#!/usr/bin/env python3
"""
83_consensus_rde_region_analysis_v01.py

Purpose
-------
Analyze the strongest and most novel candidate classes from model-generation
comparison:

1. Universal consensus candidates
2. RDE-supported multi-model candidates
3. RDE-specific candidates

Scientific question
-------------------
Which landscapes are safest empirical examples, and which are the most novel
RDE-driven discoveries?

Inputs
------
data/processed/urem_model_generation_cell_membership_v01.gpkg
data/processed/recognition_disequilibrium_equation_v01.gpkg

Outputs
-------
data/processed/consensus_rde_candidate_regions_v01.gpkg
data/processed/consensus_rde_candidate_region_summary_v01.csv
data/processed/consensus_rde_candidate_cells_v01.csv
data/processed/consensus_rde_candidate_cells_v01.gpkg
"""

from pathlib import Path
import logging
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

MEMBERSHIP_PATH = DATA / "urem_model_generation_cell_membership_v01.gpkg"
RDE_PATH = DATA / "recognition_disequilibrium_equation_v01.gpkg"

OUT_REGIONS_GPKG = DATA / "consensus_rde_candidate_regions_v01.gpkg"
OUT_REGION_SUMMARY_CSV = DATA / "consensus_rde_candidate_region_summary_v01.csv"
OUT_CELLS_CSV = DATA / "consensus_rde_candidate_cells_v01.csv"
OUT_CELLS_GPKG = DATA / "consensus_rde_candidate_cells_v01.gpkg"

CELL_BUFFER_M = 1500
MIN_REGION_AREA_KM2 = 1.0

TARGET_CLASSES = [
    "Universal consensus candidate",
    "RDE-supported multi-model candidate",
    "RDE-specific candidate",
]

logging.basicConfig(
    level=logging.INFO,
    format="[83_consensus_rde_region_analysis_v01] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


def cluster_class_cells(cells: gpd.GeoDataFrame, class_name: str) -> gpd.GeoDataFrame:
    if cells.empty:
        return gpd.GeoDataFrame(
            columns=[
                "candidate_class",
                "candidate_region_id",
                "region_area_km2",
                "geometry",
            ],
            geometry="geometry",
            crs=cells.crs,
        )

    dissolved = unary_union(cells.geometry.buffer(CELL_BUFFER_M))

    if dissolved.geom_type == "Polygon":
        geoms = [dissolved]
    else:
        geoms = list(dissolved.geoms)

    rows = []

    safe_class = (
        class_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )

    for i, geom in enumerate(geoms, start=1):
        area_km2 = geom.area / 1_000_000

        if area_km2 < MIN_REGION_AREA_KM2:
            continue

        rows.append(
            {
                "candidate_class": class_name,
                "candidate_region_id": f"{safe_class}_{i}",
                "region_area_km2": area_km2,
                "geometry": geom,
            }
        )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=cells.crs)


def main():
    log.info(f"Reading model membership: {MEMBERSHIP_PATH}")
    membership = gpd.read_file(MEMBERSHIP_PATH)

    log.info(f"Reading RDE score layer: {RDE_PATH}")
    rde = gpd.read_file(RDE_PATH)

    if rde.crs != membership.crs:
        rde = rde.to_crs(membership.crs)

    log.info(f"Membership rows: {len(membership):,}")
    log.info(f"RDE rows: {len(rde):,}")

    if "model_generation_class" not in membership.columns:
        raise KeyError("Missing model_generation_class in membership layer.")

    metric_cols = [
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
        "geometry",
    ]

    metric_cols = [c for c in metric_cols if c in rde.columns]

    cells = membership.merge(
        rde[metric_cols].drop(columns="geometry", errors="ignore"),
        on="cell_id",
        how="left",
        suffixes=("", "_rde"),
    )

    cells = gpd.GeoDataFrame(cells, geometry="geometry", crs=membership.crs)

    target_cells = cells[cells["model_generation_class"].isin(TARGET_CLASSES)].copy()

    log.info(f"Target candidate cells: {len(target_cells):,}")

    if target_cells.empty:
        raise ValueError("No target cells found for selected model generation classes.")

    all_regions = []

    for class_name in TARGET_CLASSES:
        class_cells = target_cells[target_cells["model_generation_class"] == class_name].copy()

        log.info(f"{class_name}: cells = {len(class_cells):,}")

        regions = cluster_class_cells(class_cells, class_name)
        log.info(f"{class_name}: regions = {len(regions):,}")

        all_regions.append(regions)

    regions = pd.concat(all_regions, ignore_index=True)
    regions = gpd.GeoDataFrame(regions, geometry="geometry", crs=target_cells.crs)

    # Join cells to regions for summary.
    joined = gpd.sjoin(
        target_cells,
        regions[["candidate_class", "candidate_region_id", "geometry"]],
        how="left",
        predicate="within",
    )

    agg_dict = {
        "cell_count": ("cell_id", "count"),
        "mean_model_presence_count": ("model_presence_count", "mean"),
        "mean_model_presence_share": ("model_presence_share", "mean"),
    }

    possible_metrics = [
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

    for col in possible_metrics:
        if col in joined.columns:
            agg_dict[f"mean_{col}"] = (col, "mean")
            agg_dict[f"median_{col}"] = (col, "median")

    region_summary = (
        joined.dropna(subset=["candidate_region_id"])
        .groupby(["candidate_class", "candidate_region_id"])
        .agg(**agg_dict)
        .reset_index()
    )

    region_summary["region_priority_score"] = 0

    if "mean_rde_v01_composite_score" in region_summary.columns:
        region_summary["region_priority_score"] += (
            0.50 * region_summary["mean_rde_v01_composite_score"]
        )

    if "mean_model_presence_share" in region_summary.columns:
        region_summary["region_priority_score"] += (
            0.30 * region_summary["mean_model_presence_share"]
        )

    if "mean_rde_recognition_inefficiency_subtype_v01" in region_summary.columns:
        region_summary["region_priority_score"] += (
            0.20 * region_summary["mean_rde_recognition_inefficiency_subtype_v01"]
        )

    def classify_region(row):
        c = row["candidate_class"]
        score = row["region_priority_score"]

        if c == "Universal consensus candidate":
            return "Safest empirical example"
        if c == "RDE-supported multi-model candidate" and score >= 0.75:
            return "Strong RDE methodology example"
        if c == "RDE-specific candidate" and score >= 0.70:
            return "Novel RDE research target"
        if c == "RDE-specific candidate":
            return "Exploratory RDE-specific target"
        return "Supporting example"

    region_summary["region_interpretation_class"] = region_summary.apply(
        classify_region,
        axis=1,
    )

    regions_out = regions.merge(
        region_summary,
        on=["candidate_class", "candidate_region_id"],
        how="left",
    )

    regions_out = regions_out.sort_values(
        ["region_priority_score", "region_area_km2"],
        ascending=False,
    )

    target_cells = target_cells.sort_values(
        ["model_presence_count", "rde_v01_composite_score"],
        ascending=False,
    )

    log.info(f"Writing regions GPKG: {OUT_REGIONS_GPKG}")
    regions_out.to_file(OUT_REGIONS_GPKG, driver="GPKG")

    log.info(f"Writing region summary CSV: {OUT_REGION_SUMMARY_CSV}")
    region_summary.sort_values(
        "region_priority_score",
        ascending=False,
    ).to_csv(OUT_REGION_SUMMARY_CSV, index=False)

    log.info(f"Writing cells CSV: {OUT_CELLS_CSV}")
    target_cells.drop(columns="geometry").to_csv(OUT_CELLS_CSV, index=False)

    log.info(f"Writing cells GPKG: {OUT_CELLS_GPKG}")
    target_cells.to_file(OUT_CELLS_GPKG, driver="GPKG")

    print("\nConsensus + RDE-Specific Region Analysis v01")
    print("--------------------------------------------")
    print(f"Target cells: {len(target_cells):,}")
    print(f"Regions created: {len(regions_out):,}")

    class_counts = (
        target_cells.groupby("model_generation_class")
        .size()
        .reset_index(name="cell_count")
        .sort_values("cell_count", ascending=False)
    )

    print("\nCell counts by class:")
    print(class_counts.to_string(index=False))

    print("\nTop regions:")
    display_cols = [
        "candidate_class",
        "candidate_region_id",
        "region_interpretation_class",
        "cell_count",
        "region_area_km2",
        "region_priority_score",
        "mean_rde_v01_composite_score",
        "mean_model_presence_share",
        "mean_rde_recognition_inefficiency_subtype_v01",
        "mean_physical_exceptionality_v03",
        "mean_observed_recognition_v04",
    ]
    display_cols = [c for c in display_cols if c in regions_out.columns]

    print(regions_out[display_cols].head(25).to_string(index=False))

    print("\nInterpretation")
    print("--------------")
    print("Universal consensus regions are safest examples.")
    print("RDE-supported regions are best methodology examples.")
    print("RDE-specific regions are the most novel future research targets.")
    print("")
    print("This script bridges empirical validation and the future RDE methodology paper.")


if __name__ == "__main__":
    main()