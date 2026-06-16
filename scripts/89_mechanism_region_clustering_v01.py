#!/usr/bin/env python3
"""
89_mechanism_region_clustering_v01.py

Purpose
-------
Cluster orthogonalized mechanism-class cells into interpretable mechanism regions.

This converts Script 88 from a cell-level mechanism taxonomy into landscape-scale
mechanism regions.

Scientific question
-------------------
Do Recognition Inefficiency, Opportunity Failure, and Recognition Diversion
form coherent geographic landscapes?

Inputs
------
data/processed/orthogonalized_mechanism_taxonomy_v01.gpkg

Outputs
-------
data/processed/mechanism_regions_v01.gpkg
data/processed/mechanism_regions_summary_v01.csv
data/processed/mechanism_regions_class_summary_v01.csv
data/processed/mechanism_region_cells_v01.csv
data/processed/mechanism_region_cells_v01.gpkg
data/processed/mechanism_region_framework_v01.md
"""

from pathlib import Path
import logging
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

INPUT_GPKG = DATA / "orthogonalized_mechanism_taxonomy_v01.gpkg"

OUT_REGIONS_GPKG = DATA / "mechanism_regions_v01.gpkg"
OUT_REGION_SUMMARY = DATA / "mechanism_regions_summary_v01.csv"
OUT_CLASS_SUMMARY = DATA / "mechanism_regions_class_summary_v01.csv"
OUT_CELLS_CSV = DATA / "mechanism_region_cells_v01.csv"
OUT_CELLS_GPKG = DATA / "mechanism_region_cells_v01.gpkg"
OUT_MD = DATA / "mechanism_region_framework_v01.md"

CELL_BUFFER_M = 1500
MIN_REGION_AREA_KM2 = 1.0
TOP_N_PER_CLASS = 500

TARGET_CLASSES = [
    "Recognition Inefficiency",
    "Comparative Shadowing / Recognition Diversion Candidate",
    "Opportunity Failure",
    "Transmission Failure",
    "General Under-Recognized Exceptionality",
]

logging.basicConfig(
    level=logging.INFO,
    format="[89_mechanism_region_clustering_v01] %(levelname)s: %(message)s",
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


def safe_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def cluster_cells(cells: gpd.GeoDataFrame, mechanism_class: str) -> gpd.GeoDataFrame:
    if cells.empty:
        return gpd.GeoDataFrame(
            columns=[
                "mechanism_class",
                "mechanism_region_id",
                "mechanism_region_area_km2",
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
    prefix = safe_name(mechanism_class)

    for i, geom in enumerate(geoms, start=1):
        area_km2 = geom.area / 1_000_000

        if area_km2 < MIN_REGION_AREA_KM2:
            continue

        rows.append(
            {
                "mechanism_class": mechanism_class,
                "mechanism_region_id": f"{prefix}_{i}",
                "mechanism_region_area_km2": area_km2,
                "geometry": geom,
            }
        )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=cells.crs)


def main():
    log.info(f"Reading mechanism taxonomy: {INPUT_GPKG}")
    gdf = gpd.read_file(INPUT_GPKG)

    required = [
        "cell_id",
        "orthogonalized_mechanism_class_v01",
        "orthogonalized_rde_v01",
        "mechanism_expression_score_v01",
        "P_orthogonal_v01",
        "O_base_opportunity_v01",
        "T_net_transmission_v01",
        "R_net_under_recognition_v01",
        "rde_v01_composite_score",
        "is_valid_land_candidate",
    ]

    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    valid = gdf["is_valid_land_candidate"].astype(bool)
    gdf = gdf[valid].copy()

    log.info(f"Valid rows: {len(gdf):,}")

    selected_cells = []

    for cls in TARGET_CLASSES:
        cls_cells = gdf[gdf["orthogonalized_mechanism_class_v01"] == cls].copy()

        if cls_cells.empty:
            log.warning(f"No cells for class: {cls}")
            continue

        cls_cells = (
            cls_cells.sort_values("orthogonalized_rde_v01", ascending=False)
            .head(TOP_N_PER_CLASS)
            .copy()
        )

        log.info(f"{cls}: selected cells = {len(cls_cells):,}")

        selected_cells.append(cls_cells)

    if not selected_cells:
        raise ValueError("No mechanism cells selected.")

    selected = pd.concat(selected_cells, ignore_index=True)
    selected = gpd.GeoDataFrame(selected, geometry="geometry", crs=gdf.crs)

    log.info(f"Total selected mechanism cells: {len(selected):,}")

    all_regions = []

    for cls in selected["orthogonalized_mechanism_class_v01"].unique():
        cls_cells = selected[selected["orthogonalized_mechanism_class_v01"] == cls].copy()

        regions = cluster_cells(cls_cells, cls)

        log.info(f"{cls}: regions = {len(regions):,}")

        all_regions.append(regions)

    regions = pd.concat(all_regions, ignore_index=True)
    regions = gpd.GeoDataFrame(regions, geometry="geometry", crs=gdf.crs)

    if regions.empty:
        raise ValueError("No mechanism regions created.")

    # Join selected cells to mechanism regions.
    joined = gpd.sjoin(
        selected,
        regions[["mechanism_class", "mechanism_region_id", "geometry"]],
        how="left",
        predicate="within",
    )

    # Prevent accidental cross-class joins.
    joined = joined[
        joined["orthogonalized_mechanism_class_v01"] == joined["mechanism_class"]
    ].copy()

    metric_cols = [
        "orthogonalized_rde_v01",
        "mechanism_expression_score_v01",
        "P_orthogonal_v01",
        "O_base_opportunity_v01",
        "T_net_transmission_v01",
        "R_net_under_recognition_v01",
        "rde_v01_composite_score",
    ]

    optional_cols = [
        "physical_exceptionality_v03",
        "observed_recognition_v04",
        "expected_recognition_v06",
        "recognition_transmission_index_v01",
        "opportunity_structure_index_v01",
    ]

    metric_cols += [c for c in optional_cols if c in joined.columns]

    agg_dict = {
        "cell_count": ("cell_id", "count"),
    }

    for c in metric_cols:
        agg_dict[f"mean_{c}"] = (c, "mean")
        agg_dict[f"median_{c}"] = (c, "median")

    region_summary = (
        joined.dropna(subset=["mechanism_region_id"])
        .groupby(["mechanism_class", "mechanism_region_id"])
        .agg(**agg_dict)
        .reset_index()
    )

    regions_out = regions.merge(
        region_summary,
        on=["mechanism_class", "mechanism_region_id"],
        how="left",
    )

    # Region priority score.
    score_parts = []
    for col in [
        "mean_orthogonalized_rde_v01",
        "mean_mechanism_expression_score_v01",
        "mean_rde_v01_composite_score",
    ]:
        if col in regions_out.columns:
            score_parts.append(col)

    regions_out["mechanism_region_priority_score_v01"] = regions_out[score_parts].mean(axis=1)

    def priority_class(score):
        if pd.isna(score):
            return "unclassified"
        if score >= 0.85:
            return "core mechanism region"
        if score >= 0.75:
            return "strong mechanism region"
        if score >= 0.65:
            return "promising mechanism region"
        return "secondary mechanism region"

    regions_out["mechanism_region_priority_class_v01"] = regions_out[
        "mechanism_region_priority_score_v01"
    ].apply(priority_class)

    regions_out = regions_out.sort_values(
        ["mechanism_region_priority_score_v01", "mechanism_region_area_km2"],
        ascending=False,
    )

    region_table = regions_out.drop(columns="geometry").copy()

    class_summary = (
        regions_out.groupby("mechanism_class")
        .agg(
            region_count=("mechanism_region_id", "count"),
            total_cells=("cell_count", "sum"),
            mean_region_area_km2=("mechanism_region_area_km2", "mean"),
            total_region_area_km2=("mechanism_region_area_km2", "sum"),
            mean_priority_score=("mechanism_region_priority_score_v01", "mean"),
            mean_orthogonalized_rde=("mean_orthogonalized_rde_v01", "mean"),
            mean_expression_score=("mean_mechanism_expression_score_v01", "mean"),
            mean_P=("mean_P_orthogonal_v01", "mean"),
            mean_O=("mean_O_base_opportunity_v01", "mean"),
            mean_Tnet=("mean_T_net_transmission_v01", "mean"),
            mean_under_recognition=("mean_R_net_under_recognition_v01", "mean"),
        )
        .reset_index()
        .sort_values("mean_priority_score", ascending=False)
    )

    # Outputs.
    log.info(f"Writing regions GPKG: {OUT_REGIONS_GPKG}")
    regions_out.to_file(OUT_REGIONS_GPKG, driver="GPKG")

    log.info(f"Writing region summary CSV: {OUT_REGION_SUMMARY}")
    region_table.to_csv(OUT_REGION_SUMMARY, index=False)

    log.info(f"Writing class summary CSV: {OUT_CLASS_SUMMARY}")
    class_summary.to_csv(OUT_CLASS_SUMMARY, index=False)

    log.info(f"Writing cells CSV: {OUT_CELLS_CSV}")
    selected.drop(columns="geometry").to_csv(OUT_CELLS_CSV, index=False)

    log.info(f"Writing cells GPKG: {OUT_CELLS_GPKG}")
    selected.to_file(OUT_CELLS_GPKG, driver="GPKG")

    md = []
    md.append("# Mechanism Region Clustering v01")
    md.append("")
    md.append("## Purpose")
    md.append("")
    md.append("This script converts orthogonalized mechanism cells into mechanism landscapes.")
    md.append("")
    md.append("It moves RDE from cell-level mechanism taxonomy toward geographic mechanism regions.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Target Mechanisms")
    md.append("")
    for cls in TARGET_CLASSES:
        md.append(f"- {cls}")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Class Summary")
    md.append("")
    md.append(dataframe_to_markdown(class_summary))
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Top Mechanism Regions")
    md.append("")
    display_cols = [
        "mechanism_class",
        "mechanism_region_id",
        "mechanism_region_priority_class_v01",
        "mechanism_region_priority_score_v01",
        "cell_count",
        "mechanism_region_area_km2",
        "mean_orthogonalized_rde_v01",
        "mean_P_orthogonal_v01",
        "mean_O_base_opportunity_v01",
        "mean_T_net_transmission_v01",
        "mean_R_net_under_recognition_v01",
        "mean_rde_v01_composite_score",
    ]
    display_cols = [c for c in display_cols if c in region_table.columns]
    md.append(dataframe_to_markdown(region_table[display_cols].head(40)))
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Methodological Interpretation")
    md.append("")
    md.append("Mechanism regions are candidate geographic objects for future RDE theory.")
    md.append("")
    md.append("Recognition Inefficiency and Comparative Shadowing regions are likely the most novel targets.")
    md.append("")
    md.append("Opportunity Failure and Transmission Failure regions explain under-recognition through access or diffusion limits.")

    log.info(f"Writing MD: {OUT_MD}")
    OUT_MD.write_text("\n".join(md))

    print("\nMechanism Region Clustering v01")
    print("-------------------------------")
    print(f"Selected cells: {len(selected):,}")
    print(f"Mechanism regions: {len(regions_out):,}")

    print("\nClass summary:")
    print(class_summary.to_string(index=False))

    print("\nTop regions:")
    print(region_table[display_cols].head(30).to_string(index=False))

    print("\nWrote:")
    print(f"- {OUT_REGIONS_GPKG}")
    print(f"- {OUT_REGION_SUMMARY}")
    print(f"- {OUT_CLASS_SUMMARY}")
    print(f"- {OUT_CELLS_CSV}")
    print(f"- {OUT_CELLS_GPKG}")
    print(f"- {OUT_MD}")

    print("\nInterpretation")
    print("--------------")
    print("This produces the first landscape-scale mechanism map.")
    print("Next, inspect the GPKG in QGIS and compare Recognition Inefficiency")
    print("regions against Comparative Shadowing and Opportunity Failure regions.")


if __name__ == "__main__":
    main()