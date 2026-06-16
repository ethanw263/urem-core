#!/usr/bin/env python3
"""
75_compare_v1_v2_regions.py

Purpose
-------
Compare Coastal UREM v1 discovery regions with UREM v2 opportunity-adjusted
candidate regions.

Scientific question
-------------------
Does opportunity-adjusted UREM v2 confirm the same landscapes as v1, shift
within similar landscapes, or reveal new recognition-disequilibrium regions?

Inputs
------
data/processed/v07_no_coast_discovery_regions.gpkg
data/processed/ranked_urem_v2_opportunity_adjusted_candidates.gpkg

Outputs
-------
data/processed/urem_v1_v2_region_comparison_overlay.gpkg
data/processed/urem_v2_opportunity_adjusted_regions.gpkg
data/processed/urem_v1_v2_region_comparison_summary.csv
"""

from pathlib import Path
import logging
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

V1_REGIONS_PATH = DATA / "v07_no_coast_discovery_regions.gpkg"
V2_CANDIDATES_PATH = DATA / "ranked_urem_v2_opportunity_adjusted_candidates.gpkg"

OUT_V2_REGIONS = DATA / "urem_v2_opportunity_adjusted_regions.gpkg"
OUT_OVERLAY = DATA / "urem_v1_v2_region_comparison_overlay.gpkg"
OUT_SUMMARY = DATA / "urem_v1_v2_region_comparison_summary.csv"

TOP_N = 300
CELL_BUFFER_M = 1500
MIN_REGION_AREA_KM2 = 1.0

logging.basicConfig(
    level=logging.INFO,
    format="[75_compare_v1_v2_regions] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


def get_region_col(gdf):
    for c in ["region_id", "discovery_region_id", "cluster_id", "id"]:
        if c in gdf.columns:
            return c
    return None


def cluster_cells_to_regions(cells):
    cells = cells.copy()

    dissolved = unary_union(cells.geometry.buffer(CELL_BUFFER_M))

    if dissolved.geom_type == "Polygon":
        geoms = [dissolved]
    else:
        geoms = list(dissolved.geoms)

    rows = []

    for i, geom in enumerate(geoms, start=1):
        area_km2 = geom.area / 1_000_000

        if area_km2 < MIN_REGION_AREA_KM2:
            continue

        rows.append(
            {
                "v2_region_id": i,
                "v2_region_area_km2": area_km2,
                "geometry": geom,
            }
        )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=cells.crs)


def main():
    log.info(f"Reading v1 regions: {V1_REGIONS_PATH}")
    v1 = gpd.read_file(V1_REGIONS_PATH)

    log.info(f"Reading v2 candidates: {V2_CANDIDATES_PATH}")
    v2_cells = gpd.read_file(V2_CANDIDATES_PATH).head(TOP_N)

    if v2_cells.crs != v1.crs:
        v2_cells = v2_cells.to_crs(v1.crs)

    v1_region_col = get_region_col(v1)

    if v1_region_col is None:
        raise KeyError("Could not identify v1 region ID column.")

    log.info(f"Using v1 region column: {v1_region_col}")
    log.info(f"V2 top cells used: {len(v2_cells):,}")

    log.info("Clustering v2 top cells into regions...")
    v2_regions = cluster_cells_to_regions(v2_cells)

    log.info(f"V2 regions created: {len(v2_regions):,}")

    # Add v2 cell counts and mean metrics.
    joined_v2_cells = gpd.sjoin(
        v2_cells,
        v2_regions[["v2_region_id", "geometry"]],
        how="left",
        predicate="within",
    )

    metric_cols = [
        "urem_v2_opportunity_adjusted_score",
        "physical_potential_component_v2",
        "recognition_opportunity_v2",
        "expected_recognition_opportunity_adjusted_v2",
        "recognition_disequilibrium_component_v2",
        "observed_recognition_v04",
        "expected_recognition_v06",
    ]

    valid_metric_cols = [c for c in metric_cols if c in joined_v2_cells.columns]

    agg_dict = {"v2_cell_count": ("cell_id", "count")}
    for c in valid_metric_cols:
        agg_dict[f"mean_{c}"] = (c, "mean")

    v2_summary = (
        joined_v2_cells.dropna(subset=["v2_region_id"])
        .groupby("v2_region_id")
        .agg(**agg_dict)
        .reset_index()
    )

    v2_regions = v2_regions.merge(v2_summary, on="v2_region_id", how="left")

    log.info(f"Writing v2 regions: {OUT_V2_REGIONS}")
    v2_regions.to_file(OUT_V2_REGIONS, driver="GPKG")

    # Overlay comparison.
    rows = []

    for _, r1 in v1.iterrows():
        v1_id = r1[v1_region_col]
        g1 = r1.geometry
        area1 = g1.area

        for _, r2 in v2_regions.iterrows():
            g2 = r2.geometry

            if not g1.intersects(g2):
                continue

            inter = g1.intersection(g2)
            inter_area = inter.area
            union_area = g1.union(g2).area

            rows.append(
                {
                    "v1_region_id": v1_id,
                    "v2_region_id": r2["v2_region_id"],
                    "v1_area_km2": area1 / 1_000_000,
                    "v2_area_km2": r2["v2_region_area_km2"],
                    "intersection_area_km2": inter_area / 1_000_000,
                    "v1_overlap_share": inter_area / area1 if area1 else 0,
                    "v2_overlap_share": inter_area / g2.area if g2.area else 0,
                    "iou": inter_area / union_area if union_area else 0,
                    "centroid_shift_km": g1.centroid.distance(g2.centroid) / 1000,
                    "geometry": inter,
                }
            )

    overlay = gpd.GeoDataFrame(rows, geometry="geometry", crs=v1.crs)

    if overlay.empty:
        log.warning("No direct overlaps found between v1 and v2 regions.")

    else:
        log.info(f"Writing overlay: {OUT_OVERLAY}")
        overlay.to_file(OUT_OVERLAY, driver="GPKG")

    # Region comparison summary.
    if overlay.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            overlay.groupby("v1_region_id")
            .agg(
                matched_v2_regions=("v2_region_id", "nunique"),
                max_iou=("iou", "max"),
                mean_iou=("iou", "mean"),
                max_v1_overlap_share=("v1_overlap_share", "max"),
                mean_v1_overlap_share=("v1_overlap_share", "mean"),
                min_centroid_shift_km=("centroid_shift_km", "min"),
                mean_centroid_shift_km=("centroid_shift_km", "mean"),
                total_intersection_area_km2=("intersection_area_km2", "sum"),
            )
            .reset_index()
        )

        def classify(row):
            if row["max_v1_overlap_share"] >= 0.50 or row["max_iou"] >= 0.35:
                return "Confirmed by v2"
            if row["max_v1_overlap_share"] >= 0.20 or row["min_centroid_shift_km"] <= 5:
                return "Partially shifted / nearby v2 support"
            return "Weak v2 support"

        summary["v1_v2_relationship"] = summary.apply(classify, axis=1)

    # Identify v2 regions with weak/no v1 overlap.
    if not overlay.empty:
        v2_overlap = (
            overlay.groupby("v2_region_id")
            .agg(
                max_v1_overlap_share=("v2_overlap_share", "max"),
                max_iou=("iou", "max"),
            )
            .reset_index()
        )
    else:
        v2_overlap = pd.DataFrame(columns=["v2_region_id", "max_v1_overlap_share", "max_iou"])

    v2_all = v2_regions.drop(columns="geometry").copy()
    v2_all = v2_all.merge(v2_overlap, on="v2_region_id", how="left")
    v2_all["max_v1_overlap_share"] = v2_all["max_v1_overlap_share"].fillna(0)
    v2_all["max_iou"] = v2_all["max_iou"].fillna(0)

    v2_all["v2_novelty_class"] = v2_all.apply(
        lambda r: (
            "V2 new/displaced region"
            if r["max_v1_overlap_share"] < 0.10 and r["max_iou"] < 0.10
            else "V2 overlaps v1 landscape"
        ),
        axis=1,
    )

    summary_out = {
        "v1_region_comparison": summary,
        "v2_region_novelty": v2_all,
    }

    # Flatten into one CSV with section labels.
    rows_out = []

    if not summary.empty:
        for _, row in summary.iterrows():
            d = row.to_dict()
            d["section"] = "v1_region_comparison"
            rows_out.append(d)

    for _, row in v2_all.iterrows():
        d = row.to_dict()
        d["section"] = "v2_region_novelty"
        rows_out.append(d)

    final_summary = pd.DataFrame(rows_out)

    log.info(f"Writing summary CSV: {OUT_SUMMARY}")
    final_summary.to_csv(OUT_SUMMARY, index=False)

    print("\nUREM v1 vs v2 Region Comparison")
    print("--------------------------------")
    print(f"V1 regions: {len(v1):,}")
    print(f"V2 top cells: {len(v2_cells):,}")
    print(f"V2 regions: {len(v2_regions):,}")

    if not summary.empty:
        print("\nV1 regions compared to v2:")
        print(
            summary.sort_values(
                ["v1_v2_relationship", "max_v1_overlap_share"],
                ascending=[True, False],
            ).to_string(index=False)
        )

    print("\nV2 region novelty summary:")
    print(
        v2_all[
            [
                "v2_region_id",
                "v2_region_area_km2",
                "v2_cell_count",
                "max_v1_overlap_share",
                "max_iou",
                "v2_novelty_class",
            ]
        ].sort_values("v2_region_area_km2", ascending=False).to_string(index=False)
    )

    print("\nInterpretation")
    print("--------------")
    print("If many v2 regions overlap v1, opportunity adjustment confirms v1.")
    print("If many v2 regions are new/displaced, opportunity adjustment is changing")
    print("the theory and may represent a real v2 methodological shift.")


if __name__ == "__main__":
    main()