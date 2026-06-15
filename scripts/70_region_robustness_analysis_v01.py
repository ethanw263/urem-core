#!/usr/bin/env python3
"""
70_region_robustness_analysis_v01.py

Purpose
-------
Test whether UREM discovery REGIONS are robust under alternative score scenarios.

Scientific question
-------------------
Even if exact top cells shift under perturbation, do the same geographic
discovery regions persist?

Inputs
------
data/processed/urem_robustness_cell_scenarios_v02.gpkg
data/processed/v07_no_coast_discovery_regions.gpkg

Outputs
-------
data/processed/urem_region_robustness_summary_v01.csv
data/processed/urem_region_robustness_overlay_v01.gpkg
data/processed/urem_region_robustness_scenario_regions_v01.gpkg
"""

from pathlib import Path
import logging
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

CELL_SCENARIOS_PATH = DATA / "urem_robustness_cell_scenarios_v02.gpkg"
ORIGINAL_REGIONS_PATH = DATA / "v07_no_coast_discovery_regions.gpkg"

OUT_SUMMARY_CSV = DATA / "urem_region_robustness_summary_v01.csv"
OUT_OVERLAY_GPKG = DATA / "urem_region_robustness_overlay_v01.gpkg"
OUT_SCENARIO_REGIONS_GPKG = DATA / "urem_region_robustness_scenario_regions_v01.gpkg"

TOP_N = 300
CELL_BUFFER_M = 1500
MIN_CLUSTER_AREA_KM2 = 1.0

logging.basicConfig(
    level=logging.INFO,
    format="[70_region_robustness_analysis_v01] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


def get_scenario_names(gdf):
    names = []
    prefix = f"in_top_{TOP_N}_"
    for c in gdf.columns:
        if c.startswith(prefix):
            names.append(c.replace(prefix, ""))
    return names


def cluster_top_cells(top_gdf, scenario_name):
    if top_gdf.empty:
        return gpd.GeoDataFrame(
            columns=["scenario", "scenario_region_id", "area_km2", "geometry"],
            geometry="geometry",
            crs=top_gdf.crs,
        )

    dissolved_geom = unary_union(top_gdf.geometry.buffer(CELL_BUFFER_M))

    if dissolved_geom.geom_type == "Polygon":
        geoms = [dissolved_geom]
    else:
        geoms = list(dissolved_geom.geoms)

    rows = []
    for i, geom in enumerate(geoms, start=1):
        area_km2 = geom.area / 1_000_000

        if area_km2 < MIN_CLUSTER_AREA_KM2:
            continue

        rows.append(
            {
                "scenario": scenario_name,
                "scenario_region_id": f"{scenario_name}_{i}",
                "area_km2": area_km2,
                "geometry": geom,
            }
        )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=top_gdf.crs)


def main():
    log.info(f"Reading cell scenario layer: {CELL_SCENARIOS_PATH}")
    cells = gpd.read_file(CELL_SCENARIOS_PATH)

    log.info(f"Reading original discovery regions: {ORIGINAL_REGIONS_PATH}")
    original_regions = gpd.read_file(ORIGINAL_REGIONS_PATH)

    if original_regions.crs != cells.crs:
        original_regions = original_regions.to_crs(cells.crs)

    region_col = None
    for c in ["region_id", "discovery_region_id", "cluster_id", "id"]:
        if c in original_regions.columns:
            region_col = c
            break

    if region_col is None:
        raise KeyError("Could not find region ID column in original regions.")

    scenario_names = get_scenario_names(cells)

    if not scenario_names:
        raise ValueError("No scenario top-membership columns found.")

    log.info(f"Scenarios found: {scenario_names}")

    scenario_region_layers = []

    for scenario in scenario_names:
        col = f"in_top_{TOP_N}_{scenario}"
        top_cells = cells[cells[col].astype(bool)].copy()

        log.info(f"Scenario {scenario}: top cells = {len(top_cells):,}")

        scenario_regions = cluster_top_cells(top_cells, scenario)
        log.info(f"Scenario {scenario}: regions = {len(scenario_regions):,}")

        scenario_region_layers.append(scenario_regions)

    all_scenario_regions = pd.concat(scenario_region_layers, ignore_index=True)
    all_scenario_regions = gpd.GeoDataFrame(
        all_scenario_regions,
        geometry="geometry",
        crs=cells.crs,
    )

    overlay_rows = []

    log.info("Comparing scenario regions to original discovery regions...")

    for _, orig in original_regions.iterrows():
        orig_id = orig[region_col]
        orig_geom = orig.geometry
        orig_area = orig_geom.area

        for _, scen in all_scenario_regions.iterrows():
            scen_geom = scen.geometry

            if not orig_geom.intersects(scen_geom):
                continue

            inter_area = orig_geom.intersection(scen_geom).area
            union_area = orig_geom.union(scen_geom).area

            if union_area == 0:
                iou = 0
            else:
                iou = inter_area / union_area

            overlap_share_original = inter_area / orig_area if orig_area else 0
            centroid_shift_km = orig_geom.centroid.distance(scen_geom.centroid) / 1000

            overlay_rows.append(
                {
                    region_col: orig_id,
                    "scenario": scen["scenario"],
                    "scenario_region_id": scen["scenario_region_id"],
                    "original_area_km2": orig_area / 1_000_000,
                    "scenario_area_km2": scen["area_km2"],
                    "intersection_area_km2": inter_area / 1_000_000,
                    "iou": iou,
                    "overlap_share_original": overlap_share_original,
                    "centroid_shift_km": centroid_shift_km,
                    "geometry": orig_geom.intersection(scen_geom),
                }
            )

    overlay = gpd.GeoDataFrame(overlay_rows, geometry="geometry", crs=cells.crs)

    if overlay.empty:
        raise ValueError("No overlaps found between original regions and scenario regions.")

    summary = (
        overlay.groupby(region_col)
        .agg(
            scenario_presence_count=("scenario", "nunique"),
            mean_iou=("iou", "mean"),
            max_iou=("iou", "max"),
            mean_overlap_share_original=("overlap_share_original", "mean"),
            max_overlap_share_original=("overlap_share_original", "max"),
            mean_centroid_shift_km=("centroid_shift_km", "mean"),
            min_centroid_shift_km=("centroid_shift_km", "min"),
            total_intersection_area_km2=("intersection_area_km2", "sum"),
        )
        .reset_index()
    )

    summary["scenario_presence_share"] = summary["scenario_presence_count"] / len(scenario_names)

    original_area = original_regions[[region_col, "geometry"]].copy()
    original_area["original_region_area_km2"] = original_area.geometry.area / 1_000_000
    original_area = original_area.drop(columns="geometry")

    summary = summary.merge(original_area, on=region_col, how="left")

    summary["robustness_class"] = pd.cut(
        summary["scenario_presence_share"],
        bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
        labels=[
            "weak",
            "moderate",
            "strong",
            "very_strong",
        ],
    )

    log.info(f"Writing summary CSV: {OUT_SUMMARY_CSV}")
    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    log.info(f"Writing overlay GPKG: {OUT_OVERLAY_GPKG}")
    overlay.to_file(OUT_OVERLAY_GPKG, driver="GPKG")

    log.info(f"Writing scenario regions GPKG: {OUT_SCENARIO_REGIONS_GPKG}")
    all_scenario_regions.to_file(OUT_SCENARIO_REGIONS_GPKG, driver="GPKG")

    print("\nUREM Region Robustness Summary")
    print("------------------------------")
    print(summary.sort_values("scenario_presence_share", ascending=False).to_string(index=False))

    print("\nInterpretation")
    print("--------------")
    print("scenario_presence_share measures how often each original discovery region")
    print("reappears under alternative UREM scoring scenarios.")
    print()
    print(">= 0.75 = very strong regional robustness")
    print("0.50–0.75 = strong/moderate regional robustness")
    print("< 0.50 = weak regional robustness")
    print()
    print("This is more important than exact cell overlap because UREM is increasingly")
    print("a discovery-region methodology, not a single-cell ranking tool.")


if __name__ == "__main__":
    main()