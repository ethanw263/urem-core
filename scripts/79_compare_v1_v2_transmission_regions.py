#!/usr/bin/env python3
"""
79_compare_v1_v2_transmission_regions.py

Purpose
-------
Compare Coastal UREM v1, Opportunity-adjusted v2, and Transmission-limited
disequilibrium regions.

Scientific question
-------------------
Does recognition transmission identify a distinct class of recognition
disequilibrium, or is it merely selecting remote zero-recognition cells?

Outputs
-------
data/processed/urem_transmission_limited_regions_v01.gpkg
data/processed/urem_v1_v2_transmission_region_comparison_v01.csv
data/processed/urem_v1_v2_transmission_region_overlay_v01.gpkg
"""

from pathlib import Path
import logging
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

V1_REGIONS_PATH = DATA / "v07_no_coast_discovery_regions.gpkg"
V2_REGIONS_PATH = DATA / "urem_v2_opportunity_adjusted_regions.gpkg"
TRANSMISSION_PATH = DATA / "recognition_transmission_index_v01.gpkg"

OUT_TRANS_REGIONS = DATA / "urem_transmission_limited_regions_v01.gpkg"
OUT_COMPARISON_CSV = DATA / "urem_v1_v2_transmission_region_comparison_v01.csv"
OUT_OVERLAY_GPKG = DATA / "urem_v1_v2_transmission_region_overlay_v01.gpkg"

TOP_N = 300
CELL_BUFFER_M = 1500
MIN_REGION_AREA_KM2 = 1.0

logging.basicConfig(
    level=logging.INFO,
    format="[79_compare_v1_v2_transmission_regions] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


def get_region_col(gdf):
    for c in ["region_id", "v2_region_id", "transmission_region_id", "id"]:
        if c in gdf.columns:
            return c
    return None


def cluster_cells(cells, id_col_name):
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
                id_col_name: i,
                "area_km2": area_km2,
                "geometry": geom,
            }
        )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=cells.crs)


def summarize_cells_by_region(cells, regions, region_col):
    joined = gpd.sjoin(
        cells,
        regions[[region_col, "geometry"]],
        how="left",
        predicate="within",
    )

    metric_cols = [
        "transmission_limited_disequilibrium_v01",
        "recognition_transmission_index_v01",
        "recognition_transmission_deficit_v01",
        "recognition_disequilibrium_index_v01",
        "physical_exceptionality_v03",
        "observed_recognition_v04",
        "opportunity_structure_index_v01",
    ]

    metric_cols = [c for c in metric_cols if c in joined.columns]

    agg = {"cell_count": ("cell_id", "count")}

    for c in metric_cols:
        agg[f"mean_{c}"] = (c, "mean")
        agg[f"median_{c}"] = (c, "median")

    summary = (
        joined.dropna(subset=[region_col])
        .groupby(region_col)
        .agg(**agg)
        .reset_index()
    )

    return regions.merge(summary, on=region_col, how="left")


def overlay_compare(source_regions, source_name, source_id_col, target_regions, target_name, target_id_col):
    rows = []

    for _, src in source_regions.iterrows():
        src_id = src[source_id_col]
        src_geom = src.geometry
        src_area = src_geom.area

        for _, tgt in target_regions.iterrows():
            tgt_id = tgt[target_id_col]
            tgt_geom = tgt.geometry

            if not src_geom.intersects(tgt_geom):
                continue

            inter = src_geom.intersection(tgt_geom)
            inter_area = inter.area
            union_area = src_geom.union(tgt_geom).area

            rows.append(
                {
                    "source_layer": source_name,
                    "source_region_id": src_id,
                    "target_layer": target_name,
                    "target_region_id": tgt_id,
                    "source_area_km2": src_area / 1_000_000,
                    "target_area_km2": tgt_geom.area / 1_000_000,
                    "intersection_area_km2": inter_area / 1_000_000,
                    "source_overlap_share": inter_area / src_area if src_area else 0,
                    "target_overlap_share": inter_area / tgt_geom.area if tgt_geom.area else 0,
                    "iou": inter_area / union_area if union_area else 0,
                    "centroid_shift_km": src_geom.centroid.distance(tgt_geom.centroid) / 1000,
                    "geometry": inter,
                }
            )

    return rows


def main():
    log.info(f"Reading v1 regions: {V1_REGIONS_PATH}")
    v1 = gpd.read_file(V1_REGIONS_PATH)

    log.info(f"Reading v2 regions: {V2_REGIONS_PATH}")
    v2 = gpd.read_file(V2_REGIONS_PATH)

    log.info(f"Reading transmission layer: {TRANSMISSION_PATH}")
    transmission = gpd.read_file(TRANSMISSION_PATH)

    if transmission.crs != v1.crs:
        transmission = transmission.to_crs(v1.crs)

    if v2.crs != v1.crs:
        v2 = v2.to_crs(v1.crs)

    v1_col = get_region_col(v1)
    v2_col = get_region_col(v2)

    if v1_col is None:
        raise KeyError("Could not identify v1 region column.")
    if v2_col is None:
        raise KeyError("Could not identify v2 region column.")

    top_transmission = (
        transmission.sort_values(
            "transmission_limited_disequilibrium_v01",
            ascending=False,
        )
        .head(TOP_N)
        .copy()
    )

    log.info(f"Top transmission-limited cells: {len(top_transmission):,}")

    log.info("Clustering transmission-limited cells...")
    trans_regions = cluster_cells(top_transmission, "transmission_region_id")

    log.info(f"Transmission-limited regions: {len(trans_regions):,}")

    trans_regions = summarize_cells_by_region(
        top_transmission,
        trans_regions,
        "transmission_region_id",
    )

    log.info(f"Writing transmission regions: {OUT_TRANS_REGIONS}")
    trans_regions.to_file(OUT_TRANS_REGIONS, driver="GPKG")

    overlay_rows = []
    overlay_rows += overlay_compare(trans_regions, "transmission", "transmission_region_id", v1, "v1", v1_col)
    overlay_rows += overlay_compare(trans_regions, "transmission", "transmission_region_id", v2, "v2", v2_col)

    overlay = gpd.GeoDataFrame(overlay_rows, geometry="geometry", crs=v1.crs)

    if not overlay.empty:
        log.info(f"Writing overlay: {OUT_OVERLAY_GPKG}")
        overlay.to_file(OUT_OVERLAY_GPKG, driver="GPKG")

    comparison_rows = []

    for _, tr in trans_regions.iterrows():
        tid = tr["transmission_region_id"]

        subset = overlay[
            (overlay["source_layer"] == "transmission")
            & (overlay["source_region_id"] == tid)
        ] if not overlay.empty else pd.DataFrame()

        v1_subset = subset[subset["target_layer"] == "v1"] if not subset.empty else pd.DataFrame()
        v2_subset = subset[subset["target_layer"] == "v2"] if not subset.empty else pd.DataFrame()

        max_v1_iou = v1_subset["iou"].max() if not v1_subset.empty else 0
        max_v2_iou = v2_subset["iou"].max() if not v2_subset.empty else 0

        max_v1_overlap = v1_subset["source_overlap_share"].max() if not v1_subset.empty else 0
        max_v2_overlap = v2_subset["source_overlap_share"].max() if not v2_subset.empty else 0

        if max_v1_overlap >= 0.30 or max_v2_overlap >= 0.30:
            novelty_class = "Supported by existing UREM landscapes"
        elif max_v1_overlap >= 0.10 or max_v2_overlap >= 0.10:
            novelty_class = "Partially related to existing UREM landscapes"
        else:
            novelty_class = "Transmission-specific new landscape"

        comparison_rows.append(
            {
                "transmission_region_id": tid,
                "area_km2": tr["area_km2"],
                "cell_count": tr.get("cell_count", None),
                "mean_transmission_limited_disequilibrium": tr.get(
                    "mean_transmission_limited_disequilibrium_v01",
                    None,
                ),
                "mean_recognition_transmission_index": tr.get(
                    "mean_recognition_transmission_index_v01",
                    None,
                ),
                "mean_recognition_transmission_deficit": tr.get(
                    "mean_recognition_transmission_deficit_v01",
                    None,
                ),
                "mean_recognition_disequilibrium_index": tr.get(
                    "mean_recognition_disequilibrium_index_v01",
                    None,
                ),
                "mean_physical_exceptionality": tr.get(
                    "mean_physical_exceptionality_v03",
                    None,
                ),
                "mean_observed_recognition": tr.get(
                    "mean_observed_recognition_v04",
                    None,
                ),
                "max_v1_iou": max_v1_iou,
                "max_v2_iou": max_v2_iou,
                "max_v1_overlap_share": max_v1_overlap,
                "max_v2_overlap_share": max_v2_overlap,
                "novelty_class": novelty_class,
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison = comparison.sort_values(
        ["novelty_class", "area_km2"],
        ascending=[True, False],
    )

    log.info(f"Writing comparison CSV: {OUT_COMPARISON_CSV}")
    comparison.to_csv(OUT_COMPARISON_CSV, index=False)

    print("\nV1 / V2 / Transmission Region Comparison")
    print("----------------------------------------")
    print(f"V1 regions: {len(v1):,}")
    print(f"V2 regions: {len(v2):,}")
    print(f"Transmission-limited regions: {len(trans_regions):,}")

    print("\nTransmission region comparison:")
    print(comparison.to_string(index=False))

    print("\nInterpretation")
    print("--------------")
    print("If transmission regions overlap v1/v2, transmission is reinforcing existing discoveries.")
    print("If they are mostly new, transmission may be identifying a distinct disequilibrium class.")
    print("If top transmission regions have observed_recognition near zero, later we need a")
    print("zero-recognition artifact test before treating them as final discoveries.")


if __name__ == "__main__":
    main()