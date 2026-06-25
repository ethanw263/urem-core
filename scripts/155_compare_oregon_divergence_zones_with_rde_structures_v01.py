#!/usr/bin/env python3

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

SCRIPT_NAME = "155_compare_oregon_divergence_zones_with_rde_structures_v01"

DIVERGENCE_ZONES = Path("data/processed/oregon_divergence_source_sink_zones_v01.gpkg")
CORE_BASINS = Path("data/processed/oregon_final_recognition_core_basins_v01.gpkg")
FLOW_ZONES = Path("data/processed/oregon_recognition_flow_zones_v01.gpkg")

OUTPUT_GPKG = Path("data/processed/oregon_divergence_zone_rde_comparison_v01.gpkg")
OUTPUT_CSV = Path("data/processed/oregon_divergence_zone_rde_comparison_v01.csv")


def safe_overlay_area(base_row, other_gdf):
    geom = base_row.geometry

    hits = other_gdf[other_gdf.intersects(geom)].copy()

    if hits.empty:
        return 0.0, 0

    inter_area = hits.geometry.intersection(geom).area.sum()
    return inter_area / 1_000_000, len(hits)


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    zones = gpd.read_file(DIVERGENCE_ZONES)
    basins = gpd.read_file(CORE_BASINS)
    flows = gpd.read_file(FLOW_ZONES)

    print(f"[{SCRIPT_NAME}] Divergence zones: {len(zones):,}")
    print(f"[{SCRIPT_NAME}] Core basins: {len(basins):,}")
    print(f"[{SCRIPT_NAME}] Flow zones: {len(flows):,}")

    if basins.crs != zones.crs:
        basins = basins.to_crs(zones.crs)

    if flows.crs != zones.crs:
        flows = flows.to_crs(zones.crs)

    records = []

    for _, row in zones.iterrows():
        zone_area_km2 = row.geometry.area / 1_000_000

        basin_overlap_km2, basin_count = safe_overlay_area(row, basins)
        flow_overlap_km2, flow_count = safe_overlay_area(row, flows)

        basin_overlap_pct = (
            basin_overlap_km2 / zone_area_km2
            if zone_area_km2 > 0
            else 0
        )

        flow_overlap_pct = (
            flow_overlap_km2 / zone_area_km2
            if zone_area_km2 > 0
            else 0
        )

        records.append(
            {
                "divergence_zone_id_v01": row["divergence_zone_id_v01"],
                "divergence_zone_type_v01": row["divergence_zone_type_v01"],
                "cell_count_v01": row["cell_count_v01"],
                "zone_area_km2_v01": zone_area_km2,
                "mean_divergence_norm_v01": row["mean_divergence_norm_v01"],
                "core_basin_overlap_km2_v01": basin_overlap_km2,
                "core_basin_overlap_pct_v01": basin_overlap_pct,
                "core_basin_count_intersected_v01": basin_count,
                "flow_zone_overlap_km2_v01": flow_overlap_km2,
                "flow_zone_overlap_pct_v01": flow_overlap_pct,
                "flow_zone_count_intersected_v01": flow_count,
                "geometry": row.geometry,
            }
        )

    out = gpd.GeoDataFrame(records, geometry="geometry", crs=zones.crs)

    def interpret(row):
        if row["core_basin_overlap_pct_v01"] >= 0.25 and row["flow_zone_overlap_pct_v01"] >= 0.25:
            return "basin_flow_aligned"
        elif row["core_basin_overlap_pct_v01"] >= 0.25:
            return "basin_aligned"
        elif row["flow_zone_overlap_pct_v01"] >= 0.25:
            return "flow_aligned"
        else:
            return "weakly_aligned"

    out["rde_structure_alignment_v01"] = out.apply(interpret, axis=1)

    print()
    print(f"[{SCRIPT_NAME}] Alignment counts:")
    print(out["rde_structure_alignment_v01"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] By divergence type:")
    print(
        out.groupby("divergence_zone_type_v01")[
            [
                "zone_area_km2_v01",
                "core_basin_overlap_pct_v01",
                "flow_zone_overlap_pct_v01",
            ]
        ].mean()
    )

    print()
    print(f"[{SCRIPT_NAME}] Top aligned zones:")
    print(
        out.sort_values(
            ["core_basin_overlap_pct_v01", "flow_zone_overlap_pct_v01"],
            ascending=False,
        )[
            [
                "divergence_zone_id_v01",
                "divergence_zone_type_v01",
                "cell_count_v01",
                "core_basin_overlap_pct_v01",
                "flow_zone_overlap_pct_v01",
                "rde_structure_alignment_v01",
            ]
        ]
        .head(25)
        .to_string(index=False)
    )

    print()
    print(f"[{SCRIPT_NAME}] Writing GPKG: {OUTPUT_GPKG}")
    out.to_file(OUTPUT_GPKG, driver="GPKG")

    print(f"[{SCRIPT_NAME}] Writing CSV: {OUTPUT_CSV}")
    out.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()