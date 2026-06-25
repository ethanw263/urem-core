#!/usr/bin/env python3

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

SCRIPT_NAME = "158_identify_oregon_divergence_source_sink_zones_v02"

DIVERGENCE_FIELD = Path("data/processed/oregon_recognition_divergence_field_v01.gpkg")
CORE_BASINS = Path("data/processed/oregon_final_recognition_core_basins_v01.gpkg")
FLOW_ZONES = Path("data/processed/oregon_recognition_flow_zones_v01.gpkg")
DISCOVERY_REGIONS = Path("data/processed/oregon_discovery_regions_v06.gpkg")

OUTPUT_GPKG = Path("data/processed/oregon_divergence_source_sink_zones_v02.gpkg")
OUTPUT_CSV = Path("data/processed/oregon_divergence_source_sink_zones_v02.csv")

STRONG_DIVERGENCE_THRESHOLD = 0.50
MIN_ZONE_CELLS = 5


def overlap_stats(zone_geom, other):
    hits = other[other.intersects(zone_geom)]
    if hits.empty:
        return 0, 0.0

    inter_area = hits.geometry.intersection(zone_geom).area.sum()
    pct = inter_area / zone_geom.area if zone_geom.area > 0 else 0.0

    return len(hits), pct


def confidence(row):
    strength = row["abs_mean_divergence_norm_v02"]
    cells = row["cell_count_v02"]
    flow = row["flow_zone_overlap_pct_v02"]
    basin = row["core_basin_overlap_pct_v02"]
    discovery = row["discovery_region_overlap_pct_v02"]

    if flow >= 0.25 and strength >= 0.50:
        return "high_confidence_divergence_zone"

    if discovery >= 0.25 and strength >= 0.50:
        return "high_confidence_divergence_zone"

    if basin >= 0.10 and strength >= 0.50:
        return "moderate_confidence_divergence_zone"

    if cells >= 75 and strength >= 0.60:
        return "moderate_confidence_structural_zone"

    return "structural_candidate_zone"


def interpretation(row):
    ztype = row["divergence_zone_type_v02"]
    flow = row["flow_zone_overlap_pct_v02"]
    basin = row["core_basin_overlap_pct_v02"]
    discovery = row["discovery_region_overlap_pct_v02"]

    if ztype == "source_zone":
        if flow >= 0.25:
            return "active_recognition_dispersal_source"
        if discovery >= 0.25:
            return "discovery_region_source"
        if basin >= 0.10:
            return "basin_edge_source"
        return "structural_recognition_source"

    if ztype == "sink_zone":
        if flow >= 0.25:
            return "flow_convergence_sink"
        if basin >= 0.10:
            return "recognition_accumulation_sink"
        if discovery >= 0.25:
            return "discovery_region_sink"
        return "structural_recognition_sink"

    return "unclassified"


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    gdf = gpd.read_file(DIVERGENCE_FIELD)
    basins = gpd.read_file(CORE_BASINS)
    flows = gpd.read_file(FLOW_ZONES)
    discovery = gpd.read_file(DISCOVERY_REGIONS)

    for layer_name, layer in [
        ("core basins", basins),
        ("flow zones", flows),
        ("discovery regions", discovery),
    ]:
        if layer.crs != gdf.crs:
            print(f"[{SCRIPT_NAME}] Reprojecting {layer_name}")
            if layer_name == "core basins":
                basins = layer.to_crs(gdf.crs)
            elif layer_name == "flow zones":
                flows = layer.to_crs(gdf.crs)
            elif layer_name == "discovery regions":
                discovery = layer.to_crs(gdf.crs)

    print(f"[{SCRIPT_NAME}] Input cells: {len(gdf):,}")

    required = [
        "recognition_divergence_norm_v01",
        "recognition_divergence_v01",
    ]

    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    gdf["divergence_zone_type_v02"] = np.select(
        [
            gdf["recognition_divergence_norm_v01"] >= STRONG_DIVERGENCE_THRESHOLD,
            gdf["recognition_divergence_norm_v01"] <= -STRONG_DIVERGENCE_THRESHOLD,
        ],
        [
            "source_zone",
            "sink_zone",
        ],
        default="not_candidate",
    )

    candidates = gdf[gdf["divergence_zone_type_v02"] != "not_candidate"].copy()

    print(f"[{SCRIPT_NAME}] Strong divergence candidate cells: {len(candidates):,}")
    print(candidates["divergence_zone_type_v02"].value_counts())

    zones = []

    for ztype, sub in candidates.groupby("divergence_zone_type_v02"):
        print(f"[{SCRIPT_NAME}] Building connected components for {ztype}")

        dissolved = sub.dissolve(by="divergence_zone_type_v02")
        exploded = dissolved.explode(index_parts=False).reset_index(drop=True)

        for idx, row in exploded.iterrows():
            geom = row.geometry

            cells = sub[sub.intersects(geom)].copy()
            cell_count = len(cells)

            if cell_count < MIN_ZONE_CELLS:
                continue

            mean_div = cells["recognition_divergence_v01"].mean()
            mean_norm = cells["recognition_divergence_norm_v01"].mean()
            abs_mean_norm = abs(mean_norm)

            basin_count, basin_pct = overlap_stats(geom, basins)
            flow_count, flow_pct = overlap_stats(geom, flows)
            discovery_count, discovery_pct = overlap_stats(geom, discovery)

            zone_id = f"{ztype}_{idx + 1:03d}"

            zones.append(
                {
                    "divergence_zone_id_v02": zone_id,
                    "divergence_zone_type_v02": ztype,
                    "cell_count_v02": cell_count,
                    "zone_area_km2_v02": geom.area / 1_000_000,
                    "mean_divergence_v02": mean_div,
                    "mean_divergence_norm_v02": mean_norm,
                    "abs_mean_divergence_norm_v02": abs_mean_norm,
                    "max_abs_divergence_norm_v02": cells["recognition_divergence_norm_v01"].abs().max(),
                    "core_basin_count_intersected_v02": basin_count,
                    "core_basin_overlap_pct_v02": basin_pct,
                    "flow_zone_count_intersected_v02": flow_count,
                    "flow_zone_overlap_pct_v02": flow_pct,
                    "discovery_region_count_intersected_v02": discovery_count,
                    "discovery_region_overlap_pct_v02": discovery_pct,
                    "geometry": geom,
                }
            )

    out = gpd.GeoDataFrame(zones, geometry="geometry", crs=gdf.crs)

    if out.empty:
        print(f"[{SCRIPT_NAME}] No retained divergence zones.")
        return

    out["divergence_confidence_class_v02"] = out.apply(confidence, axis=1)
    out["divergence_interpretation_v02"] = out.apply(interpretation, axis=1)

    out["divergence_evidence_score_v02"] = (
        0.35 * out["abs_mean_divergence_norm_v02"]
        + 0.25 * out["flow_zone_overlap_pct_v02"].clip(0, 1)
        + 0.20 * out["discovery_region_overlap_pct_v02"].clip(0, 1)
        + 0.10 * out["core_basin_overlap_pct_v02"].clip(0, 1)
        + 0.10 * (out["cell_count_v02"] / out["cell_count_v02"].max())
    )

    out = out.sort_values("divergence_evidence_score_v02", ascending=False).reset_index(drop=True)
    out["divergence_priority_rank_v02"] = np.arange(1, len(out) + 1)

    print()
    print(f"[{SCRIPT_NAME}] Retained zones: {len(out):,}")
    print(out["divergence_zone_type_v02"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] Confidence classes:")
    print(out["divergence_confidence_class_v02"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] Interpretations:")
    print(out["divergence_interpretation_v02"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] Top zones:")
    print(
        out[
            [
                "divergence_priority_rank_v02",
                "divergence_zone_id_v02",
                "divergence_zone_type_v02",
                "cell_count_v02",
                "mean_divergence_norm_v02",
                "flow_zone_overlap_pct_v02",
                "discovery_region_overlap_pct_v02",
                "core_basin_overlap_pct_v02",
                "divergence_confidence_class_v02",
                "divergence_interpretation_v02",
                "divergence_evidence_score_v02",
            ]
        ]
        .head(40)
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