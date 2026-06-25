#!/usr/bin/env python3

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

SCRIPT_NAME = "165_oregon_geographic_context_validation_v01"

HOTSPOTS = Path("data/processed/oregon_transition_hotspot_validation_summary_v01.gpkg")
DIVERGENCE_ZONES = Path("data/processed/oregon_divergence_source_sink_zones_v02.gpkg")
DISCOVERY_REGIONS = Path("data/processed/oregon_discovery_regions_v06.gpkg")
FLOW_ZONES = Path("data/processed/oregon_recognition_flow_zones_v01.gpkg")
NETWORK_NODES = Path("data/processed/oregon_recognition_network_node_analysis_v01.gpkg")
NETWORK_EDGES = Path("data/processed/oregon_recognition_network_edges_v01.gpkg")

OUTPUT_GPKG = Path("data/processed/oregon_geographic_context_validation_v01.gpkg")
OUTPUT_CSV = Path("data/processed/oregon_geographic_context_validation_v01.csv")
OUTPUT_SUMMARY_CSV = Path("data/processed/oregon_geographic_context_validation_summary_v01.csv")


def nearest_distance_and_id(geom, layer, id_col):
    if layer.empty:
        return np.nan, ""

    distances = layer.geometry.distance(geom)
    idx = distances.idxmin()

    return float(distances.loc[idx]), str(layer.loc[idx, id_col])


def count_within_distance(geom, layer, distance_m):
    if layer.empty:
        return 0

    return int(layer.geometry.distance(geom).le(distance_m).sum())


def context_class(row):
    src = row["intersects_source_zone_v01"]
    sink = row["intersects_sink_zone_v01"]
    flow = row["intersects_flow_zone_v01"]
    disc = row["intersects_discovery_region_v01"]

    if src and sink and flow and disc:
        return "full_transition_convergence_context"

    if src and sink and flow:
        return "source_sink_flow_transition_context"

    if src and flow and disc:
        return "source_discovery_flow_context"

    if sink and flow and disc:
        return "sink_discovery_flow_context"

    if src and sink:
        return "source_sink_boundary_context"

    if disc and flow:
        return "discovery_flow_context"

    if src and flow:
        return "source_flow_context"

    if sink and flow:
        return "sink_flow_context"

    if disc:
        return "discovery_context"

    if flow:
        return "flow_context"

    return "weak_internal_context"


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    hotspots = gpd.read_file(HOTSPOTS)
    divz = gpd.read_file(DIVERGENCE_ZONES)
    discovery = gpd.read_file(DISCOVERY_REGIONS)
    flow = gpd.read_file(FLOW_ZONES)
    nodes = gpd.read_file(NETWORK_NODES)
    edges = gpd.read_file(NETWORK_EDGES)

    layers = {
        "divergence zones": divz,
        "discovery regions": discovery,
        "flow zones": flow,
        "network nodes": nodes,
        "network edges": edges,
    }

    for name, layer in layers.items():
        if layer.crs != hotspots.crs:
            print(f"[{SCRIPT_NAME}] Reprojecting {name}")
            layers[name] = layer.to_crs(hotspots.crs)

    divz = layers["divergence zones"]
    discovery = layers["discovery regions"]
    flow = layers["flow zones"]
    nodes = layers["network nodes"]
    edges = layers["network edges"]

    print(f"[{SCRIPT_NAME}] Hotspots: {len(hotspots):,}")
    print(f"[{SCRIPT_NAME}] Divergence zones: {len(divz):,}")
    print(f"[{SCRIPT_NAME}] Discovery regions: {len(discovery):,}")
    print(f"[{SCRIPT_NAME}] Flow zones: {len(flow):,}")
    print(f"[{SCRIPT_NAME}] Network nodes: {len(nodes):,}")
    print(f"[{SCRIPT_NAME}] Network edges: {len(edges):,}")

    source_zones = divz[divz["divergence_zone_type_v02"] == "source_zone"].copy()
    sink_zones = divz[divz["divergence_zone_type_v02"] == "sink_zone"].copy()

    records = []

    for _, row in hotspots.iterrows():
        geom = row.geometry
        centroid = geom.centroid

        source_hits = source_zones[source_zones.intersects(geom)]
        sink_hits = sink_zones[sink_zones.intersects(geom)]
        discovery_hits = discovery[discovery.intersects(geom)]
        flow_hits = flow[flow.intersects(geom)]

        nearest_source_dist, nearest_source_id = nearest_distance_and_id(
            centroid,
            source_zones,
            "divergence_zone_id_v02",
        )

        nearest_sink_dist, nearest_sink_id = nearest_distance_and_id(
            centroid,
            sink_zones,
            "divergence_zone_id_v02",
        )

        nearest_discovery_dist, nearest_discovery_id = nearest_distance_and_id(
            centroid,
            discovery,
            discovery.columns[0],
        )

        nearest_flow_dist, nearest_flow_id = nearest_distance_and_id(
            centroid,
            flow,
            flow.columns[0],
        )

        nearest_node_dist, nearest_node_id = nearest_distance_and_id(
            centroid,
            nodes,
            "node_id_v01",
        )

        nearest_edge_dist, nearest_edge_id = nearest_distance_and_id(
            centroid,
            edges,
            "edge_id_v01",
        )

        local_node_count_5km = count_within_distance(
            centroid,
            nodes,
            5000,
        )

        local_node_count_10km = count_within_distance(
            centroid,
            nodes,
            10000,
        )

        local_edge_count_5km = count_within_distance(
            centroid,
            edges,
            5000,
        )

        local_edge_count_10km = count_within_distance(
            centroid,
            edges,
            10000,
        )

        rec = row.drop(labels="geometry").to_dict()

        rec.update(
            {
                "intersects_source_zone_v01": len(source_hits) > 0,
                "intersects_sink_zone_v01": len(sink_hits) > 0,
                "intersects_discovery_region_v01": len(discovery_hits) > 0,
                "intersects_flow_zone_v01": len(flow_hits) > 0,

                "source_zone_count_intersected_v01": len(source_hits),
                "sink_zone_count_intersected_v01": len(sink_hits),
                "discovery_region_count_intersected_v01": len(discovery_hits),
                "flow_zone_count_intersected_v01": len(flow_hits),

                "nearest_source_distance_m_v01": nearest_source_dist,
                "nearest_source_id_v01": nearest_source_id,
                "nearest_sink_distance_m_v01": nearest_sink_dist,
                "nearest_sink_id_v01": nearest_sink_id,
                "nearest_discovery_distance_m_v01": nearest_discovery_dist,
                "nearest_discovery_id_v01": nearest_discovery_id,
                "nearest_flow_distance_m_v01": nearest_flow_dist,
                "nearest_flow_id_v01": nearest_flow_id,
                "nearest_network_node_distance_m_v01": nearest_node_dist,
                "nearest_network_node_id_v01": nearest_node_id,
                "nearest_network_edge_distance_m_v01": nearest_edge_dist,
                "nearest_network_edge_id_v01": nearest_edge_id,

                "local_network_node_count_5km_v01": local_node_count_5km,
                "local_network_node_count_10km_v01": local_node_count_10km,
                "local_network_edge_count_5km_v01": local_edge_count_5km,
                "local_network_edge_count_10km_v01": local_edge_count_10km,

                "geometry": geom,
            }
        )

        records.append(rec)

    out = gpd.GeoDataFrame(records, geometry="geometry", crs=hotspots.crs)

    out["internal_geographic_context_class_v01"] = out.apply(
        context_class,
        axis=1,
    )

    out["source_sink_transition_flag_v01"] = (
        out["nearest_source_distance_m_v01"].le(5000)
        & out["nearest_sink_distance_m_v01"].le(5000)
    )

    out["network_dense_context_flag_v01"] = (
        out["local_network_node_count_5km_v01"] >= 3
    )

    out["internal_context_score_v01"] = (
        0.20 * out["intersects_source_zone_v01"].astype(int)
        + 0.20 * out["intersects_sink_zone_v01"].astype(int)
        + 0.20 * out["intersects_flow_zone_v01"].astype(int)
        + 0.15 * out["intersects_discovery_region_v01"].astype(int)
        + 0.15 * out["source_sink_transition_flag_v01"].astype(int)
        + 0.10 * out["network_dense_context_flag_v01"].astype(int)
    )

    out = out.sort_values(
        [
            "internal_context_score_v01",
            "hotspot_validation_score_v01",
            "mean_transition_score_v01",
        ],
        ascending=False,
    ).reset_index(drop=True)

    out["internal_context_rank_v01"] = np.arange(1, len(out) + 1)

    print()
    print(f"[{SCRIPT_NAME}] Internal context classes:")
    print(out["internal_geographic_context_class_v01"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] Source/sink transition flags:")
    print(out["source_sink_transition_flag_v01"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] Network dense context flags:")
    print(out["network_dense_context_flag_v01"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] Top internal geographic contexts:")
    print(
        out[
            [
                "internal_context_rank_v01",
                "transition_hotspot_rank_v01",
                "transition_hotspot_id_v01",
                "internal_geographic_context_class_v01",
                "internal_context_score_v01",
                "intersects_source_zone_v01",
                "intersects_sink_zone_v01",
                "intersects_flow_zone_v01",
                "intersects_discovery_region_v01",
                "nearest_source_distance_m_v01",
                "nearest_sink_distance_m_v01",
                "local_network_node_count_5km_v01",
                "local_network_edge_count_5km_v01",
            ]
        ]
        .head(34)
        .to_string(index=False)
    )

    summary = []

    summary.append({
        "metric": "total_hotspots",
        "value": len(out),
    })

    summary.append({
        "metric": "mean_internal_context_score",
        "value": out["internal_context_score_v01"].mean(),
    })

    summary.append({
        "metric": "pct_intersects_source_zone",
        "value": out["intersects_source_zone_v01"].mean(),
    })

    summary.append({
        "metric": "pct_intersects_sink_zone",
        "value": out["intersects_sink_zone_v01"].mean(),
    })

    summary.append({
        "metric": "pct_intersects_flow_zone",
        "value": out["intersects_flow_zone_v01"].mean(),
    })

    summary.append({
        "metric": "pct_intersects_discovery_region",
        "value": out["intersects_discovery_region_v01"].mean(),
    })

    summary.append({
        "metric": "pct_source_sink_transition_flag",
        "value": out["source_sink_transition_flag_v01"].mean(),
    })

    summary.append({
        "metric": "pct_network_dense_context_flag",
        "value": out["network_dense_context_flag_v01"].mean(),
    })

    for cls, count in out["internal_geographic_context_class_v01"].value_counts().items():
        summary.append({
            "metric": f"context_class_count__{cls}",
            "value": count,
        })

    summary_df = pd.DataFrame(summary)

    print()
    print(f"[{SCRIPT_NAME}] Writing GPKG: {OUTPUT_GPKG}")
    out.to_file(OUTPUT_GPKG, driver="GPKG")

    print(f"[{SCRIPT_NAME}] Writing CSV: {OUTPUT_CSV}")
    out.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Writing summary CSV: {OUTPUT_SUMMARY_CSV}")
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()