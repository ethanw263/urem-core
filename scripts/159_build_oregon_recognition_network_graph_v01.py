#!/usr/bin/env python3

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

SCRIPT_NAME = "159_build_oregon_recognition_network_graph_v01"

DIVERGENCE = Path("data/processed/oregon_divergence_source_sink_zones_v02.gpkg")
DISCOVERY = Path("data/processed/oregon_discovery_regions_v06.gpkg")
FLOW_ZONES = Path("data/processed/oregon_recognition_flow_zones_v01.gpkg")

OUTPUT_NODES_GPKG = Path("data/processed/oregon_recognition_network_nodes_v01.gpkg")
OUTPUT_EDGES_GPKG = Path("data/processed/oregon_recognition_network_edges_v01.gpkg")
OUTPUT_NODES_CSV = Path("data/processed/oregon_recognition_network_nodes_v01.csv")
OUTPUT_EDGES_CSV = Path("data/processed/oregon_recognition_network_edges_v01.csv")

MAX_EDGE_DISTANCE_M = 25000
MIN_EDGE_SCORE = 0.10


def normalize(s):
    s = pd.Series(s).astype(float)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return np.ones(len(s))
    return (s - mn) / (mx - mn)


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    div = gpd.read_file(DIVERGENCE)
    disc = gpd.read_file(DISCOVERY)
    flow = gpd.read_file(FLOW_ZONES)

    if disc.crs != div.crs:
        disc = disc.to_crs(div.crs)
    if flow.crs != div.crs:
        flow = flow.to_crs(div.crs)

    nodes = []

    for _, row in div.iterrows():
        geom = row.geometry.centroid
        nodes.append({
            "node_id_v01": f"DIV_{row['divergence_zone_id_v02']}",
            "node_type_v01": row["divergence_zone_type_v02"],
            "node_subtype_v01": row["divergence_interpretation_v02"],
            "node_confidence_v01": row["divergence_confidence_class_v02"],
            "node_strength_v01": row["divergence_evidence_score_v02"],
            "source_strength_v01": max(row["mean_divergence_norm_v02"], 0),
            "sink_strength_v01": max(-row["mean_divergence_norm_v02"], 0),
            "geometry": geom,
        })

    for i, row in disc.iterrows():
        geom = row.geometry.centroid
        strength = row.get("region_score_v06", np.nan)
        if pd.isna(strength):
            strength = 0.5
        nodes.append({
            "node_id_v01": f"DISC_{i+1:03d}",
            "node_type_v01": "discovery_region",
            "node_subtype_v01": "urem_discovery_region",
            "node_confidence_v01": "discovery_region",
            "node_strength_v01": strength,
            "source_strength_v01": 0,
            "sink_strength_v01": 0,
            "geometry": geom,
        })

    for i, row in flow.iterrows():
        geom = row.geometry.centroid
        strength = row.get("mean_recognition_flow_energy_v01", np.nan)
        if pd.isna(strength):
            strength = 0.5
        nodes.append({
            "node_id_v01": f"FLOW_{i+1:03d}",
            "node_type_v01": "flow_zone",
            "node_subtype_v01": row.get("flow_zone_type_v01", "recognition_flow_zone"),
            "node_confidence_v01": "flow_zone",
            "node_strength_v01": strength,
            "source_strength_v01": 0,
            "sink_strength_v01": 0,
            "geometry": geom,
        })

    nodes_gdf = gpd.GeoDataFrame(nodes, geometry="geometry", crs=div.crs)
    nodes_gdf["node_strength_norm_v01"] = normalize(nodes_gdf["node_strength_v01"])

    print(f"[{SCRIPT_NAME}] Nodes: {len(nodes_gdf):,}")
    print(nodes_gdf["node_type_v01"].value_counts())

    edges = []

    for i, a in nodes_gdf.iterrows():
        for j, b in nodes_gdf.iterrows():
            if i >= j:
                continue

            dist = a.geometry.distance(b.geometry)

            if dist > MAX_EDGE_DISTANCE_M:
                continue

            distance_score = 1 - (dist / MAX_EDGE_DISTANCE_M)

            source_sink_score = max(
                a["source_strength_v01"] * b["sink_strength_v01"],
                b["source_strength_v01"] * a["sink_strength_v01"],
            )

            discovery_link_score = 0
            if "discovery_region" in {a["node_type_v01"], b["node_type_v01"]}:
                discovery_link_score = 0.5

            flow_link_score = 0
            if "flow_zone" in {a["node_type_v01"], b["node_type_v01"]}:
                flow_link_score = 0.5

            edge_score = (
                0.40 * distance_score
                + 0.30 * source_sink_score
                + 0.15 * discovery_link_score
                + 0.15 * flow_link_score
            )

            if edge_score < MIN_EDGE_SCORE:
                continue

            line = gpd.GeoSeries(
                [a.geometry, b.geometry],
                crs=div.crs
            ).union_all().convex_hull

            edge_type = "proximity_edge"

            if source_sink_score > 0:
                edge_type = "source_sink_edge"
            if discovery_link_score > 0:
                edge_type = "discovery_link_edge"
            if flow_link_score > 0:
                edge_type = "flow_link_edge"

            edges.append({
                "edge_id_v01": f"EDGE_{len(edges)+1:05d}",
                "from_node_v01": a["node_id_v01"],
                "to_node_v01": b["node_id_v01"],
                "from_type_v01": a["node_type_v01"],
                "to_type_v01": b["node_type_v01"],
                "edge_distance_m_v01": dist,
                "distance_score_v01": distance_score,
                "source_sink_score_v01": source_sink_score,
                "discovery_link_score_v01": discovery_link_score,
                "flow_link_score_v01": flow_link_score,
                "edge_score_v01": edge_score,
                "edge_type_v01": edge_type,
                "geometry": line,
            })

    edges_gdf = gpd.GeoDataFrame(edges, geometry="geometry", crs=div.crs)

    print(f"[{SCRIPT_NAME}] Edges: {len(edges_gdf):,}")
    if len(edges_gdf):
        print(edges_gdf["edge_type_v01"].value_counts())

        print()
        print(f"[{SCRIPT_NAME}] Top edges:")
        print(
            edges_gdf.sort_values("edge_score_v01", ascending=False)[
                [
                    "edge_id_v01",
                    "from_node_v01",
                    "to_node_v01",
                    "edge_type_v01",
                    "edge_distance_m_v01",
                    "edge_score_v01",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    print()
    print(f"[{SCRIPT_NAME}] Writing nodes: {OUTPUT_NODES_GPKG}")
    nodes_gdf.to_file(OUTPUT_NODES_GPKG, driver="GPKG")

    print(f"[{SCRIPT_NAME}] Writing edges: {OUTPUT_EDGES_GPKG}")
    edges_gdf.to_file(OUTPUT_EDGES_GPKG, driver="GPKG")

    nodes_gdf.drop(columns="geometry").to_csv(OUTPUT_NODES_CSV, index=False)
    edges_gdf.drop(columns="geometry").to_csv(OUTPUT_EDGES_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()