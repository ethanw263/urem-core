#!/usr/bin/env python3

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

SCRIPT_NAME = "160_analyze_oregon_recognition_network_graph_v01"

NODES = Path("data/processed/oregon_recognition_network_nodes_v01.gpkg")
EDGES = Path("data/processed/oregon_recognition_network_edges_v01.gpkg")

OUTPUT_NODES_GPKG = Path("data/processed/oregon_recognition_network_node_analysis_v01.gpkg")
OUTPUT_NODES_CSV = Path("data/processed/oregon_recognition_network_node_analysis_v01.csv")
OUTPUT_SUMMARY_CSV = Path("data/processed/oregon_recognition_network_summary_v01.csv")


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    nodes = gpd.read_file(NODES)
    edges = gpd.read_file(EDGES)

    print(f"[{SCRIPT_NAME}] Nodes: {len(nodes):,}")
    print(f"[{SCRIPT_NAME}] Edges: {len(edges):,}")

    node_ids = nodes["node_id_v01"].tolist()

    stats = {
        node_id: {
            "degree_v01": 0,
            "weighted_degree_v01": 0.0,
            "source_sink_edges_v01": 0,
            "discovery_edges_v01": 0,
            "flow_edges_v01": 0,
            "proximity_edges_v01": 0,
        }
        for node_id in node_ids
    }

    for _, edge in edges.iterrows():
        a = edge["from_node_v01"]
        b = edge["to_node_v01"]
        score = edge["edge_score_v01"]
        etype = edge["edge_type_v01"]

        for node in [a, b]:
            stats[node]["degree_v01"] += 1
            stats[node]["weighted_degree_v01"] += score

            if etype == "source_sink_edge":
                stats[node]["source_sink_edges_v01"] += 1
            elif etype == "discovery_link_edge":
                stats[node]["discovery_edges_v01"] += 1
            elif etype == "flow_link_edge":
                stats[node]["flow_edges_v01"] += 1
            elif etype == "proximity_edge":
                stats[node]["proximity_edges_v01"] += 1

    stats_df = pd.DataFrame.from_dict(stats, orient="index").reset_index()
    stats_df = stats_df.rename(columns={"index": "node_id_v01"})

    out = nodes.merge(stats_df, on="node_id_v01", how="left")

    out["network_hub_score_v01"] = (
        0.50 * (
            out["weighted_degree_v01"] / out["weighted_degree_v01"].max()
            if out["weighted_degree_v01"].max() > 0 else 0
        )
        + 0.30 * (
            out["degree_v01"] / out["degree_v01"].max()
            if out["degree_v01"].max() > 0 else 0
        )
        + 0.20 * out["node_strength_norm_v01"]
    )

    def hub_class(row):
        if row["network_hub_score_v01"] >= 0.75:
            return "major_rde_network_hub"
        if row["network_hub_score_v01"] >= 0.50:
            return "moderate_rde_network_hub"
        if row["degree_v01"] > 0:
            return "minor_rde_network_node"
        return "isolated_rde_node"

    out["network_hub_class_v01"] = out.apply(hub_class, axis=1)

    out = out.sort_values("network_hub_score_v01", ascending=False).reset_index(drop=True)
    out["network_rank_v01"] = np.arange(1, len(out) + 1)

    print()
    print(f"[{SCRIPT_NAME}] Hub classes:")
    print(out["network_hub_class_v01"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] Mean network stats by node type:")
    print(
        out.groupby("node_type_v01")[
            [
                "degree_v01",
                "weighted_degree_v01",
                "source_sink_edges_v01",
                "discovery_edges_v01",
                "flow_edges_v01",
                "proximity_edges_v01",
                "network_hub_score_v01",
            ]
        ].mean()
    )

    print()
    print(f"[{SCRIPT_NAME}] Top network nodes:")
    print(
        out[
            [
                "network_rank_v01",
                "node_id_v01",
                "node_type_v01",
                "node_subtype_v01",
                "degree_v01",
                "weighted_degree_v01",
                "source_sink_edges_v01",
                "discovery_edges_v01",
                "flow_edges_v01",
                "network_hub_class_v01",
                "network_hub_score_v01",
            ]
        ]
        .head(40)
        .to_string(index=False)
    )

    summary = []

    for node_type, sub in out.groupby("node_type_v01"):
        summary.append({
            "node_type_v01": node_type,
            "node_count_v01": len(sub),
            "mean_degree_v01": sub["degree_v01"].mean(),
            "mean_weighted_degree_v01": sub["weighted_degree_v01"].mean(),
            "mean_hub_score_v01": sub["network_hub_score_v01"].mean(),
            "major_hub_count_v01": (sub["network_hub_class_v01"] == "major_rde_network_hub").sum(),
            "moderate_hub_count_v01": (sub["network_hub_class_v01"] == "moderate_rde_network_hub").sum(),
        })

    summary_df = pd.DataFrame(summary)

    print()
    print(f"[{SCRIPT_NAME}] Writing node analysis: {OUTPUT_NODES_GPKG}")
    out.to_file(OUTPUT_NODES_GPKG, driver="GPKG")

    print(f"[{SCRIPT_NAME}] Writing CSVs")
    out.drop(columns="geometry").to_csv(OUTPUT_NODES_CSV, index=False)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()