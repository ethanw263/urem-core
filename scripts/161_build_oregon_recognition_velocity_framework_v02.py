#!/usr/bin/env python3

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

SCRIPT_NAME = "161_build_oregon_recognition_velocity_framework_v02"

FLOW_FIELD = Path("data/processed/oregon_recognition_flow_field_v01.gpkg")
DIVERGENCE_FIELD = Path("data/processed/oregon_recognition_divergence_field_v01.gpkg")
NETWORK_NODES = Path("data/processed/oregon_recognition_network_node_analysis_v01.gpkg")

OUTPUT_GPKG = Path("data/processed/oregon_recognition_velocity_framework_v02.gpkg")
OUTPUT_CSV = Path("data/processed/oregon_recognition_velocity_framework_v02.csv")


def normalize(series):
    s = pd.Series(series).astype(float).fillna(0)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return np.zeros(len(s))
    return (s - mn) / (mx - mn)


def classify_velocity(row):
    v = row["recognition_velocity_score_v02"]
    d = row["recognition_disequilibrium_v01"]
    flow = row["recognition_flow_magnitude_norm_v01"]
    div = row["recognition_divergence_norm_v01"]

    if d >= 0.50 and flow < 0.20:
        return "stagnant_recognition_potential"

    if v >= 0.75:
        return "rapid_recognition_change_potential"

    if v >= 0.50:
        return "moderate_recognition_change_potential"

    if div < -0.50 and flow >= 0.25:
        return "recognition_accumulation_zone"

    if v >= 0.25:
        return "slow_recognition_change_potential"

    return "low_velocity_recognition_zone"


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    flow = gpd.read_file(FLOW_FIELD).reset_index(drop=True)
    div = gpd.read_file(DIVERGENCE_FIELD).reset_index(drop=True)
    nodes = gpd.read_file(NETWORK_NODES).reset_index(drop=True)

    print(f"[{SCRIPT_NAME}] Flow cells: {len(flow):,}")
    print(f"[{SCRIPT_NAME}] Divergence cells: {len(div):,}")
    print(f"[{SCRIPT_NAME}] Network nodes: {len(nodes):,}")

    if len(flow) != len(div):
        raise ValueError("Flow and divergence fields do not have the same number of cells.")

    if div.crs != flow.crs:
        div = div.to_crs(flow.crs)

    if nodes.crs != flow.crs:
        nodes = nodes.to_crs(flow.crs)

    required_flow = [
        "recognition_disequilibrium_v01",
        "recognition_flow_magnitude_norm_v01",
        "recognition_flow_energy_v01",
        "rde_surface_energy_v01",
    ]

    required_div = [
        "recognition_divergence_norm_v01",
        "recognition_source_strength_v01",
        "recognition_sink_strength_v01",
    ]

    missing_flow = [c for c in required_flow if c not in flow.columns]
    missing_div = [c for c in required_div if c not in div.columns]

    if missing_flow:
        raise ValueError(f"Missing flow columns: {missing_flow}")

    if missing_div:
        raise ValueError(f"Missing divergence columns: {missing_div}")

    # ------------------------------------------------------------
    # Preserve one record per flow cell.
    # Since flow and divergence are both cell-level layers from the
    # same grid, copy divergence attributes by row order after verifying
    # matching length.
    # ------------------------------------------------------------

    joined = flow.copy()

    joined["recognition_divergence_norm_v01"] = div["recognition_divergence_norm_v01"].values
    joined["recognition_source_strength_v01"] = div["recognition_source_strength_v01"].values
    joined["recognition_sink_strength_v01"] = div["recognition_sink_strength_v01"].values

    # ------------------------------------------------------------
    # Nearest network node: force exactly one nearest node per cell.
    # max_distance is intentionally omitted so every cell gets a nearest
    # node, but influence decays with distance.
    # ------------------------------------------------------------

    cell_centroids = joined.copy()
    cell_centroids["geometry"] = cell_centroids.geometry.centroid

    nodes_small = nodes[
        [
            "node_id_v01",
            "node_type_v01",
            "node_subtype_v01",
            "network_hub_score_v01",
            "network_hub_class_v01",
            "geometry",
        ]
    ].copy()

    nearest = gpd.sjoin_nearest(
        cell_centroids[["geometry"]],
        nodes_small,
        how="left",
        distance_col="nearest_network_node_distance_m_v02",
        exclusive=False,
    )

    nearest = nearest[~nearest.index.duplicated(keep="first")]
    nearest = nearest.sort_index()

    if len(nearest) != len(joined):
        raise ValueError(
            f"Nearest-node join changed row count: {len(nearest):,} vs {len(joined):,}"
        )

    nearest_attrs = nearest[
        [
            "node_id_v01",
            "node_type_v01",
            "node_subtype_v01",
            "network_hub_score_v01",
            "network_hub_class_v01",
            "nearest_network_node_distance_m_v02",
        ]
    ].reset_index(drop=True)

    joined = joined.reset_index(drop=True).join(nearest_attrs)

    joined["network_hub_influence_v02"] = (
        joined["network_hub_score_v01"].fillna(0)
        * np.exp(
            -joined["nearest_network_node_distance_m_v02"].fillna(999999)
            / 15000
        )
    )

    joined["disequilibrium_pressure_norm_v02"] = normalize(
        joined["recognition_disequilibrium_v01"]
    )

    joined["flow_velocity_norm_v02"] = normalize(
        joined["recognition_flow_magnitude_norm_v01"]
    )

    joined["surface_energy_norm_v02"] = normalize(
        joined["rde_surface_energy_v01"]
    )

    joined["divergence_push_pull_norm_v02"] = normalize(
        joined["recognition_source_strength_v01"].fillna(0)
        + joined["recognition_sink_strength_v01"].fillna(0)
    )

    joined["recognition_velocity_score_v02"] = (
        0.30 * joined["disequilibrium_pressure_norm_v02"]
        + 0.25 * joined["flow_velocity_norm_v02"]
        + 0.20 * joined["divergence_push_pull_norm_v02"]
        + 0.15 * joined["network_hub_influence_v02"]
        + 0.10 * joined["surface_energy_norm_v02"]
    )

    joined["recognition_velocity_class_v02"] = joined.apply(
        classify_velocity,
        axis=1
    )

    joined = joined.sort_values(
        "recognition_velocity_score_v02",
        ascending=False
    ).reset_index(drop=True)

    joined["recognition_velocity_rank_v02"] = np.arange(1, len(joined) + 1)

    print()
    print(f"[{SCRIPT_NAME}] Output rows: {len(joined):,}")

    if len(joined) != len(flow):
        raise ValueError("Output row count does not match input flow cell count.")

    print()
    print(f"[{SCRIPT_NAME}] Velocity classes:")
    print(joined["recognition_velocity_class_v02"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] Top velocity cells:")
    print(
        joined[
            [
                "recognition_velocity_rank_v02",
                "recognition_velocity_score_v02",
                "recognition_velocity_class_v02",
                "recognition_disequilibrium_v01",
                "recognition_flow_magnitude_norm_v01",
                "recognition_divergence_norm_v01",
                "network_hub_score_v01",
                "nearest_network_node_distance_m_v02",
                "node_id_v01",
                "node_type_v01",
            ]
        ]
        .head(40)
        .to_string(index=False)
    )

    print()
    print(f"[{SCRIPT_NAME}] Writing GPKG: {OUTPUT_GPKG}")
    joined.to_file(OUTPUT_GPKG, driver="GPKG")

    print(f"[{SCRIPT_NAME}] Writing CSV: {OUTPUT_CSV}")
    joined.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()