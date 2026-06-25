#!/usr/bin/env python3

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

SCRIPT_NAME = "162_identify_oregon_recognition_transition_hotspots_v01"

VELOCITY = Path("data/processed/oregon_recognition_velocity_framework_v02.gpkg")
DISCOVERY = Path("data/processed/oregon_discovery_regions_v06.gpkg")
DIVERGENCE_ZONES = Path("data/processed/oregon_divergence_source_sink_zones_v02.gpkg")

OUTPUT_CELLS_GPKG = Path("data/processed/oregon_recognition_transition_hotspot_cells_v01.gpkg")
OUTPUT_CELLS_CSV = Path("data/processed/oregon_recognition_transition_hotspot_cells_v01.csv")
OUTPUT_HOTSPOTS_GPKG = Path("data/processed/oregon_recognition_transition_hotspots_v01.gpkg")
OUTPUT_HOTSPOTS_CSV = Path("data/processed/oregon_recognition_transition_hotspots_v01.csv")

TOP_CELL_QUANTILE = 0.98
MIN_HOTSPOT_CELLS = 3


def normalize(series):
    s = pd.Series(series).astype(float).fillna(0)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return np.zeros(len(s))
    return (s - mn) / (mx - mn)


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    vel = gpd.read_file(VELOCITY)
    disc = gpd.read_file(DISCOVERY)
    divz = gpd.read_file(DIVERGENCE_ZONES)

    if disc.crs != vel.crs:
        disc = disc.to_crs(vel.crs)

    if divz.crs != vel.crs:
        divz = divz.to_crs(vel.crs)

    print(f"[{SCRIPT_NAME}] Velocity cells: {len(vel):,}")
    print(f"[{SCRIPT_NAME}] Discovery regions: {len(disc):,}")
    print(f"[{SCRIPT_NAME}] Divergence zones: {len(divz):,}")

    required = [
        "recognition_velocity_score_v02",
        "recognition_disequilibrium_v01",
        "recognition_flow_magnitude_norm_v01",
        "recognition_divergence_norm_v01",
        "network_hub_influence_v02",
        "rde_surface_energy_v01",
    ]

    missing = [c for c in required if c not in vel.columns]
    if missing:
        raise ValueError(f"Missing velocity columns: {missing}")

    cells = vel.copy()

    cells["transition_disequilibrium_norm_v01"] = normalize(
        cells["recognition_disequilibrium_v01"]
    )

    cells["transition_flow_norm_v01"] = normalize(
        cells["recognition_flow_magnitude_norm_v01"]
    )

    cells["transition_divergence_strength_norm_v01"] = normalize(
        cells["recognition_divergence_norm_v01"].abs()
    )

    cells["transition_network_norm_v01"] = normalize(
        cells["network_hub_influence_v02"]
    )

    cells["transition_velocity_norm_v01"] = normalize(
        cells["recognition_velocity_score_v02"]
    )

    cells["transition_surface_energy_norm_v01"] = normalize(
        cells["rde_surface_energy_v01"]
    )

    # Discovery-region overlap flag
    cells["in_discovery_region_v01"] = cells.intersects(disc.unary_union)

    # Divergence-zone overlap flag
    cells["in_divergence_zone_v01"] = cells.intersects(divz.unary_union)

    cells["transition_agreement_score_v01"] = (
        0.25 * cells["transition_velocity_norm_v01"]
        + 0.20 * cells["transition_disequilibrium_norm_v01"]
        + 0.20 * cells["transition_flow_norm_v01"]
        + 0.15 * cells["transition_divergence_strength_norm_v01"]
        + 0.10 * cells["transition_network_norm_v01"]
        + 0.10 * cells["transition_surface_energy_norm_v01"]
    )

    cells["transition_agreement_score_v01"] += np.where(
        cells["in_discovery_region_v01"],
        0.05,
        0
    )

    cells["transition_agreement_score_v01"] += np.where(
        cells["in_divergence_zone_v01"],
        0.05,
        0
    )

    threshold = cells["transition_agreement_score_v01"].quantile(TOP_CELL_QUANTILE)

    hotspot_cells = cells[
        cells["transition_agreement_score_v01"] >= threshold
    ].copy()

    hotspot_cells = hotspot_cells.sort_values(
        "transition_agreement_score_v01",
        ascending=False
    ).reset_index(drop=True)

    hotspot_cells["transition_cell_rank_v01"] = np.arange(
        1,
        len(hotspot_cells) + 1
    )

    print()
    print(f"[{SCRIPT_NAME}] Transition threshold: {threshold:.4f}")
    print(f"[{SCRIPT_NAME}] Hotspot candidate cells: {len(hotspot_cells):,}")

    print()
    print(f"[{SCRIPT_NAME}] Candidate velocity classes:")
    print(hotspot_cells["recognition_velocity_class_v02"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] In discovery region:")
    print(hotspot_cells["in_discovery_region_v01"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] In divergence zone:")
    print(hotspot_cells["in_divergence_zone_v01"].value_counts())

    # Connected hotspot regions
    dissolved = hotspot_cells.dissolve()
    exploded = dissolved.explode(index_parts=False).reset_index(drop=True)

    hotspots = []

    for i, row in exploded.iterrows():
        geom = row.geometry
        members = hotspot_cells[hotspot_cells.intersects(geom)].copy()

        if len(members) < MIN_HOTSPOT_CELLS:
            continue

        hotspots.append({
            "transition_hotspot_id_v01": f"transition_hotspot_{i+1:03d}",
            "cell_count_v01": len(members),
            "hotspot_area_km2_v01": geom.area / 1_000_000,
            "mean_transition_score_v01": members["transition_agreement_score_v01"].mean(),
            "max_transition_score_v01": members["transition_agreement_score_v01"].max(),
            "mean_velocity_score_v02": members["recognition_velocity_score_v02"].mean(),
            "mean_disequilibrium_v01": members["recognition_disequilibrium_v01"].mean(),
            "mean_flow_norm_v01": members["recognition_flow_magnitude_norm_v01"].mean(),
            "mean_abs_divergence_norm_v01": members["recognition_divergence_norm_v01"].abs().mean(),
            "mean_network_influence_v02": members["network_hub_influence_v02"].mean(),
            "pct_cells_in_discovery_region_v01": members["in_discovery_region_v01"].mean(),
            "pct_cells_in_divergence_zone_v01": members["in_divergence_zone_v01"].mean(),
            "geometry": geom,
        })

    hotspots_gdf = gpd.GeoDataFrame(
        hotspots,
        geometry="geometry",
        crs=cells.crs
    )

    if hotspots_gdf.empty:
        print(f"[{SCRIPT_NAME}] No retained transition hotspots.")
    else:
        hotspots_gdf = hotspots_gdf.sort_values(
            "mean_transition_score_v01",
            ascending=False
        ).reset_index(drop=True)

        hotspots_gdf["transition_hotspot_rank_v01"] = np.arange(
            1,
            len(hotspots_gdf) + 1
        )

        print()
        print(f"[{SCRIPT_NAME}] Retained transition hotspots: {len(hotspots_gdf):,}")

        print()
        print(f"[{SCRIPT_NAME}] Top transition hotspots:")
        print(
            hotspots_gdf[
                [
                    "transition_hotspot_rank_v01",
                    "transition_hotspot_id_v01",
                    "cell_count_v01",
                    "hotspot_area_km2_v01",
                    "mean_transition_score_v01",
                    "max_transition_score_v01",
                    "mean_velocity_score_v02",
                    "pct_cells_in_discovery_region_v01",
                    "pct_cells_in_divergence_zone_v01",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    print()
    print(f"[{SCRIPT_NAME}] Writing hotspot cells: {OUTPUT_CELLS_GPKG}")
    hotspot_cells.to_file(OUTPUT_CELLS_GPKG, driver="GPKG")

    print(f"[{SCRIPT_NAME}] Writing hotspot cells CSV: {OUTPUT_CELLS_CSV}")
    hotspot_cells.drop(columns="geometry").to_csv(OUTPUT_CELLS_CSV, index=False)

    if not hotspots_gdf.empty:
        print(f"[{SCRIPT_NAME}] Writing hotspots: {OUTPUT_HOTSPOTS_GPKG}")
        hotspots_gdf.to_file(OUTPUT_HOTSPOTS_GPKG, driver="GPKG")

        print(f"[{SCRIPT_NAME}] Writing hotspots CSV: {OUTPUT_HOTSPOTS_CSV}")
        hotspots_gdf.drop(columns="geometry").to_csv(OUTPUT_HOTSPOTS_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()