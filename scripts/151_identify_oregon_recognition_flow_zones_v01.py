#!/usr/bin/env python3
"""
151_identify_oregon_recognition_flow_zones_v01.py

Identify Oregon Recognition Flow Zones.

Purpose:
Cluster cells with meaningful recognition-flow behavior into interpretable
RDE flow zones.

Inputs:
- data/processed/oregon_recognition_flow_field_v01.gpkg

Outputs:
- data/processed/oregon_recognition_flow_zones_v01.gpkg
- data/processed/oregon_recognition_flow_zones_v01.csv
- data/processed/oregon_recognition_flow_zone_centroids_v01.gpkg
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import networkx as nx


SCRIPT_NAME = "151_identify_oregon_recognition_flow_zones_v01"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = PROCESSED_DIR / "oregon_recognition_flow_field_v01.gpkg"

OUT_GPKG = PROCESSED_DIR / "oregon_recognition_flow_zones_v01.gpkg"
OUT_CSV = PROCESSED_DIR / "oregon_recognition_flow_zones_v01.csv"
OUT_CENTROIDS_GPKG = PROCESSED_DIR / "oregon_recognition_flow_zone_centroids_v01.gpkg"

FLOW_CLASSES = {
    "high_disequilibrium_low_flow",
    "strong_flow_from_high_disequilibrium",
    "moderate_disequilibrium_strong_flow",
}

MIN_ZONE_CELLS = 3


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def zone_type(row):
    high_low = row["high_disequilibrium_low_flow_cells"]
    strong = row["strong_flow_from_high_disequilibrium_cells"]
    moderate_strong = row["moderate_disequilibrium_strong_flow_cells"]

    if strong > 0:
        return "active_high_disequilibrium_flow_zone"

    if moderate_strong > 0:
        return "moderate_active_flow_zone"

    if high_low > 0:
        return "stored_recognition_potential_zone"

    return "mixed_flow_zone"


def main():
    log("Starting Oregon recognition flow zone identification")

    if not INPUT_GPKG.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_GPKG}")

    gdf = gpd.read_file(INPUT_GPKG)

    log(f"Cells loaded: {len(gdf):,}")
    log(f"CRS: {gdf.crs}")

    flow_cells = gdf[
        gdf["recognition_flow_class_v01"].isin(FLOW_CLASSES)
    ].copy()

    log(f"Selected flow cells: {len(flow_cells):,}")

    if flow_cells.empty:
        raise ValueError("No flow cells selected.")

    flow_cells = flow_cells.reset_index(drop=True)
    flow_cells["node_id"] = flow_cells.index

    graph = nx.Graph()
    graph.add_nodes_from(flow_cells["node_id"])

    sindex = flow_cells.sindex

    log("Building flow-zone adjacency graph")

    for i, geom in enumerate(flow_cells.geometry):
        if i % 500 == 0:
            log(f"Adjacency scan {i:,}/{len(flow_cells):,}")

        candidates = list(sindex.intersection(geom.bounds))

        for j in candidates:
            if j <= i:
                continue

            other = flow_cells.geometry.iloc[j]

            if geom.touches(other) or geom.intersects(other):
                graph.add_edge(i, j)

    components = list(nx.connected_components(graph))
    retained = [c for c in components if len(c) >= MIN_ZONE_CELLS]

    log(f"Initial flow components: {len(components):,}")
    log(f"Retained flow zones: {len(retained):,}")

    records = []

    for zone_id, comp in enumerate(retained, start=1):
        sub = flow_cells.iloc[list(comp)].copy()
        geom = sub.geometry.union_all()

        counts = sub["recognition_flow_class_v01"].value_counts()

        records.append(
            {
                "recognition_flow_zone_id_v01": zone_id,
                "flow_zone_cell_count": len(sub),
                "flow_zone_area_km2": sub.geometry.area.sum() / 1_000_000,
                "mean_recognition_flow_energy_v01": sub[
                    "recognition_flow_energy_v01"
                ].mean(),
                "max_recognition_flow_energy_v01": sub[
                    "recognition_flow_energy_v01"
                ].max(),
                "mean_flow_magnitude_norm_v01": sub[
                    "recognition_flow_magnitude_norm_v01"
                ].mean(),
                "max_flow_magnitude_norm_v01": sub[
                    "recognition_flow_magnitude_norm_v01"
                ].max(),
                "mean_disequilibrium_v01": sub[
                    "recognition_disequilibrium_v01"
                ].mean(),
                "max_disequilibrium_v01": sub[
                    "recognition_disequilibrium_v01"
                ].max(),
                "mean_surface_energy_v01": sub[
                    "rde_surface_energy_v01"
                ].mean(),
                "high_disequilibrium_low_flow_cells": int(
                    counts.get("high_disequilibrium_low_flow", 0)
                ),
                "strong_flow_from_high_disequilibrium_cells": int(
                    counts.get("strong_flow_from_high_disequilibrium", 0)
                ),
                "moderate_disequilibrium_strong_flow_cells": int(
                    counts.get("moderate_disequilibrium_strong_flow", 0)
                ),
                "dominant_flow_direction_label_v01": sub[
                    "recognition_flow_direction_label_v01"
                ].value_counts().idxmax(),
                "geometry": geom,
            }
        )

    zones = gpd.GeoDataFrame(records, geometry="geometry", crs=gdf.crs)

    zones["recognition_flow_zone_type_v01"] = zones.apply(zone_type, axis=1)

    zones = zones.sort_values(
        [
            "max_recognition_flow_energy_v01",
            "mean_recognition_flow_energy_v01",
            "flow_zone_cell_count",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    zones["recognition_flow_zone_rank_v01"] = zones.index + 1

    centroids = zones.copy()
    centroids["geometry"] = centroids.geometry.representative_point()

    wgs = centroids.to_crs(4326)
    zones["longitude"] = wgs.geometry.x
    zones["latitude"] = wgs.geometry.y

    log("Flow zone type counts:")
    print(zones["recognition_flow_zone_type_v01"].value_counts())

    log("Top flow zones:")
    print(
        zones[
            [
                "recognition_flow_zone_rank_v01",
                "recognition_flow_zone_type_v01",
                "flow_zone_cell_count",
                "flow_zone_area_km2",
                "mean_recognition_flow_energy_v01",
                "max_recognition_flow_energy_v01",
                "mean_disequilibrium_v01",
                "dominant_flow_direction_label_v01",
                "longitude",
                "latitude",
            ]
        ].head(20)
    )

    for path in [OUT_GPKG, OUT_CENTROIDS_GPKG]:
        if path.exists():
            path.unlink()

    log(f"Writing GPKG: {OUT_GPKG}")
    zones.to_file(
        OUT_GPKG,
        layer="oregon_recognition_flow_zones_v01",
        driver="GPKG",
    )

    log(f"Writing CSV: {OUT_CSV}")
    zones.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log(f"Writing centroids GPKG: {OUT_CENTROIDS_GPKG}")
    centroids.to_file(
        OUT_CENTROIDS_GPKG,
        layer="oregon_recognition_flow_zone_centroids_v01",
        driver="GPKG",
    )

    log("Done")


if __name__ == "__main__":
    main()