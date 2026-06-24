#!/usr/bin/env python3
"""
146_identify_oregon_recognition_core_basins_v01.py

Purpose:
- Replace threshold-only recognition basins.
- Identify high-energy Recognition Disequilibrium cores.
- Produce compact basin nuclei suitable for interpretation.

Inputs:
- oregon_recognition_disequilibrium_surface_v01.gpkg

Outputs:
- oregon_recognition_core_basins_v01.gpkg
- oregon_recognition_core_basins_v01.csv
- oregon_recognition_core_basin_centroids_v01.gpkg
- oregon_recognition_core_basin_summary_v01.csv
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx

SCRIPT_NAME = "146_identify_oregon_recognition_core_basins_v01"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = (
    PROCESSED_DIR
    / "oregon_recognition_disequilibrium_surface_v01.gpkg"
)

OUT_BASINS_GPKG = (
    PROCESSED_DIR
    / "oregon_recognition_core_basins_v01.gpkg"
)

OUT_BASINS_CSV = (
    PROCESSED_DIR
    / "oregon_recognition_core_basins_v01.csv"
)

OUT_CENTROIDS_GPKG = (
    PROCESSED_DIR
    / "oregon_recognition_core_basin_centroids_v01.gpkg"
)

OUT_SUMMARY_CSV = (
    PROCESSED_DIR
    / "oregon_recognition_core_basin_summary_v01.csv"
)

MIN_CORE_CELLS = 3


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def classify_core(row):
    area = row["core_area_km2"]
    energy = row["mean_rde_surface_energy_v01"]

    if area >= 50:
        return "major_core_basin"

    if energy >= 0.40:
        return "high_energy_core"

    return "localized_core"


def main():

    log("Starting Oregon recognition core basins")

    gdf = gpd.read_file(INPUT_GPKG)

    log(f"Cells loaded: {len(gdf):,}")
    log(f"CRS: {gdf.crs}")

    threshold = (
        gdf["rde_surface_energy_v01"]
        .quantile(0.90)
    )

    log(
        f"Energy threshold (90th percentile): "
        f"{threshold:.6f}"
    )

    core_cells = gdf[
        gdf["rde_surface_energy_v01"] >= threshold
    ].copy()

    log(
        f"Core cells selected: "
        f"{len(core_cells):,}"
    )

    core_cells = core_cells.reset_index(drop=True)

    core_cells["node_id"] = core_cells.index

    graph = nx.Graph()

    graph.add_nodes_from(core_cells["node_id"])

    sindex = core_cells.sindex

    log("Building adjacency graph")

    for i, geom in enumerate(core_cells.geometry):

        if i % 500 == 0:
            log(f"Adjacency scan {i:,}/{len(core_cells):,}")

        candidates = list(
            sindex.intersection(geom.bounds)
        )

        for j in candidates:

            if j <= i:
                continue

            other = core_cells.geometry.iloc[j]

            if geom.touches(other) or geom.intersects(other):
                graph.add_edge(i, j)

    components = list(nx.connected_components(graph))

    retained = [
        comp
        for comp in components
        if len(comp) >= MIN_CORE_CELLS
    ]

    log(
        f"Initial components: {len(components):,}"
    )

    log(
        f"Retained core basins: {len(retained):,}"
    )

    basin_frames = []
    summary_rows = []

    for basin_id, comp in enumerate(retained, start=1):

        basin = core_cells.iloc[list(comp)].copy()

        basin["recognition_core_basin_id_v01"] = basin_id

        basin_area = len(basin)

        mean_energy = (
            basin["rde_surface_energy_v01"]
            .mean()
        )

        max_energy = (
            basin["rde_surface_energy_v01"]
            .max()
        )

        mean_diseq = (
            basin["recognition_disequilibrium_v01"]
            .mean()
        )

        max_score = (
            basin["urem_score_v06_raw"]
            .max()
        )

        best_rank = (
            basin["candidate_rank_v06"]
            .min()
            if "candidate_rank_v06" in basin.columns
            else np.nan
        )

        row = {
            "recognition_core_basin_id_v01": basin_id,
            "core_cell_count": len(basin),
            "core_area_km2": basin_area,
            "mean_rde_surface_energy_v01": mean_energy,
            "max_rde_surface_energy_v01": max_energy,
            "mean_recognition_disequilibrium_v01": mean_diseq,
            "max_urem_score_v06_raw": max_score,
            "best_candidate_rank_v06": best_rank,
        }

        summary_rows.append(row)

        basin_frames.append(basin)

    if not basin_frames:
        raise ValueError(
            "No retained core basins found."
        )

    basins = pd.concat(
        basin_frames,
        ignore_index=True
    )

    summary = pd.DataFrame(summary_rows)

    summary = summary.sort_values(
        "max_urem_score_v06_raw",
        ascending=False
    ).reset_index(drop=True)

    summary["recognition_core_rank_v01"] = (
        summary.index + 1
    )

    summary["recognition_core_type_v01"] = (
        summary.apply(classify_core, axis=1)
    )

    basins = basins.merge(
        summary[
            [
                "recognition_core_basin_id_v01",
                "recognition_core_rank_v01",
                "recognition_core_type_v01",
            ]
        ],
        on="recognition_core_basin_id_v01",
        how="left"
    )

    centroid_rows = []

    for basin_id in summary[
        "recognition_core_basin_id_v01"
    ]:

        subset = basins[
            basins[
                "recognition_core_basin_id_v01"
            ] == basin_id
        ]

        centroid_geom = (
            subset.unary_union.centroid
        )

        centroid_rows.append({
            "recognition_core_basin_id_v01": basin_id,
            "geometry": centroid_geom
        })

    centroids = gpd.GeoDataFrame(
        centroid_rows,
        geometry="geometry",
        crs=basins.crs
    )

    summary["longitude"] = (
        centroids.to_crs(4326)
        .geometry.x
        .values
    )

    summary["latitude"] = (
        centroids.to_crs(4326)
        .geometry.y
        .values
    )

    log("Summary:")
    log(
        f"Core basins retained: "
        f"{len(summary):,}"
    )

    log(
        f"Largest core basin cells: "
        f"{summary['core_cell_count'].max():,}"
    )

    log(
        f"Mean core basin area km2: "
        f"{summary['core_area_km2'].mean():.2f}"
    )

    print("\nTop 20 core basins:")
    print(summary.head(20))

    log(f"Writing GPKG: {OUT_BASINS_GPKG}")
    gpd.GeoDataFrame(
        basins,
        geometry="geometry",
        crs=gdf.crs
    ).to_file(
        OUT_BASINS_GPKG,
        driver="GPKG"
    )

    log(f"Writing CSV: {OUT_BASINS_CSV}")
    basins.drop(
        columns="geometry"
    ).to_csv(
        OUT_BASINS_CSV,
        index=False
    )

    log(
        f"Writing centroids: "
        f"{OUT_CENTROIDS_GPKG}"
    )

    centroids.to_file(
        OUT_CENTROIDS_GPKG,
        driver="GPKG"
    )

    log(
        f"Writing summary: "
        f"{OUT_SUMMARY_CSV}"
    )

    summary.to_csv(
        OUT_SUMMARY_CSV,
        index=False
    )

    log("Done")


if __name__ == "__main__":
    main()