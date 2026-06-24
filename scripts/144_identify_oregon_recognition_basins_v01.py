#!/usr/bin/env python3
"""
144_identify_oregon_recognition_basins_v01.py

Identify Oregon Recognition Disequilibrium Basins.

Purpose:
Move RDE toward field theory by identifying contiguous zones of elevated
recognition disequilibrium, analogous to basins/catchments in a potential field.

Inputs:
- data/processed/oregon_recognition_disequilibrium_surface_v01.gpkg

Outputs:
- data/processed/oregon_recognition_basins_v01.gpkg
- data/processed/oregon_recognition_basins_v01.csv
- data/processed/oregon_recognition_basin_centroids_v01.gpkg
- data/processed/oregon_recognition_basin_summary_v01.csv
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import networkx as nx


SCRIPT_NAME = "144_identify_oregon_recognition_basins_v01"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = PROCESSED_DIR / "oregon_recognition_disequilibrium_surface_v01.gpkg"

OUT_BASINS_GPKG = PROCESSED_DIR / "oregon_recognition_basins_v01.gpkg"
OUT_BASINS_CSV = PROCESSED_DIR / "oregon_recognition_basins_v01.csv"
OUT_CENTROIDS_GPKG = PROCESSED_DIR / "oregon_recognition_basin_centroids_v01.gpkg"
OUT_SUMMARY_CSV = PROCESSED_DIR / "oregon_recognition_basin_summary_v01.csv"

DISEQUILIBRIUM_THRESHOLD = 0.20
MIN_BASIN_CELLS = 5


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def basin_type(row):
    mean_d = row["mean_disequilibrium_v01"]
    max_d = row["max_disequilibrium_v01"]
    mean_energy = row["mean_surface_energy_v01"]

    if max_d >= 0.50 and mean_energy >= 0.25:
        return "high_energy_recognition_basin"

    if mean_d >= 0.30:
        return "strong_recognition_basin"

    if mean_d >= 0.20:
        return "moderate_recognition_basin"

    return "weak_recognition_basin"


def main():
    log("Starting Oregon recognition basin identification")

    if not INPUT_GPKG.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_GPKG}")

    gdf = gpd.read_file(INPUT_GPKG)

    log(f"Cells loaded: {len(gdf):,}")
    log(f"CRS: {gdf.crs}")

    if "recognition_disequilibrium_v01" not in gdf.columns:
        raise ValueError("Missing required column: recognition_disequilibrium_v01")

    basin_cells = gdf[
        gdf["recognition_disequilibrium_v01"] >= DISEQUILIBRIUM_THRESHOLD
    ].copy()

    log(f"Cells above disequilibrium threshold {DISEQUILIBRIUM_THRESHOLD}: {len(basin_cells):,}")

    if basin_cells.empty:
        raise ValueError("No cells exceeded disequilibrium threshold.")

    basin_cells = basin_cells.reset_index(drop=True)
    basin_cells["node_id"] = basin_cells.index

    log("Building connected-component graph")

    graph = nx.Graph()
    graph.add_nodes_from(basin_cells["node_id"])

    sindex = basin_cells.sindex

    for i, geom in enumerate(basin_cells.geometry):
        if i % 1000 == 0:
            log(f"Adjacency scan {i:,}/{len(basin_cells):,}")

        possible = list(sindex.intersection(geom.bounds))

        for j in possible:
            if j <= i:
                continue

            other = basin_cells.geometry.iloc[j]

            if geom.touches(other) or geom.intersects(other):
                graph.add_edge(i, j)

    components = list(nx.connected_components(graph))

    log(f"Initial basin components: {len(components):,}")

    basin_records = []
    retained_cell_frames = []

    basin_id = 1

    for comp in components:
        idx = sorted(list(comp))
        sub = basin_cells.iloc[idx].copy()

        if len(sub) < MIN_BASIN_CELLS:
            continue

        sub["recognition_basin_id_v01"] = basin_id
        retained_cell_frames.append(sub)

        geom = sub.geometry.union_all()

        record = {
            "recognition_basin_id_v01": basin_id,
            "basin_cell_count": len(sub),
            "basin_area_km2": sub.geometry.area.sum() / 1_000_000,
            "mean_disequilibrium_v01": sub["recognition_disequilibrium_v01"].mean(),
            "max_disequilibrium_v01": sub["recognition_disequilibrium_v01"].max(),
            "mean_disequilibrium_norm_v01": sub[
                "recognition_disequilibrium_norm_v01"
            ].mean(),
            "mean_gradient_magnitude_v01": sub[
                "recognition_gradient_magnitude_v01"
            ].mean(),
            "max_gradient_magnitude_v01": sub[
                "recognition_gradient_magnitude_v01"
            ].max(),
            "mean_gradient_norm_v01": sub[
                "recognition_gradient_magnitude_norm_v01"
            ].mean(),
            "mean_surface_energy_v01": sub["rde_surface_energy_v01"].mean(),
            "max_surface_energy_v01": sub["rde_surface_energy_v01"].max(),
            "mean_physical_exceptionality_v03": sub[
                "physical_exceptionality_v03"
            ].mean()
            if "physical_exceptionality_v03" in sub.columns
            else None,
            "mean_observed_recognition_v04": sub[
                "observed_recognition_v04"
            ].mean()
            if "observed_recognition_v04" in sub.columns
            else None,
            "mean_expected_recognition_v06": sub[
                "expected_recognition_v06"
            ].mean()
            if "expected_recognition_v06" in sub.columns
            else None,
            "high_basin_cell_count_v01": (
                sub["rde_surface_class_v01"] == "high_disequilibrium_basin"
            ).sum(),
            "high_gradient_cell_count_v01": (
                sub["rde_surface_class_v01"] == "high_disequilibrium_high_gradient"
            ).sum(),
            "dominant_surface_class_v01": sub[
                "rde_surface_class_v01"
            ].value_counts().idxmax(),
            "geometry": geom,
        }

        basin_records.append(record)
        basin_id += 1

    if not basin_records:
        raise ValueError("No recognition basins met minimum cell threshold.")

    basins = gpd.GeoDataFrame(
        basin_records,
        geometry="geometry",
        crs=gdf.crs,
    )

    basins["recognition_basin_type_v01"] = basins.apply(
        basin_type,
        axis=1,
    )

    basins = basins.sort_values(
        [
            "mean_surface_energy_v01",
            "mean_disequilibrium_v01",
            "basin_cell_count",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    basins["recognition_basin_rank_v01"] = basins.index + 1

    basins["recognition_basin_tier_v01"] = pd.cut(
        basins["recognition_basin_rank_v01"],
        bins=[0, 10, 25, 50, 10_000],
        labels=["top_10", "top_25", "top_50", "basin"],
    ).astype(str)

    centroids = basins.copy()
    centroids["geometry"] = centroids.geometry.representative_point()

    if retained_cell_frames:
        retained_cells = pd.concat(retained_cell_frames, ignore_index=True)
    else:
        retained_cells = basin_cells.iloc[[]].copy()

    log("Summary:")
    log(f"Recognition basins retained: {len(basins):,}")
    log(f"Cells in retained basins: {len(retained_cells):,}")
    log(f"Largest basin cell count: {basins['basin_cell_count'].max():,}")
    log(f"Mean basin area km2: {basins['basin_area_km2'].mean():.2f}")

    print("\nBasin type counts:")
    print(basins["recognition_basin_type_v01"].value_counts())

    print("\nTop 20 basins:")
    print(
        basins[
            [
                "recognition_basin_rank_v01",
                "recognition_basin_id_v01",
                "recognition_basin_type_v01",
                "basin_cell_count",
                "basin_area_km2",
                "mean_disequilibrium_v01",
                "mean_surface_energy_v01",
                "mean_physical_exceptionality_v03",
                "mean_observed_recognition_v04",
                "mean_expected_recognition_v06",
                "high_basin_cell_count_v01",
                "high_gradient_cell_count_v01",
            ]
        ].head(20)
    )

    for path in [OUT_BASINS_GPKG, OUT_CENTROIDS_GPKG]:
        if path.exists():
            path.unlink()

    log(f"Writing basins GPKG: {OUT_BASINS_GPKG}")
    basins.to_file(
        OUT_BASINS_GPKG,
        layer="oregon_recognition_basins_v01",
        driver="GPKG",
    )

    log(f"Writing basins CSV: {OUT_BASINS_CSV}")
    basins.drop(columns="geometry").to_csv(OUT_BASINS_CSV, index=False)

    log(f"Writing centroids GPKG: {OUT_CENTROIDS_GPKG}")
    centroids.to_file(
        OUT_CENTROIDS_GPKG,
        layer="oregon_recognition_basin_centroids_v01",
        driver="GPKG",
    )

    summary = pd.DataFrame(
        [
            {
                "disequilibrium_threshold": DISEQUILIBRIUM_THRESHOLD,
                "min_basin_cells": MIN_BASIN_CELLS,
                "input_cells": len(gdf),
                "threshold_cells": len(basin_cells),
                "basins_retained": len(basins),
                "cells_in_retained_basins": len(retained_cells),
                "largest_basin_cell_count": basins["basin_cell_count"].max(),
                "mean_basin_cell_count": basins["basin_cell_count"].mean(),
                "median_basin_cell_count": basins["basin_cell_count"].median(),
                "mean_basin_area_km2": basins["basin_area_km2"].mean(),
                "max_basin_area_km2": basins["basin_area_km2"].max(),
            }
        ]
    )

    log(f"Writing summary CSV: {OUT_SUMMARY_CSV}")
    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    log("Done")


if __name__ == "__main__":
    main()