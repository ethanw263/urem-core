#!/usr/bin/env python3
"""
145_oregon_recognition_basin_threshold_sensitivity_v01.py

Test recognition basin thresholds to avoid over-merged mega-basins.

Inputs:
- data/processed/oregon_recognition_disequilibrium_surface_v01.gpkg

Outputs:
- data/processed/oregon_recognition_basin_threshold_sensitivity_v01.csv
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import networkx as nx


SCRIPT_NAME = "145_oregon_recognition_basin_threshold_sensitivity_v01"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = PROCESSED_DIR / "oregon_recognition_disequilibrium_surface_v01.gpkg"
OUT_CSV = PROCESSED_DIR / "oregon_recognition_basin_threshold_sensitivity_v01.csv"

THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
MIN_BASIN_CELLS = 5


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def connected_components_for_threshold(gdf, threshold):
    cells = gdf[gdf["recognition_disequilibrium_v01"] >= threshold].copy()

    if cells.empty:
        return {
            "threshold": threshold,
            "threshold_cells": 0,
            "initial_components": 0,
            "retained_basins": 0,
            "cells_in_retained_basins": 0,
            "largest_basin_cells": 0,
            "mean_retained_basin_cells": 0,
            "median_retained_basin_cells": 0,
        }

    cells = cells.reset_index(drop=True)
    cells["node_id"] = cells.index

    graph = nx.Graph()
    graph.add_nodes_from(cells["node_id"])

    sindex = cells.sindex

    for i, geom in enumerate(cells.geometry):
        possible = list(sindex.intersection(geom.bounds))

        for j in possible:
            if j <= i:
                continue

            other = cells.geometry.iloc[j]

            if geom.touches(other) or geom.intersects(other):
                graph.add_edge(i, j)

    comps = list(nx.connected_components(graph))
    retained = [c for c in comps if len(c) >= MIN_BASIN_CELLS]

    sizes = [len(c) for c in retained]

    return {
        "threshold": threshold,
        "threshold_cells": len(cells),
        "initial_components": len(comps),
        "retained_basins": len(retained),
        "cells_in_retained_basins": sum(sizes),
        "largest_basin_cells": max(sizes) if sizes else 0,
        "mean_retained_basin_cells": sum(sizes) / len(sizes) if sizes else 0,
        "median_retained_basin_cells": pd.Series(sizes).median() if sizes else 0,
    }


def main():
    log("Starting Oregon recognition basin threshold sensitivity")

    if not INPUT_GPKG.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_GPKG}")

    gdf = gpd.read_file(INPUT_GPKG)

    log(f"Cells loaded: {len(gdf):,}")

    rows = []

    for threshold in THRESHOLDS:
        log(f"Testing threshold: {threshold}")
        rows.append(connected_components_for_threshold(gdf, threshold))

    out = pd.DataFrame(rows)

    print(out)

    log(f"Writing CSV: {OUT_CSV}")
    out.to_csv(OUT_CSV, index=False)

    log("Done")


if __name__ == "__main__":
    main()