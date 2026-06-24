#!/usr/bin/env python3
"""
147_oregon_core_basin_energy_sensitivity_v01.py

Test RDE surface-energy thresholds for Oregon recognition core basins.

Inputs:
- data/processed/oregon_recognition_disequilibrium_surface_v01.gpkg

Outputs:
- data/processed/oregon_core_basin_energy_sensitivity_v01.csv
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import networkx as nx


SCRIPT_NAME = "147_oregon_core_basin_energy_sensitivity_v01"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = PROCESSED_DIR / "oregon_recognition_disequilibrium_surface_v01.gpkg"
OUT_CSV = PROCESSED_DIR / "oregon_core_basin_energy_sensitivity_v01.csv"

PERCENTILES = [0.90, 0.925, 0.95, 0.975, 0.99]
MIN_CORE_CELLS = 3


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def analyze_threshold(gdf, percentile):
    threshold = gdf["rde_surface_energy_v01"].quantile(percentile)

    cells = gdf[gdf["rde_surface_energy_v01"] >= threshold].copy()

    if cells.empty:
        return {
            "energy_percentile": percentile,
            "energy_threshold": threshold,
            "core_cells": 0,
            "initial_components": 0,
            "retained_cores": 0,
            "cells_in_retained_cores": 0,
            "largest_core_cells": 0,
            "mean_core_cells": 0,
            "median_core_cells": 0,
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
    retained = [c for c in comps if len(c) >= MIN_CORE_CELLS]

    sizes = [len(c) for c in retained]

    return {
        "energy_percentile": percentile,
        "energy_threshold": threshold,
        "core_cells": len(cells),
        "initial_components": len(comps),
        "retained_cores": len(retained),
        "cells_in_retained_cores": sum(sizes),
        "largest_core_cells": max(sizes) if sizes else 0,
        "mean_core_cells": sum(sizes) / len(sizes) if sizes else 0,
        "median_core_cells": pd.Series(sizes).median() if sizes else 0,
    }


def main():
    log("Starting Oregon core basin energy sensitivity")

    if not INPUT_GPKG.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_GPKG}")

    gdf = gpd.read_file(INPUT_GPKG)

    rows = []

    for p in PERCENTILES:
        log(f"Testing energy percentile: {p}")
        rows.append(analyze_threshold(gdf, p))

    out = pd.DataFrame(rows)

    print(out)

    log(f"Writing CSV: {OUT_CSV}")
    out.to_csv(OUT_CSV, index=False)

    log("Done")


if __name__ == "__main__":
    main()