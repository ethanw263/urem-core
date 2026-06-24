#!/usr/bin/env python3
"""
134_cluster_oregon_discovery_regions_v06.py

Cluster Oregon UREM v06 candidate cells into connected discovery regions.

Inputs:
- data/processed/oregon_ranked_urem_candidates_v06.gpkg

Outputs:
- data/processed/oregon_discovery_regions_v06.gpkg
- data/processed/oregon_discovery_regions_v06.csv
- data/processed/oregon_discovery_region_centroids_v06.gpkg
- data/processed/oregon_discovery_region_summary_v06.csv
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import networkx as nx


SCRIPT_NAME = "134_cluster_oregon_discovery_regions_v06"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = PROCESSED_DIR / "oregon_ranked_urem_candidates_v06.gpkg"

OUT_REGIONS_GPKG = PROCESSED_DIR / "oregon_discovery_regions_v06.gpkg"
OUT_REGIONS_CSV = PROCESSED_DIR / "oregon_discovery_regions_v06.csv"
OUT_CENTROIDS_GPKG = PROCESSED_DIR / "oregon_discovery_region_centroids_v06.gpkg"
OUT_SUMMARY_CSV = PROCESSED_DIR / "oregon_discovery_region_summary_v06.csv"

MIN_REGION_CELLS = 3


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def main():
    log("Starting Oregon discovery region clustering v06")

    if not INPUT_GPKG.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_GPKG}")

    cells = gpd.read_file(INPUT_GPKG)

    log(f"Candidate cells loaded: {len(cells):,}")
    log(f"CRS: {cells.crs}")

    cells = cells.reset_index(drop=True)
    cells["node_id"] = cells.index

    log("Building spatial adjacency graph")

    sindex = cells.sindex
    graph = nx.Graph()
    graph.add_nodes_from(cells["node_id"])

    for i, geom in enumerate(cells.geometry):
        if i % 500 == 0:
            log(f"Adjacency scan {i:,}/{len(cells):,}")

        possible = list(sindex.intersection(geom.bounds))

        for j in possible:
            if j <= i:
                continue

            other = cells.geometry.iloc[j]

            if geom.touches(other) or geom.intersects(other):
                graph.add_edge(i, j)

    components = list(nx.connected_components(graph))

    log(f"Initial connected components: {len(components):,}")

    region_records = []
    region_cells = []

    region_id = 1

    for comp in components:
        idx = sorted(list(comp))
        sub = cells.iloc[idx].copy()

        if len(sub) < MIN_REGION_CELLS:
            continue

        sub["discovery_region_id_v06"] = region_id
        region_cells.append(sub)

        dissolved_geom = sub.geometry.union_all()

        record = {
            "discovery_region_id_v06": region_id,
            "cell_count": len(sub),
            "region_area_km2": sub.geometry.area.sum() / 1_000_000,
            "mean_urem_score_v06_raw": sub["urem_score_v06_raw"].mean(),
            "max_urem_score_v06_raw": sub["urem_score_v06_raw"].max(),
            "mean_urem_score_v06_fullrange": sub["urem_score_v06_fullrange"].mean(),
            "mean_physical_exceptionality_v03": sub["physical_exceptionality_v03"].mean(),
            "max_physical_exceptionality_v03": sub["physical_exceptionality_v03"].max(),
            "mean_observed_recognition_v04": sub["observed_recognition_v04"].mean(),
            "mean_expected_recognition_v06": sub["expected_recognition_v06"].mean(),
            "mean_positive_under_recognition_residual_v06": sub[
                "positive_under_recognition_residual_v06"
            ].mean(),
            "mean_recognition_total_count_3km_v04": sub[
                "recognition_total_count_3km_v04"
            ].mean()
            if "recognition_total_count_3km_v04" in sub.columns
            else None,
            "best_cell_rank_v06": sub["candidate_rank_v06"].min(),
            "best_cell_id": sub.loc[sub["candidate_rank_v06"].idxmin(), "cell_id"],
            "geometry": dissolved_geom,
        }

        region_records.append(record)
        region_id += 1

    if not region_records:
        raise ValueError("No discovery regions met minimum cell threshold.")

    regions = gpd.GeoDataFrame(
        region_records,
        geometry="geometry",
        crs=cells.crs,
    )

    regions = regions.sort_values(
        ["max_urem_score_v06_raw", "mean_urem_score_v06_raw", "cell_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    regions["discovery_region_rank_v06"] = regions.index + 1

    regions["discovery_region_tier_v06"] = pd.cut(
        regions["discovery_region_rank_v06"],
        bins=[0, 10, 25, 50, 10_000],
        labels=["top_10", "top_25", "top_50", "region"],
    ).astype(str)

    centroids = regions.copy()
    centroids["geometry"] = centroids.geometry.representative_point()

    if region_cells:
        all_region_cells = pd.concat(region_cells, ignore_index=True)
        all_region_cells = gpd.GeoDataFrame(
            all_region_cells,
            geometry="geometry",
            crs=cells.crs,
        )
    else:
        all_region_cells = cells.iloc[[]].copy()

    log("Summary:")
    log(f"Discovery regions retained: {len(regions):,}")
    log(f"Cells in retained regions: {len(all_region_cells):,}")
    log(f"Minimum region cells: {MIN_REGION_CELLS}")
    log(f"Largest region cell count: {regions['cell_count'].max():,}")

    print(
        regions[
            [
                "discovery_region_rank_v06",
                "discovery_region_id_v06",
                "cell_count",
                "region_area_km2",
                "mean_urem_score_v06_raw",
                "max_urem_score_v06_raw",
                "mean_physical_exceptionality_v03",
                "mean_observed_recognition_v04",
                "mean_expected_recognition_v06",
                "mean_positive_under_recognition_residual_v06",
                "best_cell_rank_v06",
            ]
        ].head(20)
    )

    for path in [OUT_REGIONS_GPKG, OUT_CENTROIDS_GPKG]:
        if path.exists():
            path.unlink()

    log(f"Writing regions GPKG: {OUT_REGIONS_GPKG}")
    regions.to_file(
        OUT_REGIONS_GPKG,
        layer="oregon_discovery_regions_v06",
        driver="GPKG",
    )

    log(f"Writing regions CSV: {OUT_REGIONS_CSV}")
    regions.drop(columns="geometry").to_csv(OUT_REGIONS_CSV, index=False)

    log(f"Writing centroids GPKG: {OUT_CENTROIDS_GPKG}")
    centroids.to_file(
        OUT_CENTROIDS_GPKG,
        layer="oregon_discovery_region_centroids_v06",
        driver="GPKG",
    )

    summary = pd.DataFrame(
        [
            {
                "candidate_cells_input": len(cells),
                "regions_retained": len(regions),
                "cells_in_retained_regions": len(all_region_cells),
                "min_region_cells": MIN_REGION_CELLS,
                "largest_region_cell_count": regions["cell_count"].max(),
                "mean_region_cell_count": regions["cell_count"].mean(),
                "median_region_cell_count": regions["cell_count"].median(),
                "mean_region_area_km2": regions["region_area_km2"].mean(),
                "max_region_area_km2": regions["region_area_km2"].max(),
            }
        ]
    )

    log(f"Writing summary CSV: {OUT_SUMMARY_CSV}")
    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    log("Done")


if __name__ == "__main__":
    main()