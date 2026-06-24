#!/usr/bin/env python3
"""
148_identify_oregon_final_core_basins_v01.py

Final Oregon recognition core basin extraction using 99th percentile
RDE surface energy threshold.

Inputs:
- data/processed/oregon_recognition_disequilibrium_surface_v01.gpkg

Outputs:
- data/processed/oregon_final_recognition_core_basins_v01.gpkg
- data/processed/oregon_final_recognition_core_basins_v01.csv
- data/processed/oregon_final_recognition_core_basin_centroids_v01.gpkg
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import networkx as nx

SCRIPT_NAME = "148_identify_oregon_final_core_basins_v01"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = PROCESSED_DIR / "oregon_recognition_disequilibrium_surface_v01.gpkg"

OUT_GPKG = PROCESSED_DIR / "oregon_final_recognition_core_basins_v01.gpkg"
OUT_CSV = PROCESSED_DIR / "oregon_final_recognition_core_basins_v01.csv"
OUT_CENTROIDS_GPKG = PROCESSED_DIR / "oregon_final_recognition_core_basin_centroids_v01.gpkg"

ENERGY_PERCENTILE = 0.99
MIN_CORE_CELLS = 3


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def main():
    log("Starting final Oregon recognition core basin extraction")

    gdf = gpd.read_file(INPUT_GPKG)

    threshold = gdf["rde_surface_energy_v01"].quantile(ENERGY_PERCENTILE)

    log(f"Energy percentile: {ENERGY_PERCENTILE}")
    log(f"Energy threshold: {threshold:.6f}")

    cells = gdf[gdf["rde_surface_energy_v01"] >= threshold].copy()
    cells = cells.reset_index(drop=True)
    cells["node_id"] = cells.index

    log(f"Core cells selected: {len(cells):,}")

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

    components = list(nx.connected_components(graph))
    retained = [c for c in components if len(c) >= MIN_CORE_CELLS]

    log(f"Initial components: {len(components):,}")
    log(f"Retained final cores: {len(retained):,}")

    records = []

    for core_id, comp in enumerate(retained, start=1):
        sub = cells.iloc[list(comp)].copy()

        geom = sub.geometry.union_all()

        records.append(
            {
                "final_core_basin_id_v01": core_id,
                "core_cell_count": len(sub),
                "core_area_km2": sub.geometry.area.sum() / 1_000_000,
                "mean_rde_surface_energy_v01": sub["rde_surface_energy_v01"].mean(),
                "max_rde_surface_energy_v01": sub["rde_surface_energy_v01"].max(),
                "mean_disequilibrium_v01": sub["recognition_disequilibrium_v01"].mean(),
                "max_disequilibrium_v01": sub["recognition_disequilibrium_v01"].max(),
                "mean_gradient_norm_v01": sub["recognition_gradient_magnitude_norm_v01"].mean(),
                "max_gradient_norm_v01": sub["recognition_gradient_magnitude_norm_v01"].max(),
                "mean_physical_exceptionality_v03": sub["physical_exceptionality_v03"].mean(),
                "mean_observed_recognition_v04": sub["observed_recognition_v04"].mean(),
                "mean_expected_recognition_v06": sub["expected_recognition_v06"].mean(),
                "mean_urem_score_v06_raw": sub["urem_score_v06_raw"].mean()
                if "urem_score_v06_raw" in sub.columns
                else None,
                "max_urem_score_v06_raw": sub["urem_score_v06_raw"].max()
                if "urem_score_v06_raw" in sub.columns
                else None,
                "geometry": geom,
            }
        )

    out = gpd.GeoDataFrame(records, geometry="geometry", crs=gdf.crs)

    out = out.sort_values(
        ["max_rde_surface_energy_v01", "mean_rde_surface_energy_v01"],
        ascending=[False, False],
    ).reset_index(drop=True)

    out["final_core_basin_rank_v01"] = out.index + 1

    centroids = out.copy()
    centroids["geometry"] = centroids.geometry.representative_point()

    wgs = centroids.to_crs(4326)
    out["longitude"] = wgs.geometry.x
    out["latitude"] = wgs.geometry.y

    log("Top final core basins:")
    print(
        out[
            [
                "final_core_basin_rank_v01",
                "core_cell_count",
                "core_area_km2",
                "mean_rde_surface_energy_v01",
                "mean_disequilibrium_v01",
                "mean_physical_exceptionality_v03",
                "mean_observed_recognition_v04",
                "mean_expected_recognition_v06",
                "longitude",
                "latitude",
            ]
        ]
    )

    for path in [OUT_GPKG, OUT_CENTROIDS_GPKG]:
        if path.exists():
            path.unlink()

    log(f"Writing GPKG: {OUT_GPKG}")
    out.to_file(OUT_GPKG, layer="oregon_final_recognition_core_basins_v01", driver="GPKG")

    log(f"Writing CSV: {OUT_CSV}")
    out.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log(f"Writing centroids GPKG: {OUT_CENTROIDS_GPKG}")
    centroids.to_file(
        OUT_CENTROIDS_GPKG,
        layer="oregon_final_recognition_core_basin_centroids_v01",
        driver="GPKG",
    )

    log("Done")


if __name__ == "__main__":
    main()