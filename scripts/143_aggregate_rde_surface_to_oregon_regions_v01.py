#!/usr/bin/env python3
"""
143_aggregate_rde_surface_to_oregon_regions_v01.py

Aggregate Oregon RDE disequilibrium surface metrics to discovery regions.

Inputs:
- data/processed/oregon_discovery_regions_v06.gpkg
- data/processed/oregon_recognition_disequilibrium_surface_v01.gpkg

Outputs:
- data/processed/oregon_region_rde_surface_v01.gpkg
- data/processed/oregon_region_rde_surface_v01.csv
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd


SCRIPT_NAME = "143_aggregate_rde_surface_to_oregon_regions_v01"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

REGIONS_PATH = PROCESSED_DIR / "oregon_discovery_regions_v06.gpkg"
SURFACE_PATH = PROCESSED_DIR / "oregon_recognition_disequilibrium_surface_v01.gpkg"

OUT_GPKG = PROCESSED_DIR / "oregon_region_rde_surface_v01.gpkg"
OUT_CSV = PROCESSED_DIR / "oregon_region_rde_surface_v01.csv"


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def dominant_class(series):
    if series.empty:
        return "unknown"
    return series.value_counts().idxmax()


def main():
    log("Starting Oregon region RDE surface aggregation")

    if not REGIONS_PATH.exists():
        raise FileNotFoundError(f"Missing regions: {REGIONS_PATH}")

    if not SURFACE_PATH.exists():
        raise FileNotFoundError(f"Missing surface: {SURFACE_PATH}")

    regions = gpd.read_file(REGIONS_PATH)
    surface = gpd.read_file(SURFACE_PATH)

    if surface.crs != regions.crs:
        surface = surface.to_crs(regions.crs)

    rows = []

    for _, region in regions.iterrows():
        geom = region.geometry

        cells = surface[surface.geometry.intersects(geom)].copy()

        if cells.empty:
            continue

        row = region.drop(labels="geometry").to_dict()

        row.update(
            {
                "rde_surface_cell_count_v01": len(cells),
                "region_mean_disequilibrium_v01": cells[
                    "recognition_disequilibrium_v01"
                ].mean(),
                "region_max_disequilibrium_v01": cells[
                    "recognition_disequilibrium_v01"
                ].max(),
                "region_mean_disequilibrium_norm_v01": cells[
                    "recognition_disequilibrium_norm_v01"
                ].mean(),
                "region_mean_gradient_magnitude_v01": cells[
                    "recognition_gradient_magnitude_v01"
                ].mean(),
                "region_max_gradient_magnitude_v01": cells[
                    "recognition_gradient_magnitude_v01"
                ].max(),
                "region_mean_gradient_norm_v01": cells[
                    "recognition_gradient_magnitude_norm_v01"
                ].mean(),
                "region_mean_surface_energy_v01": cells[
                    "rde_surface_energy_v01"
                ].mean(),
                "region_max_surface_energy_v01": cells[
                    "rde_surface_energy_v01"
                ].max(),
                "region_dominant_surface_class_v01": dominant_class(
                    cells["rde_surface_class_v01"]
                ),
                "region_high_basin_cell_count_v01": (
                    cells["rde_surface_class_v01"]
                    == "high_disequilibrium_basin"
                ).sum(),
                "region_high_gradient_cell_count_v01": (
                    cells["rde_surface_class_v01"]
                    == "high_disequilibrium_high_gradient"
                ).sum(),
                "geometry": geom,
            }
        )

        rows.append(row)

    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=regions.crs)

    out = out.sort_values(
        [
            "region_mean_surface_energy_v1"
            if "region_mean_surface_energy_v1" in out.columns
            else "region_mean_surface_energy_v01"
        ],
        ascending=False,
    ).reset_index(drop=True)

    out["rde_surface_region_rank_v01"] = out.index + 1

    log("Dominant surface classes:")
    print(out["region_dominant_surface_class_v01"].value_counts())

    log("Top 20 region RDE surface metrics:")
    print(
        out[
            [
                "rde_surface_region_rank_v01",
                "discovery_region_rank_v06",
                "cell_count",
                "region_mean_disequilibrium_v01",
                "region_mean_gradient_norm_v01",
                "region_mean_surface_energy_v01",
                "region_dominant_surface_class_v01",
                "region_high_basin_cell_count_v01",
                "region_high_gradient_cell_count_v01",
            ]
        ].head(20)
    )

    if OUT_GPKG.exists():
        OUT_GPKG.unlink()

    log(f"Writing GPKG: {OUT_GPKG}")
    out.to_file(
        OUT_GPKG,
        layer="oregon_region_rde_surface_v01",
        driver="GPKG",
    )

    log(f"Writing CSV: {OUT_CSV}")
    out.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")


if __name__ == "__main__":
    main()