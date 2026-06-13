#!/usr/bin/env python3
"""
Script 41: Compare UREM Hotspots v03 vs v04

Purpose:
- Compare v03 and v04 hotspot outputs.
- Identify which v03 hotspots survived in v04.
- Identify new v04-only hotspots.
- Measure nearest hotspot centroid distance.

Inputs:
- data/processed/urem_hotspots_v03.gpkg
- data/processed/urem_hotspots_v04.gpkg

Outputs:
- data/processed/urem_hotspot_comparison_v03_v04.csv
- data/processed/urem_hotspot_comparison_v03_v04.gpkg
"""

from pathlib import Path
import warnings

import geopandas as gpd
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]

V03_PATH = BASE_DIR / "data/processed/urem_hotspots_v03.gpkg"
V04_PATH = BASE_DIR / "data/processed/urem_hotspots_v04.gpkg"

OUT_CSV = BASE_DIR / "data/processed/urem_hotspot_comparison_v03_v04.csv"
OUT_GPKG = BASE_DIR / "data/processed/urem_hotspot_comparison_v03_v04.gpkg"


def log(msg: str) -> None:
    print(f"[41_compare_urem_hotspots_v03_v04] {msg}")


def prep_hotspots(gdf: gpd.GeoDataFrame, version: str) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf[f"centroid_{version}"] = gdf.geometry.centroid
    gdf = gdf.set_geometry(f"centroid_{version}")
    return gdf


def main():
    log("Starting Script 41")

    if not V03_PATH.exists():
        raise FileNotFoundError(f"Missing v03 hotspots: {V03_PATH}")
    if not V04_PATH.exists():
        raise FileNotFoundError(f"Missing v04 hotspots: {V04_PATH}")

    v03 = gpd.read_file(V03_PATH)
    v04 = gpd.read_file(V04_PATH)

    if v03.crs != v04.crs:
        v04 = v04.to_crs(v03.crs)

    log(f"v03 hotspots: {len(v03):,}")
    log(f"v04 hotspots: {len(v04):,}")

    v03_cent = prep_hotspots(v03, "v03")
    v04_cent = prep_hotspots(v04, "v04")

    rows = []

    for _, h4 in v04_cent.iterrows():
        dists = v03_cent.geometry.distance(h4.geometry)
        nearest_idx = dists.idxmin()
        nearest = v03_cent.loc[nearest_idx]

        nearest_dist_m = float(dists.loc[nearest_idx])

        overlap = v03.loc[nearest_idx].geometry.intersects(
            v04.loc[h4.name].geometry
        )

        rows.append({
            "v04_hotspot_rank": h4.get("hotspot_rank_v04"),
            "v04_hotspot_id": h4.get("hotspot_id"),
            "v04_cell_count": h4.get("cell_count"),
            "v04_hotspot_score": h4.get("hotspot_score_v04"),
            "v04_mean_urem_score": h4.get("mean_urem_score_v04"),
            "v04_mean_exceptionality": h4.get("mean_exceptionality_v02"),
            "v04_mean_observed_recognition": h4.get("mean_observed_recognition_v04"),
            "v04_mean_expected_recognition": h4.get("mean_expected_recognition_v04"),
            "v04_mean_under_recognition_residual": h4.get(
                "mean_under_recognition_residual_v04"
            ),
            "v04_mean_land_area_share": h4.get("mean_land_area_share"),

            "nearest_v03_hotspot_rank": nearest.get("hotspot_rank_v03"),
            "nearest_v03_hotspot_id": nearest.get("hotspot_id"),
            "nearest_v03_cell_count": nearest.get("cell_count"),
            "nearest_v03_hotspot_score": nearest.get("hotspot_score_v03"),
            "nearest_v03_mean_urem_score": nearest.get("mean_urem_score_v03"),
            "nearest_v03_distance_m": nearest_dist_m,
            "nearest_v03_distance_km": nearest_dist_m / 1000,
            "intersects_v03_hotspot": bool(overlap),
        })

    comparison = pd.DataFrame(rows)

    comparison["v04_status_vs_v03"] = np.where(
        comparison["intersects_v03_hotspot"],
        "survived_overlap",
        np.where(
            comparison["nearest_v03_distance_km"] <= 10,
            "near_existing_v03",
            "new_v04_hotspot",
        ),
    )

    # GeoPackage output uses v04 hotspot polygons with comparison attributes.
    out_gdf = v04.merge(
        comparison,
        left_on="hotspot_rank_v04",
        right_on="v04_hotspot_rank",
        how="left",
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    log(f"Writing CSV: {OUT_CSV}")
    comparison.to_csv(OUT_CSV, index=False)

    log(f"Writing GeoPackage: {OUT_GPKG}")
    out_gdf.to_file(OUT_GPKG, driver="GPKG")

    log("Done")

    print("\nComparison summary:")
    print(comparison["v04_status_vs_v03"].value_counts())

    print("\nTop v04 hotspots comparison:")
    print(
        comparison[
            [
                "v04_hotspot_rank",
                "v04_cell_count",
                "v04_hotspot_score",
                "v04_mean_land_area_share",
                "nearest_v03_hotspot_rank",
                "nearest_v03_distance_km",
                "intersects_v03_hotspot",
                "v04_status_vs_v03",
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()