#!/usr/bin/env python3
"""
52_compare_v04_v05b_hotspots.py

Compare UREM v04 and v05b hotspots.

Purpose:
- Determine whether v05b is a useful calibrated refinement of v04.
- Compare each v04 hotspot to the nearest v05b hotspot.
- Identify v05b hotspots that are new relative to v04.

Inputs:
- data/processed/urem_hotspots_v04.gpkg
- data/processed/urem_hotspots_v05b.gpkg
- data/processed/urem_hotspot_centroids_v04.csv
- data/processed/urem_hotspot_centroids_v05b.csv

Outputs:
- data/processed/urem_hotspot_comparison_v04_v05b.csv
- data/processed/urem_hotspot_comparison_v04_v05b.gpkg
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np


SCRIPT_NAME = "52_compare_v04_v05b_hotspots"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

V04_GPKG = PROCESSED_DIR / "urem_hotspots_v04.gpkg"
V05B_GPKG = PROCESSED_DIR / "urem_hotspots_v05b.gpkg"

V04_CSV = PROCESSED_DIR / "urem_hotspot_centroids_v04.csv"
V05B_CSV = PROCESSED_DIR / "urem_hotspot_centroids_v05b.csv"

OUT_CSV = PROCESSED_DIR / "urem_hotspot_comparison_v04_v05b.csv"
OUT_GPKG = PROCESSED_DIR / "urem_hotspot_comparison_v04_v05b.gpkg"


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def classify_distance(distance_km):
    if pd.isna(distance_km):
        return "no_v05b_match"
    if distance_km <= 5:
        return "survived"
    if distance_km <= 25:
        return "near_survived"
    return "lost_in_v05b"


def load_centroids(csv_path, version):
    df = pd.read_csv(csv_path)

    rank_col = f"hotspot_rank_{version}"
    score_col = f"hotspot_score_{version}"

    keep = [
        rank_col,
        "hotspot_id",
        "longitude",
        "latitude",
        "cell_count",
        "hotspot_area_km2",
        score_col,
        "best_cell_id",
    ]

    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    rename = {
        rank_col: f"{version}_rank",
        "hotspot_id": f"{version}_hotspot_id",
        "longitude": f"{version}_longitude",
        "latitude": f"{version}_latitude",
        "cell_count": f"{version}_cell_count",
        "hotspot_area_km2": f"{version}_area_km2",
        score_col: f"{version}_score",
        "best_cell_id": f"{version}_best_cell_id",
    }

    return df.rename(columns=rename)


def main():
    log("Starting v04 vs v05b hotspot comparison")

    for p in [V04_GPKG, V05B_GPKG, V04_CSV, V05B_CSV]:
        require_file(p)

    v04 = gpd.read_file(V04_GPKG)
    v05b = gpd.read_file(V05B_GPKG)

    if v04.crs is None or v05b.crs is None:
        raise ValueError("One hotspot layer has no CRS.")

    v04 = v04.to_crs("EPSG:3310")
    v05b = v05b.to_crs("EPSG:3310")

    log(f"v04 hotspots: {len(v04):,}")
    log(f"v05b hotspots: {len(v05b):,}")

    v04_centroids = load_centroids(V04_CSV, "v04")
    v05b_centroids = load_centroids(V05B_CSV, "v05b")

    if "hotspot_rank_v04" not in v04.columns:
        v04 = v04.reset_index(drop=True)
        v04["hotspot_rank_v04"] = range(1, len(v04) + 1)

    if "hotspot_rank_v05b" not in v05b.columns:
        v05b = v05b.reset_index(drop=True)
        v05b["hotspot_rank_v05b"] = range(1, len(v05b) + 1)

    v04_pts = v04.copy()
    v04_pts["geometry"] = v04_pts.geometry.centroid

    v05b_pts = v05b.copy()
    v05b_pts["geometry"] = v05b_pts.geometry.centroid

    rows = []

    for _, row04 in v04_pts.iterrows():
        distances_m = v05b_pts.geometry.distance(row04.geometry)
        nearest_idx = distances_m.idxmin()
        nearest_v05b = v05b_pts.loc[nearest_idx]
        distance_km = float(distances_m.loc[nearest_idx] / 1000)

        rows.append(
            {
                "v04_rank": row04.get("hotspot_rank_v04"),
                "v04_hotspot_id": row04.get("hotspot_id"),
                "nearest_v05b_rank": nearest_v05b.get("hotspot_rank_v05b"),
                "nearest_v05b_hotspot_id": nearest_v05b.get("hotspot_id"),
                "distance_to_nearest_v05b_km": distance_km,
                "comparison_class": classify_distance(distance_km),
                "geometry": row04.geometry,
            }
        )

    comparison = pd.DataFrame(rows)

    comparison = comparison.merge(
        v04_centroids,
        on=["v04_rank", "v04_hotspot_id"],
        how="left",
    )

    comparison = comparison.merge(
        v05b_centroids,
        left_on=["nearest_v05b_rank", "nearest_v05b_hotspot_id"],
        right_on=["v05b_rank", "v05b_hotspot_id"],
        how="left",
    )

    new_rows = []

    for _, row05b in v05b_pts.iterrows():
        distances_m = v04_pts.geometry.distance(row05b.geometry)
        nearest_idx = distances_m.idxmin()
        distance_km = float(distances_m.loc[nearest_idx] / 1000)

        if distance_km > 25:
            new_rows.append(
                {
                    "v04_rank": np.nan,
                    "v04_hotspot_id": np.nan,
                    "nearest_v05b_rank": row05b.get("hotspot_rank_v05b"),
                    "nearest_v05b_hotspot_id": row05b.get("hotspot_id"),
                    "distance_to_nearest_v05b_km": distance_km,
                    "comparison_class": "new_v05b_only",
                    "geometry": row05b.geometry,
                }
            )

    if new_rows:
        new_v05b = pd.DataFrame(new_rows)
        new_v05b = new_v05b.merge(
            v05b_centroids,
            left_on=["nearest_v05b_rank", "nearest_v05b_hotspot_id"],
            right_on=["v05b_rank", "v05b_hotspot_id"],
            how="left",
        )
        comparison = pd.concat([comparison, new_v05b], ignore_index=True)

    out_gdf = gpd.GeoDataFrame(comparison, geometry="geometry", crs="EPSG:3310")

    log(f"Writing CSV: {OUT_CSV}")
    out_gdf.to_crs("EPSG:4326").drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log(f"Writing GeoPackage: {OUT_GPKG}")
    out_gdf.to_file(OUT_GPKG, layer="urem_hotspot_comparison_v04_v05b", driver="GPKG")

    log("Done")

    print("\nComparison summary:")
    print(out_gdf["comparison_class"].value_counts(dropna=False).to_string())

    print("\nDistance summary km:")
    print(out_gdf["distance_to_nearest_v05b_km"].describe().to_string())


if __name__ == "__main__":
    main()