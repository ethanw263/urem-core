#!/usr/bin/env python3
"""
49_compare_v04_v05_hotspots.py

Compare UREM v04 and v05 hotspots.

Purpose:
- Determine whether v05 improved hotspot quality or collapsed onto one geography type.
- Compare each v04 hotspot to nearest v05 hotspot.
- Also identify v05 hotspots that are new relative to v04.

Inputs:
- data/processed/urem_hotspots_v04.gpkg
- data/processed/urem_hotspots_v05.gpkg
- data/processed/urem_hotspot_centroids_v04.csv
- data/processed/urem_hotspot_centroids_v05.csv

Outputs:
- data/processed/urem_hotspot_comparison_v04_v05.csv
- data/processed/urem_hotspot_comparison_v04_v05.gpkg
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np


SCRIPT_NAME = "49_compare_v04_v05_hotspots"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

V04_GPKG = PROCESSED_DIR / "urem_hotspots_v04.gpkg"
V05_GPKG = PROCESSED_DIR / "urem_hotspots_v05.gpkg"

V04_CSV = PROCESSED_DIR / "urem_hotspot_centroids_v04.csv"
V05_CSV = PROCESSED_DIR / "urem_hotspot_centroids_v05.csv"

OUT_CSV = PROCESSED_DIR / "urem_hotspot_comparison_v04_v05.csv"
OUT_GPKG = PROCESSED_DIR / "urem_hotspot_comparison_v04_v05.gpkg"


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def classify_distance(distance_km):
    if pd.isna(distance_km):
        return "no_v05_match"
    if distance_km <= 5:
        return "survived"
    if distance_km <= 25:
        return "near_survived"
    return "lost_in_v05"


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
    log("Starting v04 vs v05 hotspot comparison")

    for p in [V04_GPKG, V05_GPKG, V04_CSV, V05_CSV]:
        require_file(p)

    log(f"Reading v04 hotspots: {V04_GPKG}")
    v04 = gpd.read_file(V04_GPKG)

    log(f"Reading v05 hotspots: {V05_GPKG}")
    v05 = gpd.read_file(V05_GPKG)

    if v04.crs is None or v05.crs is None:
        raise ValueError("One hotspot layer has no CRS.")

    v04 = v04.to_crs("EPSG:3310")
    v05 = v05.to_crs("EPSG:3310")

    log(f"v04 hotspots: {len(v04):,}")
    log(f"v05 hotspots: {len(v05):,}")

    v04_centroids = load_centroids(V04_CSV, "v04")
    v05_centroids = load_centroids(V05_CSV, "v05")

    # Ensure rank columns exist in geometry layers.
    if "hotspot_rank_v04" not in v04.columns:
        v04 = v04.reset_index(drop=True)
        v04["hotspot_rank_v04"] = range(1, len(v04) + 1)

    if "hotspot_rank_v05" not in v05.columns:
        v05 = v05.reset_index(drop=True)
        v05["hotspot_rank_v05"] = range(1, len(v05) + 1)

    # Use centroids for nearest-neighbor distance.
    v04_pts = v04.copy()
    v04_pts["geometry"] = v04_pts.geometry.centroid

    v05_pts = v05.copy()
    v05_pts["geometry"] = v05_pts.geometry.centroid

    rows = []

    for _, row04 in v04_pts.iterrows():
        distances_m = v05_pts.geometry.distance(row04.geometry)

        nearest_idx = distances_m.idxmin()
        nearest_v05 = v05_pts.loc[nearest_idx]
        distance_km = float(distances_m.loc[nearest_idx] / 1000)

        rows.append(
            {
                "v04_rank": row04.get("hotspot_rank_v04"),
                "v04_hotspot_id": row04.get("hotspot_id"),
                "nearest_v05_rank": nearest_v05.get("hotspot_rank_v05"),
                "nearest_v05_hotspot_id": nearest_v05.get("hotspot_id"),
                "distance_to_nearest_v05_km": distance_km,
                "comparison_class": classify_distance(distance_km),
                "v04_geometry": row04.geometry,
            }
        )

    comparison = pd.DataFrame(rows)

    comparison = comparison.merge(v04_centroids, on=["v04_rank", "v04_hotspot_id"], how="left")
    comparison = comparison.merge(
        v05_centroids,
        left_on=["nearest_v05_rank", "nearest_v05_hotspot_id"],
        right_on=["v05_rank", "v05_hotspot_id"],
        how="left",
    )

    # Add new-v05-only rows: v05 hotspots not near any v04 hotspot.
    v05_rows = []
    for _, row05 in v05_pts.iterrows():
        distances_m = v04_pts.geometry.distance(row05.geometry)
        nearest_idx = distances_m.idxmin()
        nearest_v04 = v04_pts.loc[nearest_idx]
        distance_km = float(distances_m.loc[nearest_idx] / 1000)

        if distance_km > 25:
            v05_rows.append(
                {
                    "v04_rank": np.nan,
                    "v04_hotspot_id": np.nan,
                    "nearest_v05_rank": row05.get("hotspot_rank_v05"),
                    "nearest_v05_hotspot_id": row05.get("hotspot_id"),
                    "distance_to_nearest_v05_km": distance_km,
                    "comparison_class": "new_v05_only",
                    "v04_geometry": row05.geometry,
                }
            )

    if v05_rows:
        new_v05 = pd.DataFrame(v05_rows)
        new_v05 = new_v05.merge(
            v05_centroids,
            left_on=["nearest_v05_rank", "nearest_v05_hotspot_id"],
            right_on=["v05_rank", "v05_hotspot_id"],
            how="left",
        )
        comparison = pd.concat([comparison, new_v05], ignore_index=True)

    out_gdf = gpd.GeoDataFrame(
        comparison.drop(columns=["v04_geometry"]),
        geometry=comparison["v04_geometry"],
        crs="EPSG:3310",
    )

    out_gdf_wgs = out_gdf.to_crs("EPSG:4326")

    log(f"Writing CSV: {OUT_CSV}")
    out_gdf_wgs.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log(f"Writing GeoPackage: {OUT_GPKG}")
    out_gdf.to_file(OUT_GPKG, layer="urem_hotspot_comparison_v04_v05", driver="GPKG")

    log("Done")

    print("\nComparison summary:")
    print(out_gdf["comparison_class"].value_counts(dropna=False).to_string())

    print("\nDistance summary km:")
    print(out_gdf["distance_to_nearest_v05_km"].describe().to_string())


if __name__ == "__main__":
    main()