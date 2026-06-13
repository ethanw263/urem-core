#!/usr/bin/env python3
"""
54_compare_hotspot_overlap_v04_v05b.py

Compare v04 and v05b hotspots by polygon overlap.

Purpose:
- Determine whether v05b hotspots are refined cores inside v04 hotspots.
- Avoid misleading centroid-distance comparison.
- Classify v04 hotspots as:
  survived_by_overlap
  near_overlap
  lost_by_overlap

Inputs:
- data/processed/urem_hotspots_v04.gpkg
- data/processed/urem_hotspots_v05b.gpkg

Outputs:
- data/processed/urem_hotspot_overlap_v04_v05b.csv
- data/processed/urem_hotspot_overlap_v04_v05b.gpkg
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd


SCRIPT_NAME = "54_compare_hotspot_overlap_v04_v05b"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

V04_PATH = PROCESSED_DIR / "urem_hotspots_v04.gpkg"
V05B_PATH = PROCESSED_DIR / "urem_hotspots_v05b.gpkg"

OUT_CSV = PROCESSED_DIR / "urem_hotspot_overlap_v04_v05b.csv"
OUT_GPKG = PROCESSED_DIR / "urem_hotspot_overlap_v04_v05b.gpkg"


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def classify(overlap_share, distance_km):
    if overlap_share > 0:
        return "survived_by_overlap"
    if distance_km <= 10:
        return "near_overlap"
    return "lost_by_overlap"


def main():
    log("Starting hotspot overlap comparison v04 vs v05b")

    require_file(V04_PATH)
    require_file(V05B_PATH)

    v04 = gpd.read_file(V04_PATH).to_crs("EPSG:3310")
    v05b = gpd.read_file(V05B_PATH).to_crs("EPSG:3310")

    if "hotspot_rank_v04" not in v04.columns:
        v04 = v04.reset_index(drop=True)
        v04["hotspot_rank_v04"] = range(1, len(v04) + 1)

    if "hotspot_rank_v05b" not in v05b.columns:
        v05b = v05b.reset_index(drop=True)
        v05b["hotspot_rank_v05b"] = range(1, len(v05b) + 1)

    v04["v04_area_m2"] = v04.geometry.area
    v05b["v05b_area_m2"] = v05b.geometry.area

    rows = []

    for _, row04 in v04.iterrows():
        best = None

        for _, row05 in v05b.iterrows():
            intersection = row04.geometry.intersection(row05.geometry)
            overlap_m2 = intersection.area if not intersection.is_empty else 0.0

            v04_overlap_share = overlap_m2 / row04["v04_area_m2"] if row04["v04_area_m2"] else 0
            v05b_overlap_share = overlap_m2 / row05["v05b_area_m2"] if row05["v05b_area_m2"] else 0

            distance_km = row04.geometry.centroid.distance(row05.geometry.centroid) / 1000

            candidate = {
                "v04_rank": row04["hotspot_rank_v04"],
                "v04_hotspot_id": row04.get("hotspot_id"),
                "v05b_rank": row05["hotspot_rank_v05b"],
                "v05b_hotspot_id": row05.get("hotspot_id"),
                "overlap_area_km2": overlap_m2 / 1_000_000,
                "v04_overlap_share": v04_overlap_share,
                "v05b_overlap_share": v05b_overlap_share,
                "centroid_distance_km": distance_km,
                "geometry": row04.geometry,
            }

            if best is None:
                best = candidate
            else:
                if (
                    candidate["overlap_area_km2"] > best["overlap_area_km2"]
                    or (
                        candidate["overlap_area_km2"] == best["overlap_area_km2"]
                        and candidate["centroid_distance_km"] < best["centroid_distance_km"]
                    )
                ):
                    best = candidate

        best["overlap_class"] = classify(
            best["v04_overlap_share"],
            best["centroid_distance_km"],
        )

        rows.append(best)

    out = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:3310")

    log(f"Writing CSV: {OUT_CSV}")
    out.to_crs("EPSG:4326").drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log(f"Writing GeoPackage: {OUT_GPKG}")
    out.to_file(OUT_GPKG, layer="urem_hotspot_overlap_v04_v05b", driver="GPKG")

    log("Done")

    print("\nOverlap summary:")
    print(out["overlap_class"].value_counts(dropna=False).to_string())

    print("\nOverlap share summary:")
    print(out["v04_overlap_share"].describe().to_string())

    print("\nTop overlaps:")
    print(
        out[
            [
                "v04_rank",
                "v05b_rank",
                "overlap_area_km2",
                "v04_overlap_share",
                "v05b_overlap_share",
                "centroid_distance_km",
                "overlap_class",
            ]
        ]
        .sort_values(["overlap_area_km2", "v04_overlap_share"], ascending=False)
        .head(25)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
    