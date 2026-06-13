#!/usr/bin/env python3
"""
Script 51: Generate UREM Hotspots v05b

Purpose:
- Convert ranked UREM v05b candidate cells into spatial hotspots.
- v05b is a hybrid model: v04 recognition/residuals + v03 refinement + v05b score.
"""

from pathlib import Path
import warnings

import geopandas as gpd
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

SCRIPT_NAME = "51_generate_urem_hotspots_v05b"

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data/processed/ranked_urem_candidates_v05b.gpkg"

OUT_HOTSPOTS_GPKG = BASE_DIR / "data/processed/urem_hotspots_v05b.gpkg"
OUT_CENTROIDS_CSV = BASE_DIR / "data/processed/urem_hotspot_centroids_v05b.csv"
OUT_RANKINGS_CSV = BASE_DIR / "data/processed/urem_hotspot_rankings_v05b.csv"

TOP_PERCENTILE_THRESHOLD = 0.975
BUFFER_M = 1500
MIN_CELLS_PER_HOTSPOT = 2


def log(msg: str) -> None:
    print(f"[{SCRIPT_NAME}] {msg}")


def select_top_candidates(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Selecting top UREM v05b candidate cells")

    if "is_valid_land_candidate" in gdf.columns:
        valid = gdf["is_valid_land_candidate"].astype(str).str.lower().isin(["true", "1"])
        gdf = gdf[valid].copy()

    if "land_area_share" in gdf.columns:
        gdf = gdf[gdf["land_area_share"] >= 0.50].copy()

    if "urem_percentile_v05b" in gdf.columns:
        selected = gdf[gdf["urem_percentile_v05b"] >= TOP_PERCENTILE_THRESHOLD].copy()
    else:
        cutoff = gdf["urem_score_v05b"].quantile(TOP_PERCENTILE_THRESHOLD)
        selected = gdf[gdf["urem_score_v05b"] >= cutoff].copy()

    log(f"Selected top cells: {len(selected):,}")

    if selected.empty:
        raise ValueError("No top candidate cells selected")

    return selected


def build_hotspots(selected: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Building buffered hotspot clusters")

    working = selected.copy()
    working["geometry"] = working.geometry.buffer(BUFFER_M)

    dissolved = working.dissolve()
    exploded = dissolved.explode(index_parts=False).reset_index(drop=True)

    hotspots = gpd.GeoDataFrame(
        {"hotspot_id": range(1, len(exploded) + 1)},
        geometry=exploded.geometry,
        crs=selected.crs,
    )

    log(f"Initial hotspot polygons: {len(hotspots):,}")

    return hotspots


def attach_cells_to_hotspots(
    hotspots: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    log("Attaching candidate cells to hotspots")

    joined = gpd.sjoin(
        candidates,
        hotspots[["hotspot_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    if joined.empty:
        raise ValueError("No candidate cells joined to hotspots")

    grouped = joined.groupby("hotspot_id")

    stats = grouped.agg(
        cell_count=("cell_id", "count"),
        mean_urem_score_v05b=("urem_score_v05b", "mean"),
        max_urem_score_v05b=("urem_score_v05b", "max"),

        mean_exceptionality_v03=("physical_exceptionality_v03", "mean"),
        max_exceptionality_v03=("physical_exceptionality_v03", "max"),

        mean_observed_recognition_v04=("observed_recognition_v04", "mean"),
        mean_expected_recognition_v05b=("expected_recognition_v04", "mean"),

        mean_under_recognition_residual_v05b=(
            "positive_under_recognition_residual_v04",
            "mean",
        ),
        max_under_recognition_residual_v05b=(
            "positive_under_recognition_residual_v04",
            "max",
        ),

        mean_expected_recognition_confidence_v05b=(
            "comparable_confidence_v04",
            "mean",
        ),
        mean_recognition_confidence_v04=("recognition_cell_confidence_v04", "mean"),
        mean_land_area_share=("land_area_share", "mean"),

        mean_terrain_drama_v03=("terrain_drama_v03", "mean"),
        mean_scenic_coast_v03=("scenic_coast_v03", "mean"),
        mean_flat_coastal_edge_penalty_v03=(
            "flat_coastal_edge_penalty_v03",
            "mean",
        ),
        mean_complex_flat_shoreline_penalty_v03=(
            "complex_flat_shoreline_penalty_v03",
            "mean",
        ),

        mean_v03_refinement_bonus_v05b=("v03_refinement_bonus_v05b", "mean"),
        mean_v03_flat_edge_penalty_v05b=("v03_flat_edge_penalty_v05b", "mean"),

        best_cell_id=("cell_id", "first"),
    ).reset_index()

    hotspots = hotspots.merge(stats, on="hotspot_id", how="left")
    hotspots = hotspots[hotspots["cell_count"] >= MIN_CELLS_PER_HOTSPOT].copy()

    log(f"Hotspots after minimum cell filter: {len(hotspots):,}")

    if hotspots.empty:
        raise ValueError("No hotspots remain after filtering")

    return hotspots


def rank_hotspots(hotspots: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Ranking hotspots")

    hotspots["hotspot_area_m2"] = hotspots.geometry.area
    hotspots["hotspot_area_km2"] = hotspots["hotspot_area_m2"] / 1_000_000
    hotspots["hotspot_support_score"] = np.log1p(hotspots["cell_count"])

    hotspots["hotspot_score_v05b_raw"] = (
        0.40 * hotspots["max_urem_score_v05b"]
        + 0.25 * hotspots["mean_urem_score_v05b"]
        + 0.15 * hotspots["mean_exceptionality_v03"]
        + 0.10 * hotspots["mean_under_recognition_residual_v05b"]
        + 0.05 * hotspots["mean_v03_refinement_bonus_v05b"]
        + 0.05 * hotspots["hotspot_support_score"]
        - 0.05 * hotspots["mean_v03_flat_edge_penalty_v05b"]
    )

    mn = hotspots["hotspot_score_v05b_raw"].min()
    mx = hotspots["hotspot_score_v05b_raw"].max()

    if mx == mn:
        hotspots["hotspot_score_v05b"] = 1.0
    else:
        hotspots["hotspot_score_v05b"] = (
            (hotspots["hotspot_score_v05b_raw"] - mn) / (mx - mn)
        )

    hotspots = hotspots.sort_values("hotspot_score_v05b", ascending=False).reset_index(drop=True)
    hotspots["hotspot_rank_v05b"] = range(1, len(hotspots) + 1)

    return hotspots


def create_centroid_outputs(hotspots: gpd.GeoDataFrame) -> pd.DataFrame:
    log("Creating centroid output")

    centroids = hotspots.copy()
    centroids["centroid_geometry"] = centroids.geometry.centroid
    centroids_wgs = centroids.set_geometry("centroid_geometry").to_crs("EPSG:4326")

    out = pd.DataFrame(
        {
            "hotspot_rank_v05b": centroids_wgs["hotspot_rank_v05b"],
            "hotspot_id": centroids_wgs["hotspot_id"],
            "longitude": centroids_wgs.geometry.x,
            "latitude": centroids_wgs.geometry.y,
            "cell_count": centroids_wgs["cell_count"],
            "hotspot_area_km2": centroids_wgs["hotspot_area_km2"],
            "hotspot_score_v05b": centroids_wgs["hotspot_score_v05b"],
            "mean_urem_score_v05b": centroids_wgs["mean_urem_score_v05b"],
            "max_urem_score_v05b": centroids_wgs["max_urem_score_v05b"],
            "mean_exceptionality_v03": centroids_wgs["mean_exceptionality_v03"],
            "mean_observed_recognition_v04": centroids_wgs["mean_observed_recognition_v04"],
            "mean_expected_recognition_v05b": centroids_wgs["mean_expected_recognition_v05b"],
            "mean_under_recognition_residual_v05b": centroids_wgs[
                "mean_under_recognition_residual_v05b"
            ],
            "mean_expected_recognition_confidence_v05b": centroids_wgs[
                "mean_expected_recognition_confidence_v05b"
            ],
            "mean_recognition_confidence_v04": centroids_wgs[
                "mean_recognition_confidence_v04"
            ],
            "mean_land_area_share": centroids_wgs["mean_land_area_share"],
            "mean_terrain_drama_v03": centroids_wgs["mean_terrain_drama_v03"],
            "mean_scenic_coast_v03": centroids_wgs["mean_scenic_coast_v03"],
            "mean_flat_coastal_edge_penalty_v03": centroids_wgs[
                "mean_flat_coastal_edge_penalty_v03"
            ],
            "mean_complex_flat_shoreline_penalty_v03": centroids_wgs[
                "mean_complex_flat_shoreline_penalty_v03"
            ],
            "mean_v03_refinement_bonus_v05b": centroids_wgs[
                "mean_v03_refinement_bonus_v05b"
            ],
            "mean_v03_flat_edge_penalty_v05b": centroids_wgs[
                "mean_v03_flat_edge_penalty_v05b"
            ],
            "best_cell_id": centroids_wgs["best_cell_id"],
        }
    )

    return out


def main():
    log("Starting Script 51")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    log(f"Reading candidates: {INPUT_PATH}")
    candidates = gpd.read_file(INPUT_PATH)

    if candidates.empty:
        raise ValueError("Candidate input is empty")

    if candidates.crs is None:
        raise ValueError("Candidate input has no CRS")

    log(f"Candidate rows: {len(candidates):,}")
    log(f"CRS: {candidates.crs}")

    selected = select_top_candidates(candidates)
    hotspots = build_hotspots(selected)
    hotspots = attach_cells_to_hotspots(hotspots, selected)
    hotspots = rank_hotspots(hotspots)

    centroids = create_centroid_outputs(hotspots)

    OUT_HOTSPOTS_GPKG.parent.mkdir(parents=True, exist_ok=True)

    log(f"Writing hotspots GeoPackage: {OUT_HOTSPOTS_GPKG}")
    hotspots.to_file(OUT_HOTSPOTS_GPKG, driver="GPKG")

    log(f"Writing centroid CSV: {OUT_CENTROIDS_CSV}")
    centroids.to_csv(OUT_CENTROIDS_CSV, index=False)

    rankings = hotspots.drop(columns="geometry").copy()

    log(f"Writing ranking CSV: {OUT_RANKINGS_CSV}")
    rankings.to_csv(OUT_RANKINGS_CSV, index=False)

    log("Done")

    print("\nHotspot v05b summary:")
    print(
        hotspots[
            [
                "hotspot_rank_v05b",
                "cell_count",
                "hotspot_area_km2",
                "hotspot_score_v05b",
                "mean_urem_score_v05b",
                "max_urem_score_v05b",
                "mean_exceptionality_v03",
                "mean_under_recognition_residual_v05b",
                "mean_land_area_share",
            ]
        ].head(25).to_string(index=False)
    )


if __name__ == "__main__":
    main()