#!/usr/bin/env python3
"""
17_cluster_urem_hotspots_v01.py

Clusters top-ranked residual-based UREM candidate cells into hotspot polygons.

Input:
- data/processed/ranked_urem_algorithm_candidates_v01.gpkg

Outputs:
- data/processed/urem_hotspots_v01.gpkg
- outputs/validation/urem_hotspots_v01.csv
- outputs/validation/urem_hotspots_v01.kml

Purpose:
Instead of manually reviewing 500 individual cells, this script groups adjacent
candidate cells into larger hotspot regions for QGIS / Google Earth validation.
"""

from pathlib import Path
import logging
import sys

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data/processed/ranked_urem_algorithm_candidates_v01.gpkg"

OUTPUT_GPKG_PATH = PROJECT_ROOT / "data/processed/urem_hotspots_v01.gpkg"
OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs/validation/urem_hotspots_v01.csv"
OUTPUT_KML_PATH = PROJECT_ROOT / "outputs/validation/urem_hotspots_v01.kml"

TARGET_CRS = "EPSG:3310"
WGS84 = "EPSG:4326"

# Small buffer helps cells that touch at edges/corners cluster together.
CLUSTER_BUFFER_M = 5

MIN_CELLS_PER_HOTSPOT = 2


def setup_logger():
    logger = logging.getLogger("17_cluster_urem_hotspots_v01")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def main():
    logger = setup_logger()
    logger.info("Starting Script 17: Cluster UREM hotspots v0.1")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    logger.info(f"Reading ranked candidates: {INPUT_PATH}")

    candidates = gpd.read_file(INPUT_PATH).to_crs(TARGET_CRS)

    if candidates.empty:
        raise ValueError("Ranked candidates layer is empty.")

    required_cols = [
        "urem_algorithm_rank_v01",
        "urem_score_v01_norm",
        "urem_score_v01",
        "positive_under_recognition_residual_v01",
        "expected_recognition_v01",
        "observed_recognition_v0",
        "physical_potential_v01",
        "comparable_confidence_v01",
        "geometry",
    ]

    missing = [c for c in required_cols if c not in candidates.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Candidate cells loaded: {len(candidates):,}")
    logger.info(f"Input CRS: {candidates.crs}")

    candidates = candidates.copy()
    candidates["candidate_cell_area_m2"] = candidates.geometry.area

    logger.info("Buffering cells slightly for connected-component clustering")

    buffered = candidates.copy()
    buffered["geometry"] = buffered.geometry.buffer(CLUSTER_BUFFER_M)

    logger.info("Dissolving connected candidate cells")

    dissolved_geom = buffered.geometry.union_all()

    hotspot_parts = gpd.GeoDataFrame(
        geometry=[dissolved_geom],
        crs=TARGET_CRS,
    ).explode(index_parts=False).reset_index(drop=True)

    hotspot_parts["geometry"] = hotspot_parts.geometry.buffer(-CLUSTER_BUFFER_M)

    hotspot_parts = hotspot_parts[hotspot_parts.geometry.notna()]
    hotspot_parts = hotspot_parts[~hotspot_parts.geometry.is_empty].copy()

    hotspot_parts["hotspot_temp_id"] = range(1, len(hotspot_parts) + 1)

    logger.info(f"Raw hotspot polygons found: {len(hotspot_parts):,}")

    logger.info("Assigning candidate cells to hotspots")

    candidate_points = candidates.copy()
    candidate_points["geometry"] = candidate_points.geometry.centroid

    joined = gpd.sjoin(
        candidate_points,
        hotspot_parts[["hotspot_temp_id", "geometry"]],
        how="left",
        predicate="within",
    )

    if joined["hotspot_temp_id"].isna().any():
        logger.warning("Some candidate centroids were not assigned to hotspots.")

    joined = joined.dropna(subset=["hotspot_temp_id"]).copy()
    joined["hotspot_temp_id"] = joined["hotspot_temp_id"].astype(int)

    logger.info("Computing hotspot statistics")

    stats = (
        joined.groupby("hotspot_temp_id")
        .agg(
            cell_count=("urem_algorithm_rank_v01", "count"),
            best_rank=("urem_algorithm_rank_v01", "min"),
            mean_rank=("urem_algorithm_rank_v01", "mean"),
            max_urem_score_norm=("urem_score_v01_norm", "max"),
            mean_urem_score_norm=("urem_score_v01_norm", "mean"),
            max_urem_score=("urem_score_v01", "max"),
            mean_urem_score=("urem_score_v01", "mean"),
            mean_under_recognition_residual=(
                "positive_under_recognition_residual_v01",
                "mean",
            ),
            max_under_recognition_residual=(
                "positive_under_recognition_residual_v01",
                "max",
            ),
            mean_expected_recognition=("expected_recognition_v01", "mean"),
            mean_observed_recognition=("observed_recognition_v0", "mean"),
            mean_physical_potential=("physical_potential_v01", "mean"),
            max_physical_potential=("physical_potential_v01", "max"),
            mean_comparable_confidence=("comparable_confidence_v01", "mean"),
        )
        .reset_index()
    )

    hotspots = hotspot_parts.merge(stats, on="hotspot_temp_id", how="left")

    hotspots = hotspots[hotspots["cell_count"] >= MIN_CELLS_PER_HOTSPOT].copy()

    if hotspots.empty:
        logger.warning(
            "No hotspots met MIN_CELLS_PER_HOTSPOT. Keeping all raw hotspots instead."
        )
        hotspots = hotspot_parts.merge(stats, on="hotspot_temp_id", how="left")

    hotspots["area_m2"] = hotspots.geometry.area
    hotspots["area_km2"] = hotspots["area_m2"] / 1_000_000

    hotspots = hotspots.sort_values(
        by=[
            "max_urem_score_norm",
            "mean_urem_score_norm",
            "cell_count",
            "mean_physical_potential",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    hotspots["hotspot_rank_v01"] = hotspots.index + 1
    hotspots["hotspot_id"] = hotspots["hotspot_rank_v01"].apply(
        lambda x: f"UREM-HS-{x:03d}"
    )

    centroids_wgs84 = hotspots.geometry.centroid.to_crs(WGS84)
    hotspots["centroid_lon"] = centroids_wgs84.x
    hotspots["centroid_lat"] = centroids_wgs84.y

    # Reorder useful columns.
    preferred_cols = [
        "hotspot_id",
        "hotspot_rank_v01",
        "cell_count",
        "area_km2",
        "centroid_lat",
        "centroid_lon",
        "best_rank",
        "mean_rank",
        "max_urem_score_norm",
        "mean_urem_score_norm",
        "max_urem_score",
        "mean_urem_score",
        "max_under_recognition_residual",
        "mean_under_recognition_residual",
        "mean_expected_recognition",
        "mean_observed_recognition",
        "max_physical_potential",
        "mean_physical_potential",
        "mean_comparable_confidence",
        "geometry",
    ]

    existing_cols = [c for c in preferred_cols if c in hotspots.columns]
    hotspots = hotspots[existing_cols].copy()

    logger.info("QA summary")
    logger.info(f"Candidate cells input: {len(candidates):,}")
    logger.info(f"Hotspots output: {len(hotspots):,}")
    logger.info(f"Minimum cells per hotspot: {MIN_CELLS_PER_HOTSPOT}")
    logger.info(f"Largest hotspot cell count: {hotspots['cell_count'].max():,}")
    logger.info(f"Best hotspot max score: {hotspots['max_urem_score_norm'].max():.4f}")
    logger.info(f"Output CRS: {hotspots.crs}")

    OUTPUT_GPKG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_KML_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_GPKG_PATH.exists():
        OUTPUT_GPKG_PATH.unlink()

    hotspots.to_file(
        OUTPUT_GPKG_PATH,
        layer="urem_hotspots_v01",
        driver="GPKG",
    )

    csv_cols = [c for c in hotspots.columns if c != "geometry"]
    hotspots[csv_cols].to_csv(OUTPUT_CSV_PATH, index=False)

    # KML must be WGS84.
    hotspots_kml = hotspots.to_crs(WGS84).copy()

    # KML supports limited fields, so keep concise display fields.
    kml_cols = [
        "hotspot_id",
        "hotspot_rank_v01",
        "cell_count",
        "area_km2",
        "max_urem_score_norm",
        "mean_urem_score_norm",
        "mean_physical_potential",
        "mean_expected_recognition",
        "mean_observed_recognition",
        "geometry",
    ]

    kml_cols = [c for c in kml_cols if c in hotspots_kml.columns]
    hotspots_kml = hotspots_kml[kml_cols].copy()

    if OUTPUT_KML_PATH.exists():
        OUTPUT_KML_PATH.unlink()

    hotspots_kml.to_file(
        OUTPUT_KML_PATH,
        driver="KML",
    )

    logger.info(f"Saved GeoPackage: {OUTPUT_GPKG_PATH}")
    logger.info(f"Saved CSV: {OUTPUT_CSV_PATH}")
    logger.info(f"Saved KML: {OUTPUT_KML_PATH}")
    logger.info("Script 17 completed successfully.")


if __name__ == "__main__":
    main()