#!/usr/bin/env python3
"""
18_create_validation_sample_v01.py

Creates a manual validation sample comparing:

A. Top UREM algorithm hotspots
B. Top physical-potential-only cells
C. Random coastal cells

Output:
- outputs/validation/urem_validation_sample_v01.csv
- data/processed/urem_validation_sample_v01.gpkg

Purpose:
Help determine whether UREM is finding genuinely plausible candidates
or simply producing artifacts from golf-only recognition.
"""

from pathlib import Path
import logging
import sys

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

UREM_HOTSPOTS_PATH = PROJECT_ROOT / "data/processed/urem_hotspots_v01.gpkg"
PHYSICAL_GRID_PATH = PROJECT_ROOT / "data/processed/coastal_grid_physical_potential_v01.gpkg"
RANDOM_GRID_PATH = PROJECT_ROOT / "data/processed/coastal_grid_urem_algorithm_v01.gpkg"

OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs/validation/urem_validation_sample_v01.csv"
OUTPUT_GPKG_PATH = PROJECT_ROOT / "data/processed/urem_validation_sample_v01.gpkg"

TARGET_CRS = "EPSG:3310"
WGS84 = "EPSG:4326"

N_UREM = 50
N_PHYSICAL = 50
N_RANDOM = 50

RANDOM_SEED = 42


def setup_logger():
    logger = logging.getLogger("18_create_validation_sample_v01")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def add_centroids(gdf):
    centroids = gdf.geometry.centroid.to_crs(WGS84)
    gdf["centroid_lon"] = centroids.x
    gdf["centroid_lat"] = centroids.y
    return gdf


def main():
    logger = setup_logger()
    logger.info("Starting Script 18: Create UREM validation sample")

    for path in [UREM_HOTSPOTS_PATH, PHYSICAL_GRID_PATH, RANDOM_GRID_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    logger.info("Reading UREM hotspots")
    urem = gpd.read_file(UREM_HOTSPOTS_PATH).to_crs(TARGET_CRS)

    logger.info("Reading physical potential grid")
    physical = gpd.read_file(PHYSICAL_GRID_PATH).to_crs(TARGET_CRS)

    logger.info("Reading UREM algorithm grid")
    random_base = gpd.read_file(RANDOM_GRID_PATH).to_crs(TARGET_CRS)

    if urem.empty:
        raise ValueError("UREM hotspot file is empty.")

    if physical.empty:
        raise ValueError("Physical potential grid is empty.")

    if random_base.empty:
        raise ValueError("Random base grid is empty.")

    # -------------------------
    # A. UREM hotspot sample
    # -------------------------
    required_urem_cols = [
        "hotspot_rank_v01",
        "hotspot_id",
        "max_urem_score_norm",
        "mean_urem_score_norm",
        "mean_physical_potential",
    ]

    missing_urem = [c for c in required_urem_cols if c not in urem.columns]
    if missing_urem:
        raise ValueError(f"Missing required UREM hotspot columns: {missing_urem}")

    urem_sample = (
        urem.sort_values("hotspot_rank_v01", ascending=True)
        .head(N_UREM)
        .copy()
    )

    urem_sample["validation_group"] = "UREM_hotspot"
    urem_sample["validation_rank"] = range(1, len(urem_sample) + 1)

    # -------------------------
    # B. Physical-only sample
    # -------------------------
    if "physical_potential_v01" not in physical.columns:
        raise ValueError("Missing physical_potential_v01 in physical grid.")

    physical_sample = (
        physical.sort_values("physical_potential_v01", ascending=False)
        .head(N_PHYSICAL)
        .copy()
    )

    physical_sample["validation_group"] = "Physical_only"
    physical_sample["validation_rank"] = range(1, len(physical_sample) + 1)

    # -------------------------
    # C. Random sample
    # -------------------------
    random_sample = random_base.sample(
        n=min(N_RANDOM, len(random_base)),
        random_state=RANDOM_SEED,
    ).copy()

    random_sample["validation_group"] = "Random"
    random_sample["validation_rank"] = range(1, len(random_sample) + 1)

    # -------------------------
    # Standardize columns
    # -------------------------
    common_cols = [
        "validation_group",
        "validation_rank",
        "hotspot_id",
        "hotspot_rank_v01",
        "cell_id",
        "physical_potential_v01",
        "max_urem_score_norm",
        "mean_urem_score_norm",
        "urem_score_v01_norm",
        "expected_recognition_v01",
        "observed_recognition_v0",
        "recognition_residual_v01",
        "positive_under_recognition_residual_v01",
        "coastal_proximity_score",
        "elevation_score",
        "relief_score",
        "slope_score",
        "golf_recognition_score",
        "golf_area_km2",
        "has_golf",
        "geometry",
    ]

    samples = []

    for sample in [urem_sample, physical_sample, random_sample]:
        for col in common_cols:
            if col not in sample.columns:
                sample[col] = pd.NA
        samples.append(sample[common_cols].copy())

    validation = pd.concat(samples, ignore_index=True)
    validation = gpd.GeoDataFrame(validation, geometry="geometry", crs=TARGET_CRS)

    validation = add_centroids(validation)

    # Manual review fields.
    validation["manual_place_name"] = ""
    validation["manual_review_status"] = "unreviewed"

    validation["scenic_quality_0_3"] = ""
    validation["recreation_potential_0_3"] = ""
    validation["existing_recognition_0_3"] = ""
    validation["expert_plausibility_0_3"] = ""

    validation["false_positive_category"] = ""
    validation["notes"] = ""

    validation["google_maps_query"] = (
        validation["centroid_lat"].round(6).astype(str)
        + ","
        + validation["centroid_lon"].round(6).astype(str)
    )

    # Reorder for CSV review.
    csv_cols = [
        "validation_group",
        "validation_rank",
        "hotspot_id",
        "hotspot_rank_v01",
        "cell_id",
        "centroid_lat",
        "centroid_lon",
        "google_maps_query",
        "manual_place_name",
        "manual_review_status",
        "scenic_quality_0_3",
        "recreation_potential_0_3",
        "existing_recognition_0_3",
        "expert_plausibility_0_3",
        "false_positive_category",
        "notes",
        "physical_potential_v01",
        "max_urem_score_norm",
        "mean_urem_score_norm",
        "urem_score_v01_norm",
        "expected_recognition_v01",
        "observed_recognition_v0",
        "recognition_residual_v01",
        "positive_under_recognition_residual_v01",
        "coastal_proximity_score",
        "elevation_score",
        "relief_score",
        "slope_score",
        "golf_recognition_score",
        "golf_area_km2",
        "has_golf",
    ]

    csv_cols = [c for c in csv_cols if c in validation.columns]

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_GPKG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_GPKG_PATH.exists():
        OUTPUT_GPKG_PATH.unlink()

    validation.to_file(
        OUTPUT_GPKG_PATH,
        layer="urem_validation_sample_v01",
        driver="GPKG",
    )

    validation[csv_cols].to_csv(OUTPUT_CSV_PATH, index=False)

    logger.info("QA summary")
    logger.info(f"UREM hotspot sample: {len(urem_sample):,}")
    logger.info(f"Physical-only sample: {len(physical_sample):,}")
    logger.info(f"Random sample: {len(random_sample):,}")
    logger.info(f"Total validation rows: {len(validation):,}")
    logger.info(f"Saved GeoPackage: {OUTPUT_GPKG_PATH}")
    logger.info(f"Saved CSV: {OUTPUT_CSV_PATH}")
    logger.info("Script 18 completed successfully.")


if __name__ == "__main__":
    main()