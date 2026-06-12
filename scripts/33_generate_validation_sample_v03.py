#!/usr/bin/env python3
"""
Script 33: Generate Validation Sample v03

Purpose:
- Create a blind validation sample to test whether UREM v03 adds value.
- Compare:
  A = Physical Exceptionality only
  B = UREM v03
  C = Random coastal locations

Inputs:
- data/processed/urem_score_v03.gpkg

Outputs:
- data/processed/validation_sample_v03.gpkg
- data/processed/validation_sample_v03.csv
- data/processed/validation_sample_v03.kml
- data/processed/validation_scoring_template_v03.csv
"""

from pathlib import Path
import warnings
import random

import geopandas as gpd
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data/processed/urem_score_v03.gpkg"

OUT_GPKG = BASE_DIR / "data/processed/validation_sample_v03.gpkg"
OUT_CSV = BASE_DIR / "data/processed/validation_sample_v03.csv"
OUT_KML = BASE_DIR / "data/processed/validation_sample_v03.kml"
OUT_TEMPLATE = BASE_DIR / "data/processed/validation_scoring_template_v03.csv"

N_PER_GROUP = 25
RANDOM_SEED = 42


def log(msg: str) -> None:
    print(f"[33_generate_validation_sample_v03] {msg}")


def sample_physical_only(gdf: gpd.GeoDataFrame, n: int) -> gpd.GeoDataFrame:
    log("Sampling physical-exceptionality-only locations")

    sample = (
        gdf.sort_values("physical_exceptionality_score_v02", ascending=False)
        .head(n * 3)
        .sample(n=n, random_state=RANDOM_SEED)
        .copy()
    )

    sample["true_model_group"] = "physical_exceptionality_only"

    return sample


def sample_urem(gdf: gpd.GeoDataFrame, n: int) -> gpd.GeoDataFrame:
    log("Sampling UREM v03 locations")

    sample = (
        gdf.sort_values("urem_score_v03", ascending=False)
        .head(n * 3)
        .sample(n=n, random_state=RANDOM_SEED + 1)
        .copy()
    )

    sample["true_model_group"] = "urem_v03"

    return sample


def sample_random(gdf: gpd.GeoDataFrame, n: int) -> gpd.GeoDataFrame:
    log("Sampling random coastal locations")

    sample = gdf.sample(n=n, random_state=RANDOM_SEED + 2).copy()

    sample["true_model_group"] = "random_coastal"

    return sample


def build_validation_sample(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Building combined validation sample")

    physical = sample_physical_only(gdf, N_PER_GROUP)
    urem = sample_urem(gdf, N_PER_GROUP)
    random_sample = sample_random(gdf, N_PER_GROUP)

    sample = pd.concat(
        [physical, urem, random_sample],
        ignore_index=True,
    )

    sample = gpd.GeoDataFrame(sample, geometry="geometry", crs=gdf.crs)

    # Remove duplicate cells if overlap occurs.
    sample = sample.drop_duplicates(subset=["cell_id"]).copy()

    # If duplicates caused sample loss, fill from random pool.
    needed = (N_PER_GROUP * 3) - len(sample)
    if needed > 0:
        log(f"Duplicate overlap found. Adding {needed} replacement random cells.")

        replacements = (
            gdf[~gdf["cell_id"].isin(sample["cell_id"])]
            .sample(n=needed, random_state=RANDOM_SEED + 99)
            .copy()
        )

        replacements["true_model_group"] = "random_replacement"

        sample = pd.concat([sample, replacements], ignore_index=True)
        sample = gpd.GeoDataFrame(sample, geometry="geometry", crs=gdf.crs)

    # Shuffle for blind scoring.
    sample = sample.sample(frac=1, random_state=RANDOM_SEED + 100).reset_index(drop=True)

    sample["validation_sample_id"] = [
        f"VAL_V03_{i+1:03d}" for i in range(len(sample))
    ]

    # Blind group labels.
    blind_labels = ["Group A", "Group B", "Group C"]
    group_map = {
        "physical_exceptionality_only": "Group A",
        "urem_v03": "Group B",
        "random_coastal": "Group C",
        "random_replacement": "Group C",
    }

    # Shuffle label meaning so reviewer cannot infer.
    shuffled_labels = blind_labels.copy()
    random.seed(RANDOM_SEED)
    random.shuffle(shuffled_labels)

    true_groups = [
        "physical_exceptionality_only",
        "urem_v03",
        "random_coastal",
    ]

    blind_map = dict(zip(true_groups, shuffled_labels))
    blind_map["random_replacement"] = blind_map["random_coastal"]

    sample["blind_group"] = sample["true_model_group"].map(blind_map)

    return sample


def create_centroid_fields(sample: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Creating longitude/latitude centroid fields")

    centroids = sample.geometry.centroid
    sample["centroid_x_projected"] = centroids.x
    sample["centroid_y_projected"] = centroids.y

    sample_wgs = sample.copy()
    sample_wgs["geometry"] = sample_wgs.geometry.centroid
    sample_wgs = sample_wgs.to_crs("EPSG:4326")

    sample["longitude"] = sample_wgs.geometry.x
    sample["latitude"] = sample_wgs.geometry.y

    return sample


def create_scoring_template(sample: gpd.GeoDataFrame) -> pd.DataFrame:
    log("Creating manual scoring template")

    cols = [
        "validation_sample_id",
        "blind_group",
        "longitude",
        "latitude",
        "cell_id",
    ]

    template = sample[cols].copy()

    # Manual scoring fields.
    template["scenic_quality_1_5"] = ""
    template["recreation_potential_1_5"] = ""
    template["geographic_uniqueness_1_5"] = ""
    template["landscape_drama_1_5"] = ""
    template["water_coast_relationship_1_5"] = ""
    template["accessibility_1_5"] = ""
    template["existing_recognition_1_5"] = ""
    template["under_recognized_exceptionality_1_5"] = ""
    template["review_notes"] = ""

    return template


def create_kml(sample: gpd.GeoDataFrame) -> None:
    log("Writing KML")

    kml_gdf = sample.copy()
    kml_gdf["geometry"] = kml_gdf.geometry.centroid
    kml_gdf = kml_gdf.to_crs("EPSG:4326")

    kml_gdf["Name"] = kml_gdf["validation_sample_id"]
    kml_gdf["Description"] = (
        "Blind group: " + kml_gdf["blind_group"].astype(str)
        + " | Cell ID: " + kml_gdf["cell_id"].astype(str)
    )

    # Keep only simple KML fields.
    kml_gdf[["Name", "Description", "geometry"]].to_file(
        OUT_KML,
        driver="KML",
    )


def main():
    log("Starting Script 33")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    log(f"Reading input: {INPUT_PATH}")
    gdf = gpd.read_file(INPUT_PATH)

    if gdf.empty:
        raise ValueError("Input is empty")

    if gdf.crs is None:
        raise ValueError("Input has no CRS")

    required = [
        "cell_id",
        "physical_exceptionality_score_v02",
        "observed_recognition_v03",
        "expected_recognition_v03",
        "positive_under_recognition_residual_v03",
        "urem_score_v03",
    ]

    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    log(f"Rows: {len(gdf):,}")
    log(f"CRS: {gdf.crs}")

    sample = build_validation_sample(gdf)
    sample = create_centroid_fields(sample)

    template = create_scoring_template(sample)

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

    log(f"Writing GeoPackage: {OUT_GPKG}")
    sample.to_file(OUT_GPKG, driver="GPKG")

    log(f"Writing CSV: {OUT_CSV}")
    sample.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    create_kml(sample)

    log(f"Writing scoring template: {OUT_TEMPLATE}")
    template.to_csv(OUT_TEMPLATE, index=False)

    log("Done")

    print("\nValidation sample summary:")
    print(sample["true_model_group"].value_counts())

    print("\nBlind group counts:")
    print(sample["blind_group"].value_counts())

    print("\nFiles created:")
    print(OUT_GPKG)
    print(OUT_CSV)
    print(OUT_KML)
    print(OUT_TEMPLATE)


if __name__ == "__main__":
    main()