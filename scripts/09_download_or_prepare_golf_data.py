#!/usr/bin/env python3
"""
09_download_or_prepare_golf_data.py

UREM Phase: Recognition Gap v0.1 prep

Purpose:
- Use local California OSM PBF extract.
- Extract OSM features tagged leisure=golf_course.
- Clip to UREM 25 km coastal study area.
- Save coastal golf course layer for recognition modeling.

Inputs:
- data/raw/golf/california-latest.osm.pbf
- data/processed/study_area_25km.gpkg

Output:
- data/processed/golf_courses_ca_coastal.gpkg

CRS:
- EPSG:3310
"""

from pathlib import Path
import logging
import sys

import geopandas as gpd
import pandas as pd
from pyrosm import OSM


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PBF_PATH = PROJECT_ROOT / "data" / "raw" / "golf" / "california-latest.osm.pbf"
STUDY_AREA_PATH = PROJECT_ROOT / "data" / "processed" / "study_area_25km.gpkg"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "golf_courses_ca_coastal.gpkg"

TARGET_CRS = "EPSG:3310"
OUTPUT_LAYER = "golf_courses_ca_coastal"


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("09_download_or_prepare_golf_data")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
    )

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]

    # Fix invalid polygons where possible.
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]

    return gdf


def main() -> None:
    logger = setup_logging()

    logger.info("Starting Script 09: Prepare local golf course data")

    require_file(PBF_PATH, "local California OSM PBF")
    require_file(STUDY_AREA_PATH, "study area GeoPackage")

    logger.info(f"Using local PBF: {PBF_PATH}")
    logger.info(f"Reading study area: {STUDY_AREA_PATH}")

    study_area = gpd.read_file(STUDY_AREA_PATH)

    if study_area.empty:
        raise ValueError("Study area file loaded but contains no features.")

    study_area = study_area.to_crs(TARGET_CRS)
    study_area = clean_geometries(study_area)

    study_union = study_area.geometry.union_all()

    logger.info("Extracting OSM features where leisure=golf_course")

    osm = OSM(str(PBF_PATH))

    golf = osm.get_data_by_custom_criteria(
        custom_filter={"leisure": ["golf_course"]},
        filter_type="keep",
        keep_nodes=False,
        keep_ways=True,
        keep_relations=True,
        extra_tags=[
            "name",
            "leisure",
            "sport",
            "operator",
            "website",
            "wikidata",
            "wikipedia",
            "tourism",
            "access",
        ],
    )

    if golf is None or golf.empty:
        raise ValueError("No leisure=golf_course features found in local PBF.")

    logger.info(f"Raw golf features extracted: {len(golf):,}")

    golf = clean_geometries(golf)

    if golf.crs is None:
        logger.info("Input golf CRS missing; assuming EPSG:4326 from OSM.")
        golf = golf.set_crs("EPSG:4326")

    golf = golf.to_crs(TARGET_CRS)

    # Keep polygonal/area-like features only.
    golf = golf[
        golf.geometry.geom_type.isin(
            ["Polygon", "MultiPolygon", "GeometryCollection"]
        )
    ].copy()

    golf = clean_geometries(golf)

    logger.info(f"Golf polygon/area features retained: {len(golf):,}")

    logger.info("Clipping golf courses to 25 km coastal study area")

    golf_clipped = gpd.clip(golf, study_area)
    golf_clipped = clean_geometries(golf_clipped)
    golf_clipped = golf_clipped.to_crs(TARGET_CRS)

    if golf_clipped.empty:
        raise ValueError("Golf courses extracted, but none intersect the study area.")

    # Area QA.
    golf_clipped["area_m2"] = golf_clipped.geometry.area
    golf_clipped["area_km2"] = golf_clipped["area_m2"] / 1_000_000

    # Stable feature ID.
    if "id" in golf_clipped.columns:
        golf_clipped["osm_feature_id"] = golf_clipped["id"].astype(str)
    elif "osm_id" in golf_clipped.columns:
        golf_clipped["osm_feature_id"] = golf_clipped["osm_id"].astype(str)
    else:
        golf_clipped["osm_feature_id"] = [
            f"golf_{i + 1}" for i in range(len(golf_clipped))
        ]

    # Normalize name.
    if "name" not in golf_clipped.columns:
        golf_clipped["name"] = pd.NA

    keep_cols = [
        "osm_feature_id",
        "name",
        "leisure",
        "sport",
        "operator",
        "website",
        "wikidata",
        "wikipedia",
        "tourism",
        "access",
        "area_m2",
        "area_km2",
        "geometry",
    ]

    for col in keep_cols:
        if col not in golf_clipped.columns:
            golf_clipped[col] = pd.NA

    golf_clipped = golf_clipped[keep_cols].copy()

    # QA checks.
    total_area_km2 = golf_clipped["area_km2"].sum()
    named_count = golf_clipped["name"].notna().sum()
    unnamed_count = golf_clipped["name"].isna().sum()

    logger.info("QA summary")
    logger.info(f"Coastal golf features: {len(golf_clipped):,}")
    logger.info(f"Named features: {named_count:,}")
    logger.info(f"Unnamed features: {unnamed_count:,}")
    logger.info(f"Total clipped golf area: {total_area_km2:,.2f} km²")
    logger.info(f"CRS: {golf_clipped.crs}")

    if len(golf_clipped) < 10:
        logger.warning("Very few golf courses found. Review extraction/clipping.")
    if total_area_km2 <= 0:
        raise ValueError("Golf area is zero after clipping.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    golf_clipped.to_file(
        OUTPUT_PATH,
        layer=OUTPUT_LAYER,
        driver="GPKG",
    )

    logger.info(f"Saved output: {OUTPUT_PATH}")
    logger.info("Script 09 completed successfully.")


if __name__ == "__main__":
    main()