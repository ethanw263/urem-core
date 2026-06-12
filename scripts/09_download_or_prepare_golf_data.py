#!/usr/bin/env python3
"""
09_download_or_prepare_golf_data.py

Extract leisure=golf_course from local California OSM PBF using streaming osmium.
Clip to UREM coastal study area.
Output EPSG:3310 GeoPackage.
"""

from pathlib import Path
import logging
import sys

import geopandas as gpd
import pandas as pd
import osmium
import osmium.geom
from shapely import wkb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PBF_PATH = PROJECT_ROOT / "data/raw/golf/california-latest.osm.pbf"
STUDY_AREA_PATH = PROJECT_ROOT / "data/processed/study_area_25km.gpkg"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/golf_courses_ca_coastal.gpkg"

TARGET_CRS = "EPSG:3310"
SOURCE_CRS = "EPSG:4326"


def setup_logger():
    logger = logging.getLogger("09_download_or_prepare_golf_data")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


class GolfCourseHandler(osmium.SimpleHandler):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.wkb_factory = osmium.geom.WKBFactory()
        self.records = []
        self.seen = 0
        self.failed_geometry = 0

    def area(self, area):
        self.seen += 1

        tags = dict(area.tags)

        if tags.get("leisure") != "golf_course":
            return

        try:
            geom_wkb = self.wkb_factory.create_multipolygon(area)

            if isinstance(geom_wkb, str):
                geometry = wkb.loads(bytes.fromhex(geom_wkb))
            else:
                geometry = wkb.loads(geom_wkb)

        except Exception:
            self.failed_geometry += 1
            return

        self.records.append(
            {
                "osm_feature_id": str(area.id),
                "name": tags.get("name"),
                "leisure": tags.get("leisure"),
                "sport": tags.get("sport"),
                "operator": tags.get("operator"),
                "website": tags.get("website"),
                "wikidata": tags.get("wikidata"),
                "wikipedia": tags.get("wikipedia"),
                "tourism": tags.get("tourism"),
                "access": tags.get("access"),
                "geometry": geometry,
            }
        )


def main():
    logger = setup_logger()
    logger.info("Starting Script 09: Prepare local golf course data")

    if not PBF_PATH.exists():
        raise FileNotFoundError(f"Missing local OSM PBF: {PBF_PATH}")

    if not STUDY_AREA_PATH.exists():
        raise FileNotFoundError(f"Missing study area: {STUDY_AREA_PATH}")

    logger.info(f"Using local PBF: {PBF_PATH}")
    logger.info(f"Reading study area: {STUDY_AREA_PATH}")

    study_area = gpd.read_file(STUDY_AREA_PATH).to_crs(TARGET_CRS)

    if study_area.empty:
        raise ValueError("Study area is empty.")

    logger.info("Streaming local OSM PBF with osmium")
    logger.info("Extracting area features where leisure=golf_course")

    handler = GolfCourseHandler(logger)
    handler.apply_file(str(PBF_PATH), locations=True)

    logger.info(f"OSM area objects scanned: {handler.seen:,}")
    logger.info(f"Golf course records extracted: {len(handler.records):,}")
    logger.info(f"Failed golf geometries skipped: {handler.failed_geometry:,}")

    if not handler.records:
        raise ValueError("No leisure=golf_course area features extracted from PBF.")

    golf = gpd.GeoDataFrame(
        handler.records,
        geometry="geometry",
        crs=SOURCE_CRS,
    )

    golf = golf[golf.geometry.notna()]
    golf = golf[~golf.geometry.is_empty]
    golf = golf.to_crs(TARGET_CRS)

    logger.info(f"Valid golf geometries before clip: {len(golf):,}")

    logger.info("Clipping golf courses to UREM 25 km coastal study area")

    golf_clipped = gpd.clip(golf, study_area)

    golf_clipped = golf_clipped[golf_clipped.geometry.notna()]
    golf_clipped = golf_clipped[~golf_clipped.geometry.is_empty]
    golf_clipped = golf_clipped.to_crs(TARGET_CRS)

    if golf_clipped.empty:
        raise ValueError("Golf courses extracted, but none intersect the study area.")

    golf_clipped["area_m2"] = golf_clipped.geometry.area
    golf_clipped["area_km2"] = golf_clipped["area_m2"] / 1_000_000

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

    logger.info("QA summary")
    logger.info(f"Coastal golf features: {len(golf_clipped):,}")
    logger.info(f"Named features: {golf_clipped['name'].notna().sum():,}")
    logger.info(f"Unnamed features: {golf_clipped['name'].isna().sum():,}")
    logger.info(f"Total clipped golf area: {golf_clipped['area_km2'].sum():,.2f} km²")
    logger.info(f"CRS: {golf_clipped.crs}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    golf_clipped.to_file(
        OUTPUT_PATH,
        layer="golf_courses_ca_coastal",
        driver="GPKG",
    )

    logger.info(f"Saved output: {OUTPUT_PATH}")
    logger.info("Script 09 completed successfully.")


if __name__ == "__main__":
    main()