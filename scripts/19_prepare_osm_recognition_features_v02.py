#!/usr/bin/env python3
"""
19_prepare_osm_recognition_features_v02.py

Fast point-only OSM recognition feature extraction.

Input:
- data/raw/golf/california-latest.osm.pbf
- data/processed/study_area_25km.gpkg

Output:
- data/processed/osm_recognition_features_v02.gpkg

Extracts POINT features:
- tourism=viewpoint
- tourism=attraction
- tourism=camp_site
- tourism=picnic_site
- tourism=information
- natural=peak
"""

from pathlib import Path
import logging
import sys

import geopandas as gpd
import osmium
from shapely.geometry import Point
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PBF_PATH = PROJECT_ROOT / "data/raw/golf/california-latest.osm.pbf"
STUDY_AREA_PATH = PROJECT_ROOT / "data/processed/study_area_25km.gpkg"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/osm_recognition_features_v02.gpkg"

TARGET_CRS = "EPSG:3310"
SOURCE_CRS = "EPSG:4326"


def setup_logger():
    logger = logging.getLogger("19_prepare_osm_recognition_features_v02")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


class OSMPointRecognitionHandler(osmium.SimpleHandler):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.records = []
        self.nodes_seen = 0
        self.nodes_matched = 0
        self.failed_geometry = 0

    def node(self, n):
        self.nodes_seen += 1

        if self.nodes_seen % 1_000_000 == 0:
            self.logger.info(f"Nodes scanned: {self.nodes_seen:,}")

        tourism = None
        natural = None

        for tag in n.tags:
            if tag.k == "tourism":
                tourism = tag.v
            elif tag.k == "natural":
                natural = tag.v

        feature_type = None

        if tourism == "viewpoint":
            feature_type = "viewpoint"
        elif tourism == "attraction":
            feature_type = "attraction"
        elif tourism == "camp_site":
            feature_type = "campground"
        elif tourism == "picnic_site":
            feature_type = "picnic_site"
        elif tourism == "information":
            feature_type = "information"
        elif natural == "peak":
            feature_type = "peak"

        if feature_type is None:
            return

        try:
            if not n.location.valid():
                self.failed_geometry += 1
                return

            tags = dict(n.tags)

            self.records.append(
                {
                    "osm_id": str(n.id),
                    "osm_element_type": "node",
                    "feature_type": feature_type,
                    "name": tags.get("name"),
                    "tourism": tourism,
                    "natural": natural,
                    "leisure": tags.get("leisure"),
                    "highway": tags.get("highway"),
                    "operator": tags.get("operator"),
                    "website": tags.get("website"),
                    "wikidata": tags.get("wikidata"),
                    "wikipedia": tags.get("wikipedia"),
                    "access": tags.get("access"),
                    "geometry": Point(n.location.lon, n.location.lat),
                }
            )

            self.nodes_matched += 1

        except Exception:
            self.failed_geometry += 1


def main():
    logger = setup_logger()
    logger.info("Starting Script 19: Prepare OSM point recognition features v02")

    if not PBF_PATH.exists():
        raise FileNotFoundError(f"Missing PBF file: {PBF_PATH}")

    if not STUDY_AREA_PATH.exists():
        raise FileNotFoundError(f"Missing study area file: {STUDY_AREA_PATH}")

    logger.info(f"Using PBF: {PBF_PATH}")
    logger.info(f"Reading study area: {STUDY_AREA_PATH}")

    study_area = gpd.read_file(STUDY_AREA_PATH).to_crs(TARGET_CRS)

    logger.info("Streaming OSM PBF nodes only with osmium")
    logger.info("This version avoids locations=True for faster node scanning.")

    handler = OSMPointRecognitionHandler(logger)

    # Important:
    # Do NOT use locations=True here.
    # Nodes already contain their own lon/lat, and locations=True slows this badly.
    handler.apply_file(str(PBF_PATH))

    logger.info(f"Nodes scanned: {handler.nodes_seen:,}")
    logger.info(f"Point recognition records extracted before clip: {len(handler.records):,}")
    logger.info(f"Matched nodes: {handler.nodes_matched:,}")
    logger.info(f"Failed geometries skipped: {handler.failed_geometry:,}")

    if not handler.records:
        raise ValueError("No OSM point recognition features extracted.")

    features = gpd.GeoDataFrame(
        handler.records,
        geometry="geometry",
        crs=SOURCE_CRS,
    )

    features = features[features.geometry.notna()]
    features = features[~features.geometry.is_empty]
    features = features.to_crs(TARGET_CRS)

    logger.info("Clipping point recognition features to coastal study area")

    features_clipped = gpd.clip(features, study_area)

    features_clipped = features_clipped[
        features_clipped.geometry.notna()
    ].copy()

    features_clipped = features_clipped[
        ~features_clipped.geometry.is_empty
    ].copy()

    if features_clipped.empty:
        raise ValueError("No point recognition features intersect study area.")

    features_clipped["recognition_feature_id"] = [
        f"rec_point_{i + 1}" for i in range(len(features_clipped))
    ]

    features_clipped["area_m2"] = 0.0
    features_clipped["area_km2"] = 0.0

    keep_cols = [
        "recognition_feature_id",
        "osm_id",
        "osm_element_type",
        "feature_type",
        "name",
        "tourism",
        "natural",
        "leisure",
        "highway",
        "operator",
        "website",
        "wikidata",
        "wikipedia",
        "access",
        "area_m2",
        "area_km2",
        "geometry",
    ]

    for col in keep_cols:
        if col not in features_clipped.columns:
            features_clipped[col] = pd.NA

    features_clipped = features_clipped[keep_cols].copy()

    logger.info("QA summary")
    logger.info(f"Recognition point features after clip: {len(features_clipped):,}")
    logger.info(f"CRS: {features_clipped.crs}")

    counts = features_clipped["feature_type"].value_counts().sort_index()

    for feature_type, count in counts.items():
        logger.info(f"{feature_type}: {count:,}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    features_clipped.to_file(
        OUTPUT_PATH,
        layer="osm_recognition_features_v02",
        driver="GPKG",
    )

    logger.info(f"Saved output: {OUTPUT_PATH}")
    logger.info("Script 19 completed successfully.")


if __name__ == "__main__":
    main()