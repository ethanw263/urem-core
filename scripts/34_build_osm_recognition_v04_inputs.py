#!/usr/bin/env python3
"""
Script 34: Build OSM Recognition v04 Inputs

Memory-safe streaming OSM extractor using Python osmium.

Extracts missing recognition inputs:
- trails / paths
- beaches
- parks / recreation areas
- protected areas
- parking
- trailheads
- visitor information
- viewpoints
- named natural features

Input:
- data/raw/golf/california-latest.osm.pbf

Output:
- data/processed/osm_recognition_features_v04.gpkg
- data/processed/osm_recognition_features_v04.csv
"""

from pathlib import Path
import warnings

import geopandas as gpd
import pandas as pd
from shapely import wkb
from shapely.geometry import Point

warnings.filterwarnings("ignore")

try:
    import osmium
except ImportError:
    raise ImportError("Install osmium with: pip install osmium")


BASE_DIR = Path(__file__).resolve().parents[1]

OSM_PBF_PATH = BASE_DIR / "data/raw/golf/california-latest.osm.pbf"

OUT_GPKG = BASE_DIR / "data/processed/osm_recognition_features_v04.gpkg"
OUT_CSV = BASE_DIR / "data/processed/osm_recognition_features_v04.csv"


def log(msg: str) -> None:
    print(f"[34_build_osm_recognition_v04_inputs] {msg}")


def get_tag(tags, key):
    try:
        return tags.get(key)
    except Exception:
        return None


def classify_feature(tags):
    tourism = get_tag(tags, "tourism")
    natural = get_tag(tags, "natural")
    leisure = get_tag(tags, "leisure")
    highway = get_tag(tags, "highway")
    amenity = get_tag(tags, "amenity")
    boundary = get_tag(tags, "boundary")
    protect_class = get_tag(tags, "protect_class")
    information = get_tag(tags, "information")

    categories = []

    if highway in {"path", "footway", "track", "bridleway", "cycleway"}:
        categories.append("trail_path")

    if natural == "beach":
        categories.append("beach")

    if leisure in {"park", "nature_reserve", "recreation_ground"}:
        categories.append("park_recreation")

    if boundary in {"protected_area", "national_park"} or protect_class:
        categories.append("protected_area")

    if amenity == "parking":
        categories.append("parking")

    if tourism == "trailhead":
        categories.append("trailhead")

    if tourism == "information" or information in {"map", "board", "guidepost", "office", "visitor_centre"}:
        categories.append("visitor_information")

    if tourism == "viewpoint":
        categories.append("viewpoint")

    if natural in {
        "peak",
        "saddle",
        "ridge",
        "cliff",
        "cape",
        "bay",
        "valley",
        "spring",
        "waterfall",
    }:
        categories.append("named_natural_feature")

    if tourism in {"attraction", "camp_site", "caravan_site", "picnic_site"}:
        categories.append("tourism_recreation")

    return categories


class RecognitionV04Handler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.records = []
        self.wkb_factory = osmium.geom.WKBFactory()
        self.node_count = 0
        self.way_count = 0
        self.skipped_geometry = 0

    def add_record(self, osm_id, element_type, tags, geom):
        categories = classify_feature(tags)

        if not categories:
            return

        self.records.append(
            {
                "osm_id": str(osm_id),
                "osm_element_type": element_type,
                "recognition_categories": ",".join(categories),
                "primary_category": categories[0],
                "name": get_tag(tags, "name"),
                "tourism": get_tag(tags, "tourism"),
                "natural": get_tag(tags, "natural"),
                "leisure": get_tag(tags, "leisure"),
                "highway": get_tag(tags, "highway"),
                "amenity": get_tag(tags, "amenity"),
                "boundary": get_tag(tags, "boundary"),
                "protect_class": get_tag(tags, "protect_class"),
                "information": get_tag(tags, "information"),
                "operator": get_tag(tags, "operator"),
                "website": get_tag(tags, "website"),
                "wikidata": get_tag(tags, "wikidata"),
                "wikipedia": get_tag(tags, "wikipedia"),
                "access": get_tag(tags, "access"),
                "geometry": geom,
            }
        )

    def node(self, n):
        cats = classify_feature(n.tags)
        if not cats:
            return

        try:
            geom = Point(float(n.location.lon), float(n.location.lat))
            self.add_record(n.id, "node", n.tags, geom)
            self.node_count += 1
        except Exception:
            self.skipped_geometry += 1

    def way(self, w):
        cats = classify_feature(w.tags)
        if not cats:
            return

        try:
            is_closed = len(w.nodes) >= 4 and w.nodes[0].ref == w.nodes[-1].ref

            if is_closed:
                try:
                    geom = wkb.loads(self.wkb_factory.create_polygon(w), hex=True)
                except Exception:
                    geom = wkb.loads(self.wkb_factory.create_linestring(w), hex=True)
            else:
                geom = wkb.loads(self.wkb_factory.create_linestring(w), hex=True)

            self.add_record(w.id, "way", w.tags, geom)
            self.way_count += 1

        except Exception:
            self.skipped_geometry += 1


def main():
    log("Starting Script 34")

    if not OSM_PBF_PATH.exists():
        raise FileNotFoundError(f"OSM PBF not found: {OSM_PBF_PATH}")

    log(f"Reading OSM PBF: {OSM_PBF_PATH}")
    log("Streaming nodes and ways. This may take several minutes.")

    handler = RecognitionV04Handler()
    handler.apply_file(str(OSM_PBF_PATH), locations=True)

    log(f"Extracted node features: {handler.node_count:,}")
    log(f"Extracted way features: {handler.way_count:,}")
    log(f"Skipped geometry count: {handler.skipped_geometry:,}")
    log(f"Total extracted records: {len(handler.records):,}")

    if not handler.records:
        raise ValueError("No recognition records extracted")

    df = pd.DataFrame(handler.records)

    gdf = gpd.GeoDataFrame(
        df,
        geometry="geometry",
        crs="EPSG:4326",
    )

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    gdf["feature_area_m2"] = gdf.to_crs("EPSG:3310").geometry.area
    gdf["feature_length_m"] = gdf.to_crs("EPSG:3310").geometry.length

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

    log(f"Writing GeoPackage: {OUT_GPKG}")
    gdf.to_file(OUT_GPKG, driver="GPKG")

    log(f"Writing CSV: {OUT_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")

    print("\nCategory counts:")
    exploded = gdf.assign(
        category=gdf["recognition_categories"].str.split(",")
    ).explode("category")
    print(exploded["category"].value_counts())


if __name__ == "__main__":
    main()