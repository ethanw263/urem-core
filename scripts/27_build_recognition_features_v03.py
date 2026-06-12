#!/usr/bin/env python3
"""
Script 27: Build Recognition Features v03

Memory-safe version.

Purpose:
- Build Recognition v03 from the already-extracted OSM v02 layer.
- Avoid loading the full California PBF.
- Reclassify existing OSM features into richer recognition categories.

Inputs:
- data/processed/coastal_grid_1km.gpkg
- data/processed/osm_recognition_features_v02.gpkg

Outputs:
- data/processed/recognition_features_v03.gpkg
- data/processed/recognition_features_v03.csv
"""

from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parents[1]

GRID_PATH = BASE_DIR / "data/processed/coastal_grid_1km.gpkg"
OSM_FEATURES_PATH = BASE_DIR / "data/processed/osm_recognition_features_v02.gpkg"

OUT_GPKG = BASE_DIR / "data/processed/recognition_features_v03.gpkg"
OUT_CSV = BASE_DIR / "data/processed/recognition_features_v03.csv"


def log(msg: str) -> None:
    print(f"[27_build_recognition_features_v03] {msg}")


def safe_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    mn = s.min()
    mx = s.max()

    if mx == mn:
        return pd.Series(0.0, index=s.index)

    return (s - mn) / (mx - mn)


def safe_log_score(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    return safe_minmax(np.log1p(s))


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().str.strip()


def classify_features(osm: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Classifying existing OSM v02 features into Recognition v03 categories")

    for col in [
        "feature_type",
        "tourism",
        "natural",
        "leisure",
        "highway",
        "name",
        "wikipedia",
        "wikidata",
        "access",
    ]:
        if col not in osm.columns:
            osm[col] = ""

    feature_type = normalize_text(osm["feature_type"])
    tourism = normalize_text(osm["tourism"])
    natural = normalize_text(osm["natural"])
    leisure = normalize_text(osm["leisure"])
    highway = normalize_text(osm["highway"])
    name = normalize_text(osm["name"])
    wikipedia = normalize_text(osm["wikipedia"])
    wikidata = normalize_text(osm["wikidata"])

    osm["cat_viewpoint"] = (
        feature_type.eq("viewpoint")
        | tourism.eq("viewpoint")
    )

    osm["cat_peak"] = (
        feature_type.eq("peak")
        | natural.eq("peak")
    )

    osm["cat_attraction"] = (
        feature_type.eq("attraction")
        | tourism.eq("attraction")
    )

    osm["cat_campground"] = (
        feature_type.isin(["campground", "camp_site"])
        | tourism.isin(["camp_site", "caravan_site"])
    )

    osm["cat_picnic"] = (
        feature_type.isin(["picnic_site", "picnic"])
        | tourism.eq("picnic_site")
        | leisure.eq("picnic_table")
    )

    osm["cat_information"] = (
        feature_type.eq("information")
        | tourism.eq("information")
    )

    osm["cat_park_or_recreation"] = (
        leisure.isin(["park", "nature_reserve", "recreation_ground"])
        | feature_type.isin(["park", "nature_reserve", "recreation_ground"])
    )

    osm["cat_trail_or_path"] = (
        highway.isin(["path", "footway", "track", "bridleway", "cycleway"])
        | feature_type.isin(["trail", "path", "footway", "track"])
    )

    osm["cat_beach"] = (
        natural.eq("beach")
        | feature_type.eq("beach")
    )

    osm["cat_named_natural"] = (
        natural.isin([
            "peak",
            "saddle",
            "ridge",
            "cliff",
            "cape",
            "bay",
            "valley",
            "spring",
            "waterfall",
            "beach",
        ])
        | feature_type.isin([
            "peak",
            "saddle",
            "ridge",
            "cliff",
            "cape",
            "bay",
            "valley",
            "spring",
            "waterfall",
            "beach",
        ])
    )

    osm["cat_wiki"] = (
        wikipedia.ne("")
        | wikidata.ne("")
    )

    osm["cat_named"] = name.ne("")

    category_cols = [c for c in osm.columns if c.startswith("cat_")]

    for c in category_cols:
        log(f"{c}: {int(osm[c].sum()):,} features")

    return osm


def count_category_within_radius(
    grid: gpd.GeoDataFrame,
    features: gpd.GeoDataFrame,
    category_col: str,
    out_prefix: str,
    radius_m: int = 3000,
) -> pd.Series:
    out_col = f"{out_prefix}_count_3km"

    subset = features[features[category_col]].copy()

    if subset.empty:
        return pd.Series(0, index=grid.index, name=out_col)

    subset = subset.to_crs(grid.crs)
    subset = subset[["geometry"]].copy()

    buffers = grid[["cell_id", "geometry"]].copy()
    buffers["geometry"] = buffers.geometry.centroid.buffer(radius_m)

    joined = gpd.sjoin(
        buffers,
        subset,
        how="left",
        predicate="intersects",
    )

    matched = joined[joined["index_right"].notna()]
    counts = matched.groupby("cell_id").size()

    result = grid["cell_id"].map(counts).fillna(0).astype(int)
    result.name = out_col

    return result


def compute_counts(grid: gpd.GeoDataFrame, osm: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Computing Recognition v03 counts within 3km")

    categories = {
        "viewpoints": "cat_viewpoint",
        "peaks": "cat_peak",
        "attractions": "cat_attraction",
        "campgrounds": "cat_campground",
        "picnic_sites": "cat_picnic",
        "information": "cat_information",
        "parks_recreation": "cat_park_or_recreation",
        "trails_paths": "cat_trail_or_path",
        "beaches": "cat_beach",
        "named_natural_features": "cat_named_natural",
        "wiki_features": "cat_wiki",
        "named_features": "cat_named",
    }

    for out_prefix, category_col in categories.items():
        log(f"Counting {out_prefix}")

        grid[f"{out_prefix}_count_3km"] = count_category_within_radius(
            grid=grid,
            features=osm,
            category_col=category_col,
            out_prefix=out_prefix,
            radius_m=3000,
        )

        grid[f"{out_prefix}_score_v03"] = safe_log_score(
            grid[f"{out_prefix}_count_3km"]
        )

        log(
            f"{out_prefix}: total={grid[f'{out_prefix}_count_3km'].sum():,}, "
            f"max={grid[f'{out_prefix}_count_3km'].max():,}"
        )

    return grid


def compute_group_scores(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("Computing Recognition v03 group scores")

    grid["recreation_recognition_score_v03"] = (
        0.25 * grid["trails_paths_score_v03"]
        + 0.20 * grid["parks_recreation_score_v03"]
        + 0.15 * grid["beaches_score_v03"]
        + 0.15 * grid["campgrounds_score_v03"]
        + 0.10 * grid["picnic_sites_score_v03"]
        + 0.15 * grid["named_natural_features_score_v03"]
    )

    grid["tourism_recognition_score_v03"] = (
        0.35 * grid["attractions_score_v03"]
        + 0.30 * grid["viewpoints_score_v03"]
        + 0.20 * grid["information_score_v03"]
        + 0.15 * grid["wiki_features_score_v03"]
    )

    grid["natural_recognition_score_v03"] = (
        0.40 * grid["peaks_score_v03"]
        + 0.35 * grid["named_natural_features_score_v03"]
        + 0.25 * grid["beaches_score_v03"]
    )

    grid["general_recognition_score_v03"] = (
        0.60 * grid["named_features_score_v03"]
        + 0.40 * grid["wiki_features_score_v03"]
    )

    grid["observed_recognition_preview_score_v03_raw"] = (
        0.35 * grid["recreation_recognition_score_v03"]
        + 0.30 * grid["tourism_recognition_score_v03"]
        + 0.20 * grid["natural_recognition_score_v03"]
        + 0.15 * grid["general_recognition_score_v03"]
    )

    grid["observed_recognition_preview_score_v03"] = safe_minmax(
        grid["observed_recognition_preview_score_v03_raw"]
    )

    count_cols = [c for c in grid.columns if c.endswith("_count_3km")]
    grid["recognition_feature_total_count_3km"] = grid[count_cols].sum(axis=1)

    grid["recognition_v03_feature_coverage"] = (
        (grid[count_cols] > 0).sum(axis=1) / len(count_cols)
    )

    return grid


def main():
    log("Starting Script 27 from existing OSM v02 features")

    if not GRID_PATH.exists():
        raise FileNotFoundError(f"Grid not found: {GRID_PATH}")

    if not OSM_FEATURES_PATH.exists():
        raise FileNotFoundError(f"OSM features not found: {OSM_FEATURES_PATH}")

    log(f"Reading grid: {GRID_PATH}")
    grid = gpd.read_file(GRID_PATH)

    if grid.empty:
        raise ValueError("Grid is empty")

    if grid.crs is None:
        raise ValueError("Grid has no CRS")

    if "cell_id" not in grid.columns:
        grid["cell_id"] = range(len(grid))

    log(f"Grid rows: {len(grid):,}")
    log(f"Grid CRS: {grid.crs}")

    log(f"Reading existing OSM recognition features: {OSM_FEATURES_PATH}")
    osm = gpd.read_file(OSM_FEATURES_PATH)

    if osm.empty:
        raise ValueError("OSM recognition features file is empty")

    if osm.crs is None:
        raise ValueError("OSM recognition features file has no CRS")

    log(f"OSM v02 feature rows: {len(osm):,}")

    osm = classify_features(osm)
    grid = compute_counts(grid, osm)
    grid = compute_group_scores(grid)

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

    log(f"Writing GeoPackage: {OUT_GPKG}")
    grid.to_file(OUT_GPKG, driver="GPKG")

    log(f"Writing CSV: {OUT_CSV}")
    grid.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")

    summary_cols = [
        "viewpoints_count_3km",
        "peaks_count_3km",
        "attractions_count_3km",
        "campgrounds_count_3km",
        "picnic_sites_count_3km",
        "information_count_3km",
        "parks_recreation_count_3km",
        "trails_paths_count_3km",
        "beaches_count_3km",
        "named_natural_features_count_3km",
        "wiki_features_count_3km",
        "named_features_count_3km",
        "recognition_feature_total_count_3km",
        "observed_recognition_preview_score_v03",
    ]

    print("\nRecognition v03 summary:")
    print(grid[summary_cols].describe())


if __name__ == "__main__":
    main()