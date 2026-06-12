#!/usr/bin/env python3

from pathlib import Path
import logging
import sys

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRID_PATH = PROJECT_ROOT / "data/processed/coastal_grid_proxy_features.gpkg"
GOLF_PATH = PROJECT_ROOT / "data/processed/golf_courses_ca_coastal.gpkg"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/coastal_grid_golf_recognition.gpkg"

TARGET_CRS = "EPSG:3310"


def setup_logger():
    logger = logging.getLogger("10_compute_golf_recognition")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def minmax(series):
    smin = series.min()
    smax = series.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        return series * 0
    return (series - smin) / (smax - smin)


def main():
    logger = setup_logger()
    logger.info("Starting Script 10: Compute golf recognition proxy")

    if not GRID_PATH.exists():
        raise FileNotFoundError(f"Missing grid file: {GRID_PATH}")
    if not GOLF_PATH.exists():
        raise FileNotFoundError(f"Missing golf file: {GOLF_PATH}")

    grid = gpd.read_file(GRID_PATH).to_crs(TARGET_CRS)
    golf = gpd.read_file(GOLF_PATH).to_crs(TARGET_CRS)

    if grid.empty:
        raise ValueError("Grid is empty.")
    if golf.empty:
        raise ValueError("Golf layer is empty.")

    logger.info(f"Grid cells loaded: {len(grid):,}")
    logger.info(f"Golf features loaded: {len(golf):,}")

    if "cell_id" not in grid.columns:
        grid["cell_id"] = range(1, len(grid) + 1)

    grid["cell_area_m2"] = grid.geometry.area

    grid_base = grid[["cell_id", "geometry"]].copy()
    golf_base = golf[["osm_feature_id", "geometry"]].copy()

    logger.info("Intersecting golf courses with grid cells")

    intersections = gpd.overlay(
        grid_base,
        golf_base,
        how="intersection",
        keep_geom_type=False,
    )

    if intersections.empty:
        logger.warning("No golf/grid intersections found.")
        grid["golf_course_count"] = 0
        grid["golf_area_m2"] = 0.0
    else:
        intersections["intersect_area_m2"] = intersections.geometry.area

        area_by_cell = (
            intersections.groupby("cell_id")["intersect_area_m2"]
            .sum()
            .reset_index()
            .rename(columns={"intersect_area_m2": "golf_area_m2"})
        )

        count_by_cell = (
            intersections.groupby("cell_id")["osm_feature_id"]
            .nunique()
            .reset_index()
            .rename(columns={"osm_feature_id": "golf_course_count"})
        )

        grid = grid.merge(area_by_cell, on="cell_id", how="left")
        grid = grid.merge(count_by_cell, on="cell_id", how="left")

        grid["golf_area_m2"] = grid["golf_area_m2"].fillna(0.0)
        grid["golf_course_count"] = grid["golf_course_count"].fillna(0).astype(int)

    grid["golf_area_km2"] = grid["golf_area_m2"] / 1_000_000
    grid["golf_area_share"] = grid["golf_area_m2"] / grid["cell_area_m2"]

    grid["golf_area_share"] = grid["golf_area_share"].clip(lower=0, upper=1)

    grid["golf_recognition_score"] = minmax(grid["golf_area_share"]).fillna(0)

    logger.info("QA summary")
    logger.info(f"Cells with golf: {(grid['golf_area_m2'] > 0).sum():,}")
    logger.info(f"Total golf area assigned: {grid['golf_area_km2'].sum():,.2f} km²")
    logger.info(f"Max golf area share: {grid['golf_area_share'].max():.4f}")
    logger.info(f"Max recognition score: {grid['golf_recognition_score'].max():.4f}")

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    grid.to_file(
        OUTPUT_PATH,
        layer="coastal_grid_golf_recognition",
        driver="GPKG",
    )

    logger.info(f"Saved output: {OUTPUT_PATH}")
    logger.info("Script 10 completed successfully.")


if __name__ == "__main__":
    main()