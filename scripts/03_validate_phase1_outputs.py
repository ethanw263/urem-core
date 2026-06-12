#!/usr/bin/env python3
"""
Phase 1 Script 03: Validate Phase 1 Outputs

Inputs:
    data/processed/study_area_25km.gpkg
    data/processed/coastal_grid_1km.gpkg

Output:
    outputs/validation/phase1_validation_summary.csv

Purpose:
    Validate that the Phase 1 study area and 1 km analysis grid were created
    correctly before moving into terrain, golf, or MVF variable generation.
"""

from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd


# -----------------------------
# Project paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VALIDATION_DIR = PROJECT_ROOT / "outputs" / "validation"

STUDY_AREA_PATH = PROCESSED_DIR / "study_area_25km.gpkg"
GRID_PATH = PROCESSED_DIR / "coastal_grid_1km.gpkg"

OUTPUT_SUMMARY = VALIDATION_DIR / "phase1_validation_summary.csv"

VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Constants
# -----------------------------

CRS_CA_ALBERS = "EPSG:3310"


# -----------------------------
# Helpers
# -----------------------------

def log(message: str) -> None:
    print(f"[03_validate_phase1_outputs] {message}")


def add_check(results: list, check_name: str, passed: bool, value, notes: str = "") -> None:
    results.append(
        {
            "check": check_name,
            "passed": passed,
            "value": value,
            "notes": notes,
        }
    )


def load_layer(path: Path, label: str) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")

    log(f"Reading {label}: {path}")
    gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError(f"{label} is empty: {path}")

    return gdf


def main() -> None:
    log("Starting Phase 1 validation")

    results = []

    try:
        study = load_layer(STUDY_AREA_PATH, "study area")
        grid = load_layer(GRID_PATH, "coastal grid")

        # CRS checks
        study_crs = study.crs.to_string() if study.crs else None
        grid_crs = grid.crs.to_string() if grid.crs else None

        add_check(
            results,
            "study_area_crs_is_epsg_3310",
            study_crs == CRS_CA_ALBERS,
            study_crs,
        )

        add_check(
            results,
            "grid_crs_is_epsg_3310",
            grid_crs == CRS_CA_ALBERS,
            grid_crs,
        )

        # Feature count checks
        add_check(
            results,
            "study_area_has_one_feature",
            len(study) == 1,
            len(study),
        )

        add_check(
            results,
            "grid_has_cells",
            len(grid) > 0,
            len(grid),
        )

        add_check(
            results,
            "grid_cell_count_reasonable",
            50_000 <= len(grid) <= 80_000,
            len(grid),
            "Expected rough range for 1 km coastal California grid.",
        )

        # Required fields
        required_grid_fields = {"cell_id", "centroid_x", "centroid_y", "geometry"}
        missing_fields = required_grid_fields - set(grid.columns)

        add_check(
            results,
            "grid_required_fields_present",
            len(missing_fields) == 0,
            ",".join(sorted(missing_fields)) if missing_fields else "none",
        )

        # Unique cell IDs
        if "cell_id" in grid.columns:
            duplicate_count = grid["cell_id"].duplicated().sum()
            add_check(
                results,
                "cell_id_unique",
                duplicate_count == 0,
                duplicate_count,
            )
        else:
            add_check(
                results,
                "cell_id_unique",
                False,
                "cell_id missing",
            )

        # Geometry validity
        study_invalid = (~study.geometry.is_valid).sum()
        grid_invalid = (~grid.geometry.is_valid).sum()

        add_check(
            results,
            "study_area_geometries_valid",
            study_invalid == 0,
            study_invalid,
        )

        add_check(
            results,
            "grid_geometries_valid",
            grid_invalid == 0,
            grid_invalid,
        )

        # Empty geometry checks
        study_empty = study.geometry.is_empty.sum()
        grid_empty = grid.geometry.is_empty.sum()

        add_check(
            results,
            "study_area_no_empty_geometries",
            study_empty == 0,
            study_empty,
        )

        add_check(
            results,
            "grid_no_empty_geometries",
            grid_empty == 0,
            grid_empty,
        )

        # Area checks
        study_area_km2 = study.geometry.area.sum() / 1_000_000
        grid_area_km2 = grid.geometry.area.sum() / 1_000_000
        area_difference_km2 = abs(study_area_km2 - grid_area_km2)

        add_check(
            results,
            "study_area_area_reasonable",
            55_000 <= study_area_km2 <= 65_000,
            round(study_area_km2, 3),
        )

        add_check(
            results,
            "grid_area_matches_study_area",
            area_difference_km2 <= 1,
            round(area_difference_km2, 6),
            "Difference should be near zero because grid cells are clipped to study area.",
        )

        # Zero-area grid cells
        zero_area_cells = (grid.geometry.area <= 0).sum()

        add_check(
            results,
            "grid_no_zero_area_cells",
            zero_area_cells == 0,
            zero_area_cells,
        )

        # Intersection check
        study_union = study.geometry.union_all()
        non_intersecting_cells = (~grid.intersects(study_union)).sum()

        add_check(
            results,
            "all_grid_cells_intersect_study_area",
            non_intersecting_cells == 0,
            non_intersecting_cells,
        )

        # Write summary
        summary = pd.DataFrame(results)
        summary.to_csv(OUTPUT_SUMMARY, index=False)

        failed = summary[summary["passed"] == False]

        log(f"Validation summary written: {OUTPUT_SUMMARY}")

        if failed.empty:
            log("All Phase 1 validation checks passed.")
        else:
            log("Some validation checks failed:")
            print(failed.to_string(index=False))
            sys.exit(1)

    except Exception as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
    