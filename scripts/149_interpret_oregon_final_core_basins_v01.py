#!/usr/bin/env python3
"""
149_interpret_oregon_final_core_basins_v01.py

Create interpretation template for final Oregon recognition core basins.

Inputs:
- data/processed/oregon_final_recognition_core_basins_v01.gpkg

Outputs:
- data/processed/oregon_final_core_basin_interpretation_v01.csv
- data/processed/oregon_final_core_basin_interpretation_v01.gpkg
"""

from pathlib import Path
import geopandas as gpd


SCRIPT_NAME = "149_interpret_oregon_final_core_basins_v01"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = PROCESSED_DIR / "oregon_final_recognition_core_basins_v01.gpkg"

OUT_GPKG = PROCESSED_DIR / "oregon_final_core_basin_interpretation_v01.gpkg"
OUT_CSV = PROCESSED_DIR / "oregon_final_core_basin_interpretation_v01.csv"


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def priority(rank):
    if rank <= 5:
        return "primary_review"
    if rank <= 10:
        return "secondary_review"
    return "tertiary_review"


def main():
    log("Starting final Oregon core basin interpretation template")

    gdf = gpd.read_file(INPUT_GPKG)

    gdf["review_priority_v01"] = gdf["final_core_basin_rank_v01"].apply(priority)

    gdf["manual_core_basin_name_v01"] = ""
    gdf["manual_nearest_town_v01"] = ""
    gdf["manual_nearest_landmark_v01"] = ""
    gdf["manual_likely_landscape_system_v01"] = ""
    gdf["manual_likely_rde_interpretation_v01"] = ""
    gdf["manual_validation_status_v01"] = ""
    gdf["manual_notes_v01"] = ""

    keep_cols = [
        "final_core_basin_rank_v01",
        "final_core_basin_id_v01",
        "review_priority_v01",
        "core_cell_count",
        "core_area_km2",
        "mean_rde_surface_energy_v01",
        "max_rde_surface_energy_v01",
        "mean_disequilibrium_v01",
        "max_disequilibrium_v01",
        "mean_gradient_norm_v01",
        "max_gradient_norm_v01",
        "mean_physical_exceptionality_v03",
        "mean_observed_recognition_v04",
        "mean_expected_recognition_v06",
        "mean_urem_score_v06_raw",
        "max_urem_score_v06_raw",
        "longitude",
        "latitude",
        "manual_core_basin_name_v01",
        "manual_nearest_town_v01",
        "manual_nearest_landmark_v01",
        "manual_likely_landscape_system_v01",
        "manual_likely_rde_interpretation_v01",
        "manual_validation_status_v01",
        "manual_notes_v01",
        "geometry",
    ]

    keep_cols = [c for c in keep_cols if c in gdf.columns]

    out = gdf[keep_cols].sort_values("final_core_basin_rank_v01").copy()

    if OUT_GPKG.exists():
        OUT_GPKG.unlink()

    log(f"Writing GPKG: {OUT_GPKG}")
    out.to_file(
        OUT_GPKG,
        layer="oregon_final_core_basin_interpretation_v01",
        driver="GPKG",
    )

    log(f"Writing CSV: {OUT_CSV}")
    out.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")

    print(
        out[
            [
                "final_core_basin_rank_v01",
                "core_cell_count",
                "core_area_km2",
                "mean_rde_surface_energy_v01",
                "mean_disequilibrium_v01",
                "longitude",
                "latitude",
            ]
        ]
    )


if __name__ == "__main__":
    main()