#!/usr/bin/env python3
"""
136_oregon_region_validation_v01.py

Create Oregon discovery region validation template.

Inputs:
- data/processed/oregon_discovery_region_interpretation_v01.gpkg

Outputs:
- data/processed/oregon_region_validation_template_v01.csv
- data/processed/oregon_region_validation_template_v01.gpkg
"""

from pathlib import Path
import geopandas as gpd


SCRIPT_NAME = "136_oregon_region_validation_v01"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_GPKG = PROCESSED_DIR / "oregon_discovery_region_interpretation_v01.gpkg"

OUT_GPKG = PROCESSED_DIR / "oregon_region_validation_template_v01.gpkg"
OUT_CSV = PROCESSED_DIR / "oregon_region_validation_template_v01.csv"


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def suggested_validation_priority(rank):
    if rank <= 10:
        return "required_review"
    if rank <= 20:
        return "high_priority"
    return "secondary_review"


def main():
    log("Starting Oregon region validation template")

    if not INPUT_GPKG.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_GPKG}")

    gdf = gpd.read_file(INPUT_GPKG)

    log(f"Regions loaded: {len(gdf):,}")
    log(f"CRS: {gdf.crs}")

    gdf["validation_priority_v01"] = (
        gdf["discovery_region_rank_v06"]
        .apply(suggested_validation_priority)
    )

    gdf["manual_place_name_v01"] = ""
    gdf["manual_known_destination_status_v01"] = ""
    gdf["manual_under_recognition_status_v01"] = ""
    gdf["manual_false_positive_flag_v01"] = ""
    gdf["manual_reasoning_notes_v01"] = ""
    gdf["manual_validation_score_0_5_v01"] = ""

    keep_cols = [
        "discovery_region_rank_v06",
        "discovery_region_id_v06",
        "discovery_region_tier_v06",
        "review_priority_group_v01",
        "validation_priority_v01",
        "manual_region_type_v01",
        "cell_count",
        "region_area_km2",
        "longitude",
        "latitude",
        "mean_urem_score_v06_raw",
        "max_urem_score_v06_raw",
        "mean_physical_exceptionality_v03",
        "mean_observed_recognition_v04",
        "mean_expected_recognition_v06",
        "mean_positive_under_recognition_residual_v06",
        "mean_recognition_total_count_3km_v04",
        "best_cell_rank_v06",
        "best_cell_id",
        "manual_place_name_v01",
        "manual_known_destination_status_v01",
        "manual_under_recognition_status_v01",
        "manual_false_positive_flag_v01",
        "manual_reasoning_notes_v01",
        "manual_validation_score_0_5_v01",
        "geometry",
    ]

    keep_cols = [c for c in keep_cols if c in gdf.columns]
    out = gdf[keep_cols].sort_values("discovery_region_rank_v06").copy()

    if OUT_GPKG.exists():
        OUT_GPKG.unlink()

    log(f"Writing GPKG: {OUT_GPKG}")
    out.to_file(
        OUT_GPKG,
        layer="oregon_region_validation_template_v01",
        driver="GPKG",
    )

    log(f"Writing CSV: {OUT_CSV}")
    out.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")

    print(
        out[
            [
                "discovery_region_rank_v06",
                "manual_region_type_v01",
                "cell_count",
                "region_area_km2",
                "longitude",
                "latitude",
                "mean_urem_score_v06_raw",
                "mean_observed_recognition_v04",
                "mean_expected_recognition_v06",
            ]
        ].head(20)
    )


if __name__ == "__main__":
    main()
    