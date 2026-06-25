#!/usr/bin/env python3

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

SCRIPT_NAME = "163_build_oregon_transition_hotspot_review_package_v01"

HOTSPOTS = Path("data/processed/oregon_recognition_transition_hotspots_v01.gpkg")
HOTSPOT_CELLS = Path("data/processed/oregon_recognition_transition_hotspot_cells_v01.gpkg")

OUTPUT_GPKG = Path("data/processed/oregon_transition_hotspot_review_package_v01.gpkg")
OUTPUT_CSV = Path("data/processed/oregon_transition_hotspot_review_package_v01.csv")


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    hotspots = gpd.read_file(HOTSPOTS)
    cells = gpd.read_file(HOTSPOT_CELLS)

    if cells.crs != hotspots.crs:
        cells = cells.to_crs(hotspots.crs)

    print(f"[{SCRIPT_NAME}] Hotspots: {len(hotspots):,}")
    print(f"[{SCRIPT_NAME}] Hotspot cells: {len(cells):,}")

    hotspots["centroid_x_v01"] = hotspots.geometry.centroid.x
    hotspots["centroid_y_v01"] = hotspots.geometry.centroid.y

    review = hotspots.copy()

    review["review_priority_v01"] = np.select(
        [
            review["mean_transition_score_v01"] >= 0.75,
            review["mean_transition_score_v01"] >= 0.65,
            review["mean_transition_score_v01"] >= 0.60,
        ],
        [
            "highest_priority_review",
            "high_priority_review",
            "standard_priority_review",
        ],
        default="lower_priority_review",
    )

    review["model_confidence_summary_v01"] = np.select(
        [
            (review["pct_cells_in_discovery_region_v01"] >= 0.50)
            & (review["pct_cells_in_divergence_zone_v01"] >= 0.75),

            (review["pct_cells_in_divergence_zone_v01"] >= 0.75),

            (review["pct_cells_in_discovery_region_v01"] >= 0.50),
        ],
        [
            "discovery_and_divergence_supported",
            "divergence_supported",
            "discovery_supported",
        ],
        default="model_supported_only",
    )

    review["manual_review_status"] = ""
    review["manual_review_notes"] = ""
    review["known_destination"] = ""
    review["coastal_landscape_type"] = ""
    review["artifact_flag"] = ""
    review["artifact_notes"] = ""
    review["interesting_feature"] = ""
    review["human_interest_score_1_to_5"] = ""
    review["geographic_validity_score_1_to_5"] = ""
    review["final_manual_verdict"] = ""

    keep_cols = [
        "transition_hotspot_rank_v01",
        "transition_hotspot_id_v01",
        "review_priority_v01",
        "model_confidence_summary_v01",
        "cell_count_v01",
        "hotspot_area_km2_v01",
        "mean_transition_score_v01",
        "max_transition_score_v01",
        "mean_velocity_score_v02",
        "mean_disequilibrium_v01",
        "mean_flow_norm_v01",
        "mean_abs_divergence_norm_v01",
        "mean_network_influence_v02",
        "pct_cells_in_discovery_region_v01",
        "pct_cells_in_divergence_zone_v01",
        "centroid_x_v01",
        "centroid_y_v01",
        "manual_review_status",
        "manual_review_notes",
        "known_destination",
        "coastal_landscape_type",
        "artifact_flag",
        "artifact_notes",
        "interesting_feature",
        "human_interest_score_1_to_5",
        "geographic_validity_score_1_to_5",
        "final_manual_verdict",
        "geometry",
    ]

    missing = [c for c in keep_cols if c not in review.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    review = review[keep_cols].copy()

    review = review.sort_values(
        [
            "transition_hotspot_rank_v01",
            "mean_transition_score_v01",
        ],
        ascending=[True, False],
    ).reset_index(drop=True)

    print()
    print(f"[{SCRIPT_NAME}] Review priority counts:")
    print(review["review_priority_v01"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] Model confidence summary:")
    print(review["model_confidence_summary_v01"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] Top review items:")
    print(
        review[
            [
                "transition_hotspot_rank_v01",
                "transition_hotspot_id_v01",
                "review_priority_v01",
                "model_confidence_summary_v01",
                "cell_count_v01",
                "mean_transition_score_v01",
                "pct_cells_in_discovery_region_v01",
                "pct_cells_in_divergence_zone_v01",
            ]
        ]
        .head(40)
        .to_string(index=False)
    )

    print()
    print(f"[{SCRIPT_NAME}] Writing GPKG: {OUTPUT_GPKG}")
    review.to_file(OUTPUT_GPKG, driver="GPKG")

    print(f"[{SCRIPT_NAME}] Writing CSV: {OUTPUT_CSV}")
    review.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()