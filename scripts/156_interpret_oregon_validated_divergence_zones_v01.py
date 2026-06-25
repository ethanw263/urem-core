#!/usr/bin/env python3

from pathlib import Path
import geopandas as gpd
import numpy as np

SCRIPT_NAME = "156_interpret_oregon_validated_divergence_zones_v01"

INPUT = Path("data/processed/oregon_validated_divergence_zones_v01.gpkg")

OUTPUT_GPKG = Path("data/processed/oregon_validated_divergence_zone_interpretation_v01.gpkg")
OUTPUT_CSV = Path("data/processed/oregon_validated_divergence_zone_interpretation_v01.csv")


def confidence_class(row):
    flow = row["flow_zone_overlap_pct_v01"]
    basin = row["core_basin_overlap_pct_v01"]
    cells = row["cell_count_v01"]
    strength = abs(row["mean_divergence_norm_v01"])

    if flow >= 0.50 and basin >= 0.10:
        return "high_confidence_divergence_zone"

    if flow >= 0.25 and strength >= 0.50:
        return "high_confidence_divergence_zone"

    if basin >= 0.10 and strength >= 0.50:
        return "moderate_confidence_divergence_zone"

    if cells >= 75 and strength >= 0.60:
        return "moderate_confidence_divergence_zone"

    return "structural_candidate_zone"


def role_label(row):
    ztype = row["divergence_zone_type_v01"]
    flow = row["flow_zone_overlap_pct_v01"]
    basin = row["core_basin_overlap_pct_v01"]

    if ztype == "source_zone":
        if flow >= 0.25:
            return "active_recognition_dispersal_source"
        if basin >= 0.10:
            return "basin_edge_recognition_source"
        return "structural_recognition_source"

    if ztype == "sink_zone":
        if basin >= 0.10:
            return "recognition_accumulation_sink"
        if flow >= 0.25:
            return "flow_convergence_sink"
        return "structural_recognition_sink"

    return "unclassified_divergence_zone"


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    gdf = gpd.read_file(INPUT)

    print(f"[{SCRIPT_NAME}] Input validated zones: {len(gdf):,}")

    required = [
        "divergence_zone_id_v01",
        "divergence_zone_type_v01",
        "cell_count_v01",
        "zone_area_km2_v01",
        "mean_divergence_norm_v01",
        "core_basin_overlap_pct_v01",
        "flow_zone_overlap_pct_v01",
        "divergence_validation_reason_v01",
    ]

    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    gdf["abs_mean_divergence_norm_v01"] = gdf["mean_divergence_norm_v01"].abs()

    gdf["divergence_confidence_class_v01"] = gdf.apply(confidence_class, axis=1)
    gdf["divergence_interpretation_v01"] = gdf.apply(role_label, axis=1)

    gdf["centroid_x_v01"] = gdf.geometry.centroid.x
    gdf["centroid_y_v01"] = gdf.geometry.centroid.y

    gdf["divergence_priority_score_v01"] = (
        0.40 * gdf["abs_mean_divergence_norm_v01"]
        + 0.25 * gdf["flow_zone_overlap_pct_v01"].clip(0, 1)
        + 0.20 * gdf["core_basin_overlap_pct_v01"].clip(0, 1)
        + 0.15 * (
            gdf["cell_count_v01"] / gdf["cell_count_v01"].max()
        )
    )

    gdf = gdf.sort_values(
        "divergence_priority_score_v01",
        ascending=False
    ).reset_index(drop=True)

    gdf["divergence_priority_rank_v01"] = np.arange(1, len(gdf) + 1)

    print()
    print(f"[{SCRIPT_NAME}] Confidence classes:")
    print(gdf["divergence_confidence_class_v01"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] Interpretations:")
    print(gdf["divergence_interpretation_v01"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] By source/sink type:")
    print(
        gdf.groupby("divergence_zone_type_v01")[
            [
                "cell_count_v01",
                "zone_area_km2_v01",
                "abs_mean_divergence_norm_v01",
                "flow_zone_overlap_pct_v01",
                "core_basin_overlap_pct_v01",
                "divergence_priority_score_v01",
            ]
        ].mean()
    )

    print()
    print(f"[{SCRIPT_NAME}] Top interpreted divergence zones:")
    print(
        gdf[
            [
                "divergence_priority_rank_v01",
                "divergence_zone_id_v01",
                "divergence_zone_type_v01",
                "cell_count_v01",
                "mean_divergence_norm_v01",
                "flow_zone_overlap_pct_v01",
                "core_basin_overlap_pct_v01",
                "divergence_confidence_class_v01",
                "divergence_interpretation_v01",
                "divergence_priority_score_v01",
            ]
        ]
        .head(32)
        .to_string(index=False)
    )

    print()
    print(f"[{SCRIPT_NAME}] Writing GPKG: {OUTPUT_GPKG}")
    gdf.to_file(OUTPUT_GPKG, driver="GPKG")

    print(f"[{SCRIPT_NAME}] Writing CSV: {OUTPUT_CSV}")
    gdf.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()