#!/usr/bin/env python3

from pathlib import Path
import geopandas as gpd
import numpy as np

SCRIPT_NAME = "155b_filter_oregon_validated_divergence_zones_v01"

INPUT = Path("data/processed/oregon_divergence_zone_rde_comparison_v01.gpkg")

OUTPUT_GPKG = Path("data/processed/oregon_validated_divergence_zones_v01.gpkg")
OUTPUT_CSV = Path("data/processed/oregon_validated_divergence_zones_v01.csv")


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    gdf = gpd.read_file(INPUT)

    print(f"[{SCRIPT_NAME}] Input zones: {len(gdf):,}")

    required = [
        "divergence_zone_id_v01",
        "divergence_zone_type_v01",
        "cell_count_v01",
        "zone_area_km2_v01",
        "mean_divergence_norm_v01",
        "core_basin_overlap_pct_v01",
        "flow_zone_overlap_pct_v01",
        "rde_structure_alignment_v01",
    ]

    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    gdf["abs_mean_divergence_norm_v01"] = gdf["mean_divergence_norm_v01"].abs()

    large_zone_threshold = gdf["cell_count_v01"].quantile(0.75)
    strong_div_threshold = gdf["abs_mean_divergence_norm_v01"].quantile(0.75)

    gdf["validated_by_flow_v01"] = gdf["flow_zone_overlap_pct_v01"] >= 0.25
    gdf["validated_by_basin_v01"] = gdf["core_basin_overlap_pct_v01"] >= 0.10

    gdf["validated_by_large_strong_structure_v01"] = (
        (gdf["cell_count_v01"] >= large_zone_threshold)
        & (gdf["abs_mean_divergence_norm_v01"] >= strong_div_threshold)
    )

    gdf["validated_divergence_zone_v01"] = (
        gdf["validated_by_flow_v01"]
        | gdf["validated_by_basin_v01"]
        | gdf["validated_by_large_strong_structure_v01"]
    )

    def reason(row):
        reasons = []

        if row["validated_by_flow_v01"]:
            reasons.append("flow_aligned")

        if row["validated_by_basin_v01"]:
            reasons.append("basin_aligned")

        if row["validated_by_large_strong_structure_v01"]:
            reasons.append("large_strong_structure")

        if not reasons:
            return "not_validated"

        return "+".join(reasons)

    gdf["divergence_validation_reason_v01"] = gdf.apply(reason, axis=1)

    valid = gdf[gdf["validated_divergence_zone_v01"]].copy()

    print()
    print(f"[{SCRIPT_NAME}] Validated zones: {len(valid):,}")
    print(valid["divergence_zone_type_v01"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] Validation reasons:")
    print(valid["divergence_validation_reason_v01"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] Large-zone threshold: {large_zone_threshold:.2f}")
    print(f"[{SCRIPT_NAME}] Strong-divergence threshold: {strong_div_threshold:.4f}")

    print()
    print(f"[{SCRIPT_NAME}] Top validated zones:")
    print(
        valid.sort_values(
            [
                "flow_zone_overlap_pct_v01",
                "core_basin_overlap_pct_v01",
                "cell_count_v01",
            ],
            ascending=False,
        )[
            [
                "divergence_zone_id_v01",
                "divergence_zone_type_v01",
                "cell_count_v01",
                "mean_divergence_norm_v01",
                "core_basin_overlap_pct_v01",
                "flow_zone_overlap_pct_v01",
                "divergence_validation_reason_v01",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    print()
    print(f"[{SCRIPT_NAME}] Writing GPKG: {OUTPUT_GPKG}")
    valid.to_file(OUTPUT_GPKG, driver="GPKG")

    print(f"[{SCRIPT_NAME}] Writing CSV: {OUTPUT_CSV}")
    valid.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()