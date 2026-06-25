#!/usr/bin/env python3

from pathlib import Path
import geopandas as gpd
import numpy as np

SCRIPT_NAME = "157_compare_oregon_divergence_zones_with_discovery_regions_v01"

DIVERGENCE = Path("data/processed/oregon_validated_divergence_zone_interpretation_v01.gpkg")
DISCOVERY = Path("data/processed/oregon_discovery_regions_v06.gpkg")

OUTPUT_GPKG = Path("data/processed/oregon_divergence_discovery_region_comparison_v01.gpkg")
OUTPUT_CSV = Path("data/processed/oregon_divergence_discovery_region_comparison_v01.csv")


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    div = gpd.read_file(DIVERGENCE)
    disc = gpd.read_file(DISCOVERY)

    if disc.crs != div.crs:
        disc = disc.to_crs(div.crs)

    print(f"[{SCRIPT_NAME}] Divergence zones: {len(div):,}")
    print(f"[{SCRIPT_NAME}] Discovery regions: {len(disc):,}")

    records = []

    for _, row in div.iterrows():
        geom = row.geometry
        zone_area = geom.area

        hits = disc[disc.intersects(geom)].copy()

        if hits.empty:
            overlap_area = 0.0
            overlap_pct = 0.0
            discovery_count = 0
        else:
            overlap_area = hits.geometry.intersection(geom).area.sum()
            overlap_pct = overlap_area / zone_area if zone_area > 0 else 0.0
            discovery_count = len(hits)

        record = row.drop(labels="geometry").to_dict()

        record.update(
            {
                "discovery_region_count_intersected_v01": discovery_count,
                "discovery_region_overlap_km2_v01": overlap_area / 1_000_000,
                "discovery_region_overlap_pct_v01": overlap_pct,
                "divergence_discovery_link_v01": (
                    "strong_discovery_link"
                    if overlap_pct >= 0.25
                    else "moderate_discovery_link"
                    if overlap_pct >= 0.10
                    else "weak_or_no_discovery_link"
                ),
                "geometry": geom,
            }
        )

        records.append(record)

    out = gpd.GeoDataFrame(records, geometry="geometry", crs=div.crs)

    print()
    print(f"[{SCRIPT_NAME}] Discovery-link counts:")
    print(out["divergence_discovery_link_v01"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] By divergence type:")
    print(
        out.groupby("divergence_zone_type_v01")[
            [
                "discovery_region_count_intersected_v01",
                "discovery_region_overlap_pct_v01",
                "divergence_priority_score_v01",
            ]
        ].mean()
    )

    print()
    print(f"[{SCRIPT_NAME}] Top linked divergence zones:")
    print(
        out.sort_values(
            [
                "discovery_region_overlap_pct_v01",
                "divergence_priority_score_v01",
            ],
            ascending=False,
        )[
            [
                "divergence_priority_rank_v01",
                "divergence_zone_id_v01",
                "divergence_zone_type_v01",
                "divergence_confidence_class_v01",
                "divergence_interpretation_v01",
                "cell_count_v01",
                "discovery_region_count_intersected_v01",
                "discovery_region_overlap_pct_v01",
                "divergence_discovery_link_v01",
                "divergence_priority_score_v01",
            ]
        ]
        .head(32)
        .to_string(index=False)
    )

    print()
    print(f"[{SCRIPT_NAME}] Writing GPKG: {OUTPUT_GPKG}")
    out.to_file(OUTPUT_GPKG, driver="GPKG")

    print(f"[{SCRIPT_NAME}] Writing CSV: {OUTPUT_CSV}")
    out.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()