#!/usr/bin/env python3

"""
154_identify_oregon_divergence_source_sink_zones_v01.py

Groups recognition divergence cells into spatially coherent source/sink zones.

Input:
    data/processed/oregon_recognition_divergence_field_v01.gpkg

Outputs:
    data/processed/oregon_divergence_source_sink_zones_v01.gpkg
    data/processed/oregon_divergence_source_sink_zones_v01.csv
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np


SCRIPT_NAME = "154_identify_oregon_divergence_source_sink_zones_v01"

INPUT_GPKG = Path("data/processed/oregon_recognition_divergence_field_v01.gpkg")

OUTPUT_GPKG = Path("data/processed/oregon_divergence_source_sink_zones_v01.gpkg")
OUTPUT_CSV = Path("data/processed/oregon_divergence_source_sink_zones_v01.csv")

SOURCE_CLASSES = {
    "strong_source",
    "moderate_source",
}

SINK_CLASSES = {
    "strong_sink",
    "moderate_sink",
}

MIN_ZONE_CELLS = 3


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    gdf = gpd.read_file(INPUT_GPKG)

    print(f"[{SCRIPT_NAME}] Rows: {len(gdf):,}")
    print(f"[{SCRIPT_NAME}] CRS: {gdf.crs}")

    required = [
        "recognition_divergence_class_v01",
        "recognition_divergence_v01",
        "recognition_divergence_norm_v01",
        "recognition_source_strength_v01",
        "recognition_sink_strength_v01",
    ]

    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    gdf["divergence_zone_type_v01"] = np.select(
        [
            gdf["recognition_divergence_class_v01"].isin(SOURCE_CLASSES),
            gdf["recognition_divergence_class_v01"].isin(SINK_CLASSES),
        ],
        [
            "source_zone_candidate",
            "sink_zone_candidate",
        ],
        default="not_zone_candidate",
    )

    candidates = gdf[
        gdf["divergence_zone_type_v01"].isin(
            ["source_zone_candidate", "sink_zone_candidate"]
        )
    ].copy()

    print(f"[{SCRIPT_NAME}] Candidate cells: {len(candidates):,}")

    zones = []

    for zone_type, subset in candidates.groupby("divergence_zone_type_v01"):
        print(f"[{SCRIPT_NAME}] Dissolving {zone_type}: {len(subset):,} cells")

        dissolved = subset.dissolve(
            by="divergence_zone_type_v01",
            as_index=False
        )

        exploded = dissolved.explode(index_parts=False).reset_index(drop=True)

        for i, row in exploded.iterrows():
            geom = row.geometry

            cells_in_zone = subset[subset.intersects(geom)]

            n_cells = len(cells_in_zone)

            if n_cells < MIN_ZONE_CELLS:
                continue

            zone_id = f"{zone_type.replace('_candidate', '')}_{i + 1:03d}"

            zones.append(
                {
                    "divergence_zone_id_v01": zone_id,
                    "divergence_zone_type_v01": zone_type.replace("_candidate", ""),
                    "cell_count_v01": n_cells,
                    "mean_divergence_v01": cells_in_zone["recognition_divergence_v01"].mean(),
                    "mean_divergence_norm_v01": cells_in_zone["recognition_divergence_norm_v01"].mean(),
                    "max_source_strength_v01": cells_in_zone["recognition_source_strength_v01"].max(),
                    "max_sink_strength_v01": cells_in_zone["recognition_sink_strength_v01"].max(),
                    "mean_source_strength_v01": cells_in_zone["recognition_source_strength_v01"].mean(),
                    "mean_sink_strength_v01": cells_in_zone["recognition_sink_strength_v01"].mean(),
                    "geometry": geom,
                }
            )

    zones_gdf = gpd.GeoDataFrame(
        zones,
        geometry="geometry",
        crs=gdf.crs,
    )

    if len(zones_gdf) == 0:
        print(f"[{SCRIPT_NAME}] No retained zones found.")
        return

    zones_gdf["zone_area_km2_v01"] = zones_gdf.geometry.area / 1_000_000

    zones_gdf = zones_gdf.sort_values(
        ["divergence_zone_type_v01", "cell_count_v01"],
        ascending=[True, False],
    ).reset_index(drop=True)

    print()
    print(f"[{SCRIPT_NAME}] Retained zones: {len(zones_gdf):,}")
    print(zones_gdf["divergence_zone_type_v01"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] Largest zones:")
    print(
        zones_gdf[
            [
                "divergence_zone_id_v01",
                "divergence_zone_type_v01",
                "cell_count_v01",
                "mean_divergence_norm_v01",
                "zone_area_km2_v01",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print()
    print(f"[{SCRIPT_NAME}] Writing GPKG: {OUTPUT_GPKG}")
    zones_gdf.to_file(OUTPUT_GPKG, driver="GPKG")

    print(f"[{SCRIPT_NAME}] Writing CSV: {OUTPUT_CSV}")
    zones_gdf.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()