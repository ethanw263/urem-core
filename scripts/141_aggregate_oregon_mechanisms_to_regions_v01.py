#!/usr/bin/env python3
"""
141_aggregate_oregon_mechanisms_to_regions_v01.py

Aggregate Oregon cell-level RDE mechanism components to discovery regions.

Inputs:
- data/processed/oregon_discovery_regions_v06.gpkg
- data/processed/oregon_rde_mechanism_components_v01.gpkg

Outputs:
- data/processed/oregon_region_mechanisms_v01.gpkg
- data/processed/oregon_region_mechanisms_v01.csv
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd


SCRIPT_NAME = "141_aggregate_oregon_mechanisms_to_regions_v01"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

REGIONS_PATH = PROCESSED_DIR / "oregon_discovery_regions_v06.gpkg"
MECHANISMS_PATH = PROCESSED_DIR / "oregon_rde_mechanism_components_v01.gpkg"

OUT_GPKG = PROCESSED_DIR / "oregon_region_mechanisms_v01.gpkg"
OUT_CSV = PROCESSED_DIR / "oregon_region_mechanisms_v01.csv"


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def dominant_region_mechanism(row):
    shares = {
        "opportunity_failure": row["region_opportunity_failure_share_v01"],
        "transmission_failure": row["region_transmission_failure_share_v01"],
        "recognition_inefficiency": row["region_recognition_inefficiency_share_v01"],
        "comparative_shadowing": row["region_comparative_shadowing_share_v01"],
    }

    best = max(shares, key=shares.get)
    best_value = shares[best]

    if best_value < 0.40:
        return "mixed_mechanism"

    return best


def main():
    log("Starting Oregon region mechanism aggregation")

    if not REGIONS_PATH.exists():
        raise FileNotFoundError(f"Missing regions: {REGIONS_PATH}")

    if not MECHANISMS_PATH.exists():
        raise FileNotFoundError(f"Missing mechanisms: {MECHANISMS_PATH}")

    regions = gpd.read_file(REGIONS_PATH)
    mech = gpd.read_file(MECHANISMS_PATH)

    log(f"Regions: {len(regions):,}")
    log(f"Mechanism cells: {len(mech):,}")

    if regions.crs != mech.crs:
        mech = mech.to_crs(regions.crs)

    region_rows = []

    for _, region in regions.iterrows():
        rid = region["discovery_region_id_v06"]
        geom = region.geometry

        candidates = mech[mech.geometry.intersects(geom)].copy()

        if candidates.empty:
            continue

        if "urem_score_v06_raw" in candidates.columns:
            weights = candidates["urem_score_v06_raw"].fillna(0)
        else:
            weights = pd.Series(1.0, index=candidates.index)

        if weights.sum() == 0:
            weights = pd.Series(1.0, index=candidates.index)

        def wmean(col):
            return (candidates[col].fillna(0) * weights).sum() / weights.sum()

        row = region.drop(labels="geometry").to_dict()

        row.update(
            {
                "mechanism_cell_count_v01": len(candidates),
                "region_opportunity_failure_share_v01": wmean(
                    "opportunity_failure_share_v01"
                ),
                "region_transmission_failure_share_v01": wmean(
                    "transmission_failure_share_v01"
                ),
                "region_recognition_inefficiency_share_v01": wmean(
                    "recognition_inefficiency_share_v01"
                ),
                "region_comparative_shadowing_share_v01": wmean(
                    "comparative_shadowing_share_v01"
                ),
                "region_mean_opportunity_proxy_v01": wmean(
                    "opportunity_proxy_v01"
                ),
                "region_mean_transmission_proxy_v01": wmean(
                    "transmission_proxy_v01"
                ),
                "region_mean_shadow_pressure_v01": wmean(
                    "shadow_pressure_v01"
                ),
                "region_mean_mechanism_confidence_v01": wmean(
                    "mechanism_confidence_v01"
                ),
                "geometry": geom,
            }
        )

        region_rows.append(row)

    out = gpd.GeoDataFrame(region_rows, geometry="geometry", crs=regions.crs)

    out["region_dominant_mechanism_v01"] = out.apply(
        dominant_region_mechanism,
        axis=1,
    )

    out["region_dominant_mechanism_share_v01"] = out[
        [
            "region_opportunity_failure_share_v01",
            "region_transmission_failure_share_v01",
            "region_recognition_inefficiency_share_v01",
            "region_comparative_shadowing_share_v01",
        ]
    ].max(axis=1)

    out = out.sort_values("discovery_region_rank_v06").reset_index(drop=True)

    log("Dominant region mechanism counts:")
    print(out["region_dominant_mechanism_v01"].value_counts())

    log("Top 20 region mechanisms:")
    print(
        out[
            [
                "discovery_region_rank_v06",
                "cell_count",
                "region_dominant_mechanism_v01",
                "region_dominant_mechanism_share_v01",
                "region_opportunity_failure_share_v01",
                "region_transmission_failure_share_v01",
                "region_recognition_inefficiency_share_v01",
                "region_comparative_shadowing_share_v01",
            ]
        ].head(20)
    )

    if OUT_GPKG.exists():
        OUT_GPKG.unlink()

    log(f"Writing GPKG: {OUT_GPKG}")
    out.to_file(
        OUT_GPKG,
        layer="oregon_region_mechanisms_v01",
        driver="GPKG",
    )

    log(f"Writing CSV: {OUT_CSV}")
    out.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log("Done")


if __name__ == "__main__":
    main()