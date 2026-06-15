#!/usr/bin/env python3
"""
Script 57: Top 500 Feature Dominance Audit

Purpose:
Determine which variables are driving coastal dominance in v05b.

Compares:

1. Top 500 v05b cells
2. Inland top cells (>10 km from coast)
3. All ranked candidates

Outputs:
- feature_dominance_audit_v05b.csv

"""

from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    BASE_DIR
    / "data/processed/urem_score_v05b.gpkg"
)

OUT_CSV = (
    BASE_DIR
    / "data/processed/feature_dominance_audit_v05b.csv"
)

TOP_N = 500
INLAND_THRESHOLD_M = 10000


def log(msg):
    print(f"[57_top500_feature_dominance_audit] {msg}")


def safe_mean(df, col):
    if col not in df.columns:
        return np.nan
    return pd.to_numeric(df[col], errors="coerce").mean()


def main():

    log("Starting feature dominance audit")

    gdf = gpd.read_file(INPUT_PATH)

    valid = gdf[
        gdf["passes_land_filter_v05b"]
    ].copy()

    ranked = valid.sort_values(
        "urem_score_v05b",
        ascending=False
    )

    top500 = ranked.head(TOP_N).copy()

    inland = ranked[
        ranked["distance_to_coast_m"]
        > INLAND_THRESHOLD_M
    ].copy()

    log(f"Valid cells: {len(valid):,}")
    log(f"Top 500 cells: {len(top500):,}")
    log(f"Inland cells: {len(inland):,}")

    candidate_features = [

        "urem_score_v05b",

        "physical_exceptionality_v03",

        "observed_recognition_v04",

        "expected_recognition_v04",
        
        "positive_under_recognition_residual_v04",

        "terrain_drama_v03",

        "scenic_coast_v03",

        "distance_to_coast_m",

        "elevation_m",

        "local_relief_m",

        "slope_deg",

        "land_area_share",

        "cliff_proximity_v03",

        "beach_proximity_v03",

        "flat_coastal_edge_penalty_v03",

    ]

    rows = []

    for feature in candidate_features:

        if feature not in gdf.columns:
            continue

        top_mean = safe_mean(top500, feature)
        inland_mean = safe_mean(inland, feature)
        ranked_mean = safe_mean(ranked, feature)

        rows.append(
            {
                "feature": feature,
                "top500_mean": top_mean,
                "inland_mean": inland_mean,
                "ranked_mean": ranked_mean,
                "top_minus_inland":
                    top_mean - inland_mean,
                "top_div_inland":
                    (
                        top_mean / inland_mean
                        if inland_mean not in [0, np.nan]
                        else np.nan
                    )
            }
        )

    out = pd.DataFrame(rows)

    out["abs_difference"] = (
        out["top_minus_inland"]
        .abs()
    )

    out = out.sort_values(
        "abs_difference",
        ascending=False
    )

    OUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    log(f"Writing CSV: {OUT_CSV}")

    out.to_csv(
        OUT_CSV,
        index=False
    )

    log("Done")

    print("\nTop feature differences:")
    print(
        out[
            [
                "feature",
                "top500_mean",
                "inland_mean",
                "top_minus_inland"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()