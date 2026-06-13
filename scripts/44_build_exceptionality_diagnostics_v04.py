#!/usr/bin/env python3
"""
44_build_exceptionality_diagnostics_v04.py

Diagnose why UREM v04 hotspots are ranking highly.

This script does NOT change the model.
It summarizes top hotspot representative points and candidate cells so we can see
whether false positives are driven by:
- water adjacency
- wetland/slough geometry
- urban/industrial shoreline
- weak naturalness filtering
- low terrain/scenic signal
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd


SCRIPT_NAME = "44_build_exceptionality_diagnostics_v04"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

REP_POINTS = PROCESSED_DIR / "urem_hotspot_representative_points_v04.gpkg"
CANDIDATES = PROCESSED_DIR / "ranked_urem_candidates_v04.gpkg"
HOTSPOTS = PROCESSED_DIR / "urem_hotspots_v04.gpkg"

OUT_CSV = PROCESSED_DIR / "exceptionality_diagnostics_v04.csv"
OUT_GPKG = PROCESSED_DIR / "exceptionality_diagnostics_v04.gpkg"

TOP_N = 25


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def find_col(df, names):
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def classify_landscape(row):
    """
    Initial rough diagnostic classification.
    This is intentionally simple and review-oriented.
    """

    text_parts = []

    for col in row.index:
        val = row[col]
        if pd.notna(val):
            text_parts.append(str(val).lower())

    text = " ".join(text_parts)

    if any(x in text for x in ["industrial", "port", "harbor", "airport", "commercial"]):
        return "urban_industrial_possible"

    if any(x in text for x in ["residential", "suburban", "urban"]):
        return "urban_residential_possible"

    if any(x in text for x in ["wetland", "marsh", "slough", "estuary", "tidal"]):
        return "wetland_slough_possible"

    if any(x in text for x in ["beach", "coast", "shore", "bay", "water"]):
        return "water_adjacent_possible"

    return "unclassified"


def main():
    log("Starting Script 44: Exceptionality diagnostics v04")

    require_file(REP_POINTS)
    require_file(CANDIDATES)
    require_file(HOTSPOTS)

    log(f"Reading representative points: {REP_POINTS}")
    reps = gpd.read_file(REP_POINTS)

    log(f"Reading candidates: {CANDIDATES}")
    candidates = gpd.read_file(CANDIDATES)

    log(f"Reading hotspots: {HOTSPOTS}")
    hotspots = gpd.read_file(HOTSPOTS)

    reps = reps.sort_values("hotspot_rank_v04").head(TOP_N).copy()

    score_cols = {
        "urem_score_v04": ["urem_score_v04", "urem_score", "urem_score_norm"],
        "physical_exceptionality_v04": [
            "physical_exceptionality_v04",
            "physical_exceptionality",
            "exceptionality_score_v02",
            "exceptionality_score",
        ],
        "observed_recognition_v04": [
            "observed_recognition_v04",
            "observed_recognition",
            "recognition_score_v04",
        ],
        "expected_recognition_v04": [
            "expected_recognition_v04",
            "expected_recognition",
        ],
        "under_recognition_residual_v04": [
            "under_recognition_residual_v04",
            "under_recognition_residual",
            "recognition_residual",
        ],
        "land_share_v04": [
            "land_share_v04",
            "land_share",
            "valid_land_share",
        ],
    }

    for standard, possible in score_cols.items():
        col = find_col(reps, possible)
        if col and col != standard:
            reps[standard] = reps[col]
        elif standard not in reps.columns:
            reps[standard] = pd.NA

    reps["diagnostic_landscape_guess"] = reps.apply(classify_landscape, axis=1)

    reps_3310 = reps.to_crs("EPSG:3310")
    candidates_3310 = candidates.to_crs("EPSG:3310")

    buffer_rows = []

    for _, rep in reps_3310.iterrows():
        rank = rep["hotspot_rank_v04"]
        geom = rep.geometry.buffer(3000)

        nearby = candidates_3310[candidates_3310.intersects(geom)].copy()

        row = {
            "hotspot_rank_v04": rank,
            "nearby_candidate_count_3km": len(nearby),
        }

        for standard, possible in score_cols.items():
            col = find_col(nearby, possible)
            if col and len(nearby) > 0:
                vals = pd.to_numeric(nearby[col], errors="coerce")
                row[f"nearby_mean_{standard}"] = vals.mean()
                row[f"nearby_max_{standard}"] = vals.max()
            else:
                row[f"nearby_mean_{standard}"] = pd.NA
                row[f"nearby_max_{standard}"] = pd.NA

        buffer_rows.append(row)

    buffer_df = pd.DataFrame(buffer_rows)

    out = reps.merge(buffer_df, on="hotspot_rank_v04", how="left")

    out["manual_region_quality_1_5"] = ""
    out["manual_point_quality_1_5"] = ""
    out["manual_urem_success_1_5"] = ""
    out["dominant_landscape_type"] = ""
    out["likely_failure_mode"] = ""
    out["review_notes"] = ""

    keep_cols = [
        "hotspot_rank_v04",
        "cell_id",
        "representative_lat",
        "representative_lon",
        "urem_score_v04",
        "physical_exceptionality_v04",
        "observed_recognition_v04",
        "expected_recognition_v04",
        "under_recognition_residual_v04",
        "land_share_v04",
        "diagnostic_landscape_guess",
        "nearby_candidate_count_3km",
        "nearby_mean_urem_score_v04",
        "nearby_max_urem_score_v04",
        "nearby_mean_physical_exceptionality_v04",
        "nearby_max_physical_exceptionality_v04",
        "nearby_mean_observed_recognition_v04",
        "nearby_mean_expected_recognition_v04",
        "nearby_mean_under_recognition_residual_v04",
        "nearby_mean_land_share_v04",
        "manual_region_quality_1_5",
        "manual_point_quality_1_5",
        "manual_urem_success_1_5",
        "dominant_landscape_type",
        "likely_failure_mode",
        "review_notes",
        "geometry",
    ]

    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols]

    log(f"Writing CSV: {OUT_CSV}")
    out.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log(f"Writing GeoPackage: {OUT_GPKG}")
    out.to_file(OUT_GPKG, layer="exceptionality_diagnostics_v04", driver="GPKG")

    log("Done")


if __name__ == "__main__":
    main()