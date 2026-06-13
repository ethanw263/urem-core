#!/usr/bin/env python3
"""
43_refine_hotspot_representative_points_v04.py

Create refined representative review points for UREM v04 hotspots.

Why this exists:
- Hotspot polygon centroids can fall in misleading locations, including water.
- A hotspot may be valid as a region but poorly represented by its centroid.
- This script selects the best actual UREM candidate cell inside each hotspot.

Inputs:
- data/processed/urem_hotspots_v04.gpkg
- data/processed/ranked_urem_candidates_v04.gpkg

Outputs:
- data/processed/urem_hotspot_representative_points_v04.csv
- data/processed/urem_hotspot_representative_points_v04.gpkg
- data/processed/urem_hotspot_representative_points_v04.kml
"""

from pathlib import Path
import warnings

import geopandas as gpd
import pandas as pd


SCRIPT_NAME = "43_refine_hotspot_representative_points_v04"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

HOTSPOTS_GPKG = PROCESSED_DIR / "urem_hotspots_v04.gpkg"
CANDIDATES_GPKG = PROCESSED_DIR / "ranked_urem_candidates_v04.gpkg"

OUT_CSV = PROCESSED_DIR / "urem_hotspot_representative_points_v04.csv"
OUT_GPKG = PROCESSED_DIR / "urem_hotspot_representative_points_v04.gpkg"
OUT_KML = PROCESSED_DIR / "urem_hotspot_representative_points_v04.kml"

TOP_N = 25


def log(msg: str) -> None:
    print(f"[{SCRIPT_NAME}] {msg}")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def ensure_rank(hotspots: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    hotspots = hotspots.copy()

    rank_col = find_col(hotspots, ["hotspot_rank_v04", "hotspot_rank", "rank"])
    mean_score_col = find_col(
        hotspots,
        ["mean_urem_score_v04", "mean_urem_score", "urem_score_mean"],
    )

    if rank_col:
        hotspots["hotspot_rank_v04"] = pd.to_numeric(hotspots[rank_col], errors="coerce")
    else:
        if mean_score_col:
            hotspots = hotspots.sort_values(mean_score_col, ascending=False)
        hotspots = hotspots.reset_index(drop=True)
        hotspots["hotspot_rank_v04"] = range(1, len(hotspots) + 1)

    return hotspots


def standardize_candidate_fields(candidates: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    candidates = candidates.copy()

    mappings = {
        "cell_id": ["cell_id", "grid_id", "id"],
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

    for standard_col, possible_cols in mappings.items():
        source_col = find_col(candidates, possible_cols)
        if source_col:
            candidates[standard_col] = candidates[source_col]
        elif standard_col not in candidates.columns:
            candidates[standard_col] = pd.NA

    return candidates


def build_representative_points(
    hotspots: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Assign candidate cells to hotspots, then choose the best actual cell per hotspot.

    Selection priority:
    1. highest UREM score
    2. highest physical exceptionality
    3. highest land share
    """

    if hotspots.crs is None:
        warnings.warn("Hotspots have no CRS. Assuming EPSG:4326.")
        hotspots = hotspots.set_crs("EPSG:4326")

    if candidates.crs is None:
        warnings.warn("Candidates have no CRS. Assuming hotspot CRS.")
        candidates = candidates.set_crs(hotspots.crs)

    hotspots = hotspots.to_crs("EPSG:3310")
    candidates = candidates.to_crs("EPSG:3310")

    hotspots_simple = hotspots[["hotspot_rank_v04", "geometry"]].copy()

    log("Spatially joining candidates to hotspots")
    joined = gpd.sjoin(
        candidates,
        hotspots_simple,
        how="inner",
        predicate="intersects",
    )

    if joined.empty:
        raise ValueError("No candidate cells intersected hotspots.")

    score_col = "urem_score_v04"
    exceptionality_col = "physical_exceptionality_v04"
    land_col = "land_share_v04"

    for col in [score_col, exceptionality_col, land_col]:
        joined[col] = pd.to_numeric(joined[col], errors="coerce")

    joined = joined.sort_values(
        ["hotspot_rank_v04", score_col, exceptionality_col, land_col],
        ascending=[True, False, False, False],
    )

    best = joined.groupby("hotspot_rank_v04").head(1).copy()

    log(f"Representative points selected: {len(best):,}")

    best_points = best.copy()
    best_points["geometry"] = best_points.geometry.centroid

    best_points = best_points.to_crs("EPSG:4326")
    best_points["representative_lon"] = best_points.geometry.x
    best_points["representative_lat"] = best_points.geometry.y

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
        "geometry",
    ]

    keep_cols = [c for c in keep_cols if c in best_points.columns]
    best_points = best_points[keep_cols].copy()

    return best_points


def add_review_columns(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    points = points.copy()

    review_cols = [
        "dominant_landscape_type",
        "representative_point_quality_1_5",
        "hotspot_region_quality_1_5",
        "scenic_quality_1_5",
        "geographic_uniqueness_1_5",
        "recreation_potential_1_5",
        "landscape_drama_1_5",
        "water_coast_relationship_1_5",
        "accessibility_1_5",
        "existing_recognition_1_5",
        "under_recognized_exceptionality_1_5",
        "hotspot_representation_failure",
        "failure_mode_notes",
        "reviewer_notes",
    ]

    for col in review_cols:
        points[col] = ""

    return points


def main() -> None:
    log("Starting Script 43: refine hotspot representative points")

    require_file(HOTSPOTS_GPKG)
    require_file(CANDIDATES_GPKG)

    log(f"Reading hotspots: {HOTSPOTS_GPKG}")
    hotspots = gpd.read_file(HOTSPOTS_GPKG)

    log(f"Reading ranked candidates: {CANDIDATES_GPKG}")
    candidates = gpd.read_file(CANDIDATES_GPKG)

    log(f"Hotspots: {len(hotspots):,}")
    log(f"Ranked candidates: {len(candidates):,}")

    hotspots = ensure_rank(hotspots)
    hotspots = hotspots.sort_values("hotspot_rank_v04").head(TOP_N).copy()

    candidates = standardize_candidate_fields(candidates)

    points = build_representative_points(hotspots, candidates)
    points = add_review_columns(points)

    log(f"Writing CSV: {OUT_CSV}")
    points.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log(f"Writing GeoPackage: {OUT_GPKG}")
    points.to_file(
        OUT_GPKG,
        layer="urem_hotspot_representative_points_v04",
        driver="GPKG",
    )

    log(f"Writing KML: {OUT_KML}")
    kml = points.copy()
    kml["Name"] = "UREM v04 Hotspot " + kml["hotspot_rank_v04"].astype(str)
    kml["Description"] = (
        "Best candidate cell: " + kml["cell_id"].astype(str)
        + " | UREM score: " + kml["urem_score_v04"].astype(str)
        + " | Land share: " + kml["land_share_v04"].astype(str)
    )

    try:
        kml[["Name", "Description", "geometry"]].to_file(OUT_KML, driver="KML")
    except Exception as exc:
        warnings.warn(f"KML export failed: {exc}")

    log("Done")
    log(f"Outputs written to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()