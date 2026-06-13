#!/usr/bin/env python3
"""
42_generate_hotspot_review_package_v04.py

Generate a hotspot-level review package for UREM v04.

Purpose:
- Package the top v04 hotspots for manual review in QGIS / Google Earth.
- Preserve model outputs without changing scores.
- Add blank manual review columns for ground-truth scoring.

Inputs:
- data/processed/urem_hotspots_v04.gpkg
- data/processed/urem_hotspot_centroids_v04.csv
- data/processed/ranked_urem_candidates_v04.gpkg  optional, if available

Outputs:
- data/processed/top_urem_hotspots_review_v04.csv
- data/processed/top_urem_hotspots_review_v04.gpkg
- data/processed/top_urem_hotspots_review_v04.kml
- data/processed/hotspot_review_scoring_template_v04.csv
"""

from pathlib import Path
import warnings

import geopandas as gpd
import pandas as pd


SCRIPT_NAME = "42_generate_hotspot_review_package_v04"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

HOTSPOTS_GPKG = PROCESSED_DIR / "urem_hotspots_v04.gpkg"
CENTROIDS_CSV = PROCESSED_DIR / "urem_hotspot_centroids_v04.csv"
RANKED_CANDIDATES_GPKG = PROCESSED_DIR / "ranked_urem_candidates_v04.gpkg"

OUT_CSV = PROCESSED_DIR / "top_urem_hotspots_review_v04.csv"
OUT_GPKG = PROCESSED_DIR / "top_urem_hotspots_review_v04.gpkg"
OUT_KML = PROCESSED_DIR / "top_urem_hotspots_review_v04.kml"
OUT_TEMPLATE = PROCESSED_DIR / "hotspot_review_scoring_template_v04.csv"

TOP_N = 25


MANUAL_REVIEW_COLUMNS = {
    "scenic_quality_1_5": "",
    "geographic_uniqueness_1_5": "",
    "recreation_potential_1_5": "",
    "landscape_drama_1_5": "",
    "water_coast_relationship_1_5": "",
    "accessibility_1_5": "",
    "existing_recognition_1_5": "",
    "under_recognized_exceptionality_1_5": "",
    "reviewer_notes": "",
    "failure_mode_notes": "",
}


FAILURE_MODE_COLUMNS = {
    "failure_already_recognized_model_missed": "",
    "failure_not_actually_exceptional": "",
    "failure_private_or_inaccessible": "",
    "failure_agricultural_or_open_land": "",
    "failure_cluster_too_broad": "",
    "failure_data_artifact": "",
    "failure_water_or_boundary_issue": "",
}


def log(msg: str) -> None:
    print(f"[{SCRIPT_NAME}] {msg}")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Return the first matching column from a list of possible names.
    Keeps the script resilient to small naming differences.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def safe_numeric(df: pd.DataFrame, col: str | None) -> pd.Series | None:
    if col is None:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def add_area_fields(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add hotspot area fields.

    Uses projected CRS EPSG:3310 for California area calculations.
    """
    gdf = gdf.copy()

    if gdf.crs is None:
        warnings.warn("Hotspot GeoPackage has no CRS. Assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")

    area_gdf = gdf.to_crs("EPSG:3310")
    gdf["hotspot_area_sq_m"] = area_gdf.geometry.area
    gdf["hotspot_area_sq_km"] = gdf["hotspot_area_sq_m"] / 1_000_000
    gdf["hotspot_area_acres"] = gdf["hotspot_area_sq_m"] * 0.000247105

    return gdf


def add_centroids(gdf: gpd.GeoDataFrame, centroids_df: pd.DataFrame | None) -> gpd.GeoDataFrame:
    """
    Add centroid latitude/longitude.

    Prefer existing centroid CSV if it has a hotspot ID/rank match.
    Otherwise compute geometric centroid from hotspot geometry.
    """
    gdf = gdf.copy()

    hotspot_id_col = find_col(gdf, ["hotspot_id", "hotspot_rank_v04", "hotspot_rank", "cluster_id"])
    centroid_id_col = find_col(centroids_df, ["hotspot_id", "hotspot_rank_v04", "hotspot_rank", "cluster_id"]) if centroids_df is not None else None

    lat_col = find_col(centroids_df, ["centroid_lat", "latitude", "lat"]) if centroids_df is not None else None
    lon_col = find_col(centroids_df, ["centroid_lon", "longitude", "lon", "lng"]) if centroids_df is not None else None

    if centroids_df is not None and hotspot_id_col and centroid_id_col and lat_col and lon_col:
        keep = centroids_df[[centroid_id_col, lat_col, lon_col]].copy()
        keep = keep.rename(
            columns={
                centroid_id_col: hotspot_id_col,
                lat_col: "centroid_lat",
                lon_col: "centroid_lon",
            }
        )
        gdf = gdf.merge(keep, on=hotspot_id_col, how="left")

    if "centroid_lat" not in gdf.columns or "centroid_lon" not in gdf.columns:
        centroid_geom = gdf.to_crs("EPSG:4326").geometry.centroid
        gdf["centroid_lon"] = centroid_geom.x
        gdf["centroid_lat"] = centroid_geom.y
    else:
        missing = gdf["centroid_lat"].isna() | gdf["centroid_lon"].isna()
        if missing.any():
            centroid_geom = gdf.to_crs("EPSG:4326").geometry.centroid
            gdf.loc[missing, "centroid_lon"] = centroid_geom.x[missing]
            gdf.loc[missing, "centroid_lat"] = centroid_geom.y[missing]

    return gdf


def add_rank_field(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Ensure hotspot_rank_v04 exists.
    """
    gdf = gdf.copy()

    existing_rank = find_col(gdf, ["hotspot_rank_v04", "hotspot_rank", "rank"])
    mean_score = find_col(gdf, ["mean_urem_score_v04", "mean_urem_score", "urem_score_mean"])
    max_score = find_col(gdf, ["max_urem_score_v04", "max_urem_score", "urem_score_max"])

    if existing_rank:
        gdf["hotspot_rank_v04"] = pd.to_numeric(gdf[existing_rank], errors="coerce")
    else:
        sort_cols = []
        ascending = []

        if mean_score:
            sort_cols.append(mean_score)
            ascending.append(False)
        if max_score:
            sort_cols.append(max_score)
            ascending.append(False)

        if sort_cols:
            gdf = gdf.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
        else:
            gdf = gdf.reset_index(drop=True)

        gdf["hotspot_rank_v04"] = range(1, len(gdf) + 1)

    return gdf


def standardize_summary_fields(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Create standardized review columns where possible.
    """
    gdf = gdf.copy()

    mappings = {
        "cell_count": ["cell_count", "n_cells", "num_cells", "count"],
        "mean_urem_score_v04": ["mean_urem_score_v04", "mean_urem_score", "urem_score_mean"],
        "max_urem_score_v04": ["max_urem_score_v04", "max_urem_score", "urem_score_max"],
        "mean_physical_exceptionality_v04": [
            "mean_physical_exceptionality_v04",
            "mean_physical_exceptionality",
            "physical_exceptionality_mean",
            "exceptionality_score_mean",
        ],
        "mean_observed_recognition_v04": [
            "mean_observed_recognition_v04",
            "mean_observed_recognition",
            "observed_recognition_mean",
        ],
        "mean_expected_recognition_v04": [
            "mean_expected_recognition_v04",
            "mean_expected_recognition",
            "expected_recognition_mean",
        ],
        "mean_under_recognition_residual_v04": [
            "mean_under_recognition_residual_v04",
            "mean_under_recognition_residual",
            "under_recognition_residual_mean",
            "recognition_residual_mean",
        ],
        "mean_land_share_v04": [
            "mean_land_share_v04",
            "mean_land_share",
            "land_share_mean",
        ],
        "best_cell_id": [
            "best_cell_id",
            "top_cell_id",
            "max_cell_id",
            "cell_id",
        ],
    }

    for standard_col, candidates in mappings.items():
        source_col = find_col(gdf, candidates)
        if source_col:
            gdf[standard_col] = gdf[source_col]
        elif standard_col not in gdf.columns:
            gdf[standard_col] = pd.NA

    return gdf


def add_nearby_candidate_count(
    hotspots: gpd.GeoDataFrame,
    ranked_candidates_path: Path,
    radius_m: float = 5000,
) -> gpd.GeoDataFrame:
    """
    Optional diagnostic:
    Count how many ranked candidate cells are near each hotspot centroid.

    This helps distinguish isolated hotspots from broader candidate regions.
    """
    hotspots = hotspots.copy()

    if not ranked_candidates_path.exists():
        hotspots["nearby_candidate_cell_count_5km"] = pd.NA
        return hotspots

    try:
        candidates = gpd.read_file(ranked_candidates_path)
    except Exception as exc:
        warnings.warn(f"Could not read ranked candidates file: {exc}")
        hotspots["nearby_candidate_cell_count_5km"] = pd.NA
        return hotspots

    if candidates.empty:
        hotspots["nearby_candidate_cell_count_5km"] = 0
        return hotspots

    if hotspots.crs is None:
        hotspots = hotspots.set_crs("EPSG:4326")

    if candidates.crs is None:
        candidates = candidates.set_crs(hotspots.crs)

    hs_proj = hotspots.to_crs("EPSG:3310")
    cand_proj = candidates.to_crs("EPSG:3310")

    centroids = hs_proj.copy()
    centroids["geometry"] = centroids.geometry.centroid
    buffers = centroids.copy()
    buffers["geometry"] = buffers.geometry.buffer(radius_m)

    joined = gpd.sjoin(
        cand_proj[["geometry"]],
        buffers[["hotspot_rank_v04", "geometry"]],
        how="inner",
        predicate="within",
    )

    counts = joined.groupby("hotspot_rank_v04").size().rename("nearby_candidate_cell_count_5km")
    hotspots = hotspots.merge(counts, on="hotspot_rank_v04", how="left")
    hotspots["nearby_candidate_cell_count_5km"] = hotspots["nearby_candidate_cell_count_5km"].fillna(0).astype(int)

    return hotspots


def add_manual_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    for col, default in MANUAL_REVIEW_COLUMNS.items():
        if col not in gdf.columns:
            gdf[col] = default

    for col, default in FAILURE_MODE_COLUMNS.items():
        if col not in gdf.columns:
            gdf[col] = default

    return gdf


def choose_output_columns(gdf: gpd.GeoDataFrame) -> list[str]:
    preferred = [
        "hotspot_rank_v04",
        "centroid_lat",
        "centroid_lon",
        "hotspot_area_sq_km",
        "hotspot_area_acres",
        "cell_count",
        "mean_urem_score_v04",
        "max_urem_score_v04",
        "mean_physical_exceptionality_v04",
        "mean_observed_recognition_v04",
        "mean_expected_recognition_v04",
        "mean_under_recognition_residual_v04",
        "mean_land_share_v04",
        "best_cell_id",
        "nearby_candidate_cell_count_5km",
    ]

    manual = list(MANUAL_REVIEW_COLUMNS.keys()) + list(FAILURE_MODE_COLUMNS.keys())

    available = [c for c in preferred + manual if c in gdf.columns]
    return available


def main() -> None:
    log("Starting Script 42: Generate v04 hotspot review package")

    require_file(HOTSPOTS_GPKG)

    log(f"Reading hotspots: {HOTSPOTS_GPKG}")
    hotspots = gpd.read_file(HOTSPOTS_GPKG)

    if hotspots.empty:
        raise ValueError("Hotspots file is empty.")

    log(f"Input hotspots: {len(hotspots):,}")

    centroids_df = None
    if CENTROIDS_CSV.exists():
        log(f"Reading centroids: {CENTROIDS_CSV}")
        centroids_df = pd.read_csv(CENTROIDS_CSV)
    else:
        log("Centroids CSV not found. Will compute centroids from geometry.")

    hotspots = add_rank_field(hotspots)
    hotspots = standardize_summary_fields(hotspots)
    hotspots = add_area_fields(hotspots)
    hotspots = add_centroids(hotspots, centroids_df)
    hotspots = add_nearby_candidate_count(hotspots, RANKED_CANDIDATES_GPKG)
    hotspots = add_manual_columns(hotspots)

    hotspots = hotspots.sort_values("hotspot_rank_v04").head(TOP_N).reset_index(drop=True)

    out_cols = choose_output_columns(hotspots)

    csv_df = pd.DataFrame(hotspots.drop(columns="geometry", errors="ignore"))
    csv_df = csv_df[out_cols]

    log(f"Writing CSV: {OUT_CSV}")
    csv_df.to_csv(OUT_CSV, index=False)

    log(f"Writing scoring template: {OUT_TEMPLATE}")
    csv_df.to_csv(OUT_TEMPLATE, index=False)

    log(f"Writing GeoPackage: {OUT_GPKG}")
    hotspots.to_file(OUT_GPKG, layer="top_urem_hotspots_review_v04", driver="GPKG")

    log(f"Writing KML: {OUT_KML}")
    kml_gdf = hotspots.to_crs("EPSG:4326").copy()
    kml_gdf["Name"] = "UREM v04 Hotspot " + kml_gdf["hotspot_rank_v04"].astype(str)
    kml_gdf["Description"] = (
        "Mean UREM: " + kml_gdf["mean_urem_score_v04"].astype(str)
        + " | Mean land share: " + kml_gdf["mean_land_share_v04"].astype(str)
        + " | Area sq km: " + kml_gdf["hotspot_area_sq_km"].round(3).astype(str)
    )

    try:
        kml_gdf[["Name", "Description", "geometry"]].to_file(OUT_KML, driver="KML")
    except Exception as exc:
        warnings.warn(f"KML export failed: {exc}")

    log("Review package complete")
    log(f"Top hotspots exported: {len(hotspots):,}")
    log(f"Outputs written to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()