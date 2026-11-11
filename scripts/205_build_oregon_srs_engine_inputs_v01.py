from pathlib import Path
import geopandas as gpd
import pandas as pd

SCRIPT = "205_build_oregon_srs_engine_inputs_v01"

BASE = Path("data/processed")

GRID_FP = BASE / "oregon_coast_grid_v01.gpkg"
FEATURES_FP = BASE / "oregon_exceptionality_features_v03.gpkg"
PHI_FP = BASE / "oregon_exceptionality_score_v03.gpkg"
Y_FP = BASE / "oregon_recognition_score_v04.gpkg"

OUT_GPKG = BASE / "oregon_srs_engine_inputs_v01.gpkg"
OUT_SCHEMA = BASE / "oregon_srs_engine_inputs_schema_v01.csv"
OUT_QA = BASE / "oregon_srs_engine_inputs_qa_v01.txt"

KEY = "cell_id"


def log(msg):
    print(f"[{SCRIPT}] {msg}")


def read_layer(fp):
    if not fp.exists():
        raise FileNotFoundError(f"Missing required file: {fp}")
    return gpd.read_file(fp)


def require_unique(gdf, name):
    if KEY not in gdf.columns:
        raise ValueError(f"{name} missing {KEY}")
    if gdf[KEY].isna().any():
        raise ValueError(f"{name} has null {KEY}")
    if gdf[KEY].duplicated().any():
        raise ValueError(f"{name} has duplicate {KEY}")


def select_existing(df, cols):
    return [c for c in cols if c in df.columns]


def main():
    log("Reading authoritative Oregon SRS input layers")

    grid = read_layer(GRID_FP)
    features = read_layer(FEATURES_FP)
    phi = read_layer(PHI_FP)
    y = read_layer(Y_FP)

    for name, gdf in {
        "grid": grid,
        "features": features,
        "phi": phi,
        "recognition": y,
    }.items():
        require_unique(gdf, name)
        log(f"{name}: rows={len(gdf):,}, crs={gdf.crs}")

    if not (grid.crs == features.crs == phi.crs == y.crs):
        raise ValueError("CRS mismatch among input layers")

    grid_cols = [
        KEY, "domain", "grid_m", "area_sq_km", "centroid_x", "centroid_y", "geometry"
    ]

    feature_cols = [
        KEY,
        "elevation_m",
        "local_relief_m",
        "slope_deg",
        "slope_pct",
        "dist_coastline_m",
        "coastline_length_3km_m",
        "coastline_complexity_3km",
    ]

    phi_cols = [
        KEY,
        "physical_exceptionality_v03",
        "physical_exceptionality_raw_v03",
        "exceptionality_v03_rank",
        "base_exceptionality_v02_norm",
        "terrain_drama_v03",
        "scenic_coast_v03",
        "relief_norm_v03",
        "slope_norm_v03",
        "coast_proximity_v03",
        "beach_proximity_v03",
        "cliff_proximity_v03",
        "coast_complexity_v03",
        "flat_coastal_edge_penalty_v03",
        "complex_flat_shoreline_penalty_v03",
    ]

    y_cols = [
        KEY,
        "land_area_share",
        "water_area_share",
        "centroid_on_land",
        "is_valid_land_candidate",
        "trail_path_count_3km",
        "trail_path_score_v04",
        "beach_count_3km",
        "beach_score_v04",
        "park_recreation_count_3km",
        "park_recreation_score_v04",
        "protected_area_count_3km",
        "protected_area_score_v04",
        "parking_count_3km",
        "parking_score_v04",
        "trailhead_count_3km",
        "trailhead_score_v04",
        "visitor_information_count_3km",
        "visitor_information_score_v04",
        "viewpoint_count_3km",
        "viewpoint_score_v04",
        "named_natural_feature_count_3km",
        "named_natural_feature_score_v04",
        "tourism_recreation_count_3km",
        "tourism_recreation_score_v04",
        "recognition_access_v04",
        "recognition_recreation_v04",
        "recognition_natural_v04",
        "recognition_total_count_3km_v04",
        "recognition_category_coverage_v04",
        "recognition_trails_access_v04",
        "recognition_natural_protected_v04",
        "observed_recognition_v04_raw",
        "observed_recognition_v04",
        "recognition_dataset_confidence_v04",
        "recognition_cell_confidence_v04",
        "recognition_valid_land_flag_v04",
        "observed_recognition_tier_v04",
    ]

    log("Building canonical engine input table")

    out = grid[select_existing(grid, grid_cols)].copy()

    features_df = pd.DataFrame(features[select_existing(features, feature_cols)]).drop(columns="geometry", errors="ignore")
    phi_df = pd.DataFrame(phi[select_existing(phi, phi_cols)]).drop(columns="geometry", errors="ignore")
    y_df = pd.DataFrame(y[select_existing(y, y_cols)]).drop(columns="geometry", errors="ignore")

    out = out.merge(features_df, on=KEY, how="left", validate="one_to_one")
    out = out.merge(phi_df, on=KEY, how="left", validate="one_to_one")
    out = out.merge(y_df, on=KEY, how="left", validate="one_to_one")

    out["phi_physical_v01"] = out["physical_exceptionality_v03"]
    out["y_observed_v01"] = out["observed_recognition_v04"]

    out["srs_valid_engine_row_v01"] = (
        out["phi_physical_v01"].notna()
        & out["y_observed_v01"].notna()
        & out["is_valid_land_candidate"].fillna(False).astype(bool)
    )

    schema_rows = []
    for col in out.columns:
        if col == "geometry":
            role = "geometry"
            allowed = "all"
        elif col in ["cell_id", "domain", "grid_m", "area_sq_km", "centroid_x", "centroid_y"]:
            role = "place_domain"
            allowed = "all"
        elif col in ["phi_physical_v01", "physical_exceptionality_v03"]:
            role = "phi_provisional"
            allowed = "pointwise, relational, constrained"
        elif col in ["y_observed_v01", "observed_recognition_v04", "observed_recognition_v04_raw"]:
            role = "realized_recognition_target"
            allowed = "target_only"
        elif "recognition" in col or col.endswith("_score_v04") or col.endswith("_count_3km"):
            role = "recognition_trace"
            allowed = "target_components_or_validation_only"
        elif col in ["is_valid_land_candidate", "recognition_valid_land_flag_v04", "land_area_share", "water_area_share"]:
            role = "validity_mask"
            allowed = "filter_or_qa"
        else:
            role = "recognizability_feature"
            allowed = "pointwise, relational"

        schema_rows.append({
            "column": col,
            "role": role,
            "allowed_use": allowed,
            "source_note": "assembled by Script 205 from authoritative Oregon processed layers",
        })

    schema = pd.DataFrame(schema_rows)

    log("Writing outputs")
    out.to_file(OUT_GPKG, driver="GPKG")
    schema.to_csv(OUT_SCHEMA, index=False)

    qa_lines = []
    qa_lines.append(f"{SCRIPT} QA")
    qa_lines.append("=" * 80)
    qa_lines.append(f"Output: {OUT_GPKG}")
    qa_lines.append(f"Rows: {len(out):,}")
    qa_lines.append(f"Columns: {len(out.columns)}")
    qa_lines.append(f"CRS: {out.crs}")
    qa_lines.append(f"Unique cell_id: {out[KEY].nunique():,}")
    qa_lines.append(f"Duplicate cell_id: {out[KEY].duplicated().sum():,}")
    qa_lines.append("")
    qa_lines.append("Core missingness:")
    for col in [
        "phi_physical_v01",
        "y_observed_v01",
        "physical_exceptionality_v03",
        "observed_recognition_v04",
        "is_valid_land_candidate",
        "srs_valid_engine_row_v01",
    ]:
        if col in out.columns:
            qa_lines.append(f"{col}: missing={out[col].isna().sum():,}")

    qa_lines.append("")
    qa_lines.append("Valid engine rows:")
    qa_lines.append(str(out["srs_valid_engine_row_v01"].value_counts(dropna=False).to_dict()))

    numeric = out.select_dtypes(include="number")
    qa_lines.append("")
    qa_lines.append("Top numeric null rates:")
    qa_lines.append(numeric.isna().mean().sort_values(ascending=False).head(20).to_string())

    OUT_QA.write_text("\n".join(qa_lines), encoding="utf-8")

    log(f"Created {OUT_GPKG}")
    log(f"Created {OUT_SCHEMA}")
    log(f"Created {OUT_QA}")
    log("Done")


if __name__ == "__main__":
    main()