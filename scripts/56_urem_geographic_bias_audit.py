#!/usr/bin/env python3
"""
56_urem_geographic_bias_audit.py

Audit geographic bias in UREM v05b.

Purpose:
- Quantify whether top UREM v05b cells are overly coastal.
- Compare top-ranked v05b candidates against:
  1. all valid land cells
  2. all v05b ranked candidates
  3. top 500 v05b cells
  4. inland-only high scoring cells

Inputs:
- data/processed/urem_score_v05b.gpkg
- data/processed/ranked_urem_candidates_v05b.gpkg

Outputs:
- data/processed/urem_geographic_bias_audit_v05b.csv
- data/processed/top_500_urem_cells_v05b_bias_audit.gpkg
- data/processed/top_500_inland_urem_cells_v05b.gpkg
- data/processed/top_500_inland_urem_cells_v05b.csv
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd


SCRIPT_NAME = "56_urem_geographic_bias_audit"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

ALL_SCORE_GPKG = PROCESSED_DIR / "urem_score_v05b.gpkg"
RANKED_GPKG = PROCESSED_DIR / "ranked_urem_candidates_v05b.gpkg"

OUT_AUDIT_CSV = PROCESSED_DIR / "urem_geographic_bias_audit_v05b.csv"
OUT_TOP500_GPKG = PROCESSED_DIR / "top_500_urem_cells_v05b_bias_audit.gpkg"
OUT_INLAND_GPKG = PROCESSED_DIR / "top_500_inland_urem_cells_v05b.gpkg"
OUT_INLAND_CSV = PROCESSED_DIR / "top_500_inland_urem_cells_v05b.csv"

TOP_N = 500
INLAND_DISTANCE_M = 10_000


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def safe_num(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    log(f"Missing column: {col}. Using default {default}.")
    return pd.Series(default, index=df.index)


def valid_land_mask(df):
    if "is_valid_land_candidate" in df.columns:
        valid = df["is_valid_land_candidate"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
    else:
        valid = pd.Series(True, index=df.index)

    if "land_area_share" in df.columns:
        valid = valid & (safe_num(df, "land_area_share") >= 0.50)

    return valid


def summarize_group(name, df):
    out = {"group": name, "row_count": len(df)}

    if len(df) == 0:
        return out

    numeric_cols = [
        "distance_to_coast_m",
        "dist_coastline_m",
        "dist_beach_m",
        "dist_cliff_m",
        "elevation_m",
        "local_relief_m",
        "slope_deg",
        "physical_exceptionality_v03",
        "physical_exceptionality_score_v02",
        "observed_recognition_v04",
        "expected_recognition_v04",
        "positive_under_recognition_residual_v04",
        "urem_score_v05b",
        "urem_score_v04",
        "terrain_drama_v03",
        "scenic_coast_v03",
        "flat_coastal_edge_penalty_v03",
        "land_area_share",
    ]

    for col in numeric_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            out[f"{col}_mean"] = s.mean()
            out[f"{col}_median"] = s.median()
            out[f"{col}_p90"] = s.quantile(0.90)

    coast_col = "distance_to_coast_m" if "distance_to_coast_m" in df.columns else None
    if coast_col:
        d = pd.to_numeric(df[coast_col], errors="coerce")
        out["pct_within_1km_coast"] = (d <= 1_000).mean()
        out["pct_within_5km_coast"] = (d <= 5_000).mean()
        out["pct_within_10km_coast"] = (d <= 10_000).mean()
        out["pct_inland_gt_10km"] = (d > 10_000).mean()
        out["pct_inland_gt_25km"] = (d > 25_000).mean()
        out["pct_inland_gt_50km"] = (d > 50_000).mean()

    return out


def main():
    log("Starting geographic bias audit v05b")

    require_file(ALL_SCORE_GPKG)
    require_file(RANKED_GPKG)

    log(f"Reading all score cells: {ALL_SCORE_GPKG}")
    all_cells = gpd.read_file(ALL_SCORE_GPKG)

    log(f"Reading ranked candidates: {RANKED_GPKG}")
    ranked = gpd.read_file(RANKED_GPKG)

    log(f"All cells: {len(all_cells):,}")
    log(f"Ranked candidates: {len(ranked):,}")

    if "urem_score_v05b" not in ranked.columns:
        raise ValueError("ranked candidates missing urem_score_v05b")

    ranked = ranked.sort_values("urem_score_v05b", ascending=False).copy()

    all_valid = all_cells[valid_land_mask(all_cells)].copy()
    ranked_valid = ranked[valid_land_mask(ranked)].copy()
    top500 = ranked_valid.head(TOP_N).copy()

    log(f"Valid land cells: {len(all_valid):,}")
    log(f"Valid ranked candidates: {len(ranked_valid):,}")
    log(f"Top {TOP_N} cells: {len(top500):,}")

    audit_rows = [
        summarize_group("all_cells", all_cells),
        summarize_group("all_valid_land_cells", all_valid),
        summarize_group("ranked_v05b_candidates", ranked_valid),
        summarize_group(f"top_{TOP_N}_v05b_cells", top500),
    ]

    if "distance_to_coast_m" in ranked_valid.columns:
        inland = ranked_valid[
            pd.to_numeric(ranked_valid["distance_to_coast_m"], errors="coerce")
            > INLAND_DISTANCE_M
        ].copy()

        top_inland = inland.sort_values("urem_score_v05b", ascending=False).head(TOP_N)

        audit_rows.append(
            summarize_group(f"top_{TOP_N}_inland_v05b_cells_gt_10km", top_inland)
        )

        log(f"Inland ranked cells >10 km from coast: {len(inland):,}")
        log(f"Writing inland top cells GPKG: {OUT_INLAND_GPKG}")
        top_inland.to_file(
            OUT_INLAND_GPKG,
            layer="top_500_inland_urem_cells_v05b",
            driver="GPKG",
        )

        log(f"Writing inland top cells CSV: {OUT_INLAND_CSV}")
        top_inland.drop(columns="geometry").to_csv(OUT_INLAND_CSV, index=False)
    else:
        log("distance_to_coast_m missing, skipping inland extract.")

    audit = pd.DataFrame(audit_rows)

    log(f"Writing audit CSV: {OUT_AUDIT_CSV}")
    audit.to_csv(OUT_AUDIT_CSV, index=False)

    log(f"Writing top 500 GPKG: {OUT_TOP500_GPKG}")
    top500.to_file(
        OUT_TOP500_GPKG,
        layer="top_500_urem_cells_v05b_bias_audit",
        driver="GPKG",
    )

    log("Done")

    print("\nGeographic bias audit:")
    display_cols = [
        "group",
        "row_count",
        "distance_to_coast_m_mean",
        "distance_to_coast_m_median",
        "pct_within_1km_coast",
        "pct_within_5km_coast",
        "pct_within_10km_coast",
        "pct_inland_gt_10km",
        "elevation_m_mean",
        "local_relief_m_mean",
        "slope_deg_mean",
        "observed_recognition_v04_mean",
        "urem_score_v05b_mean",
    ]

    display_cols = [c for c in display_cols if c in audit.columns]
    print(audit[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()