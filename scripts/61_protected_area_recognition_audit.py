#!/usr/bin/env python3
"""
61_protected_area_recognition_audit.py

Audit whether the current v06 exceptional-residual review cells are already
inside PAD-US protected areas.

Inputs:
- data/processed/v06_review_package_exceptional_residual.gpkg
- data/processed/california_protected_areas.gpkg

Outputs:
- data/processed/protected_area_audit_v06_exceptional_residual.csv
- data/processed/protected_area_audit_v06_exceptional_residual.gpkg
- data/processed/protected_area_summary_v06_exceptional_residual.csv
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd

SCRIPT_NAME = "61_protected_area_recognition_audit"

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

UREM_INPUT = PROCESSED_DIR / "v06_review_package_exceptional_residual.gpkg"
PADUS_INPUT = PROCESSED_DIR / "california_protected_areas.gpkg"

OUTPUT_GPKG = PROCESSED_DIR / "protected_area_audit_v06_exceptional_residual.gpkg"
OUTPUT_CSV = PROCESSED_DIR / "protected_area_audit_v06_exceptional_residual.csv"
SUMMARY_CSV = PROCESSED_DIR / "protected_area_summary_v06_exceptional_residual.csv"


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def main():
    log("Starting protected-area audit for v06 exceptional-residual review package")

    urem = gpd.read_file(UREM_INPUT)
    padus = gpd.read_file(PADUS_INPUT)

    log(f"UREM review rows: {len(urem):,}")
    log(f"PAD-US rows: {len(padus):,}")

    if urem.crs != padus.crs:
        padus = padus.to_crs(urem.crs)

    joined = gpd.sjoin(
        urem,
        padus,
        how="left",
        predicate="intersects",
    )

    joined["inside_protected_area"] = joined["index_right"].notna()

    # Cell-level summary, deduplicated because a cell can intersect multiple PAD-US polygons.
    cell_summary = (
        joined.groupby("cell_id")
        .agg(
            review_rank=("review_rank", "first"),
            longitude=("longitude", "first"),
            latitude=("latitude", "first"),
            physical_exceptionality_v03=("physical_exceptionality_v03", "first"),
            observed_recognition_v04=("observed_recognition_v04", "first"),
            expected_recognition_v06_raw=("expected_recognition_v06_raw", "first"),
            positive_under_recognition_residual_v06=(
                "positive_under_recognition_residual_v06",
                "first",
            ),
            any_protected=("inside_protected_area", "max"),
            padus_intersection_count=("inside_protected_area", "sum"),
            owner_codes=("Own_Name", lambda x: "; ".join(sorted(set(x.dropna().astype(str))))),
            manager_codes=("Mang_Name", lambda x: "; ".join(sorted(set(x.dropna().astype(str))))),
            owner_names=("d_Own_Name", lambda x: "; ".join(sorted(set(x.dropna().astype(str))))),
            manager_names=("d_Mang_Name", lambda x: "; ".join(sorted(set(x.dropna().astype(str))))),
            gap_statuses=("GAP_Sts", lambda x: "; ".join(sorted(set(x.dropna().astype(str))))),
            gap_descriptions=("d_GAP_Sts", lambda x: "; ".join(sorted(set(x.dropna().astype(str))))),
            geometry=("geometry", "first"),
        )
        .reset_index()
    )

    cell_summary = gpd.GeoDataFrame(cell_summary, geometry="geometry", crs=urem.crs)

    cell_summary["recognition_protection_class"] = "Not Protected"
    cell_summary.loc[
        cell_summary["any_protected"] & cell_summary["owner_codes"].str.contains("NPS", na=False),
        "recognition_protection_class",
    ] = "National Park Service"
    cell_summary.loc[
        cell_summary["any_protected"]
        & (cell_summary["recognition_protection_class"] == "Not Protected"),
        "recognition_protection_class",
    ] = "Other Protected"

    summary = (
        cell_summary.groupby("recognition_protection_class")
        .size()
        .reset_index(name="cell_count")
        .sort_values("cell_count", ascending=False)
    )

    summary["pct_of_review_cells"] = summary["cell_count"] / len(cell_summary)

    log(f"Writing CSV: {OUTPUT_CSV}")
    cell_summary.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

    log(f"Writing GeoPackage: {OUTPUT_GPKG}")
    cell_summary.to_file(OUTPUT_GPKG, layer="protected_area_audit_v06_exceptional_residual", driver="GPKG")

    log(f"Writing summary: {SUMMARY_CSV}")
    summary.to_csv(SUMMARY_CSV, index=False)

    log("Done")

    print("\nProtected Area Summary, deduplicated by cell:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()