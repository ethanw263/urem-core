#!/usr/bin/env python3
"""
Script 62: PAD-US Recognition Characterization Audit

Purpose:
Determine whether top UREM locations are:
1. National Park Service lands
2. Other protected lands
3. Unprotected lands

Input:
- protected_area_audit_top100.gpkg

Outputs:
- padus_recognition_characterization.csv
- padus_owner_summary.csv
- padus_manager_summary.csv
- padus_gap_summary.csv
- padus_recognition_summary.csv
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

SCRIPT_NAME = "62_padus_recognition_characterization"

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_GPKG = (
    BASE_DIR
    / "data/processed/protected_area_audit_top100.gpkg"
)

DETAIL_CSV = (
    BASE_DIR
    / "data/processed/padus_recognition_characterization.csv"
)

OWNER_CSV = (
    BASE_DIR
    / "data/processed/padus_owner_summary.csv"
)

MANAGER_CSV = (
    BASE_DIR
    / "data/processed/padus_manager_summary.csv"
)

GAP_CSV = (
    BASE_DIR
    / "data/processed/padus_gap_summary.csv"
)

RECOGNITION_SUMMARY_CSV = (
    BASE_DIR
    / "data/processed/padus_recognition_summary.csv"
)


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def classify_row(row):

    if pd.isna(row.get("Own_Name")):
        return "Not Protected"

    owner = str(row.get("Own_Name", "")).strip().upper()

    if owner == "NPS":
        return "National Park Service"

    return "Other Protected"


def main():

    log("Starting PAD-US characterization audit")

    gdf = gpd.read_file(INPUT_GPKG)

    log(f"Rows: {len(gdf):,}")

    gdf["recognition_class"] = gdf.apply(
        classify_row,
        axis=1,
    )

    # -----------------------------------------
    # Detail export
    # -----------------------------------------

    detail_cols = [
        c for c in [
            "cell_id",
            "inside_protected_area",
            "Own_Name",
            "Mang_Name",
            "d_Own_Name",
            "d_Mang_Name",
            "GAP_Sts",
            "d_GAP_Sts",
            "recognition_class",
        ]
        if c in gdf.columns
    ]

    detail = gdf[detail_cols].copy()

    # -----------------------------------------
    # Owner summary
    # -----------------------------------------

    owner_summary = (
        gdf.groupby("d_Own_Name", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False,
        )
    )

    # -----------------------------------------
    # Manager summary
    # -----------------------------------------

    manager_summary = (
        gdf.groupby("d_Mang_Name", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False,
        )
    )

    # -----------------------------------------
    # GAP summary
    # -----------------------------------------

    gap_summary = (
        gdf.groupby("d_GAP_Sts", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False,
        )
    )

    # -----------------------------------------
    # Recognition summary
    # -----------------------------------------

    recognition_summary = (
        gdf.groupby("recognition_class")
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False,
        )
    )

    # -----------------------------------------
    # Export
    # -----------------------------------------

    log(f"Writing: {DETAIL_CSV}")
    detail.to_csv(
        DETAIL_CSV,
        index=False,
    )

    log(f"Writing: {OWNER_CSV}")
    owner_summary.to_csv(
        OWNER_CSV,
        index=False,
    )

    log(f"Writing: {MANAGER_CSV}")
    manager_summary.to_csv(
        MANAGER_CSV,
        index=False,
    )

    log(f"Writing: {GAP_CSV}")
    gap_summary.to_csv(
        GAP_CSV,
        index=False,
    )

    log(f"Writing: {RECOGNITION_SUMMARY_CSV}")
    recognition_summary.to_csv(
        RECOGNITION_SUMMARY_CSV,
        index=False,
    )

    print("\nRecognition Summary:")
    print(recognition_summary.to_string(index=False))

    log("Done")


if __name__ == "__main__":
    main()