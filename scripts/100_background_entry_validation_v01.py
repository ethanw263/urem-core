#!/usr/bin/env python3
"""
100_background_entry_validation_v01.py

Purpose
-------
Test whether the 99 mechanism regions are statistically distinct from the
broader cell-level RDE universe.

This validates whether RDE mechanism regions represent a meaningful entry
class rather than only an internally partitioned set.

Inputs
------
data/processed/mechanism_regions_v01.gpkg
data/processed/orthogonalized_rde_dimensions_v01.gpkg

Outputs
-------
data/processed/rde_background_entry_validation_v01.csv
data/processed/rde_background_entry_feature_tests_v01.csv
data/processed/rde_background_entry_summary_v01.csv
data/processed/rde_background_entry_qa_v01.txt
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp


SCRIPT_NAME = "100_background_entry_validation_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

REGIONS_GPKG = PROCESSED / "mechanism_regions_v01.gpkg"
CELL_UNIVERSE_GPKG = PROCESSED / "orthogonalized_rde_dimensions_v01.gpkg"

OUTPUT_CELL_VALIDATION = PROCESSED / "rde_background_entry_validation_v01.csv"
OUTPUT_FEATURE_TESTS = PROCESSED / "rde_background_entry_feature_tests_v01.csv"
OUTPUT_SUMMARY = PROCESSED / "rde_background_entry_summary_v01.csv"
OUTPUT_QA = PROCESSED / "rde_background_entry_qa_v01.txt"


FEATURE_CANDIDATES = {
    "physical_potential": [
        "P_orthogonal_v01",
        "physical_exceptionality_v03",
        "physical_potential",
        "p_orthogonal",
    ],
    "opportunity": [
        "O_base_opportunity_v01",
        "opportunity_structure_index_v01",
        "o_base_opportunity",
    ],
    "transmission": [
        "T_net_transmission_v01",
        "recognition_transmission_index_v01",
        "t_net_transmission",
    ],
    "under_recognition": [
        "R_net_under_recognition_v01",
        "under_recognition",
        "recognition_deficit",
    ],
    "rde_composite": [
        "orthogonalized_rde_v01",
        "rde_v01_composite_score",
        "rde_composite_score",
    ],
    "observed_recognition": [
        "observed_recognition_v04",
        "recognition_score_v04",
    ],
    "expected_recognition": [
        "expected_recognition_v06",
        "expected_recognition",
    ],
}


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]

    for c in df.columns:
        lc = c.lower()
        for cand in candidates:
            if cand.lower() in lc:
                return c

    return None


def load_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if not REGIONS_GPKG.exists():
        raise FileNotFoundError(f"Missing mechanism regions: {REGIONS_GPKG}")

    if not CELL_UNIVERSE_GPKG.exists():
        raise FileNotFoundError(f"Missing cell universe: {CELL_UNIVERSE_GPKG}")

    log.info("Reading mechanism regions: %s", REGIONS_GPKG)
    regions = gpd.read_file(REGIONS_GPKG)

    log.info("Reading cell universe: %s", CELL_UNIVERSE_GPKG)
    cells = gpd.read_file(CELL_UNIVERSE_GPKG)

    if regions.crs != cells.crs:
        regions = regions.to_crs(cells.crs)

    log.info("Regions: %s", len(regions))
    log.info("Cells: %s", len(cells))

    return regions, cells


def label_mechanism_cells(
    regions: gpd.GeoDataFrame,
    cells: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    log.info("Spatially labeling cells inside mechanism regions")

    cells_small = cells.copy()
    cells_small["cell_universe_id_tmp"] = np.arange(len(cells_small))

    join_cols = ["mechanism_region_id", "mechanism_class", "geometry"]

    joined = gpd.sjoin(
        cells_small,
        regions[join_cols],
        how="left",
        predicate="intersects",
    )

    joined["is_mechanism_region_cell"] = joined["mechanism_region_id"].notna().astype(int)

    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    log.info(
        "Mechanism-region cells: %s | Background cells: %s",
        int(joined["is_mechanism_region_cell"].sum()),
        int((joined["is_mechanism_region_cell"] == 0).sum()),
    )

    return joined


def build_feature_frame(labeled: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = pd.DataFrame()

    rows["is_mechanism_region_cell"] = labeled["is_mechanism_region_cell"]

    if "mechanism_region_id" in labeled.columns:
        rows["mechanism_region_id"] = labeled["mechanism_region_id"]

    if "mechanism_class" in labeled.columns:
        rows["mechanism_class"] = labeled["mechanism_class"]

    found = {}

    for feature_name, candidates in FEATURE_CANDIDATES.items():
        col = find_col(labeled, candidates)
        found[feature_name] = col

        if col is not None:
            rows[feature_name] = pd.to_numeric(labeled[col], errors="coerce")
        else:
            rows[feature_name] = np.nan

    log.info("Feature column mapping:")
    for k, v in found.items():
        log.info("  %s -> %s", k, v)

    usable = [
        c for c in FEATURE_CANDIDATES
        if rows[c].notna().sum() > 0
    ]

    if len(usable) < 3:
        raise ValueError(f"Too few usable validation features found: {usable}")

    return rows


def cliffs_delta(x: pd.Series, y: pd.Series, max_pairs: int = 2_000_000) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna().to_numpy()
    y = pd.to_numeric(y, errors="coerce").dropna().to_numpy()

    if len(x) == 0 or len(y) == 0:
        return np.nan

    rng = np.random.default_rng(42)

    if len(x) * len(y) > max_pairs:
        xs = rng.choice(x, size=min(len(x), 2000), replace=False)
        ys = rng.choice(y, size=min(len(y), 2000), replace=False)
    else:
        xs = x
        ys = y

    diff = xs[:, None] - ys[None, :]
    greater = np.sum(diff > 0)
    less = np.sum(diff < 0)

    return float((greater - less) / diff.size)


def test_feature(df: pd.DataFrame, feature: str) -> dict:
    mech = df.loc[df["is_mechanism_region_cell"] == 1, feature].dropna()
    bg = df.loc[df["is_mechanism_region_cell"] == 0, feature].dropna()

    if len(mech) < 5 or len(bg) < 5:
        return {
            "feature": feature,
            "mechanism_cell_count": len(mech),
            "background_cell_count": len(bg),
            "mechanism_mean": np.nan,
            "background_mean": np.nan,
            "mean_difference": np.nan,
            "mechanism_median": np.nan,
            "background_median": np.nan,
            "median_difference": np.nan,
            "mannwhitney_p": np.nan,
            "ks_p": np.nan,
            "cliffs_delta": np.nan,
            "effect_direction": "insufficient_data",
        }

    mw = mannwhitneyu(mech, bg, alternative="two-sided")
    ks = ks_2samp(mech, bg)

    mech_mean = float(mech.mean())
    bg_mean = float(bg.mean())

    delta = cliffs_delta(mech, bg)

    if mech_mean > bg_mean:
        direction = "mechanism_higher"
    elif mech_mean < bg_mean:
        direction = "background_higher"
    else:
        direction = "no_difference"

    return {
        "feature": feature,
        "mechanism_cell_count": int(len(mech)),
        "background_cell_count": int(len(bg)),
        "mechanism_mean": mech_mean,
        "background_mean": bg_mean,
        "mean_difference": mech_mean - bg_mean,
        "mechanism_median": float(mech.median()),
        "background_median": float(bg.median()),
        "median_difference": float(mech.median() - bg.median()),
        "mannwhitney_p": float(mw.pvalue),
        "ks_p": float(ks.pvalue),
        "cliffs_delta": delta,
        "effect_direction": direction,
    }


def classify_support(row: pd.Series) -> str:
    p = row.get("mannwhitney_p", np.nan)
    delta = abs(row.get("cliffs_delta", np.nan))

    if pd.isna(p) or pd.isna(delta):
        return "Insufficient Data"

    if p < 0.001 and delta >= 0.33:
        return "Strong Entry Evidence"
    if p < 0.01 and delta >= 0.20:
        return "Moderate Entry Evidence"
    if p < 0.05 and delta >= 0.10:
        return "Weak Entry Evidence"

    return "Low Entry Evidence"


def run_feature_tests(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for feature in FEATURE_CANDIDATES:
        if feature not in feature_df.columns:
            continue
        if feature_df[feature].notna().sum() == 0:
            continue

        rows.append(test_feature(feature_df, feature))

    tests = pd.DataFrame(rows)
    tests["entry_evidence_class"] = tests.apply(classify_support, axis=1)

    tests = tests.sort_values(
        ["entry_evidence_class", "cliffs_delta"],
        ascending=[True, False],
    )

    return tests


def build_summary(tests: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    mech_cells = int((feature_df["is_mechanism_region_cell"] == 1).sum())
    bg_cells = int((feature_df["is_mechanism_region_cell"] == 0).sum())

    rows = [
        {
            "summary_metric": "mechanism_region_cells",
            "value": mech_cells,
        },
        {
            "summary_metric": "background_cells",
            "value": bg_cells,
        },
        {
            "summary_metric": "tested_features",
            "value": len(tests),
        },
        {
            "summary_metric": "strong_entry_evidence_features",
            "value": int((tests["entry_evidence_class"] == "Strong Entry Evidence").sum()),
        },
        {
            "summary_metric": "moderate_or_stronger_entry_features",
            "value": int(tests["entry_evidence_class"].isin([
                "Strong Entry Evidence",
                "Moderate Entry Evidence",
            ]).sum()),
        },
        {
            "summary_metric": "physical_potential_support",
            "value": tests.loc[
                tests["feature"] == "physical_potential",
                "entry_evidence_class",
            ].iloc[0] if "physical_potential" in tests["feature"].values else "Not Tested",
        },
        {
            "summary_metric": "rde_composite_support",
            "value": tests.loc[
                tests["feature"] == "rde_composite",
                "entry_evidence_class",
            ].iloc[0] if "rde_composite" in tests["feature"].values else "Not Tested",
        },
    ]

    return pd.DataFrame(rows)


def write_outputs(
    feature_df: pd.DataFrame,
    tests: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    log.info("Writing cell validation frame: %s", OUTPUT_CELL_VALIDATION)
    feature_df.to_csv(OUTPUT_CELL_VALIDATION, index=False)

    log.info("Writing feature tests: %s", OUTPUT_FEATURE_TESTS)
    tests.to_csv(OUTPUT_FEATURE_TESTS, index=False)

    log.info("Writing summary: %s", OUTPUT_SUMMARY)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    qa = []
    qa.append("RDE Background Entry Validation v01 QA")
    qa.append("=" * 50)
    qa.append("")
    qa.append(f"Mechanism regions: {REGIONS_GPKG}")
    qa.append(f"Cell universe: {CELL_UNIVERSE_GPKG}")
    qa.append("")
    qa.append("Summary:")
    qa.append(summary.to_string(index=False))
    qa.append("")
    qa.append("Feature tests:")
    qa.append(tests.to_string(index=False))
    qa.append("")
    qa.append("Interpretation:")
    qa.append(
        "If physical_potential and rde_composite are higher in mechanism-region cells "
        "than background cells, the selected mechanism regions are structurally distinct "
        "from the broader geography rather than merely internally partitioned."
    )

    log.info("Writing QA: %s", OUTPUT_QA)
    OUTPUT_QA.write_text("\n".join(qa), encoding="utf-8")


def main() -> None:
    log.info("Starting Script 100: background entry validation")

    regions, cells = load_inputs()
    labeled = label_mechanism_cells(regions, cells)
    feature_df = build_feature_frame(labeled)
    tests = run_feature_tests(feature_df)
    summary = build_summary(tests, feature_df)

    write_outputs(feature_df, tests, summary)

    log.info("Done")

    print("\nRDE Background Entry Validation Summary:")
    print(summary.to_string(index=False))

    print("\nFeature Tests:")
    print(
        tests[
            [
                "feature",
                "mechanism_mean",
                "background_mean",
                "mean_difference",
                "mannwhitney_p",
                "cliffs_delta",
                "entry_evidence_class",
            ]
        ].to_string(index=False)
    )

    print("\nCreated:")
    print(f"  {OUTPUT_CELL_VALIDATION}")
    print(f"  {OUTPUT_FEATURE_TESTS}")
    print(f"  {OUTPUT_SUMMARY}")
    print(f"  {OUTPUT_QA}")


if __name__ == "__main__":
    main()