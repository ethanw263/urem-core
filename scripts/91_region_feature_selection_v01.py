#!/usr/bin/env python3
"""
91_region_feature_selection_v01.py

Purpose
-------
Compress the huge enriched mechanism-region table into an interpretable
region feature matrix for typology modeling.

Input:
data/processed/enriched_mechanism_regions_v01.csv

Outputs:
data/processed/region_feature_matrix_v01.csv
data/processed/region_feature_matrix_summary_v01.csv
data/processed/region_feature_selection_audit_v01.csv
data/processed/region_feature_selection_qa_v01.txt
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


SCRIPT_NAME = "91_region_feature_selection_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUT_CSV = PROCESSED / "enriched_mechanism_regions_v01.csv"

OUTPUT_MATRIX = PROCESSED / "region_feature_matrix_v01.csv"
OUTPUT_SUMMARY = PROCESSED / "region_feature_matrix_summary_v01.csv"
OUTPUT_AUDIT = PROCESSED / "region_feature_selection_audit_v01.csv"
OUTPUT_QA = PROCESSED / "region_feature_selection_qa_v01.txt"


KEEP_ID_COLS = [
    "mechanism_class",
    "mechanism_region_id",
    "mechanism_region_area_km2",
    "cell_count",
    "mechanism_region_priority_score_v01",
    "mechanism_region_priority_class_v01",
]


FEATURE_GROUPS: Dict[str, List[str]] = {
    "physical": [
        "physical", "phys", "exceptionality", "p_orthogonal", "relief",
        "slope", "elevation", "terrain", "rugged", "drama",
    ],
    "coastal": [
        "coast", "coastal", "shore", "ocean", "beach", "marine",
    ],
    "opportunity": [
        "opportunity", "o_base", "accessibility", "access", "infrastructure",
        "road", "parking", "trail", "urban", "settlement",
    ],
    "transmission": [
        "transmission", "t_net", "institutional", "park", "protected",
        "tourism", "recreation", "visitor", "viewpoint", "attraction",
    ],
    "recognition": [
        "recognition", "observed", "expected", "under_recognition",
        "r_net", "deficit", "recognition_score",
    ],
    "shadow_context": [
        "shadow", "diversion", "neighbor", "nearby", "destination",
        "recognized", "comparative",
    ],
    "geometry_context": [
        "area", "compact", "perim", "centroid", "lat", "lon", "cell_count",
    ],
}


MAX_FEATURES_PER_GROUP = {
    "physical": 12,
    "coastal": 8,
    "opportunity": 14,
    "transmission": 14,
    "recognition": 14,
    "shadow_context": 8,
    "geometry_context": 8,
}


BAD_COLUMN_PATTERNS = [
    "matched_cell_count",
    "_mat_cnt",
    "_std",
    "_min",
    "_max",
    "available_feature_count",
    "_n",
]


PREFERRED_STATS = [
    "_mean",
    "_median",
]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def load_input() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_CSV}. Script 91 must write the CSV before this script runs."
        )

    log.info("Reading enriched CSV: %s", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    log.info("Rows: %s | Columns: %s", len(df), len(df.columns))
    return df


def make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    seen = {}
    cols = []

    for c in df.columns:
        if c not in seen:
            seen[c] = 0
            cols.append(c)
        else:
            seen[c] += 1
            cols.append(f"{c}_{seen[c]}")

    df.columns = cols
    return df


def is_numeric_feature(df: pd.DataFrame, col: str) -> bool:
    if col in KEEP_ID_COLS:
        return False

    try:
        s = pd.to_numeric(df[col], errors="coerce")
    except Exception:
        return False

    return s.notna().sum() >= max(10, int(len(df) * 0.5))


def is_bad_column(col: str) -> bool:
    lc = norm(col)
    return any(p in lc for p in BAD_COLUMN_PATTERNS)


def group_for_column(col: str) -> List[str]:
    lc = norm(col)
    groups = []

    for group, keywords in FEATURE_GROUPS.items():
        if any(norm(k) in lc for k in keywords):
            groups.append(group)

    return groups


def feature_quality_score(df: pd.DataFrame, col: str, group: str) -> float:
    s = pd.to_numeric(df[col], errors="coerce")

    nonnull_ratio = s.notna().mean()
    variance = float(s.var(skipna=True)) if s.notna().sum() > 1 else 0.0
    unique_ratio = s.nunique(dropna=True) / max(s.notna().sum(), 1)

    stat_bonus = 0.0
    lc = norm(col)

    if any(stat.replace("_", "") in lc for stat in PREFERRED_STATS):
        stat_bonus += 0.25

    if "bundle" in lc:
        stat_bonus += 0.35

    if group in ["physical", "opportunity", "transmission", "recognition"]:
        if "orthogonal" in lc or "rde" in lc or "index" in lc or "score" in lc:
            stat_bonus += 0.20

    if is_bad_column(col):
        stat_bonus -= 0.75

    if variance <= 1e-12:
        return -999

    return (
        2.0 * nonnull_ratio
        + 1.2 * min(unique_ratio, 1.0)
        + 0.4 * np.log1p(abs(variance))
        + stat_bonus
    )


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    candidate_cols = [
        c for c in df.columns
        if is_numeric_feature(df, c)
    ]

    log.info("Candidate numeric feature columns: %s", len(candidate_cols))

    for col in candidate_cols:
        groups = group_for_column(col)

        if not groups:
            continue

        for group in groups:
            rows.append(
                {
                    "column": col,
                    "feature_group": group,
                    "quality_score": feature_quality_score(df, col, group),
                    "non_null_count": int(pd.to_numeric(df[col], errors="coerce").notna().sum()),
                    "variance": float(pd.to_numeric(df[col], errors="coerce").var(skipna=True)),
                    "mean": float(pd.to_numeric(df[col], errors="coerce").mean(skipna=True)),
                    "min": float(pd.to_numeric(df[col], errors="coerce").min(skipna=True)),
                    "max": float(pd.to_numeric(df[col], errors="coerce").max(skipna=True)),
                }
            )

    audit = pd.DataFrame(rows)

    if audit.empty:
        raise ValueError("No grouped features were found.")

    audit = audit.sort_values(
        ["feature_group", "quality_score"],
        ascending=[True, False],
    )

    selected = []

    for group, sub in audit.groupby("feature_group"):
        n = MAX_FEATURES_PER_GROUP.get(group, 10)
        chosen = sub[sub["quality_score"] > -100].head(n)
        selected.extend(chosen["column"].tolist())
        log.info("Selected %s features for %s", len(chosen), group)

    selected = list(dict.fromkeys(selected))

    id_cols = [c for c in KEEP_ID_COLS if c in df.columns]

    matrix = df[id_cols + selected].copy()

    return matrix, audit


def add_clean_interpretive_features(matrix: pd.DataFrame) -> pd.DataFrame:
    matrix = matrix.copy()

    def find_first(patterns: List[str]):
        cols = list(matrix.columns)
        for p in patterns:
            pnorm = norm(p)
            for c in cols:
                if pnorm in norm(c):
                    return c
        return None

    p_col = find_first(["mean_P_orthogonal_v01", "physical_exceptionality"])
    o_col = find_first(["mean_O_base_opportunity_v01", "opportunity_structure"])
    t_col = find_first(["mean_T_net_transmission_v01", "recognition_transmission"])
    r_col = find_first(["mean_R_net_under_recognition_v01", "under_recognition"])
    obs_col = find_first(["mean_observed_recognition_v04", "observed_recognition"])
    exp_col = find_first(["mean_expected_recognition_v06", "expected_recognition"])

    if p_col:
        matrix["core_physical_potential"] = pd.to_numeric(matrix[p_col], errors="coerce")
    if o_col:
        matrix["core_opportunity"] = pd.to_numeric(matrix[o_col], errors="coerce")
    if t_col:
        matrix["core_transmission"] = pd.to_numeric(matrix[t_col], errors="coerce")
    if r_col:
        matrix["core_under_recognition"] = pd.to_numeric(matrix[r_col], errors="coerce")
    if obs_col:
        matrix["core_observed_recognition"] = pd.to_numeric(matrix[obs_col], errors="coerce")
    if exp_col:
        matrix["core_expected_recognition"] = pd.to_numeric(matrix[exp_col], errors="coerce")

    if {"core_expected_recognition", "core_observed_recognition"}.issubset(matrix.columns):
        matrix["core_recognition_gap"] = (
            matrix["core_expected_recognition"] - matrix["core_observed_recognition"]
        )

    if {"core_physical_potential", "core_under_recognition"}.issubset(matrix.columns):
        matrix["core_latent_exceptionality"] = (
            matrix["core_physical_potential"] * matrix["core_under_recognition"]
        )

    if {"core_transmission", "core_under_recognition"}.issubset(matrix.columns):
        matrix["core_transmission_failure"] = (
            matrix["core_transmission"] * matrix["core_under_recognition"]
        )

    if {"core_opportunity", "core_under_recognition"}.issubset(matrix.columns):
        matrix["core_opportunity_adjusted_gap"] = (
            matrix["core_opportunity"] * matrix["core_under_recognition"]
        )

    return matrix


def create_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        c for c in matrix.columns
        if c not in KEEP_ID_COLS and pd.api.types.is_numeric_dtype(matrix[c])
    ]

    rows = []

    for col in numeric_cols:
        s = pd.to_numeric(matrix[col], errors="coerce")
        rows.append(
            {
                "feature": col,
                "non_null_count": int(s.notna().sum()),
                "mean": float(s.mean(skipna=True)),
                "median": float(s.median(skipna=True)),
                "std": float(s.std(skipna=True)),
                "min": float(s.min(skipna=True)),
                "max": float(s.max(skipna=True)),
            }
        )

    return pd.DataFrame(rows).sort_values("feature")


def write_outputs(matrix: pd.DataFrame, summary: pd.DataFrame, audit: pd.DataFrame) -> None:
    log.info("Writing matrix: %s", OUTPUT_MATRIX)
    matrix.to_csv(OUTPUT_MATRIX, index=False)

    log.info("Writing summary: %s", OUTPUT_SUMMARY)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    log.info("Writing audit: %s", OUTPUT_AUDIT)
    audit.to_csv(OUTPUT_AUDIT, index=False)

    qa = []
    qa.append("Region Feature Selection v01 QA")
    qa.append("=" * 40)
    qa.append("")
    qa.append(f"Input CSV: {INPUT_CSV}")
    qa.append(f"Output matrix: {OUTPUT_MATRIX}")
    qa.append(f"Region rows: {len(matrix)}")
    qa.append(f"Selected matrix columns: {len(matrix.columns)}")
    qa.append("")
    qa.append("Mechanism counts:")
    if "mechanism_class" in matrix.columns:
        qa.append(matrix["mechanism_class"].value_counts().to_string())
    qa.append("")
    qa.append("Selected feature columns:")
    for c in matrix.columns:
        qa.append(f"- {c}")

    log.info("Writing QA: %s", OUTPUT_QA)
    OUTPUT_QA.write_text("\n".join(qa), encoding="utf-8")


def main() -> None:
    log.info("Starting Script 91b: region feature selection v01")

    df = load_input()
    df = make_unique_columns(df)

    matrix, audit = select_features(df)
    matrix = add_clean_interpretive_features(matrix)

    summary = create_summary(matrix)

    write_outputs(matrix, summary, audit)

    log.info("Done")
    print("\nCreated region feature matrix:")
    print(f"  {OUTPUT_MATRIX}")
    print(f"  {OUTPUT_SUMMARY}")
    print(f"  {OUTPUT_AUDIT}")
    print(f"  {OUTPUT_QA}")
    print(f"\nRows: {len(matrix)}")
    print(f"Columns: {len(matrix.columns)}")


if __name__ == "__main__":
    main()