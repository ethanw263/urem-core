#!/usr/bin/env python3
"""
96_theory_readiness_filter_v01.py

Purpose
-------
Convert RDE theory validation results into a paper-ready readiness hierarchy.

Input:
data/processed/rde_theory_validation_v01.csv

Outputs:
data/processed/rde_theory_readiness_filter_v01.csv
data/processed/rde_theory_readiness_summary_v01.csv
data/processed/rde_core_validated_theory_v01.csv
data/processed/rde_emerging_theory_v01.csv
data/processed/rde_exploratory_holdout_theory_v01.csv
data/processed/rde_theory_readiness_qa_v01.txt
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_NAME = "96_theory_readiness_filter_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUT_VALIDATION = PROCESSED / "rde_theory_validation_v01.csv"

OUTPUT_FILTER = PROCESSED / "rde_theory_readiness_filter_v01.csv"
OUTPUT_SUMMARY = PROCESSED / "rde_theory_readiness_summary_v01.csv"
OUTPUT_CORE = PROCESSED / "rde_core_validated_theory_v01.csv"
OUTPUT_EMERGING = PROCESSED / "rde_emerging_theory_v01.csv"
OUTPUT_HOLDOUT = PROCESSED / "rde_exploratory_holdout_theory_v01.csv"
OUTPUT_QA = PROCESSED / "rde_theory_readiness_qa_v01.txt"


CORE_MIN_SCORE = 0.60
CORE_MIN_REGIONS = 5

EMERGING_MIN_SCORE = 0.45
EMERGING_MIN_REGIONS = 2


def load_validation() -> pd.DataFrame:
    if not INPUT_VALIDATION.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_VALIDATION}")

    log.info("Reading validation results: %s", INPUT_VALIDATION)
    df = pd.read_csv(INPUT_VALIDATION, low_memory=False)

    required = [
        "mechanism",
        "archetype",
        "region_count",
        "overall_evidence_score",
        "evidence_class",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Validation file missing required columns: {missing}")

    log.info("Validation rows: %s", len(df))
    return df


def assign_readiness(row: pd.Series) -> str:
    score = row.get("overall_evidence_score")
    regions = row.get("region_count")

    try:
        score = float(score)
    except Exception:
        score = np.nan

    try:
        regions = int(regions)
    except Exception:
        regions = 0

    if pd.isna(score):
        return "Exploratory / Holdout Theory"

    if score >= CORE_MIN_SCORE and regions >= CORE_MIN_REGIONS:
        return "Core Validated Theory"

    if score >= EMERGING_MIN_SCORE and regions >= EMERGING_MIN_REGIONS:
        return "Emerging Theory"

    return "Exploratory / Holdout Theory"


def assign_recommendation(row: pd.Series) -> str:
    readiness = row["theory_readiness_tier"]
    mechanism = row["mechanism"]
    archetype = row["archetype"]

    if readiness == "Core Validated Theory":
        return (
            "Use as a main paper finding. Treat as part of the current validated "
            "Recognition Disequilibrium framework."
        )

    if readiness == "Emerging Theory":
        return (
            "Keep in the paper as a secondary or exploratory finding. Requires "
            "additional validation, sensitivity testing, or richer mechanism variables."
        )

    if "Comparative Shadowing" in str(mechanism):
        return (
            "Hold out from main claims for now. Improve shadow/diversion variables "
            "before treating this as a validated mechanism."
        )

    if "single" in str(row.get("evidence_class", "")).lower() or row.get("region_count", 0) == 1:
        return (
            "Do not generalize yet. Treat as a case-study candidate or anomaly for "
            "future qualitative review."
        )

    return (
        "Retain for future testing but do not use as a central theoretical claim yet."
    )


def assign_publication_role(row: pd.Series) -> str:
    tier = row["theory_readiness_tier"]

    if tier == "Core Validated Theory":
        return "Main Results / Theory Section"

    if tier == "Emerging Theory":
        return "Exploratory Results / Discussion"

    return "Appendix / Future Work / Case Study Candidate"


def apply_readiness_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["theory_readiness_tier"] = df.apply(assign_readiness, axis=1)
    df["recommended_use"] = df.apply(assign_recommendation, axis=1)
    df["publication_role"] = df.apply(assign_publication_role, axis=1)

    tier_order = {
        "Core Validated Theory": 1,
        "Emerging Theory": 2,
        "Exploratory / Holdout Theory": 3,
    }

    df["readiness_rank"] = df["theory_readiness_tier"].map(tier_order)

    df = df.sort_values(
        ["readiness_rank", "mechanism", "overall_evidence_score", "region_count"],
        ascending=[True, True, False, False],
    )

    return df


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["theory_readiness_tier", "mechanism"])
        .agg(
            archetype_count=("archetype", "count"),
            total_regions=("region_count", "sum"),
            mean_evidence_score=("overall_evidence_score", "mean"),
            max_evidence_score=("overall_evidence_score", "max"),
        )
        .reset_index()
        .sort_values(
            ["theory_readiness_tier", "mean_evidence_score"],
            ascending=[True, False],
        )
    )

    return summary


def write_outputs(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    log.info("Writing readiness filter: %s", OUTPUT_FILTER)
    df.to_csv(OUTPUT_FILTER, index=False)

    log.info("Writing readiness summary: %s", OUTPUT_SUMMARY)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    core = df[df["theory_readiness_tier"] == "Core Validated Theory"].copy()
    emerging = df[df["theory_readiness_tier"] == "Emerging Theory"].copy()
    holdout = df[df["theory_readiness_tier"] == "Exploratory / Holdout Theory"].copy()

    log.info("Writing core validated theory: %s", OUTPUT_CORE)
    core.to_csv(OUTPUT_CORE, index=False)

    log.info("Writing emerging theory: %s", OUTPUT_EMERGING)
    emerging.to_csv(OUTPUT_EMERGING, index=False)

    log.info("Writing exploratory holdout theory: %s", OUTPUT_HOLDOUT)
    holdout.to_csv(OUTPUT_HOLDOUT, index=False)

    qa = []
    qa.append("RDE Theory Readiness Filter v01 QA")
    qa.append("=" * 45)
    qa.append("")
    qa.append(f"Input: {INPUT_VALIDATION}")
    qa.append(f"Total archetypes evaluated: {len(df)}")
    qa.append("")
    qa.append("Readiness counts:")
    qa.append(df["theory_readiness_tier"].value_counts().to_string())
    qa.append("")
    qa.append("Readiness summary:")
    qa.append(summary.to_string(index=False))
    qa.append("")
    qa.append("Core validated theory:")
    if len(core) > 0:
        qa.append(core[["mechanism", "archetype", "region_count", "overall_evidence_score"]].to_string(index=False))
    else:
        qa.append("None")
    qa.append("")
    qa.append("Emerging theory:")
    if len(emerging) > 0:
        qa.append(emerging[["mechanism", "archetype", "region_count", "overall_evidence_score"]].to_string(index=False))
    else:
        qa.append("None")
    qa.append("")
    qa.append("Exploratory / holdout theory:")
    if len(holdout) > 0:
        qa.append(holdout[["mechanism", "archetype", "region_count", "overall_evidence_score", "evidence_class"]].to_string(index=False))
    else:
        qa.append("None")

    log.info("Writing QA: %s", OUTPUT_QA)
    OUTPUT_QA.write_text("\n".join(qa), encoding="utf-8")


def main() -> None:
    log.info("Starting Script 96: theory readiness filter")

    validation = load_validation()
    filtered = apply_readiness_filter(validation)
    summary = make_summary(filtered)

    write_outputs(filtered, summary)

    log.info("Done")

    print("\nRDE Theory Readiness Summary:")
    print(summary.to_string(index=False))

    print("\nReadiness Counts:")
    print(filtered["theory_readiness_tier"].value_counts().to_string())

    print("\nCreated:")
    print(f"  {OUTPUT_FILTER}")
    print(f"  {OUTPUT_SUMMARY}")
    print(f"  {OUTPUT_CORE}")
    print(f"  {OUTPUT_EMERGING}")
    print(f"  {OUTPUT_HOLDOUT}")
    print(f"  {OUTPUT_QA}")


if __name__ == "__main__":
    main()