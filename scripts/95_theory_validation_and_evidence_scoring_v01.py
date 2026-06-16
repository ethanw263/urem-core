#!/usr/bin/env python3
"""
95_theory_validation_and_evidence_scoring_v01.py

Purpose
-------
Validate Recognition Disequilibrium theory archetypes.

This script tests whether each archetype from Script 94 has a coherent
empirical evidence signature.

Inputs
------
data/processed/recognition_disequilibrium_theory_table_v01.csv
data/processed/mechanism_region_typology_v02.csv
data/processed/recognition_inefficiency_deep_typology_v01.csv

Outputs
-------
data/processed/rde_theory_validation_v01.csv
data/processed/rde_theory_validation_summary_v01.csv
data/processed/rde_theory_validation_qa_v01.txt
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_NAME = "95_theory_validation_and_evidence_scoring_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUT_THEORY = PROCESSED / "recognition_disequilibrium_theory_table_v01.csv"
INPUT_92 = PROCESSED / "mechanism_region_typology_v02.csv"
INPUT_93 = PROCESSED / "recognition_inefficiency_deep_typology_v01.csv"

OUTPUT_VALIDATION = PROCESSED / "rde_theory_validation_v01.csv"
OUTPUT_SUMMARY = PROCESSED / "rde_theory_validation_summary_v01.csv"
OUTPUT_QA = PROCESSED / "rde_theory_validation_qa_v01.txt"


EVIDENCE_FEATURES = [
    "sig_physical",
    "sig_coastal",
    "sig_opportunity",
    "sig_transmission",
    "sig_recognition_deficit",
    "sig_expected_recognition",
    "sig_shadow",
    "sig_scale",
    "sig_latent_exceptionality",
    "sig_transmission_failure",
    "sig_opportunity_failure",
    "sig_shadow_diversion",
]


EXPECTED_SIGNATURES = {
    "General Comparative Shadowing Landscape": {
        "high": ["sig_shadow", "sig_recognition_deficit"],
        "low": [],
    },
    "Coastal Recognition Diversion Landscape": {
        "high": ["sig_shadow", "sig_coastal", "sig_recognition_deficit"],
        "low": [],
    },
    "Recognition Sink Landscape": {
        "high": ["sig_shadow", "sig_recognition_deficit"],
        "low": ["sig_observed_recognition"],
    },
    "Transmission Diverted Landscape": {
        "high": ["sig_shadow", "sig_transmission"],
        "low": [],
    },
    "General Opportunity Failure Landscape": {
        "high": ["sig_recognition_deficit"],
        "low": ["sig_opportunity"],
    },
    "High-Potential Opportunity Gap Landscape": {
        "high": ["sig_physical", "sig_recognition_deficit", "sig_opportunity_failure"],
        "low": ["sig_opportunity"],
    },
    "Low-Opportunity Low-Transmission Landscape": {
        "high": ["sig_recognition_deficit"],
        "low": ["sig_opportunity", "sig_transmission"],
    },
    "Diffuse Recognition Inefficiency": {
        "high": ["sig_recognition_deficit"],
        "low": [],
    },
    "Coastal Recognition Inefficiency": {
        "high": ["sig_coastal", "sig_recognition_deficit"],
        "low": [],
    },
    "Physical Exceptionality Recognition Lag": {
        "high": ["sig_physical", "sig_recognition_deficit"],
        "low": [],
    },
    "Large-Area Latent Recognition Failure": {
        "high": ["sig_scale", "sig_recognition_deficit"],
        "low": [],
    },
    "Coastal Hidden-Gem Recognition Failure": {
        "high": ["sig_coastal", "sig_physical", "sig_recognition_deficit"],
        "low": [],
    },
    "Shadow-Contaminated Recognition Inefficiency": {
        "high": ["sig_shadow", "sig_recognition_deficit"],
        "low": [],
    },
    "Transmission Bottleneck Exceptional Landscape": {
        "high": ["sig_physical", "sig_recognition_deficit"],
        "low": ["sig_transmission"],
    },
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in [INPUT_THEORY, INPUT_92, INPUT_93]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    log.info("Reading theory table: %s", INPUT_THEORY)
    theory = pd.read_csv(INPUT_THEORY)

    log.info("Reading typology v02: %s", INPUT_92)
    df92 = pd.read_csv(INPUT_92, low_memory=False)

    log.info("Reading RI deep typology: %s", INPUT_93)
    df93 = pd.read_csv(INPUT_93, low_memory=False)

    return theory, df92, df93


def safe_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return np.nan
    return float(pd.to_numeric(df[col], errors="coerce").mean())


def safe_std(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return np.nan
    return float(pd.to_numeric(df[col], errors="coerce").std())


def safe_median(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return np.nan
    return float(pd.to_numeric(df[col], errors="coerce").median())


def coefficient_of_variation(mean: float, std: float) -> float:
    if pd.isna(mean) or pd.isna(std) or abs(mean) < 1e-12:
        return np.nan
    return float(std / abs(mean))


def get_regions_for_archetype(
    archetype: str,
    df92: pd.DataFrame,
    df93: pd.DataFrame,
) -> pd.DataFrame:
    if "ri_deep_archetype_v01" in df93.columns:
        sub93 = df93[df93["ri_deep_archetype_v01"] == archetype].copy()
        if len(sub93) > 0:
            return sub93

    if "region_archetype_v02" in df92.columns:
        sub92 = df92[df92["region_archetype_v02"] == archetype].copy()
        if len(sub92) > 0:
            return sub92

    return pd.DataFrame()


def normalize_across_archetypes(values: pd.Series, invert: bool = False) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce")

    mn = s.min(skipna=True)
    mx = s.max(skipna=True)

    if pd.isna(mn) or pd.isna(mx) or abs(mx - mn) < 1e-12:
        out = pd.Series(0.5, index=s.index)
    else:
        out = (s - mn) / (mx - mn)

    if invert:
        out = 1 - out

    return out


def raw_evidence_for_archetype(
    archetype: str,
    regions: pd.DataFrame,
) -> dict:
    expected = EXPECTED_SIGNATURES.get(archetype, {"high": [], "low": []})

    high_features = expected["high"]
    low_features = expected["low"]

    high_scores = []
    low_scores = []

    for col in high_features:
        if col in regions.columns:
            high_scores.append(safe_mean(regions, col))

    for col in low_features:
        if col in regions.columns:
            low_scores.append(1 - safe_mean(regions, col))

    if high_scores or low_scores:
        signature_match_raw = float(np.nanmean(high_scores + low_scores))
    else:
        signature_match_raw = np.nan

    feature_means = {
        f"mean_{col}": safe_mean(regions, col)
        for col in EVIDENCE_FEATURES
        if col in regions.columns
    }

    feature_stds = {
        f"std_{col}": safe_std(regions, col)
        for col in EVIDENCE_FEATURES
        if col in regions.columns
    }

    cvs = []
    for col in EVIDENCE_FEATURES:
        if col in regions.columns:
            m = safe_mean(regions, col)
            sd = safe_std(regions, col)
            cv = coefficient_of_variation(m, sd)
            if not pd.isna(cv):
                cvs.append(cv)

    internal_coherence_raw = 1 - float(np.nanmean(cvs)) if cvs else np.nan

    if pd.isna(internal_coherence_raw):
        internal_coherence_raw = np.nan
    else:
        internal_coherence_raw = max(0.0, min(1.0, internal_coherence_raw))

    region_count = len(regions)

    if region_count >= 10:
        sample_support_raw = 1.0
    elif region_count >= 5:
        sample_support_raw = 0.75
    elif region_count >= 2:
        sample_support_raw = 0.45
    else:
        sample_support_raw = 0.20

    return {
        "archetype": archetype,
        "region_count": region_count,
        "expected_high_features": ", ".join(high_features),
        "expected_low_features": ", ".join(low_features),
        "signature_match_raw": signature_match_raw,
        "internal_coherence_raw": internal_coherence_raw,
        "sample_support_raw": sample_support_raw,
        **feature_means,
        **feature_stds,
    }


def classify_evidence(score: float, region_count: int) -> str:
    if pd.isna(score):
        return "Insufficient Evidence"

    if region_count == 1:
        if score >= 0.70:
            return "Promising Single-Region Class"
        return "Weak Single-Region Class"

    if score >= 0.75:
        return "Strong Evidence"
    if score >= 0.60:
        return "Moderate Evidence"
    if score >= 0.45:
        return "Weak / Emerging Evidence"

    return "Low Evidence"


def validate(theory: pd.DataFrame, df92: pd.DataFrame, df93: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, trow in theory.iterrows():
        archetype = trow["archetype"]
        mechanism = trow["mechanism"]

        regions = get_regions_for_archetype(archetype, df92, df93)

        if len(regions) == 0:
            continue

        raw = raw_evidence_for_archetype(archetype, regions)
        raw["mechanism"] = mechanism
        raw["signature"] = trow.get("signature", "")
        raw["theory_contribution"] = trow.get("theory_contribution", "")
        rows.append(raw)

    validation = pd.DataFrame(rows)

    validation["signature_match_score"] = normalize_across_archetypes(
        validation["signature_match_raw"]
    )

    validation["internal_coherence_score"] = normalize_across_archetypes(
        validation["internal_coherence_raw"]
    )

    validation["sample_support_score"] = validation["sample_support_raw"]

    validation["overall_evidence_score"] = (
        0.45 * validation["signature_match_score"]
        + 0.30 * validation["internal_coherence_score"]
        + 0.25 * validation["sample_support_score"]
    )

    validation["evidence_class"] = validation.apply(
        lambda r: classify_evidence(
            r["overall_evidence_score"],
            int(r["region_count"]),
        ),
        axis=1,
    )

    validation = validation.sort_values(
        ["mechanism", "overall_evidence_score", "region_count"],
        ascending=[True, False, False],
    )

    return validation


def make_summary(validation: pd.DataFrame) -> pd.DataFrame:
    summary = (
        validation.groupby("mechanism")
        .agg(
            archetype_count=("archetype", "count"),
            total_regions=("region_count", "sum"),
            mean_evidence_score=("overall_evidence_score", "mean"),
            max_evidence_score=("overall_evidence_score", "max"),
            strong_or_moderate_count=(
                "evidence_class",
                lambda s: int(s.isin(["Strong Evidence", "Moderate Evidence"]).sum()),
            ),
        )
        .reset_index()
        .sort_values("mean_evidence_score", ascending=False)
    )

    return summary


def write_outputs(validation: pd.DataFrame, summary: pd.DataFrame) -> None:
    log.info("Writing validation: %s", OUTPUT_VALIDATION)
    validation.to_csv(OUTPUT_VALIDATION, index=False)

    log.info("Writing summary: %s", OUTPUT_SUMMARY)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    qa = []
    qa.append("RDE Theory Validation v01 QA")
    qa.append("=" * 45)
    qa.append("")
    qa.append(f"Validated archetypes: {len(validation)}")
    qa.append("")
    qa.append("Mechanism validation summary:")
    qa.append(summary.to_string(index=False))
    qa.append("")
    qa.append("Archetype validation:")
    qa.append(
        validation[
            [
                "mechanism",
                "archetype",
                "region_count",
                "overall_evidence_score",
                "evidence_class",
            ]
        ].to_string(index=False)
    )

    log.info("Writing QA: %s", OUTPUT_QA)
    OUTPUT_QA.write_text("\n".join(qa), encoding="utf-8")


def main() -> None:
    log.info("Starting Script 95: theory validation and evidence scoring")

    theory, df92, df93 = load_inputs()

    validation = validate(theory, df92, df93)
    summary = make_summary(validation)

    write_outputs(validation, summary)

    log.info("Done")

    print("\nRDE Theory Validation Summary:")
    print(summary.to_string(index=False))

    print("\nArchetype Evidence Classes:")
    print(
        validation[
            [
                "mechanism",
                "archetype",
                "region_count",
                "overall_evidence_score",
                "evidence_class",
            ]
        ].to_string(index=False)
    )

    print("\nCreated:")
    print(f"  {OUTPUT_VALIDATION}")
    print(f"  {OUTPUT_SUMMARY}")
    print(f"  {OUTPUT_QA}")


if __name__ == "__main__":
    main()