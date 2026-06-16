#!/usr/bin/env python3
"""
93_recognition_inefficiency_deep_typology_v01.py

Purpose
-------
Deep typology for Recognition Inefficiency regions only.

Input:
data/processed/mechanism_region_typology_v02.csv

Outputs:
data/processed/recognition_inefficiency_deep_typology_v01.csv
data/processed/recognition_inefficiency_deep_typology_summary_v01.csv
data/processed/recognition_inefficiency_deep_typology_qa_v01.txt
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


SCRIPT_NAME = "93_recognition_inefficiency_deep_typology_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUT_CSV = PROCESSED / "mechanism_region_typology_v02.csv"

OUTPUT_CSV = PROCESSED / "recognition_inefficiency_deep_typology_v01.csv"
OUTPUT_SUMMARY = PROCESSED / "recognition_inefficiency_deep_typology_summary_v01.csv"
OUTPUT_QA = PROCESSED / "recognition_inefficiency_deep_typology_qa_v01.txt"


CORE_FEATURES = [
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


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def load_input() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_CSV}")

    log.info("Reading typology v02: %s", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    if "mechanism_class" not in df.columns:
        raise ValueError("Input must contain mechanism_class.")

    ri = df[df["mechanism_class"].astype(str).str.contains("Recognition Inefficiency", case=False, na=False)].copy()

    if len(ri) == 0:
        raise ValueError("No Recognition Inefficiency regions found.")

    log.info("Recognition Inefficiency regions: %s", len(ri))
    return ri


def add_within_ri_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in CORE_FEATURES:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        df[f"{c}_pct_ri"] = s.rank(pct=True)

    return df


def choose_k(x: np.ndarray, n: int) -> int:
    if n < 8:
        return 1

    max_k = min(6, n - 1)
    best_k = 2
    best_score = -999

    for k in range(2, max_k + 1):
        try:
            labels = KMeans(n_clusters=k, random_state=42, n_init=100).fit_predict(x)
            score = silhouette_score(x, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    return best_k


def add_deep_clusters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cols = [c for c in CORE_FEATURES if c in df.columns]
    xdf = df[cols].apply(pd.to_numeric, errors="coerce")
    xdf = xdf.fillna(xdf.median()).fillna(0)

    x = StandardScaler().fit_transform(xdf)
    k = choose_k(x, len(df))

    labels = KMeans(n_clusters=k, random_state=42, n_init=100).fit_predict(x)

    df["ri_deep_cluster_v01"] = labels
    log.info("Recognition Inefficiency deep clusters: k=%s", k)

    return df


def v(row: pd.Series, col: str) -> float:
    try:
        return float(row.get(col, np.nan))
    except Exception:
        return np.nan


def hi(row: pd.Series, col: str, t: float = 0.67) -> bool:
    x = v(row, col)
    return not np.isnan(x) and x >= t


def lo(row: pd.Series, col: str, t: float = 0.33) -> bool:
    x = v(row, col)
    return not np.isnan(x) and x <= t


def assign_deep_archetype(row: pd.Series) -> str:
    coastal_hi = hi(row, "sig_coastal_pct_ri")
    coastal_lo = lo(row, "sig_coastal_pct_ri")

    physical_hi = hi(row, "sig_physical_pct_ri")
    opportunity_hi = hi(row, "sig_opportunity_pct_ri")
    opportunity_lo = lo(row, "sig_opportunity_pct_ri")

    transmission_hi = hi(row, "sig_transmission_pct_ri")
    transmission_lo = lo(row, "sig_transmission_pct_ri")

    deficit_hi = hi(row, "sig_recognition_deficit_pct_ri")
    expected_hi = hi(row, "sig_expected_recognition_pct_ri")
    shadow_hi = hi(row, "sig_shadow_pct_ri")
    scale_hi = hi(row, "sig_scale_pct_ri")
    latent_hi = hi(row, "sig_latent_exceptionality_pct_ri")
    trans_fail_hi = hi(row, "sig_transmission_failure_pct_ri")

    if coastal_hi and physical_hi and deficit_hi:
        return "Coastal Hidden-Gem Recognition Failure"

    if transmission_hi and deficit_hi and expected_hi:
        return "High-Transmission Recognition Breakdown"

    if physical_hi and latent_hi and coastal_lo:
        return "Interior Latent Exceptional Landscape"

    if scale_hi and deficit_hi:
        return "Large-Area Latent Recognition Failure"

    if shadow_hi and deficit_hi:
        return "Shadow-Contaminated Recognition Inefficiency"

    if opportunity_hi and transmission_hi and deficit_hi:
        return "Fully Enabled Recognition Failure"

    if opportunity_lo and transmission_hi:
        return "Opportunity Bottleneck Despite Transmission"

    if transmission_lo and physical_hi:
        return "Transmission Bottleneck Exceptional Landscape"

    if coastal_hi:
        return "Coastal Recognition Inefficiency"

    if physical_hi:
        return "Physical Exceptionality Recognition Lag"

    return "Diffuse Recognition Inefficiency"


def deep_strength(row: pd.Series) -> float:
    cols = [
        "sig_physical",
        "sig_recognition_deficit",
        "sig_latent_exceptionality",
        "sig_transmission_failure",
        "sig_expected_recognition",
    ]

    vals = []
    for c in cols:
        if c in row.index:
            x = v(row, c)
            if not np.isnan(x):
                vals.append(x)

    if not vals:
        return np.nan

    return float(np.mean(vals))


def add_deep_archetypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ri_deep_archetype_v01"] = df.apply(assign_deep_archetype, axis=1)
    df["ri_deep_strength_v01"] = df.apply(deep_strength, axis=1)

    df["ri_deep_code_v01"] = (
        df["ri_deep_archetype_v01"].astype(str).map(norm)
        + "__cluster_"
        + df["ri_deep_cluster_v01"].astype(str)
    )

    return df


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("ri_deep_archetype_v01")
        .agg(
            region_count=("mechanism_region_id", "count"),
            mean_strength=("ri_deep_strength_v01", "mean"),
            mean_physical=("sig_physical", "mean"),
            mean_coastal=("sig_coastal", "mean"),
            mean_opportunity=("sig_opportunity", "mean"),
            mean_transmission=("sig_transmission", "mean"),
            mean_recognition_deficit=("sig_recognition_deficit", "mean"),
            mean_expected_recognition=("sig_expected_recognition", "mean"),
            mean_shadow=("sig_shadow", "mean"),
            mean_scale=("sig_scale", "mean"),
        )
        .reset_index()
        .sort_values(["region_count", "mean_strength"], ascending=[False, False])
    )

    return summary


def write_outputs(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    log.info("Writing CSV: %s", OUTPUT_CSV)
    df.to_csv(OUTPUT_CSV, index=False)

    log.info("Writing summary: %s", OUTPUT_SUMMARY)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    qa = []
    qa.append("Recognition Inefficiency Deep Typology v01 QA")
    qa.append("=" * 50)
    qa.append("")
    qa.append(f"Input: {INPUT_CSV}")
    qa.append(f"Recognition Inefficiency regions: {len(df)}")
    qa.append(f"Deep archetypes: {df['ri_deep_archetype_v01'].nunique()}")
    qa.append("")
    qa.append("Archetype counts:")
    qa.append(df["ri_deep_archetype_v01"].value_counts().to_string())
    qa.append("")
    qa.append("Cluster counts:")
    qa.append(df["ri_deep_cluster_v01"].value_counts().sort_index().to_string())
    qa.append("")
    qa.append("Summary:")
    qa.append(summary.to_string(index=False))

    log.info("Writing QA: %s", OUTPUT_QA)
    OUTPUT_QA.write_text("\n".join(qa), encoding="utf-8")


def main() -> None:
    log.info("Starting Script 93: recognition inefficiency deep typology")

    df = load_input()
    df = add_within_ri_percentiles(df)
    df = add_deep_clusters(df)
    df = add_deep_archetypes(df)

    summary = make_summary(df)

    write_outputs(df, summary)

    log.info("Done")

    print("\nRecognition Inefficiency deep typology summary:")
    print(summary.to_string(index=False))

    print("\nCreated:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_SUMMARY}")
    print(f"  {OUTPUT_QA}")


if __name__ == "__main__":
    main()