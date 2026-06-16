#!/usr/bin/env python3
"""
92_mechanism_region_typology_v02.py

Purpose
-------
Create improved mechanism-region archetypes from the compact 76-column
region feature matrix created by Script 91b.

Input:
data/processed/region_feature_matrix_v01.csv

Outputs:
data/processed/mechanism_region_typology_v02.csv
data/processed/mechanism_region_typology_summary_v02.csv
data/processed/mechanism_region_typology_profiles_v02.csv
data/processed/mechanism_region_typology_qa_v02.txt
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


SCRIPT_NAME = "92_mechanism_region_typology_v02"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUT_MATRIX = PROCESSED / "region_feature_matrix_v01.csv"

OUTPUT_CSV = PROCESSED / "mechanism_region_typology_v02.csv"
OUTPUT_SUMMARY = PROCESSED / "mechanism_region_typology_summary_v02.csv"
OUTPUT_PROFILES = PROCESSED / "mechanism_region_typology_profiles_v02.csv"
OUTPUT_QA = PROCESSED / "mechanism_region_typology_qa_v02.txt"


ID_COLS = [
    "mechanism_class",
    "mechanism_region_id",
    "mechanism_region_area_km2",
    "cell_count",
    "mechanism_region_priority_score_v01",
    "mechanism_region_priority_class_v01",
]


SIGNATURE_GROUPS: Dict[str, List[str]] = {
    "physical": [
        "physical", "phys", "exceptionality", "p_orthogonal",
        "relief", "slope", "elevation", "terrain", "rugged", "drama",
        "core_physical_potential",
    ],
    "coastal": [
        "coast", "coastal", "shore", "ocean", "beach", "marine",
    ],
    "opportunity": [
        "opportunity", "o_base", "accessibility", "access",
        "infrastructure", "road", "parking", "trail", "urban",
        "settlement", "core_opportunity",
    ],
    "transmission": [
        "transmission", "t_net", "institutional", "park",
        "protected", "tourism", "recreation", "visitor", "viewpoint",
        "attraction", "core_transmission",
    ],
    "recognition_deficit": [
        "under_recognition", "r_net", "deficit", "gap",
        "core_under_recognition", "core_recognition_gap",
    ],
    "observed_recognition": [
        "observed_recognition", "observed", "obs",
        "core_observed_recognition",
    ],
    "expected_recognition": [
        "expected_recognition", "expected", "exp",
        "core_expected_recognition",
    ],
    "shadow": [
        "shadow", "diversion", "neighbor", "nearby", "destination",
        "recognized", "comparative",
    ],
    "scale": [
        "area", "cell_count", "perim", "compact",
    ],
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def load_matrix() -> pd.DataFrame:
    if not INPUT_MATRIX.exists():
        raise FileNotFoundError(f"Missing input matrix: {INPUT_MATRIX}")

    log.info("Reading matrix: %s", INPUT_MATRIX)
    df = pd.read_csv(INPUT_MATRIX, low_memory=False)

    if "mechanism_class" not in df.columns:
        raise ValueError("Input matrix must contain mechanism_class.")

    if "mechanism_region_id" not in df.columns:
        raise ValueError("Input matrix must contain mechanism_region_id.")

    log.info("Rows: %s | Columns: %s", len(df), len(df.columns))
    return df


def numeric_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if c in ID_COLS:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(5, int(len(df) * 0.3)):
            cols.append(c)
    return cols


def group_cols(df: pd.DataFrame, group: str) -> List[str]:
    keywords = SIGNATURE_GROUPS[group]
    out = []

    for c in df.columns:
        lc = norm(c)
        if any(norm(k) in lc for k in keywords):
            try:
                pd.to_numeric(df[c], errors="coerce")
                out.append(c)
            except Exception:
                pass

    return list(dict.fromkeys(out))


def normalized_mean(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    if not cols:
        return pd.Series(np.nan, index=df.index)

    values = df[cols].apply(pd.to_numeric, errors="coerce")

    scaled = pd.DataFrame(index=df.index)
    for c in values.columns:
        s = values[c]
        mn = s.min(skipna=True)
        mx = s.max(skipna=True)
        if pd.isna(mn) or pd.isna(mx) or abs(mx - mn) < 1e-12:
            scaled[c] = np.nan
        else:
            scaled[c] = (s - mn) / (mx - mn)

    return scaled.mean(axis=1)


def add_signature_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for group in SIGNATURE_GROUPS:
        cols = group_cols(df, group)
        df[f"sig_{group}"] = normalized_mean(df, cols)
        df[f"sig_{group}_feature_count"] = len(cols)
        log.info("%s signature uses %s columns", group, len(cols))

    if {"sig_expected_recognition", "sig_observed_recognition"}.issubset(df.columns):
        df["sig_expectation_gap"] = (
            df["sig_expected_recognition"] - df["sig_observed_recognition"]
        )

    if {"sig_physical", "sig_recognition_deficit"}.issubset(df.columns):
        df["sig_latent_exceptionality"] = (
            df["sig_physical"] * df["sig_recognition_deficit"]
        )

    if {"sig_transmission", "sig_recognition_deficit"}.issubset(df.columns):
        df["sig_transmission_failure"] = (
            df["sig_transmission"] * df["sig_recognition_deficit"]
        )

    if {"sig_opportunity", "sig_recognition_deficit"}.issubset(df.columns):
        df["sig_opportunity_failure"] = (
            (1 - df["sig_opportunity"]) * df["sig_recognition_deficit"]
        )

    if {"sig_shadow", "sig_recognition_deficit"}.issubset(df.columns):
        df["sig_shadow_diversion"] = (
            df["sig_shadow"] * df["sig_recognition_deficit"]
        )

    return df


def percentile_within_mechanism(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("mechanism_class")[col].rank(pct=True, method="average")


def add_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    sig_cols = [
        c for c in df.columns
        if c.startswith("sig_") and not c.endswith("_feature_count")
    ]

    for c in sig_cols:
        df[f"{c}_pct_mech"] = percentile_within_mechanism(df, c)

    return df


def choose_k(x: np.ndarray, n: int) -> int:
    if n < 7:
        return 1

    max_k = min(5, n - 1)
    best_k = 2
    best_score = -999

    for k in range(2, max_k + 1):
        try:
            labels = KMeans(n_clusters=k, random_state=42, n_init=50).fit_predict(x)
            score = silhouette_score(x, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    return best_k


def add_clusters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["typology_cluster_v02"] = -1

    cluster_cols = [
        "sig_physical",
        "sig_coastal",
        "sig_opportunity",
        "sig_transmission",
        "sig_recognition_deficit",
        "sig_observed_recognition",
        "sig_expected_recognition",
        "sig_shadow",
        "sig_scale",
        "sig_latent_exceptionality",
        "sig_transmission_failure",
        "sig_opportunity_failure",
        "sig_shadow_diversion",
    ]

    cluster_cols = [c for c in cluster_cols if c in df.columns]

    for mechanism, sub in df.groupby("mechanism_class"):
        idx = sub.index
        xdf = sub[cluster_cols].apply(pd.to_numeric, errors="coerce")
        xdf = xdf.fillna(xdf.median()).fillna(0)

        if len(xdf) < 7:
            df.loc[idx, "typology_cluster_v02"] = 0
            continue

        x = StandardScaler().fit_transform(xdf)
        k = choose_k(x, len(xdf))

        if k <= 1:
            labels = np.zeros(len(xdf), dtype=int)
        else:
            labels = KMeans(n_clusters=k, random_state=42, n_init=100).fit_predict(x)

        df.loc[idx, "typology_cluster_v02"] = labels
        log.info("Mechanism '%s': n=%s, k=%s", mechanism, len(xdf), k)

    return df


def val(row: pd.Series, col: str) -> float:
    v = row.get(col, np.nan)
    try:
        return float(v)
    except Exception:
        return np.nan


def hi(row: pd.Series, col: str, threshold: float = 0.67) -> bool:
    v = val(row, col)
    return not np.isnan(v) and v >= threshold


def lo(row: pd.Series, col: str, threshold: float = 0.33) -> bool:
    v = val(row, col)
    return not np.isnan(v) and v <= threshold


def assign_archetype(row: pd.Series) -> str:
    mech_raw = str(row["mechanism_class"])
    mech = mech_raw.replace(" Candidate", "").strip()

    coastal = hi(row, "sig_coastal_pct_mech")
    terrain = hi(row, "sig_physical_pct_mech")
    physical = hi(row, "sig_physical_pct_mech")
    opportunity_hi = hi(row, "sig_opportunity_pct_mech")
    opportunity_lo = lo(row, "sig_opportunity_pct_mech")
    transmission_hi = hi(row, "sig_transmission_pct_mech")
    transmission_lo = lo(row, "sig_transmission_pct_mech")
    deficit_hi = hi(row, "sig_recognition_deficit_pct_mech")
    shadow_hi = hi(row, "sig_shadow_pct_mech")
    scale_hi = hi(row, "sig_scale_pct_mech")
    observed_lo = lo(row, "sig_observed_recognition_pct_mech")

    if mech == "Recognition Inefficiency":
        if coastal and physical and deficit_hi:
            return "Coastal Hidden Exceptional Landscape"
        if physical and transmission_hi and observed_lo:
            return "Transmission-Rich Recognition Failure Landscape"
        if scale_hi and deficit_hi:
            return "Large Latent Recognition Landscape"
        if shadow_hi and deficit_hi:
            return "Shadowed Recognition Inefficiency Landscape"
        if opportunity_hi and transmission_hi and deficit_hi:
            return "High-Opportunity Recognition Failure Landscape"
        if physical and deficit_hi:
            return "Latent Exceptional Landscape"
        return "General Recognition Inefficiency Landscape"

    if mech == "Opportunity Failure":
        if opportunity_lo and physical and deficit_hi:
            return "High-Potential Opportunity Gap Landscape"
        if opportunity_lo and coastal:
            return "Coastal Access-Limited Landscape"
        if opportunity_lo and transmission_lo:
            return "Low-Opportunity Low-Transmission Landscape"
        if scale_hi and opportunity_lo:
            return "Large Opportunity-Constrained Landscape"
        if physical and opportunity_lo:
            return "Terrain-Constrained Opportunity Failure Landscape"
        return "General Opportunity Failure Landscape"

    if mech == "Comparative Shadowing / Recognition Diversion":
        if shadow_hi and coastal:
            return "Coastal Recognition Diversion Landscape"
        if shadow_hi and deficit_hi:
            return "Recognition Sink Landscape"
        if transmission_hi and shadow_hi:
            return "Transmission Diverted Landscape"
        if observed_lo and deficit_hi:
            return "Suppressed Recognition Landscape"
        return "General Comparative Shadowing Landscape"

    return "Background / Mixed Recognition Landscape"


def archetype_strength(row: pd.Series) -> float:
    mech = str(row["mechanism_class"])

    if mech == "Recognition Inefficiency":
        cols = [
            "sig_physical",
            "sig_transmission",
            "sig_recognition_deficit",
            "sig_latent_exceptionality",
            "sig_transmission_failure",
        ]
    elif mech == "Opportunity Failure":
        cols = [
            "sig_physical",
            "sig_recognition_deficit",
            "sig_opportunity_failure",
        ]
    elif mech == "Comparative Shadowing / Recognition Diversion":
        cols = [
            "sig_shadow",
            "sig_recognition_deficit",
            "sig_shadow_diversion",
        ]
    else:
        cols = [
            "sig_physical",
            "sig_recognition_deficit",
        ]

    vals = [val(row, c) for c in cols if c in row.index]
    vals = [v for v in vals if not np.isnan(v)]

    if not vals:
        return np.nan

    return float(np.mean(vals))


def add_archetypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["region_archetype_v02"] = df.apply(assign_archetype, axis=1)
    df["archetype_strength_v02"] = df.apply(archetype_strength, axis=1)

    df["typology_code_v02"] = (
        df["mechanism_class"].astype(str).map(norm)
        + "__"
        + df["region_archetype_v02"].astype(str).map(norm)
    )

    return df


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["mechanism_class", "region_archetype_v02"], dropna=False)
        .agg(
            region_count=("mechanism_region_id", "count"),
            mean_archetype_strength=("archetype_strength_v02", "mean"),
            mean_physical=("sig_physical", "mean"),
            mean_coastal=("sig_coastal", "mean"),
            mean_opportunity=("sig_opportunity", "mean"),
            mean_transmission=("sig_transmission", "mean"),
            mean_recognition_deficit=("sig_recognition_deficit", "mean"),
            mean_shadow=("sig_shadow", "mean"),
            mean_scale=("sig_scale", "mean"),
        )
        .reset_index()
        .sort_values(
            ["mechanism_class", "region_count", "mean_archetype_strength"],
            ascending=[True, False, False],
        )
    )

    return summary


def make_profiles(df: pd.DataFrame) -> pd.DataFrame:
    sig_cols = [
        c for c in df.columns
        if c.startswith("sig_") and not c.endswith("_feature_count")
    ]

    rows = []

    for arch, sub in df.groupby("region_archetype_v02"):
        row = {
            "region_archetype_v02": arch,
            "dominant_mechanism": sub["mechanism_class"].mode().iloc[0],
            "region_count": len(sub),
            "mean_archetype_strength_v02": sub["archetype_strength_v02"].mean(),
        }

        for c in sig_cols:
            row[f"mean_{c}"] = pd.to_numeric(sub[c], errors="coerce").mean()

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["dominant_mechanism", "region_count", "mean_archetype_strength_v02"],
        ascending=[True, False, False],
    )


def write_outputs(df: pd.DataFrame, summary: pd.DataFrame, profiles: pd.DataFrame) -> None:
    log.info("Writing typology CSV: %s", OUTPUT_CSV)
    df.to_csv(OUTPUT_CSV, index=False)

    log.info("Writing summary CSV: %s", OUTPUT_SUMMARY)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    log.info("Writing profiles CSV: %s", OUTPUT_PROFILES)
    profiles.to_csv(OUTPUT_PROFILES, index=False)

    qa = []
    qa.append("Mechanism Region Typology v02 QA")
    qa.append("=" * 45)
    qa.append("")
    qa.append(f"Input matrix: {INPUT_MATRIX}")
    qa.append(f"Region rows: {len(df)}")
    qa.append(f"Output columns: {len(df.columns)}")
    qa.append("")
    qa.append("Mechanism counts:")
    qa.append(df["mechanism_class"].value_counts().to_string())
    qa.append("")
    qa.append("Archetype counts:")
    qa.append(df["region_archetype_v02"].value_counts().to_string())
    qa.append("")
    qa.append("Summary:")
    qa.append(summary.to_string(index=False))
    qa.append("")
    qa.append("Signature feature counts:")
    for c in df.columns:
        if c.endswith("_feature_count"):
            qa.append(f"{c}: {int(df[c].max())}")

    log.info("Writing QA TXT: %s", OUTPUT_QA)
    OUTPUT_QA.write_text("\n".join(qa), encoding="utf-8")


def main() -> None:
    log.info("Starting Script 92: mechanism region typology v02")

    df = load_matrix()
    df = add_signature_scores(df)
    df = add_percentiles(df)
    df = add_clusters(df)
    df = add_archetypes(df)

    summary = make_summary(df)
    profiles = make_profiles(df)

    write_outputs(df, summary, profiles)

    log.info("Done")

    print("\nTypology v02 summary:")
    print(
        summary[
            [
                "mechanism_class",
                "region_archetype_v02",
                "region_count",
                "mean_archetype_strength",
            ]
        ].to_string(index=False)
    )

    print("\nCreated:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_SUMMARY}")
    print(f"  {OUTPUT_PROFILES}")
    print(f"  {OUTPUT_QA}")


if __name__ == "__main__":
    main()