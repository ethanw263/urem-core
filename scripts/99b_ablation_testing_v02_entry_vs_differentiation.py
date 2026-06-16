#!/usr/bin/env python3
"""
99b_ablation_testing_v02_entry_vs_differentiation.py

Purpose
-------
Clarify the role of Physical Potential P in the RDE framework.

Script 99 showed that O/T/R best differentiate mechanism classes among the
99 selected mechanism regions. That does NOT mean P is unimportant.

This script separates:

1. Entry role:
   Does P help distinguish mechanism regions from the broader candidate universe?

2. Differentiation role:
   Once regions are already selected, do O/T/R better distinguish mechanism type?

Inputs
------
data/processed/mechanism_region_typology_v02.csv
data/processed/region_feature_matrix_v01.csv

Outputs
-------
data/processed/rde_ablation_entry_vs_differentiation_v02.csv
data/processed/rde_ablation_entry_role_v02.csv
data/processed/rde_ablation_differentiation_role_v02.csv
data/processed/rde_ablation_v02_qa.txt
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler


SCRIPT_NAME = "99b_ablation_testing_v02_entry_vs_differentiation"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUT_TYPOLOGY = PROCESSED / "mechanism_region_typology_v02.csv"
INPUT_MATRIX = PROCESSED / "region_feature_matrix_v01.csv"

OUTPUT_COMBINED = PROCESSED / "rde_ablation_entry_vs_differentiation_v02.csv"
OUTPUT_ENTRY = PROCESSED / "rde_ablation_entry_role_v02.csv"
OUTPUT_DIFF = PROCESSED / "rde_ablation_differentiation_role_v02.csv"
OUTPUT_QA = PROCESSED / "rde_ablation_v02_qa.txt"


RANDOM_SEED = 42
K = 3


ENTRY_MODELS = {
    "P_entry_only": ["sig_physical"],
    "P_R_entry": ["sig_physical", "sig_recognition_deficit"],
    "P_O_R_entry": ["sig_physical", "sig_opportunity", "sig_recognition_deficit"],
    "P_T_R_entry": ["sig_physical", "sig_transmission", "sig_recognition_deficit"],
    "Full_entry": [
        "sig_physical",
        "sig_opportunity",
        "sig_transmission",
        "sig_recognition_deficit",
        "sig_expected_recognition",
    ],
    "No_P_entry": [
        "sig_opportunity",
        "sig_transmission",
        "sig_recognition_deficit",
        "sig_expected_recognition",
    ],
}


DIFF_MODELS = {
    "P_only_diff": ["sig_physical"],
    "O_T_R_diff": ["sig_opportunity", "sig_transmission", "sig_recognition_deficit"],
    "P_O_R_diff": ["sig_physical", "sig_opportunity", "sig_recognition_deficit"],
    "P_T_R_diff": ["sig_physical", "sig_transmission", "sig_recognition_deficit"],
    "Full_P_O_T_R_diff": [
        "sig_physical",
        "sig_opportunity",
        "sig_transmission",
        "sig_recognition_deficit",
    ],
    "Full_context_diff": [
        "sig_physical",
        "sig_opportunity",
        "sig_transmission",
        "sig_recognition_deficit",
        "sig_coastal",
        "sig_shadow",
        "sig_scale",
        "sig_expected_recognition",
    ],
    "No_P_diff": [
        "sig_opportunity",
        "sig_transmission",
        "sig_recognition_deficit",
        "sig_coastal",
        "sig_shadow",
        "sig_scale",
    ],
    "No_O_diff": [
        "sig_physical",
        "sig_transmission",
        "sig_recognition_deficit",
        "sig_coastal",
        "sig_shadow",
        "sig_scale",
    ],
    "No_T_diff": [
        "sig_physical",
        "sig_opportunity",
        "sig_recognition_deficit",
        "sig_coastal",
        "sig_shadow",
        "sig_scale",
    ],
    "No_R_diff": [
        "sig_physical",
        "sig_opportunity",
        "sig_transmission",
        "sig_coastal",
        "sig_shadow",
        "sig_scale",
    ],
}


def canonical_mechanism(x: object) -> str:
    s = str(x).replace(" Candidate", "").strip()

    if "Recognition Inefficiency" in s:
        return "Recognition Inefficiency"
    if "Opportunity Failure" in s:
        return "Opportunity Failure"
    if "Comparative Shadowing" in s or "Recognition Diversion" in s:
        return "Comparative Shadowing / Recognition Diversion"

    return s


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not INPUT_TYPOLOGY.exists():
        raise FileNotFoundError(f"Missing typology input: {INPUT_TYPOLOGY}")

    if not INPUT_MATRIX.exists():
        raise FileNotFoundError(f"Missing matrix input: {INPUT_MATRIX}")

    log.info("Reading typology: %s", INPUT_TYPOLOGY)
    typology = pd.read_csv(INPUT_TYPOLOGY, low_memory=False)

    log.info("Reading region matrix: %s", INPUT_MATRIX)
    matrix = pd.read_csv(INPUT_MATRIX, low_memory=False)

    typology = typology.copy()
    typology["canonical_mechanism"] = typology["mechanism_class"].map(canonical_mechanism)

    log.info("Typology rows: %s", len(typology))
    log.info("Matrix rows: %s", len(matrix))

    return typology, matrix


def prepare_x(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required model columns: {missing}")

    x = df[cols].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median()).fillna(0)

    return x


def purity_score(true_labels: pd.Series, pred_labels: np.ndarray) -> float:
    tmp = pd.DataFrame({"true": true_labels.values, "pred": pred_labels})
    correct = 0

    for _, sub in tmp.groupby("pred"):
        correct += sub["true"].value_counts().max()

    return float(correct / len(tmp))


def normalized_entropy(true_labels: pd.Series, pred_labels: np.ndarray) -> float:
    tmp = pd.DataFrame({"true": true_labels.values, "pred": pred_labels})
    classes = true_labels.nunique()

    if classes <= 1:
        return 0.0

    weighted_entropy = 0.0

    for _, sub in tmp.groupby("pred"):
        probs = sub["true"].value_counts(normalize=True)
        entropy = -float((probs * np.log(probs + 1e-12)).sum())
        weighted_entropy += (entropy / np.log(classes)) * (len(sub) / len(tmp))

    return float(weighted_entropy)


def cluster_balance(labels: np.ndarray) -> float:
    counts = pd.Series(labels).value_counts(normalize=True)

    if len(counts) <= 1:
        return 0.0

    entropy = -float((counts * np.log(counts + 1e-12)).sum())

    return float(entropy / np.log(len(counts)))


def evaluate_clustering(
    df: pd.DataFrame,
    true_col: str,
    model_name: str,
    cols: list[str],
    role: str,
) -> dict:
    x = prepare_x(df, cols)
    x_scaled = StandardScaler().fit_transform(x)

    labels = KMeans(
        n_clusters=K,
        random_state=RANDOM_SEED,
        n_init=100,
    ).fit_predict(x_scaled)

    silhouette = float(silhouette_score(x_scaled, labels))
    ch = float(calinski_harabasz_score(x_scaled, labels))
    purity = purity_score(df[true_col], labels)
    entropy = normalized_entropy(df[true_col], labels)
    balance = cluster_balance(labels)

    composite = (
        0.35 * purity
        + 0.25 * (1 - entropy)
        + 0.25 * max(silhouette, 0)
        + 0.15 * balance
    )

    return {
        "role": role,
        "model_name": model_name,
        "features": ", ".join(cols),
        "feature_count": len(cols),
        "target": true_col,
        "purity": purity,
        "normalized_entropy": entropy,
        "silhouette": silhouette,
        "calinski_harabasz": ch,
        "cluster_balance": balance,
        "composite_score": composite,
    }


def build_entry_dataset(typology: pd.DataFrame) -> pd.DataFrame:
    """
    Practical entry test using selected mechanism regions.

    Since the current pipeline does not preserve a full background-region table
    at the same region scale, this entry test asks whether high/low physical
    potential separates stronger vs weaker RDE expression within selected regions.

    This is not a full background validation. It is an entry-role diagnostic.
    """

    df = typology.copy()

    if "archetype_strength_v02" in df.columns:
        strength = pd.to_numeric(df["archetype_strength_v02"], errors="coerce")
    elif "mechanism_region_priority_score_v01" in df.columns:
        strength = pd.to_numeric(df["mechanism_region_priority_score_v01"], errors="coerce")
    else:
        strength = pd.to_numeric(df["sig_recognition_deficit"], errors="coerce")

    df["entry_strength"] = strength

    q1 = df["entry_strength"].quantile(0.33)
    q2 = df["entry_strength"].quantile(0.67)

    def label(v):
        if pd.isna(v):
            return "middle_entry"
        if v <= q1:
            return "low_entry"
        if v >= q2:
            return "high_entry"
        return "middle_entry"

    df["entry_class"] = df["entry_strength"].map(label)

    return df


def run_entry_role_tests(typology: pd.DataFrame) -> pd.DataFrame:
    entry_df = build_entry_dataset(typology)

    rows = []
    for model_name, cols in ENTRY_MODELS.items():
        log.info("Evaluating entry-role model: %s", model_name)
        rows.append(
            evaluate_clustering(
                entry_df,
                true_col="entry_class",
                model_name=model_name,
                cols=cols,
                role="entry_role",
            )
        )

    out = pd.DataFrame(rows).sort_values("composite_score", ascending=False)
    out["rank"] = out["composite_score"].rank(ascending=False, method="dense").astype(int)

    return out


def run_differentiation_tests(typology: pd.DataFrame) -> pd.DataFrame:
    diff_df = typology.copy()

    rows = []
    for model_name, cols in DIFF_MODELS.items():
        log.info("Evaluating differentiation-role model: %s", model_name)
        rows.append(
            evaluate_clustering(
                diff_df,
                true_col="canonical_mechanism",
                model_name=model_name,
                cols=cols,
                role="differentiation_role",
            )
        )

    out = pd.DataFrame(rows).sort_values("composite_score", ascending=False)
    out["rank"] = out["composite_score"].rank(ascending=False, method="dense").astype(int)

    return out


def component_interpretation(entry: pd.DataFrame, diff: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def score(df, model):
        sub = df[df["model_name"] == model]
        if len(sub) == 0:
            return np.nan
        return float(sub["composite_score"].iloc[0])

    entry_full = score(entry, "Full_entry")
    entry_no_p = score(entry, "No_P_entry")
    diff_full = score(diff, "Full_context_diff")
    diff_no_p = score(diff, "No_P_diff")

    rows.append(
        {
            "component": "Physical Potential P",
            "entry_full_score": entry_full,
            "entry_without_component_score": entry_no_p,
            "entry_importance_drop": entry_full - entry_no_p if pd.notna(entry_full) and pd.notna(entry_no_p) else np.nan,
            "differentiation_full_score": diff_full,
            "differentiation_without_component_score": diff_no_p,
            "differentiation_importance_drop": diff_full - diff_no_p if pd.notna(diff_full) and pd.notna(diff_no_p) else np.nan,
            "interpretation": (
                "If entry drop is larger than differentiation drop, P functions mainly "
                "as an entry/eligibility condition rather than as a mechanism differentiator."
            ),
        }
    )

    return pd.DataFrame(rows)


def write_outputs(
    entry: pd.DataFrame,
    diff: pd.DataFrame,
    combined: pd.DataFrame,
    interpretation: pd.DataFrame,
) -> None:
    log.info("Writing combined ablation v02: %s", OUTPUT_COMBINED)
    combined.to_csv(OUTPUT_COMBINED, index=False)

    log.info("Writing entry role: %s", OUTPUT_ENTRY)
    entry.to_csv(OUTPUT_ENTRY, index=False)

    log.info("Writing differentiation role: %s", OUTPUT_DIFF)
    diff.to_csv(OUTPUT_DIFF, index=False)

    qa = []
    qa.append("RDE Ablation v02: Entry vs Differentiation QA")
    qa.append("=" * 55)
    qa.append("")
    qa.append("Entry-role model rankings:")
    qa.append(entry.to_string(index=False))
    qa.append("")
    qa.append("Differentiation-role model rankings:")
    qa.append(diff.to_string(index=False))
    qa.append("")
    qa.append("Component interpretation:")
    qa.append(interpretation.to_string(index=False))
    qa.append("")
    qa.append("Important interpretation:")
    qa.append(
        "This script does not replace full external/background validation. "
        "It is an internal diagnostic separating P as an entry condition from "
        "O/T/R as mechanism differentiators."
    )

    log.info("Writing QA: %s", OUTPUT_QA)
    OUTPUT_QA.write_text("\n".join(qa), encoding="utf-8")


def main() -> None:
    log.info("Starting Script 99b: entry vs differentiation ablation")

    typology, matrix = load_inputs()

    entry = run_entry_role_tests(typology)
    diff = run_differentiation_tests(typology)

    combined = pd.concat([entry, diff], ignore_index=True)
    interpretation = component_interpretation(entry, diff)

    write_outputs(entry, diff, combined, interpretation)

    log.info("Done")

    print("\nEntry-role model rankings:")
    print(
        entry[
            [
                "rank",
                "model_name",
                "feature_count",
                "purity",
                "normalized_entropy",
                "silhouette",
                "composite_score",
            ]
        ].to_string(index=False)
    )

    print("\nDifferentiation-role model rankings:")
    print(
        diff[
            [
                "rank",
                "model_name",
                "feature_count",
                "purity",
                "normalized_entropy",
                "silhouette",
                "composite_score",
            ]
        ].to_string(index=False)
    )

    print("\nInterpretation:")
    print(interpretation.to_string(index=False))

    print("\nCreated:")
    print(f"  {OUTPUT_COMBINED}")
    print(f"  {OUTPUT_ENTRY}")
    print(f"  {OUTPUT_DIFF}")
    print(f"  {OUTPUT_QA}")


if __name__ == "__main__":
    main()