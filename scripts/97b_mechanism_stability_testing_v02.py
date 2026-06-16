#!/usr/bin/env python3
"""
97b_mechanism_stability_testing_v02.py

Purpose
-------
Baseline-anchored stability test for RDE mechanism regions.

This is fairer than Script 97 v01 because it does NOT replace the original
orthogonalized mechanism taxonomy with a new simplified classifier.

Instead, it asks:

If each region's signature vector is perturbed, does it remain closest to
its original mechanism/archetype profile?

Inputs
------
data/processed/mechanism_region_typology_v02.csv

Outputs
-------
data/processed/rde_mechanism_stability_v02.csv
data/processed/rde_archetype_stability_v02.csv
data/processed/rde_region_stability_v02.csv
data/processed/rde_stability_summary_v02.csv
data/processed/rde_stability_qa_v02.txt
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


SCRIPT_NAME = "97b_mechanism_stability_testing_v02"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUT_TYPOLOGY = PROCESSED / "mechanism_region_typology_v02.csv"

OUTPUT_MECHANISM = PROCESSED / "rde_mechanism_stability_v02.csv"
OUTPUT_ARCHETYPE = PROCESSED / "rde_archetype_stability_v02.csv"
OUTPUT_REGION = PROCESSED / "rde_region_stability_v02.csv"
OUTPUT_SUMMARY = PROCESSED / "rde_stability_summary_v02.csv"
OUTPUT_QA = PROCESSED / "rde_stability_qa_v02.txt"


RANDOM_SEED = 42
N_ITERATIONS = 500
PERTURBATION_LEVELS = [0.03, 0.05, 0.10, 0.15]

SIGNATURE_COLS = [
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

REQUIRED_COLS = [
    "mechanism_region_id",
    "mechanism_class",
    "region_archetype_v02",
]


def canonical_mechanism(x: object) -> str:
    s = str(x).replace(" Candidate", "").strip()

    if "Recognition Inefficiency" in s:
        return "Recognition Inefficiency"
    if "Opportunity Failure" in s:
        return "Opportunity Failure"
    if "Comparative Shadowing" in s or "Recognition Diversion" in s:
        return "Comparative Shadowing / Recognition Diversion"

    return s


def load_input() -> pd.DataFrame:
    if not INPUT_TYPOLOGY.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_TYPOLOGY}")

    log.info("Reading typology v02: %s", INPUT_TYPOLOGY)
    df = pd.read_csv(INPUT_TYPOLOGY, low_memory=False)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    available = [c for c in SIGNATURE_COLS if c in df.columns]
    if len(available) < 5:
        raise ValueError(
            f"Need at least 5 signature columns. Found only: {available}"
        )

    df = df.copy()
    df["canonical_mechanism"] = df["mechanism_class"].map(canonical_mechanism)

    log.info("Rows: %s", len(df))
    log.info("Signature columns used: %s", len(available))

    return df


def prepare_signature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], StandardScaler]:
    cols = [c for c in SIGNATURE_COLS if c in df.columns]

    x = df[cols].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median()).fillna(0)

    scaler = StandardScaler()
    xs = pd.DataFrame(
        scaler.fit_transform(x),
        columns=cols,
        index=df.index,
    )

    return xs, cols, scaler


def make_centroids(
    df: pd.DataFrame,
    x_scaled: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    temp = x_scaled.copy()
    temp[group_col] = df[group_col].values

    centroids = temp.groupby(group_col).mean()

    return centroids


def classify_to_nearest_centroid(
    x_scaled: pd.DataFrame,
    centroids: pd.DataFrame,
) -> pd.Series:
    dists = pairwise_distances(x_scaled, centroids, metric="euclidean")
    nearest_idx = np.argmin(dists, axis=1)
    labels = centroids.index.to_numpy()[nearest_idx]

    return pd.Series(labels, index=x_scaled.index)


def perturb_original_space(
    df: pd.DataFrame,
    cols: list[str],
    level: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median()).fillna(0)

    noise = rng.normal(loc=0, scale=level, size=x.shape)
    perturbed = np.clip(x.to_numpy() + noise, 0, 1)

    return pd.DataFrame(perturbed, columns=cols, index=df.index)


def run_simulation(
    df: pd.DataFrame,
    cols: list[str],
    scaler: StandardScaler,
    mechanism_centroids: pd.DataFrame,
    archetype_centroids: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    baseline_mech = df["canonical_mechanism"].reset_index(drop=True)
    baseline_arch = df["region_archetype_v02"].reset_index(drop=True)
    region_ids = df["mechanism_region_id"].reset_index(drop=True)

    for level in PERTURBATION_LEVELS:
        log.info("Running baseline-anchored perturbation level: %s", level)

        for i in range(N_ITERATIONS):
            perturbed = perturb_original_space(df, cols, level, rng)

            perturbed_scaled = pd.DataFrame(
                scaler.transform(perturbed),
                columns=cols,
                index=df.index,
            )

            sim_mech = classify_to_nearest_centroid(
                perturbed_scaled,
                mechanism_centroids,
            ).reset_index(drop=True)

            sim_arch = classify_to_nearest_centroid(
                perturbed_scaled,
                archetype_centroids,
            ).reset_index(drop=True)

            out = pd.DataFrame(
                {
                    "mechanism_region_id": region_ids,
                    "baseline_mechanism": baseline_mech,
                    "baseline_archetype": baseline_arch,
                    "sim_mechanism": sim_mech,
                    "sim_archetype": sim_arch,
                    "mechanism_stable": sim_mech == baseline_mech,
                    "archetype_stable": sim_arch == baseline_arch,
                    "perturbation_level": level,
                    "iteration": i + 1,
                }
            )

            rows.append(out)

    return pd.concat(rows, ignore_index=True)


def summarize_region(results: pd.DataFrame) -> pd.DataFrame:
    region = (
        results.groupby(
            [
                "mechanism_region_id",
                "baseline_mechanism",
                "baseline_archetype",
                "perturbation_level",
            ]
        )
        .agg(
            mechanism_stability_rate=("mechanism_stable", "mean"),
            archetype_stability_rate=("archetype_stable", "mean"),
            modal_sim_mechanism=("sim_mechanism", lambda s: s.mode().iloc[0]),
            modal_sim_archetype=("sim_archetype", lambda s: s.mode().iloc[0]),
        )
        .reset_index()
    )

    return region


def summarize_mechanism(results: pd.DataFrame) -> pd.DataFrame:
    mechanism = (
        results.groupby(["baseline_mechanism", "perturbation_level"])
        .agg(
            region_count=("mechanism_region_id", "nunique"),
            mechanism_stability_rate=("mechanism_stable", "mean"),
            archetype_stability_rate=("archetype_stable", "mean"),
        )
        .reset_index()
        .rename(columns={"baseline_mechanism": "mechanism"})
    )

    return mechanism


def summarize_archetype(results: pd.DataFrame) -> pd.DataFrame:
    archetype = (
        results.groupby(["baseline_mechanism", "baseline_archetype", "perturbation_level"])
        .agg(
            region_count=("mechanism_region_id", "nunique"),
            mechanism_stability_rate=("mechanism_stable", "mean"),
            archetype_stability_rate=("archetype_stable", "mean"),
        )
        .reset_index()
        .rename(
            columns={
                "baseline_mechanism": "mechanism",
                "baseline_archetype": "archetype",
            }
        )
    )

    return archetype


def classify_stability(rate: float) -> str:
    if pd.isna(rate):
        return "Unknown"
    if rate >= 0.90:
        return "Highly Stable"
    if rate >= 0.75:
        return "Stable"
    if rate >= 0.60:
        return "Moderately Stable"
    if rate >= 0.45:
        return "Weakly Stable"
    return "Unstable"


def add_classes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mechanism_stability_class"] = df["mechanism_stability_rate"].map(classify_stability)
    df["archetype_stability_class"] = df["archetype_stability_rate"].map(classify_stability)
    return df


def make_summary(
    mechanism: pd.DataFrame,
    archetype: pd.DataFrame,
    region: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for level in PERTURBATION_LEVELS:
        m = mechanism[mechanism["perturbation_level"] == level]
        a = archetype[archetype["perturbation_level"] == level]
        r = region[region["perturbation_level"] == level]

        rows.append(
            {
                "perturbation_level": level,
                "mean_mechanism_stability": m["mechanism_stability_rate"].mean(),
                "mean_archetype_stability": a["archetype_stability_rate"].mean(),
                "highly_stable_region_count": int((r["mechanism_stability_rate"] >= 0.90).sum()),
                "stable_or_better_region_count": int((r["mechanism_stability_rate"] >= 0.75).sum()),
                "moderate_or_better_region_count": int((r["mechanism_stability_rate"] >= 0.60).sum()),
                "unstable_region_count": int((r["mechanism_stability_rate"] < 0.45).sum()),
            }
        )

    return pd.DataFrame(rows)


def write_outputs(
    mechanism: pd.DataFrame,
    archetype: pd.DataFrame,
    region: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    log.info("Writing mechanism stability: %s", OUTPUT_MECHANISM)
    mechanism.to_csv(OUTPUT_MECHANISM, index=False)

    log.info("Writing archetype stability: %s", OUTPUT_ARCHETYPE)
    archetype.to_csv(OUTPUT_ARCHETYPE, index=False)

    log.info("Writing region stability: %s", OUTPUT_REGION)
    region.to_csv(OUTPUT_REGION, index=False)

    log.info("Writing summary: %s", OUTPUT_SUMMARY)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    qa = []
    qa.append("RDE Baseline-Anchored Stability Testing v02 QA")
    qa.append("=" * 55)
    qa.append("")
    qa.append(f"Input typology: {INPUT_TYPOLOGY}")
    qa.append(f"Iterations per perturbation level: {N_ITERATIONS}")
    qa.append(f"Perturbation levels: {PERTURBATION_LEVELS}")
    qa.append("")
    qa.append("Overall stability summary:")
    qa.append(summary.to_string(index=False))
    qa.append("")
    qa.append("Mechanism stability:")
    qa.append(mechanism.to_string(index=False))
    qa.append("")
    qa.append("Archetype stability:")
    qa.append(archetype.to_string(index=False))

    log.info("Writing QA: %s", OUTPUT_QA)
    OUTPUT_QA.write_text("\n".join(qa), encoding="utf-8")


def main() -> None:
    log.info("Starting Script 97b: baseline-anchored mechanism stability testing")

    df = load_input()
    x_scaled, cols, scaler = prepare_signature_matrix(df)

    mechanism_centroids = make_centroids(df, x_scaled, "canonical_mechanism")
    archetype_centroids = make_centroids(df, x_scaled, "region_archetype_v02")

    baseline_mech_pred = classify_to_nearest_centroid(x_scaled, mechanism_centroids)
    baseline_arch_pred = classify_to_nearest_centroid(x_scaled, archetype_centroids)

    mech_baseline_accuracy = (
        baseline_mech_pred.values == df["canonical_mechanism"].values
    ).mean()

    arch_baseline_accuracy = (
        baseline_arch_pred.values == df["region_archetype_v02"].values
    ).mean()

    log.info("Baseline nearest-centroid mechanism accuracy: %.3f", mech_baseline_accuracy)
    log.info("Baseline nearest-centroid archetype accuracy: %.3f", arch_baseline_accuracy)

    results = run_simulation(
        df=df,
        cols=cols,
        scaler=scaler,
        mechanism_centroids=mechanism_centroids,
        archetype_centroids=archetype_centroids,
    )

    region = add_classes(summarize_region(results))
    mechanism = add_classes(summarize_mechanism(results))
    archetype = add_classes(summarize_archetype(results))
    summary = make_summary(mechanism, archetype, region)

    summary["baseline_mechanism_centroid_accuracy"] = mech_baseline_accuracy
    summary["baseline_archetype_centroid_accuracy"] = arch_baseline_accuracy

    write_outputs(mechanism, archetype, region, summary)

    log.info("Done")

    print("\nRDE Baseline-Anchored Stability Summary:")
    print(summary.to_string(index=False))

    print("\nMechanism Stability:")
    print(mechanism.to_string(index=False))

    print("\nCreated:")
    print(f"  {OUTPUT_MECHANISM}")
    print(f"  {OUTPUT_ARCHETYPE}")
    print(f"  {OUTPUT_REGION}")
    print(f"  {OUTPUT_SUMMARY}")
    print(f"  {OUTPUT_QA}")


if __name__ == "__main__":
    main()