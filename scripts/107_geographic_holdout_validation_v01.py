#!/usr/bin/env python3
"""
107_geographic_holdout_validation_v01.py

Purpose
-------
Test geographic transferability of RDE mechanism classes.

Leave one geographic macro-zone out, learn mechanism signatures from all
remaining zones, then classify held-out regions by nearest mechanism centroid.

Inputs
------
data/processed/region_feature_matrix_v01.csv
data/processed/mechanism_region_typology_v02.csv
data/processed/rde_geographic_landscape_interpretation_v01.csv

Outputs
-------
data/processed/rde_geographic_holdout_validation_v01.csv
data/processed/rde_geographic_holdout_summary_v01.csv
data/processed/rde_geographic_holdout_mechanism_summary_v01.csv
data/processed/rde_geographic_holdout_qa_v01.txt
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


SCRIPT_NAME = "107_geographic_holdout_validation_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUT_MATRIX = PROCESSED / "region_feature_matrix_v01.csv"
INPUT_TYPOLOGY = PROCESSED / "mechanism_region_typology_v02.csv"
INPUT_LANDSCAPE = PROCESSED / "rde_geographic_landscape_interpretation_v01.csv"

OUTPUT_REGION = PROCESSED / "rde_geographic_holdout_validation_v01.csv"
OUTPUT_HOLDOUT = PROCESSED / "rde_geographic_holdout_summary_v01.csv"
OUTPUT_MECHANISM = PROCESSED / "rde_geographic_holdout_mechanism_summary_v01.csv"
OUTPUT_QA = PROCESSED / "rde_geographic_holdout_qa_v01.txt"


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

MIN_TRAIN_PER_MECHANISM = 2


def canonical_mechanism(x: object) -> str:
    s = str(x).replace(" Candidate", "").strip()

    if "Recognition Inefficiency" in s:
        return "Recognition Inefficiency"
    if "Opportunity Failure" in s:
        return "Opportunity Failure"
    if "Comparative Shadowing" in s or "Recognition Diversion" in s:
        return "Comparative Shadowing"

    return s


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in [INPUT_MATRIX, INPUT_TYPOLOGY, INPUT_LANDSCAPE]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    log.info("Reading matrix: %s", INPUT_MATRIX)
    matrix = pd.read_csv(INPUT_MATRIX, low_memory=False)

    log.info("Reading typology: %s", INPUT_TYPOLOGY)
    typology = pd.read_csv(INPUT_TYPOLOGY, low_memory=False)

    log.info("Reading landscape interpretation: %s", INPUT_LANDSCAPE)
    landscape = pd.read_csv(INPUT_LANDSCAPE, low_memory=False)

    for df_name, df in [
        ("matrix", matrix),
        ("typology", typology),
        ("landscape", landscape),
    ]:
        if "mechanism_region_id" not in df.columns:
            raise ValueError(f"{df_name} missing mechanism_region_id")
        df["mechanism_region_id"] = df["mechanism_region_id"].astype(str)

    return matrix, typology, landscape


def build_analysis_table(
    matrix: pd.DataFrame,
    typology: pd.DataFrame,
    landscape: pd.DataFrame,
) -> pd.DataFrame:
    typ = typology.copy()
    land = landscape.copy()
    mat = matrix.copy()

    if "canonical_mechanism" not in typ.columns:
        typ["canonical_mechanism"] = typ["mechanism_class"].map(canonical_mechanism)
    else:
        typ["canonical_mechanism"] = typ["canonical_mechanism"].map(canonical_mechanism)

    keep_typ = [
        "mechanism_region_id",
        "mechanism_class",
        "canonical_mechanism",
        "region_archetype_v02",
        "archetype_strength_v02",
    ]
    keep_typ = [c for c in keep_typ if c in typ.columns]

    keep_typ = [
        "mechanism_region_id",
        "mechanism_class",
        "canonical_mechanism",
        "region_archetype_v02",
        "archetype_strength_v02",
    ]

    sig_cols_in_typology = [c for c in SIGNATURE_COLS if c in typ.columns]
    keep_typ = keep_typ + sig_cols_in_typology
    keep_typ = [c for c in keep_typ if c in typ.columns]

    keep_land = [
        "mechanism_region_id",
        "geographic_macro_zone_v01",
        "geographic_landscape_type_v01",
        "external_validation_priority_score",
        "external_validation_priority_tier",
        "validation_centroid_lat",
        "validation_centroid_lon",
    ]

    keep_land = [c for c in keep_land if c in land.columns]

    df = mat.merge(typ[keep_typ], on="mechanism_region_id", how="left")
    df = df.merge(land[keep_land], on="mechanism_region_id", how="left")

    if "canonical_mechanism" not in df.columns or df["canonical_mechanism"].isna().all():
        raise ValueError("Unable to build canonical_mechanism.")

    if "geographic_macro_zone_v01" not in df.columns:
        raise ValueError("Missing geographic_macro_zone_v01 from landscape interpretation.")

    available_sig = [c for c in SIGNATURE_COLS if c in df.columns]
    if len(available_sig) < 5:
        raise ValueError(f"Too few signature columns found: {available_sig}")

    log.info("Analysis rows: %s", len(df))
    log.info("Signature columns used: %s", len(available_sig))
    log.info("Holdout zones: %s", df["geographic_macro_zone_v01"].nunique())

    return df


def prepare_scaled_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = train[cols].apply(pd.to_numeric, errors="coerce")
    x_test = test[cols].apply(pd.to_numeric, errors="coerce")

    med = x_train.median()
    x_train = x_train.fillna(med).fillna(0)
    x_test = x_test.fillna(med).fillna(0)

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train),
        columns=cols,
        index=train.index,
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        columns=cols,
        index=test.index,
    )

    return x_train_scaled, x_test_scaled


def mechanism_centroids(
    train: pd.DataFrame,
    x_train_scaled: pd.DataFrame,
) -> pd.DataFrame:
    tmp = x_train_scaled.copy()
    tmp["canonical_mechanism"] = train["canonical_mechanism"].values

    counts = tmp["canonical_mechanism"].value_counts()
    valid = counts[counts >= MIN_TRAIN_PER_MECHANISM].index.tolist()
    tmp = tmp[tmp["canonical_mechanism"].isin(valid)].copy()

    centroids = tmp.groupby("canonical_mechanism").mean()

    return centroids


def classify_nearest(
    x_test_scaled: pd.DataFrame,
    centroids: pd.DataFrame,
) -> pd.DataFrame:
    dists = pairwise_distances(x_test_scaled, centroids, metric="euclidean")
    labels = centroids.index.to_numpy()

    nearest_idx = np.argmin(dists, axis=1)
    sorted_dists = np.sort(dists, axis=1)

    nearest = labels[nearest_idx]
    nearest_distance = sorted_dists[:, 0]
    second_distance = sorted_dists[:, 1] if dists.shape[1] > 1 else np.nan
    margin = second_distance - nearest_distance

    out = pd.DataFrame(
        {
            "predicted_mechanism": nearest,
            "nearest_centroid_distance": nearest_distance,
            "second_nearest_centroid_distance": second_distance,
            "classification_margin": margin,
        },
        index=x_test_scaled.index,
    )

    return out


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(a, b) / denom)


def centroid_similarity_by_mechanism(
    train: pd.DataFrame,
    test: pd.DataFrame,
    x_train_scaled: pd.DataFrame,
    x_test_scaled: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for mech in sorted(train["canonical_mechanism"].dropna().unique()):
        train_idx = train["canonical_mechanism"] == mech
        test_idx = test["canonical_mechanism"] == mech

        if train_idx.sum() < MIN_TRAIN_PER_MECHANISM or test_idx.sum() < 1:
            continue

        train_cent = x_train_scaled.loc[train_idx.values].mean(axis=0).to_numpy()
        test_cent = x_test_scaled.loc[test_idx.values].mean(axis=0).to_numpy()

        sim = cosine_similarity(train_cent, test_cent)

        rows.append(
            {
                "mechanism": mech,
                "train_count": int(train_idx.sum()),
                "test_count": int(test_idx.sum()),
                "centroid_cosine_similarity": sim,
            }
        )

    return pd.DataFrame(rows)


def run_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [c for c in SIGNATURE_COLS if c in df.columns]
    region_rows = []
    sim_rows = []

    zones = sorted(df["geographic_macro_zone_v01"].dropna().unique())

    for zone in zones:
        log.info("Running holdout zone: %s", zone)

        train = df[df["geographic_macro_zone_v01"] != zone].copy()
        test = df[df["geographic_macro_zone_v01"] == zone].copy()

        if len(test) == 0 or len(train) == 0:
            continue

        x_train_scaled, x_test_scaled = prepare_scaled_features(train, test, cols)
        centroids = mechanism_centroids(train, x_train_scaled)

        if len(centroids) < 2:
            log.warning("Skipping %s because too few mechanism centroids.", zone)
            continue

        pred = classify_nearest(x_test_scaled, centroids)

        out = test.copy()
        out["holdout_zone"] = zone
        out["train_region_count"] = len(train)
        out["test_region_count"] = len(test)
        out["predicted_mechanism"] = pred["predicted_mechanism"]
        out["nearest_centroid_distance"] = pred["nearest_centroid_distance"]
        out["second_nearest_centroid_distance"] = pred["second_nearest_centroid_distance"]
        out["classification_margin"] = pred["classification_margin"]
        out["mechanism_correct"] = out["predicted_mechanism"] == out["canonical_mechanism"]

        region_rows.append(out)

        sims = centroid_similarity_by_mechanism(train, test, x_train_scaled, x_test_scaled)
        if len(sims) > 0:
            sims["holdout_zone"] = zone
            sim_rows.append(sims)

    region_results = pd.concat(region_rows, ignore_index=True)

    if sim_rows:
        similarity = pd.concat(sim_rows, ignore_index=True)
    else:
        similarity = pd.DataFrame()

    return region_results, similarity


def build_holdout_summary(
    region_results: pd.DataFrame,
    similarity: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for zone, sub in region_results.groupby("holdout_zone"):
        acc = float(sub["mechanism_correct"].mean())
        mean_margin = float(pd.to_numeric(sub["classification_margin"], errors="coerce").mean())

        if len(similarity) > 0:
            sim_sub = similarity[similarity["holdout_zone"] == zone]
            mean_sim = float(sim_sub["centroid_cosine_similarity"].mean()) if len(sim_sub) else np.nan
        else:
            mean_sim = np.nan

        if pd.isna(mean_sim):
            transfer = acc
        else:
            transfer = 0.5 * acc + 0.5 * ((mean_sim + 1) / 2)

        rows.append(
            {
                "holdout_zone": zone,
                "test_region_count": len(sub),
                "mechanism_accuracy": acc,
                "mean_classification_margin": mean_margin,
                "mean_centroid_cosine_similarity": mean_sim,
                "transferability_score": transfer,
                "recognition_inefficiency_count": int((sub["canonical_mechanism"] == "Recognition Inefficiency").sum()),
                "opportunity_failure_count": int((sub["canonical_mechanism"] == "Opportunity Failure").sum()),
                "comparative_shadowing_count": int((sub["canonical_mechanism"] == "Comparative Shadowing").sum()),
            }
        )

    summary = pd.DataFrame(rows).sort_values("transferability_score", ascending=False)
    return summary


def build_mechanism_summary(
    region_results: pd.DataFrame,
    similarity: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for mech, sub in region_results.groupby("canonical_mechanism"):
        recall = float(sub["mechanism_correct"].mean())
        count = len(sub)

        if len(similarity) > 0:
            sim_sub = similarity[similarity["mechanism"] == mech]
            mean_sim = float(sim_sub["centroid_cosine_similarity"].mean()) if len(sim_sub) else np.nan
        else:
            mean_sim = np.nan

        if pd.isna(mean_sim):
            transfer = recall
        else:
            transfer = 0.5 * recall + 0.5 * ((mean_sim + 1) / 2)

        rows.append(
            {
                "mechanism": mech,
                "region_count": count,
                "leave_zone_out_recall": recall,
                "mean_centroid_cosine_similarity": mean_sim,
                "mechanism_transferability_score": transfer,
            }
        )

    out = pd.DataFrame(rows).sort_values("mechanism_transferability_score", ascending=False)
    return out


def classify_transferability(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    if score >= 0.80:
        return "Strong Transferability"
    if score >= 0.65:
        return "Moderate Transferability"
    if score >= 0.50:
        return "Weak Transferability"
    return "Low Transferability"


def trim_region_output(region_results: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "mechanism_region_id",
        "holdout_zone",
        "geographic_macro_zone_v01",
        "geographic_landscape_type_v01",
        "canonical_mechanism",
        "predicted_mechanism",
        "mechanism_correct",
        "nearest_centroid_distance",
        "second_nearest_centroid_distance",
        "classification_margin",
        "external_validation_priority_score",
        "external_validation_priority_tier",
        "validation_centroid_lat",
        "validation_centroid_lon",
        "region_archetype_v02",
        "archetype_strength_v02",
    ]

    return region_results[[c for c in keep if c in region_results.columns]].copy()


def write_outputs(
    region_results: pd.DataFrame,
    holdout_summary: pd.DataFrame,
    mechanism_summary: pd.DataFrame,
) -> None:
    region_export = trim_region_output(region_results)

    holdout_summary = holdout_summary.copy()
    mechanism_summary = mechanism_summary.copy()

    holdout_summary["transferability_class"] = holdout_summary[
        "transferability_score"
    ].map(classify_transferability)

    mechanism_summary["transferability_class"] = mechanism_summary[
        "mechanism_transferability_score"
    ].map(classify_transferability)

    log.info("Writing region holdout results: %s", OUTPUT_REGION)
    region_export.to_csv(OUTPUT_REGION, index=False)

    log.info("Writing holdout summary: %s", OUTPUT_HOLDOUT)
    holdout_summary.to_csv(OUTPUT_HOLDOUT, index=False)

    log.info("Writing mechanism summary: %s", OUTPUT_MECHANISM)
    mechanism_summary.to_csv(OUTPUT_MECHANISM, index=False)

    qa = []
    qa.append("RDE Geographic Holdout Validation v01 QA")
    qa.append("=" * 50)
    qa.append("")
    qa.append(f"Region rows tested: {len(region_export)}")
    qa.append("")
    qa.append("Holdout summary:")
    qa.append(holdout_summary.to_string(index=False))
    qa.append("")
    qa.append("Mechanism summary:")
    qa.append(mechanism_summary.to_string(index=False))
    qa.append("")
    qa.append("Confusion table:")
    qa.append(
        pd.crosstab(
            region_export["canonical_mechanism"],
            region_export["predicted_mechanism"],
        ).to_string()
    )

    log.info("Writing QA: %s", OUTPUT_QA)
    OUTPUT_QA.write_text("\n".join(qa), encoding="utf-8")

    print("\nGeographic Holdout Summary:")
    print(holdout_summary.to_string(index=False))

    print("\nMechanism Transferability Summary:")
    print(mechanism_summary.to_string(index=False))

    print("\nConfusion Table:")
    print(
        pd.crosstab(
            region_export["canonical_mechanism"],
            region_export["predicted_mechanism"],
        ).to_string()
    )


def main() -> None:
    log.info("Starting Script 107: geographic holdout validation")

    matrix, typology, landscape = load_inputs()
    df = build_analysis_table(matrix, typology, landscape)

    region_results, similarity = run_holdout(df)

    holdout_summary = build_holdout_summary(region_results, similarity)
    mechanism_summary = build_mechanism_summary(region_results, similarity)

    write_outputs(region_results, holdout_summary, mechanism_summary)

    log.info("Done")

    print("\nCreated:")
    print(f"  {OUTPUT_REGION}")
    print(f"  {OUTPUT_HOLDOUT}")
    print(f"  {OUTPUT_MECHANISM}")
    print(f"  {OUTPUT_QA}")


if __name__ == "__main__":
    main()