#!/usr/bin/env python3
"""
105_external_proxy_validation_v01.py

Purpose
-------
Create an objective external-proxy validation layer for RDE candidates.

This script does NOT manually score regions.

It uses available processed recognition/proxy layers to test whether RDE
candidate regions have recognition proxy levels that are low relative to their
physical/RDE strength.

Inputs
------
data/processed/rde_external_validation_candidates_v01.csv
data/processed/mechanism_regions_v01.gpkg
data/processed/recognition_score_v04.gpkg
data/processed/orthogonalized_rde_dimensions_v01.gpkg

Outputs
-------
data/processed/rde_external_proxy_validation_v01.csv
data/processed/rde_external_proxy_validation_summary_v01.csv
data/processed/rde_external_proxy_validation_mechanism_summary_v01.csv
data/processed/rde_external_proxy_validation_qa_v01.txt
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SCRIPT_NAME = "105_external_proxy_validation_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUT_CANDIDATES = PROCESSED / "rde_external_validation_candidates_v01.csv"
INPUT_REGIONS = PROCESSED / "mechanism_regions_v01.gpkg"
INPUT_RECOGNITION = PROCESSED / "recognition_score_v04.gpkg"
INPUT_RDE_CELLS = PROCESSED / "orthogonalized_rde_dimensions_v01.gpkg"

OUTPUT_REGION_VALIDATION = PROCESSED / "rde_external_proxy_validation_v01.csv"
OUTPUT_SUMMARY = PROCESSED / "rde_external_proxy_validation_summary_v01.csv"
OUTPUT_MECH_SUMMARY = PROCESSED / "rde_external_proxy_validation_mechanism_summary_v01.csv"
OUTPUT_QA = PROCESSED / "rde_external_proxy_validation_qa_v01.txt"


def canonical_mechanism(x: object) -> str:
    s = str(x).replace(" Candidate", "").strip()
    if "Recognition Inefficiency" in s:
        return "Recognition Inefficiency"
    if "Opportunity Failure" in s:
        return "Opportunity Failure"
    if "Comparative Shadowing" in s or "Recognition Diversion" in s:
        return "Comparative Shadowing"
    return s


def load_inputs():
    for p in [INPUT_CANDIDATES, INPUT_REGIONS, INPUT_RECOGNITION, INPUT_RDE_CELLS]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    log.info("Reading candidates: %s", INPUT_CANDIDATES)
    candidates = pd.read_csv(INPUT_CANDIDATES, low_memory=False)

    log.info("Reading mechanism regions: %s", INPUT_REGIONS)
    regions = gpd.read_file(INPUT_REGIONS)

    log.info("Reading recognition layer: %s", INPUT_RECOGNITION)
    recognition = gpd.read_file(INPUT_RECOGNITION)

    log.info("Reading RDE cell layer: %s", INPUT_RDE_CELLS)
    rde_cells = gpd.read_file(INPUT_RDE_CELLS)

    if regions.crs != recognition.crs:
        recognition = recognition.to_crs(regions.crs)

    if regions.crs != rde_cells.crs:
        rde_cells = rde_cells.to_crs(regions.crs)

    candidates["mechanism_region_id"] = candidates["mechanism_region_id"].astype(str)
    regions["mechanism_region_id"] = regions["mechanism_region_id"].astype(str)

    log.info("Candidates: %s", len(candidates))
    log.info("Regions: %s", len(regions))
    log.info("Recognition cells: %s", len(recognition))
    log.info("RDE cells: %s", len(rde_cells))

    return candidates, regions, recognition, rde_cells


def find_numeric_cols(gdf: gpd.GeoDataFrame, keywords: list[str]) -> list[str]:
    cols = []
    for c in gdf.columns:
        if c == "geometry":
            continue
        if not pd.api.types.is_numeric_dtype(gdf[c]):
            continue
        lc = c.lower()
        if any(k.lower() in lc for k in keywords):
            cols.append(c)
    return cols


def summarize_layer_within_regions(
    regions: gpd.GeoDataFrame,
    layer: gpd.GeoDataFrame,
    prefix: str,
    numeric_cols: list[str],
) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame({"mechanism_region_id": regions["mechanism_region_id"]})

    joined = gpd.sjoin(
        layer[numeric_cols + ["geometry"]],
        regions[["mechanism_region_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    if len(joined) == 0:
        return pd.DataFrame({"mechanism_region_id": regions["mechanism_region_id"]})

    agg = {}
    for c in numeric_cols:
        agg[c] = ["mean", "median", "max", "sum"]

    out = joined.groupby("mechanism_region_id").agg(agg)
    out.columns = [f"{prefix}_{c}_{stat}" for c, stat in out.columns]
    out = out.reset_index()

    counts = (
        joined.groupby("mechanism_region_id")
        .size()
        .reset_index(name=f"{prefix}_matched_cell_count")
    )

    out = out.merge(counts, on="mechanism_region_id", how="left")

    return out


def build_region_proxy_table(
    candidates: pd.DataFrame,
    regions: gpd.GeoDataFrame,
    recognition: gpd.GeoDataFrame,
    rde_cells: gpd.GeoDataFrame,
) -> pd.DataFrame:
    rec_cols = find_numeric_cols(
        recognition,
        [
            "recognition",
            "tourism",
            "trail",
            "park",
            "beach",
            "view",
            "recreation",
            "protected",
            "score",
        ],
    )

    rde_cols = find_numeric_cols(
        rde_cells,
        [
            "P_orthogonal",
            "O_base",
            "T_net",
            "R_net",
            "observed_recognition",
            "expected_recognition",
            "orthogonalized_rde",
        ],
    )

    log.info("Recognition proxy columns: %s", len(rec_cols))
    log.info("RDE proxy columns: %s", len(rde_cols))

    rec_summary = summarize_layer_within_regions(
        regions,
        recognition,
        "external_recognition_proxy",
        rec_cols,
    )

    rde_summary = summarize_layer_within_regions(
        regions,
        rde_cells,
        "rde_cell",
        rde_cols,
    )

    base = candidates.copy()
    base["mechanism_region_id"] = base["mechanism_region_id"].astype(str)

    out = base.merge(rec_summary, on="mechanism_region_id", how="left")
    out = out.merge(rde_summary, on="mechanism_region_id", how="left")

    if "canonical_mechanism" not in out.columns:
        out["canonical_mechanism"] = out.get("mechanism_class", "").map(canonical_mechanism)
    else:
        out["canonical_mechanism"] = out["canonical_mechanism"].map(canonical_mechanism)

    return out


def choose_best_col(df: pd.DataFrame, patterns: list[str]) -> str | None:
    candidates = []
    for c in df.columns:
        lc = c.lower()
        if any(p.lower() in lc for p in patterns):
            if pd.api.types.is_numeric_dtype(df[c]) or pd.to_numeric(df[c], errors="coerce").notna().sum() > 0:
                candidates.append(c)

    if not candidates:
        return None

    priority = ["_mean", "_median", "_sum", "_max"]
    for p in priority:
        for c in candidates:
            if p in c.lower():
                return c

    return candidates[0]


def normalize_series(s: pd.Series, invert: bool = False) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mn = x.min(skipna=True)
    mx = x.max(skipna=True)

    if pd.isna(mn) or pd.isna(mx) or abs(mx - mn) < 1e-12:
        out = pd.Series(0.5, index=x.index)
    else:
        out = (x - mn) / (mx - mn)

    if invert:
        out = 1 - out

    return out


def score_external_validation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    physical_col = choose_best_col(df, ["mean_P_orthogonal", "P_orthogonal", "physical"])
    deficit_col = choose_best_col(df, ["mean_R_net", "R_net_under", "under_recognition"])
    rde_col = choose_best_col(df, ["orthogonalized_rde", "rde_composite"])
    observed_col = choose_best_col(df, ["observed_recognition", "recognition_score_v04"])
    expected_col = choose_best_col(df, ["expected_recognition"])

    external_recognition_cols = [
        c for c in df.columns
        if c.startswith("external_recognition_proxy_")
        and c.endswith("_mean")
    ]

    if external_recognition_cols:
        rec_values = df[external_recognition_cols].apply(pd.to_numeric, errors="coerce")
        df["external_proxy_recognition_mean"] = rec_values.mean(axis=1)
    elif observed_col:
        df["external_proxy_recognition_mean"] = pd.to_numeric(df[observed_col], errors="coerce")
    else:
        df["external_proxy_recognition_mean"] = np.nan

    if physical_col:
        df["validation_physical_norm"] = normalize_series(df[physical_col])
    else:
        df["validation_physical_norm"] = np.nan

    if deficit_col:
        df["validation_deficit_norm"] = normalize_series(df[deficit_col])
    else:
        df["validation_deficit_norm"] = np.nan

    if rde_col:
        df["validation_rde_norm"] = normalize_series(df[rde_col])
    else:
        df["validation_rde_norm"] = np.nan

    df["validation_external_recognition_norm"] = normalize_series(
        df["external_proxy_recognition_mean"]
    )

    df["validation_external_under_recognition_norm"] = normalize_series(
        df["external_proxy_recognition_mean"],
        invert=True,
    )

    if expected_col and observed_col:
        df["validation_expected_minus_observed"] = (
            pd.to_numeric(df[expected_col], errors="coerce")
            - pd.to_numeric(df[observed_col], errors="coerce")
        )
        df["validation_expected_gap_norm"] = normalize_series(
            df["validation_expected_minus_observed"]
        )
    else:
        df["validation_expected_gap_norm"] = np.nan

    components = [
        "validation_physical_norm",
        "validation_deficit_norm",
        "validation_rde_norm",
        "validation_external_under_recognition_norm",
        "validation_expected_gap_norm",
    ]

    df["external_proxy_validation_score"] = df[components].mean(axis=1)

    def classify(v):
        if pd.isna(v):
            return "Insufficient Proxy Data"
        if v >= 0.70:
            return "Strong External Proxy Support"
        if v >= 0.55:
            return "Moderate External Proxy Support"
        if v >= 0.40:
            return "Weak / Mixed External Proxy Support"
        return "Low External Proxy Support"

    df["external_proxy_validation_class"] = df["external_proxy_validation_score"].map(classify)

    df["external_proxy_validation_note"] = (
        "Automated proxy validation only. This is not manual ground truth. "
        "Use as objective preliminary evidence before deeper external review."
    )

    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "metric": "regions_evaluated",
            "value": len(df),
        },
        {
            "metric": "strong_external_proxy_support",
            "value": int((df["external_proxy_validation_class"] == "Strong External Proxy Support").sum()),
        },
        {
            "metric": "moderate_or_stronger_external_proxy_support",
            "value": int(df["external_proxy_validation_class"].isin([
                "Strong External Proxy Support",
                "Moderate External Proxy Support",
            ]).sum()),
        },
        {
            "metric": "mean_external_proxy_validation_score",
            "value": float(df["external_proxy_validation_score"].mean()),
        },
    ]

    return pd.DataFrame(rows)


def build_mechanism_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("canonical_mechanism")
        .agg(
            region_count=("mechanism_region_id", "count"),
            mean_external_proxy_validation_score=("external_proxy_validation_score", "mean"),
            strong_support_count=(
                "external_proxy_validation_class",
                lambda s: int((s == "Strong External Proxy Support").sum()),
            ),
            moderate_or_stronger_count=(
                "external_proxy_validation_class",
                lambda s: int(s.isin([
                    "Strong External Proxy Support",
                    "Moderate External Proxy Support",
                ]).sum()),
            ),
            mean_external_recognition_proxy=("external_proxy_recognition_mean", "mean"),
            mean_external_under_recognition_norm=("validation_external_under_recognition_norm", "mean"),
        )
        .reset_index()
        .sort_values("mean_external_proxy_validation_score", ascending=False)
    )

    return summary


def trim_output(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "mechanism_region_id",
        "canonical_mechanism",
        "external_validation_priority_score",
        "external_validation_priority_tier",
        "validation_centroid_lat",
        "validation_centroid_lon",
        "mean_P_orthogonal_v01",
        "mean_O_base_opportunity_v01",
        "mean_T_net_transmission_v01",
        "mean_R_net_under_recognition_v01",
        "mean_observed_recognition_v04",
        "mean_expected_recognition_v06",
        "external_proxy_recognition_mean",
        "validation_physical_norm",
        "validation_deficit_norm",
        "validation_rde_norm",
        "validation_external_recognition_norm",
        "validation_external_under_recognition_norm",
        "validation_expected_gap_norm",
        "external_proxy_validation_score",
        "external_proxy_validation_class",
        "external_proxy_validation_note",
    ]

    cols = [c for c in preferred if c in df.columns]
    return df[cols].copy()


def write_outputs(df: pd.DataFrame, summary: pd.DataFrame, mech_summary: pd.DataFrame) -> None:
    trimmed = trim_output(df)

    log.info("Writing external proxy validation: %s", OUTPUT_REGION_VALIDATION)
    trimmed.to_csv(OUTPUT_REGION_VALIDATION, index=False)

    log.info("Writing summary: %s", OUTPUT_SUMMARY)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    log.info("Writing mechanism summary: %s", OUTPUT_MECH_SUMMARY)
    mech_summary.to_csv(OUTPUT_MECH_SUMMARY, index=False)

    qa = []
    qa.append("RDE External Proxy Validation v01 QA")
    qa.append("=" * 45)
    qa.append("")
    qa.append("Summary:")
    qa.append(summary.to_string(index=False))
    qa.append("")
    qa.append("Mechanism summary:")
    qa.append(mech_summary.to_string(index=False))
    qa.append("")
    qa.append("Validation class counts:")
    qa.append(trimmed["external_proxy_validation_class"].value_counts().to_string())
    qa.append("")
    qa.append("Top 20 validated candidates:")
    qa.append(
        trimmed.sort_values("external_proxy_validation_score", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    log.info("Writing QA: %s", OUTPUT_QA)
    OUTPUT_QA.write_text("\n".join(qa), encoding="utf-8")


def main() -> None:
    log.info("Starting Script 105: external proxy validation")

    candidates, regions, recognition, rde_cells = load_inputs()

    proxy_table = build_region_proxy_table(
        candidates=candidates,
        regions=regions,
        recognition=recognition,
        rde_cells=rde_cells,
    )

    scored = score_external_validation(proxy_table)
    summary = build_summary(scored)
    mech_summary = build_mechanism_summary(scored)

    write_outputs(scored, summary, mech_summary)

    log.info("Done")

    print("\nExternal Proxy Validation Summary:")
    print(summary.to_string(index=False))

    print("\nMechanism Summary:")
    print(mech_summary.to_string(index=False))

    print("\nValidation Class Counts:")
    print(scored["external_proxy_validation_class"].value_counts().to_string())

    print("\nCreated:")
    print(f"  {OUTPUT_REGION_VALIDATION}")
    print(f"  {OUTPUT_SUMMARY}")
    print(f"  {OUTPUT_MECH_SUMMARY}")
    print(f"  {OUTPUT_QA}")


if __name__ == "__main__":
    main()