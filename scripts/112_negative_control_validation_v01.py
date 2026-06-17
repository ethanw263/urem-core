#!/usr/bin/env python3
"""
112_negative_control_validation_v01.py

Purpose
-------
Test whether RDE candidate regions are distinguishable from famous / already-recognized
landscapes.

This is a critical reviewer-facing validation.

Core question:
    Is RDE merely identifying beautiful/high-potential places,
    or is it identifying recognition disequilibrium?

Negative controls should represent places that are physically exceptional AND already
well-recognized. If the model is behaving correctly, negative controls should generally show:

    High physical potential
    Higher observed recognition
    Lower under-recognition
    Lower RDE disequilibrium

compared with RDE mechanism regions.

Inputs
------
data/processed/rde_external_validation_candidates_v01.csv
data/processed/orthogonalized_rde_dimensions_v01.gpkg
data/processed/recognition_score_v04.gpkg

Outputs
-------
data/processed/rde_negative_control_validation_v01.csv
data/processed/rde_negative_control_summary_v01.csv
data/processed/rde_negative_control_feature_tests_v01.csv
data/processed/rde_negative_control_qa_v01.txt
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp


SCRIPT_NAME = "112_negative_control_validation_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUT_RDE_CANDIDATES = PROCESSED / "rde_external_validation_candidates_v01.csv"
INPUT_RDE_CELLS = PROCESSED / "orthogonalized_rde_dimensions_v01.gpkg"
INPUT_RECOGNITION = PROCESSED / "recognition_score_v04.gpkg"

OUTPUT_VALIDATION = PROCESSED / "rde_negative_control_validation_v01.csv"
OUTPUT_SUMMARY = PROCESSED / "rde_negative_control_summary_v01.csv"
OUTPUT_TESTS = PROCESSED / "rde_negative_control_feature_tests_v01.csv"
OUTPUT_QA = PROCESSED / "rde_negative_control_qa_v01.txt"


NEGATIVE_CONTROLS = [
    {"control_id": "nc_big_sur_core", "control_name": "Big Sur Core", "control_type": "recognized_coastal_landscape", "lat": 36.2704, "lon": -121.8081, "radius_km": 15, "recognition_rationale": "Iconic California coastal landscape with extensive tourism recognition."},
    {"control_id": "nc_point_reyes", "control_name": "Point Reyes National Seashore", "control_type": "recognized_coastal_landscape", "lat": 38.0685, "lon": -122.8790, "radius_km": 14, "recognition_rationale": "National seashore and highly recognized Bay Area coastal destination."},
    {"control_id": "nc_monterey_carmel", "control_name": "Monterey / Carmel Coast", "control_type": "recognized_coastal_landscape", "lat": 36.5552, "lon": -121.9233, "radius_km": 12, "recognition_rationale": "Highly recognized coastal tourism region including Monterey and Carmel."},
    {"control_id": "nc_santa_barbara_coast", "control_name": "Santa Barbara Coast", "control_type": "recognized_coastal_landscape", "lat": 34.4208, "lon": -119.6982, "radius_km": 12, "recognition_rationale": "Major recognized coastal destination with strong tourism identity."},
    {"control_id": "nc_laguna_beach", "control_name": "Laguna Beach", "control_type": "recognized_coastal_landscape", "lat": 33.5427, "lon": -117.7854, "radius_km": 8, "recognition_rationale": "Famous Southern California coastal destination."},
    {"control_id": "nc_santa_monica_malibu", "control_name": "Santa Monica / Malibu Coast", "control_type": "recognized_coastal_landscape", "lat": 34.0259, "lon": -118.7798, "radius_km": 15, "recognition_rationale": "Globally recognized Los Angeles coastal landscape."},
    {"control_id": "nc_yosemite_valley", "control_name": "Yosemite Valley", "control_type": "recognized_exceptional_landscape", "lat": 37.7456, "lon": -119.5936, "radius_km": 10, "recognition_rationale": "Globally recognized physically exceptional landscape."},
    {"control_id": "nc_lake_tahoe_south", "control_name": "South Lake Tahoe", "control_type": "recognized_exceptional_landscape", "lat": 38.9399, "lon": -119.9772, "radius_km": 12, "recognition_rationale": "Highly recognized lake/mountain recreation destination."},
    {"control_id": "nc_golden_gate_marin_headlands", "control_name": "Golden Gate / Marin Headlands", "control_type": "recognized_coastal_landscape", "lat": 37.8270, "lon": -122.4990, "radius_km": 10, "recognition_rationale": "Iconic recognized coastal/headland landscape adjacent to San Francisco."},
    {"control_id": "nc_joshua_tree", "control_name": "Joshua Tree National Park", "control_type": "recognized_exceptional_landscape", "lat": 33.8734, "lon": -115.9010, "radius_km": 18, "recognition_rationale": "Highly recognized desert landscape and national park."},
]


FEATURE_CANDIDATES = {
    "physical_potential": ["P_orthogonal_v01", "physical_exceptionality_v03", "physical_potential"],
    "opportunity": ["O_base_opportunity_v01", "opportunity_structure_index_v01"],
    "transmission": ["T_net_transmission_v01", "recognition_transmission_index_v01"],
    "under_recognition": ["R_net_under_recognition_v01", "under_recognition", "recognition_deficit"],
    "rde_composite": ["orthogonalized_rde_v01", "rde_v01_composite_score", "rde_composite_score"],
    "observed_recognition": ["observed_recognition_v04", "recognition_score_v04"],
    "expected_recognition": ["expected_recognition_v06", "expected_recognition"],
}


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in df.columns:
        lc = c.lower()
        for cand in candidates:
            if cand.lower() in lc:
                return c
    return None


def canonical_mechanism(x: object) -> str:
    s = str(x).replace(" Candidate", "").strip()
    if "Recognition Inefficiency" in s:
        return "Recognition Inefficiency"
    if "Opportunity Failure" in s:
        return "Opportunity Failure"
    if "Comparative Shadowing" in s or "Recognition Diversion" in s:
        return "Comparative Shadowing"
    return s


def load_inputs() -> tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    for p in [INPUT_RDE_CANDIDATES, INPUT_RDE_CELLS, INPUT_RECOGNITION]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    log.info("Reading RDE candidates: %s", INPUT_RDE_CANDIDATES)
    candidates = pd.read_csv(INPUT_RDE_CANDIDATES, low_memory=False)

    log.info("Reading RDE cells: %s", INPUT_RDE_CELLS)
    rde_cells = gpd.read_file(INPUT_RDE_CELLS)

    log.info("Reading recognition cells: %s", INPUT_RECOGNITION)
    recognition = gpd.read_file(INPUT_RECOGNITION)

    if rde_cells.crs != recognition.crs:
        recognition = recognition.to_crs(rde_cells.crs)

    return candidates, rde_cells, recognition


def make_control_gdf(rde_cells: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    controls = pd.DataFrame(NEGATIVE_CONTROLS)
    gdf = gpd.GeoDataFrame(
        controls,
        geometry=gpd.points_from_xy(controls["lon"], controls["lat"]),
        crs="EPSG:4326",
    )
    gdf = gdf.to_crs(rde_cells.crs)
    gdf["geometry"] = gdf.apply(lambda r: r.geometry.buffer(float(r["radius_km"]) * 1000.0), axis=1)
    return gdf


def extract_features_from_cells(cells: gpd.GeoDataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=cells.index)
    found = {}
    for feature, candidates in FEATURE_CANDIDATES.items():
        col = find_col(cells, candidates)
        found[feature] = col
        if col is not None:
            out[feature] = pd.to_numeric(cells[col], errors="coerce")
        else:
            out[feature] = np.nan

    log.info("Feature column mapping:")
    for k, v in found.items():
        log.info("  %s -> %s", k, v)

    return out


def summarize_rde_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    df = candidates.copy()
    if "canonical_mechanism" not in df.columns:
        if "mechanism_class" in df.columns:
            df["canonical_mechanism"] = df["mechanism_class"].map(canonical_mechanism)
        else:
            df["canonical_mechanism"] = "Unknown"

    mapping = {
        "mean_P_orthogonal_v01": "physical_potential",
        "mean_O_base_opportunity_v01": "opportunity",
        "mean_T_net_transmission_v01": "transmission",
        "mean_R_net_under_recognition_v01": "under_recognition",
        "mean_orthogonalized_rde_v01": "rde_composite",
        "mean_observed_recognition_v04": "observed_recognition",
        "mean_expected_recognition_v06": "expected_recognition",
    }

    rows = []
    for _, r in df.iterrows():
        row = {
            "sample_id": r.get("mechanism_region_id", ""),
            "sample_name": r.get("mechanism_region_id", ""),
            "sample_type": "rde_candidate_region",
            "group": "RDE Candidate",
            "canonical_mechanism": r.get("canonical_mechanism", ""),
            "control_type": "",
            "recognition_rationale": "",
            "lat": r.get("validation_centroid_lat", np.nan),
            "lon": r.get("validation_centroid_lon", np.nan),
            "radius_km": np.nan,
            "cell_count": r.get("cell_count", np.nan),
        }
        for src, dst in mapping.items():
            row[dst] = pd.to_numeric(pd.Series([r.get(src, np.nan)]), errors="coerce").iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_controls(
    controls: gpd.GeoDataFrame,
    rde_cells: gpd.GeoDataFrame,
    recognition: gpd.GeoDataFrame,
) -> pd.DataFrame:
    cells = rde_cells.copy()
    feature_df = extract_features_from_cells(cells)

    cells_for_join = cells[["geometry"]].copy()
    for c in feature_df.columns:
        cells_for_join[c] = feature_df[c]

    joined = gpd.sjoin(
        cells_for_join,
        controls[["control_id", "control_name", "control_type", "recognition_rationale", "lat", "lon", "radius_km", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    if len(joined) == 0:
        raise ValueError("No cells intersected negative-control buffers. Check CRS/radii.")

    rows = []
    for control_id, sub in joined.groupby("control_id"):
        first = sub.iloc[0]
        row = {
            "sample_id": control_id,
            "sample_name": first["control_name"],
            "sample_type": "negative_control",
            "group": "Recognized Negative Control",
            "canonical_mechanism": "Negative Control",
            "control_type": first["control_type"],
            "recognition_rationale": first["recognition_rationale"],
            "lat": first["lat"],
            "lon": first["lon"],
            "radius_km": first["radius_km"],
            "cell_count": int(len(sub)),
        }
        for feature in FEATURE_CANDIDATES.keys():
            if feature in sub.columns:
                row[feature] = float(pd.to_numeric(sub[feature], errors="coerce").mean())
            else:
                row[feature] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def normalize_against_all(df: pd.DataFrame, col: str, invert: bool = False) -> pd.Series:
    x = pd.to_numeric(df[col], errors="coerce")
    mn = x.min(skipna=True)
    mx = x.max(skipna=True)

    if pd.isna(mn) or pd.isna(mx) or abs(mx - mn) < 1e-12:
        out = pd.Series(0.5, index=df.index)
    else:
        out = (x - mn) / (mx - mn)

    if invert:
        out = 1 - out
    return out


def add_validation_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in [
        "physical_potential",
        "opportunity",
        "transmission",
        "under_recognition",
        "rde_composite",
        "observed_recognition",
        "expected_recognition",
    ]:
        if c not in out.columns:
            out[c] = np.nan

    out["norm_physical_potential"] = normalize_against_all(out, "physical_potential")
    out["norm_under_recognition"] = normalize_against_all(out, "under_recognition")
    out["norm_rde_composite"] = normalize_against_all(out, "rde_composite")
    out["norm_observed_recognition"] = normalize_against_all(out, "observed_recognition")
    out["norm_expected_recognition"] = normalize_against_all(out, "expected_recognition")

    out["expected_minus_observed"] = (
        pd.to_numeric(out["expected_recognition"], errors="coerce")
        - pd.to_numeric(out["observed_recognition"], errors="coerce")
    )
    out["norm_expected_gap"] = normalize_against_all(out, "expected_minus_observed")

    out["rde_positive_signature_score"] = out[
        [
            "norm_physical_potential",
            "norm_under_recognition",
            "norm_rde_composite",
            "norm_expected_gap",
        ]
    ].mean(axis=1)

    out["recognized_control_signature_score"] = out[
        [
            "norm_physical_potential",
            "norm_observed_recognition",
            "norm_expected_recognition",
        ]
    ].mean(axis=1)

    out["disequilibrium_vs_recognition_contrast"] = (
        out["rde_positive_signature_score"] - out["recognized_control_signature_score"]
    )

    def classify(row: pd.Series) -> str:
        sample_type = row.get("sample_type", "")
        rde_score = row.get("rde_positive_signature_score", np.nan)
        recognized_score = row.get("recognized_control_signature_score", np.nan)
        observed = row.get("norm_observed_recognition", np.nan)
        under = row.get("norm_under_recognition", np.nan)

        if sample_type == "negative_control":
            if recognized_score >= 0.55 and under <= 0.55:
                return "Good Negative Control"
            if observed >= 0.55:
                return "Recognized but Disequilibrium-Mixed"
            return "Weak / Ambiguous Negative Control"

        if sample_type == "rde_candidate_region":
            if rde_score >= 0.60 and under >= 0.55:
                return "RDE-Consistent Candidate"
            if rde_score >= 0.45:
                return "Mixed RDE Candidate"
            return "Weak / Ambiguous RDE Candidate"

        return "Unknown"

    out["negative_control_validation_class"] = out.apply(classify, axis=1)
    return out


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
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def interpret_feature(feature: str, diff: float, delta: float) -> str:
    if pd.isna(diff) or pd.isna(delta):
        return "Insufficient data"

    if feature in [
        "under_recognition",
        "rde_composite",
        "rde_positive_signature_score",
        "expected_minus_observed",
        "norm_under_recognition",
        "norm_rde_composite",
    ]:
        if diff > 0:
            return "Supports RDE: candidates exceed recognized controls on disequilibrium."
        return "Does not support expected RDE separation."

    if feature in [
        "observed_recognition",
        "recognized_control_signature_score",
        "norm_observed_recognition",
    ]:
        if diff < 0:
            return "Supports negative control: recognized controls exceed RDE candidates on recognition."
        return "Does not support expected recognized-control separation."

    if feature == "physical_potential":
        return "Diagnostic: both groups may be physically strong; separation is not expected to rely only on physical potential."

    return "Diagnostic comparison."


def test_feature(df: pd.DataFrame, feature: str) -> dict:
    rde = df.loc[df["sample_type"] == "rde_candidate_region", feature].dropna()
    ctrl = df.loc[df["sample_type"] == "negative_control", feature].dropna()

    if len(rde) < 3 or len(ctrl) < 3:
        return {
            "feature": feature,
            "rde_count": len(rde),
            "negative_control_count": len(ctrl),
            "rde_mean": np.nan,
            "negative_control_mean": np.nan,
            "mean_difference_rde_minus_control": np.nan,
            "mannwhitney_p": np.nan,
            "ks_p": np.nan,
            "cliffs_delta": np.nan,
            "interpretation": "Insufficient data",
        }

    mw = mannwhitneyu(rde, ctrl, alternative="two-sided")
    ks = ks_2samp(rde, ctrl)
    delta = cliffs_delta(rde, ctrl)

    rde_mean = float(rde.mean())
    ctrl_mean = float(ctrl.mean())
    diff = rde_mean - ctrl_mean

    return {
        "feature": feature,
        "rde_count": int(len(rde)),
        "negative_control_count": int(len(ctrl)),
        "rde_mean": rde_mean,
        "negative_control_mean": ctrl_mean,
        "mean_difference_rde_minus_control": diff,
        "rde_median": float(rde.median()),
        "negative_control_median": float(ctrl.median()),
        "median_difference_rde_minus_control": float(rde.median() - ctrl.median()),
        "mannwhitney_p": float(mw.pvalue),
        "ks_p": float(ks.pvalue),
        "cliffs_delta": delta,
        "direction": "RDE higher" if diff > 0 else ("Negative controls higher" if diff < 0 else "No difference"),
        "interpretation": interpret_feature(feature, diff, delta),
    }


def build_feature_tests(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "physical_potential",
        "observed_recognition",
        "expected_recognition",
        "under_recognition",
        "rde_composite",
        "expected_minus_observed",
        "rde_positive_signature_score",
        "recognized_control_signature_score",
        "disequilibrium_vs_recognition_contrast",
    ]

    return pd.DataFrame([test_feature(df, f) for f in features if f in df.columns])


def build_summary(df: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    rde = df[df["sample_type"] == "rde_candidate_region"].copy()
    ctrl = df[df["sample_type"] == "negative_control"].copy()

    rows = [
        {"metric": "rde_candidate_count", "value": len(rde)},
        {"metric": "negative_control_count", "value": len(ctrl)},
        {
            "metric": "rde_consistent_candidate_count",
            "value": int((rde["negative_control_validation_class"] == "RDE-Consistent Candidate").sum()),
        },
        {
            "metric": "good_negative_control_count",
            "value": int((ctrl["negative_control_validation_class"] == "Good Negative Control").sum()),
        },
        {
            "metric": "mean_rde_positive_signature_rde_candidates",
            "value": float(rde["rde_positive_signature_score"].mean()),
        },
        {
            "metric": "mean_rde_positive_signature_negative_controls",
            "value": float(ctrl["rde_positive_signature_score"].mean()),
        },
        {
            "metric": "mean_recognized_signature_rde_candidates",
            "value": float(rde["recognized_control_signature_score"].mean()),
        },
        {
            "metric": "mean_recognized_signature_negative_controls",
            "value": float(ctrl["recognized_control_signature_score"].mean()),
        },
    ]

    rde_sig = tests[tests["feature"] == "rde_positive_signature_score"]
    rec_sig = tests[tests["feature"] == "recognized_control_signature_score"]

    if len(rde_sig) > 0:
        rows.append(
            {
                "metric": "rde_signature_test_interpretation",
                "value": rde_sig["interpretation"].iloc[0],
            }
        )

    if len(rec_sig) > 0:
        rows.append(
            {
                "metric": "recognized_signature_test_interpretation",
                "value": rec_sig["interpretation"].iloc[0],
            }
        )

    return pd.DataFrame(rows)


def trim_validation_output(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "sample_id",
        "sample_name",
        "sample_type",
        "group",
        "canonical_mechanism",
        "control_type",
        "recognition_rationale",
        "lat",
        "lon",
        "radius_km",
        "cell_count",
        "physical_potential",
        "opportunity",
        "transmission",
        "observed_recognition",
        "expected_recognition",
        "under_recognition",
        "rde_composite",
        "expected_minus_observed",
        "rde_positive_signature_score",
        "recognized_control_signature_score",
        "disequilibrium_vs_recognition_contrast",
        "negative_control_validation_class",
    ]

    return df[[c for c in keep if c in df.columns]].copy()


def write_outputs(df: pd.DataFrame, summary: pd.DataFrame, tests: pd.DataFrame) -> None:
    trimmed = trim_validation_output(df)

    log.info("Writing validation: %s", OUTPUT_VALIDATION)
    trimmed.to_csv(OUTPUT_VALIDATION, index=False)

    log.info("Writing summary: %s", OUTPUT_SUMMARY)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    log.info("Writing feature tests: %s", OUTPUT_TESTS)
    tests.to_csv(OUTPUT_TESTS, index=False)

    qa = []
    qa.append("RDE Negative Control Validation v01 QA")
    qa.append("=" * 50)
    qa.append("")
    qa.append("Purpose:")
    qa.append(
        "Test whether RDE candidate regions differ from famous/recognized landscapes, "
        "showing that RDE is not merely identifying physically beautiful places."
    )
    qa.append("")
    qa.append("Summary:")
    qa.append(summary.to_string(index=False))
    qa.append("")
    qa.append("Feature tests:")
    qa.append(tests.to_string(index=False))
    qa.append("")
    qa.append("Negative controls:")
    qa.append(
        trimmed[trimmed["sample_type"] == "negative_control"]
        .sort_values("recognized_control_signature_score", ascending=False)
        .to_string(index=False)
    )
    qa.append("")
    qa.append("Top RDE-consistent candidates:")
    qa.append(
        trimmed[trimmed["sample_type"] == "rde_candidate_region"]
        .sort_values("rde_positive_signature_score", ascending=False)
        .head(30)
        .to_string(index=False)
    )
    qa.append("")
    qa.append("Interpretation warning:")
    qa.append(
        "Negative controls use approximate centroid-radius buffers. For manuscript use, "
        "replace with official boundaries or manually validated polygons where possible."
    )

    log.info("Writing QA: %s", OUTPUT_QA)
    OUTPUT_QA.write_text("\n".join(qa), encoding="utf-8")


def main() -> None:
    log.info("Starting Script 112: negative control validation")

    candidates, rde_cells, recognition = load_inputs()

    controls = make_control_gdf(rde_cells)
    log.info("Negative controls: %s", len(controls))

    rde_summary = summarize_rde_candidates(candidates)
    control_summary = summarize_controls(controls, rde_cells, recognition)

    combined = pd.concat([rde_summary, control_summary], ignore_index=True)
    scored = add_validation_scores(combined)

    tests = build_feature_tests(scored)
    summary = build_summary(scored, tests)

    write_outputs(scored, summary, tests)

    log.info("Done")

    print("\nNegative Control Validation Summary:")
    print(summary.to_string(index=False))

    print("\nFeature Tests:")
    print(
        tests[
            [
                "feature",
                "rde_mean",
                "negative_control_mean",
                "mean_difference_rde_minus_control",
                "mannwhitney_p",
                "cliffs_delta",
                "interpretation",
            ]
        ].to_string(index=False)
    )

    print("\nCreated:")
    print(f"  {OUTPUT_VALIDATION}")
    print(f"  {OUTPUT_SUMMARY}")
    print(f"  {OUTPUT_TESTS}")
    print(f"  {OUTPUT_QA}")


if __name__ == "__main__":
    main()
