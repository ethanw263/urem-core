#!/usr/bin/env python3
"""
67_matched_counterfactual_validation_v01.py

Validates top UREM v07 no-coast candidates against the FULL v07 no-coast
universe, not merely against the ranked candidate subset.
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

UNIVERSE_PATH = DATA / "urem_score_v07_no_coast.gpkg"
CANDIDATE_PATH = DATA / "ranked_urem_candidates_v07_no_coast.gpkg"
REGION_PATH = DATA / "v07_no_coast_discovery_regions.gpkg"

OUT_CSV = DATA / "matched_counterfactual_validation_v01.csv"
OUT_GPKG = DATA / "matched_counterfactual_validation_v01.gpkg"
OUT_REGION_CSV = DATA / "matched_counterfactual_region_summary_v01.csv"
OUT_REGION_GPKG = DATA / "matched_counterfactual_region_summary_v01.gpkg"

TOP_N = 300
TARGET_MATCHES = 500
MIN_MATCHES = 100

logging.basicConfig(
    level=logging.INFO,
    format="[67_matched_counterfactual_validation_v01] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(skipna=True)
    if std == 0 or pd.isna(std):
        return s * 0
    return (s - s.mean(skipna=True)) / std


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n)
    running = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * n / rank)
        adjusted[order[i]] = running

    return np.clip(adjusted, 0, 1)


def pick_col(df, options, required=True):
    for c in options:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required column. Tried: {options}")
    return None


def main():
    log.info(f"Reading full validation universe: {UNIVERSE_PATH}")
    universe = gpd.read_file(UNIVERSE_PATH)

    log.info(f"Reading ranked candidates: {CANDIDATE_PATH}")
    candidates = gpd.read_file(CANDIDATE_PATH)

    log.info(f"Universe rows: {len(universe):,}")
    log.info(f"Candidate rows: {len(candidates):,}")
    log.info(f"Universe CRS: {universe.crs}")

    if len(universe) <= len(candidates):
        raise ValueError(
            "Universe file is not larger than candidate file. "
            "This means validation would still be circular."
        )

    score_col = pick_col(
        candidates,
        ["urem_score_v07_no_coast", "urem_score", "urem_score_v07"],
    )

    observed_col = pick_col(
        universe,
        ["observed_recognition_v04", "recognition_score_v04", "observed_recognition"],
    )

    expected_col = pick_col(
        universe,
        ["expected_recognition_v06", "expected_recognition_v05", "expected_recognition"],
    )

    match_features = [
        c for c in [
            "terrain_drama_v03",
            "local_relief_m",
            "slope_deg",
            "elevation_m",
            "physical_exceptionality_v03",
            "distance_to_coast_km",
            "dist_to_coast_km",
        ]
        if c in universe.columns and c in candidates.columns
    ]

    if len(match_features) < 4:
        raise ValueError(f"Not enough matching features found: {match_features}")

    log.info(f"Score column: {score_col}")
    log.info(f"Observed column: {observed_col}")
    log.info(f"Expected column: {expected_col}")
    log.info(f"Matching features: {match_features}")

    universe = universe.copy()
    candidates = candidates.copy()

    universe["recognition_gap_validation"] = (
        pd.to_numeric(universe[expected_col], errors="coerce")
        - pd.to_numeric(universe[observed_col], errors="coerce")
    )

    candidates["recognition_gap_validation"] = (
        pd.to_numeric(candidates[expected_col], errors="coerce")
        - pd.to_numeric(candidates[observed_col], errors="coerce")
    )

    required = match_features + [
        observed_col,
        expected_col,
        "recognition_gap_validation",
    ]

    universe_clean = universe.dropna(subset=required).copy()
    candidates_clean = candidates.dropna(subset=required + [score_col]).copy()

    candidates_clean = candidates_clean.sort_values(score_col, ascending=False).head(TOP_N)

    log.info(f"Clean universe rows: {len(universe_clean):,}")
    log.info(f"Candidates selected: {len(candidates_clean):,}")

    for c in match_features:
        mean = universe_clean[c].mean()
        std = universe_clean[c].std()
        if std == 0 or pd.isna(std):
            universe_clean[f"z_{c}"] = 0
            candidates_clean[f"z_{c}"] = 0
        else:
            universe_clean[f"z_{c}"] = (universe_clean[c] - mean) / std
            candidates_clean[f"z_{c}"] = (candidates_clean[c] - mean) / std

    z_cols = [f"z_{c}" for c in match_features]

    universe_matrix = universe_clean[z_cols].to_numpy(dtype=float)
    universe_gap = universe_clean["recognition_gap_validation"].to_numpy(dtype=float)

    rows = []

    log.info("Running matched counterfactual validation against full universe...")

    for i, (idx, cand) in enumerate(candidates_clean.iterrows(), start=1):
        cand_vec = cand[z_cols].to_numpy(dtype=float)

        dists = np.sqrt(((universe_matrix - cand_vec) ** 2).sum(axis=1))
        order = np.argsort(dists)

        matched_positions = order[:TARGET_MATCHES]

        if len(matched_positions) < MIN_MATCHES:
            log.warning(f"Candidate {idx} has only {len(matched_positions)} matches.")

        matched_gaps = universe_gap[matched_positions]
        cand_gap = float(cand["recognition_gap_validation"])

        percentile = float((matched_gaps <= cand_gap).mean())
        p_value = float(((matched_gaps >= cand_gap).sum() + 1) / (len(matched_gaps) + 1))

        row = {
            "source_index": idx,
            "urem_score": float(cand[score_col]),
            "candidate_gap": cand_gap,
            "observed_recognition": float(cand[observed_col]),
            "expected_recognition": float(cand[expected_col]),
            "matched_gap_mean": float(np.mean(matched_gaps)),
            "matched_gap_median": float(np.median(matched_gaps)),
            "matched_gap_std": float(np.std(matched_gaps)),
            "matched_gap_percentile": percentile,
            "empirical_p_value": p_value,
            "match_pool_n": int(len(matched_positions)),
            "mean_match_distance": float(np.mean(dists[matched_positions])),
        }

        for c in match_features:
            row[c] = cand[c]

        rows.append(row)

        if i % 50 == 0:
            log.info(f"Validated {i}/{len(candidates_clean)} candidates...")

    results = pd.DataFrame(rows)
    results["fdr_q_value"] = bh_fdr(results["empirical_p_value"])
    results["counterfactual_significant_05"] = results["fdr_q_value"] <= 0.05
    results["counterfactual_significant_10"] = results["fdr_q_value"] <= 0.10

    results["counterfactual_strength"] = pd.cut(
        results["matched_gap_percentile"],
        bins=[-np.inf, 0.50, 0.75, 0.90, 0.95, np.inf],
        labels=["weak", "moderate", "strong", "very_strong", "exceptional"],
    )

    out_gdf = candidates_clean.reset_index(drop=True).copy()

    for col in results.columns:
        out_gdf[col] = results[col].values

    log.info(f"Writing CSV: {OUT_CSV}")
    results.to_csv(OUT_CSV, index=False)

    log.info(f"Writing GPKG: {OUT_GPKG}")
    out_gdf.to_file(OUT_GPKG, driver="GPKG")

    if REGION_PATH.exists():
        log.info(f"Reading discovery regions: {REGION_PATH}")
        regions = gpd.read_file(REGION_PATH)

        if regions.crs != out_gdf.crs:
            regions = regions.to_crs(out_gdf.crs)

        joined = gpd.sjoin(out_gdf, regions, how="left", predicate="within")

        region_col = None
        for c in ["region_id", "discovery_region_id", "cluster_id", "id"]:
            if c in joined.columns:
                region_col = c
                break

        if region_col:
            summary = (
                joined.dropna(subset=[region_col])
                .groupby(region_col)
                .agg(
                    validated_cells=("candidate_gap", "count"),
                    mean_urem_score=("urem_score", "mean"),
                    mean_candidate_gap=("candidate_gap", "mean"),
                    median_candidate_gap=("candidate_gap", "median"),
                    mean_matched_gap_percentile=("matched_gap_percentile", "mean"),
                    median_matched_gap_percentile=("matched_gap_percentile", "median"),
                    share_90_plus=("matched_gap_percentile", lambda x: (x >= 0.90).mean()),
                    share_95_plus=("matched_gap_percentile", lambda x: (x >= 0.95).mean()),
                    significant_cells_05=("counterfactual_significant_05", "sum"),
                    significant_cells_10=("counterfactual_significant_10", "sum"),
                    mean_observed_recognition=("observed_recognition", "mean"),
                    mean_expected_recognition=("expected_recognition", "mean"),
                )
                .reset_index()
            )

            summary["share_significant_05"] = (
                summary["significant_cells_05"] / summary["validated_cells"]
            )
            summary["share_significant_10"] = (
                summary["significant_cells_10"] / summary["validated_cells"]
            )

            regions_out = regions.merge(summary, on=region_col, how="left")

            log.info(f"Writing region CSV: {OUT_REGION_CSV}")
            summary.to_csv(OUT_REGION_CSV, index=False)

            log.info(f"Writing region GPKG: {OUT_REGION_GPKG}")
            regions_out.to_file(OUT_REGION_GPKG, driver="GPKG")

    print("\nMatched Counterfactual Validation Summary")
    print("-----------------------------------------")
    print(f"Universe rows: {len(universe_clean):,}")
    print(f"Validated candidates: {len(results):,}")
    print(f"Median matched-gap percentile: {results['matched_gap_percentile'].median():.3f}")
    print(f"Mean matched-gap percentile: {results['matched_gap_percentile'].mean():.3f}")
    print(f"Share >= 90th percentile: {(results['matched_gap_percentile'] >= 0.90).mean():.2%}")
    print(f"Share >= 95th percentile: {(results['matched_gap_percentile'] >= 0.95).mean():.2%}")
    print(f"FDR significant q <= 0.05: {results['counterfactual_significant_05'].mean():.2%}")
    print(f"FDR significant q <= 0.10: {results['counterfactual_significant_10'].mean():.2%}")

    print("\nInterpretation")
    print("--------------")
    print("If percentiles are now high, UREM is finding genuine recognition-gap anomalies.")
    print("If percentiles remain near 0.50, UREM is mostly ranking physical exceptionality.")


if __name__ == "__main__":
    main()