from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np

from scipy.stats import mannwhitneyu, kruskal, chi2_contingency


SCRIPT = "218_oregon_srs_persistence_regime_diagnostics_v01"

BASE = Path("data/processed")

INPUT_FP = BASE / "oregon_srs_relational_persistence_common_population_v01.gpkg"

OUT_CONTINUOUS = BASE / "oregon_srs_persistence_regime_continuous_diagnostics_v01.csv"
OUT_CATEGORICAL = BASE / "oregon_srs_persistence_regime_categorical_diagnostics_v01.csv"
OUT_PROFILE = BASE / "oregon_srs_persistence_regime_profiles_v01.csv"
OUT_QA = BASE / "oregon_srs_persistence_regime_diagnostics_qa_v01.txt"

REGIME = "persistence_regime_v01"

REL = "persistent_relational"
PT = "persistent_pointwise"


def log(msg):
    print(f"[{SCRIPT}] {msg}")


def rank_biserial(u, n1, n2):
    if n1 == 0 or n2 == 0:
        return np.nan
    return (2 * u / (n1 * n2)) - 1


def cramers_v(table):
    chi2, _, _, _ = chi2_contingency(table)
    n = table.sum()
    if n == 0:
        return np.nan
    r, k = table.shape
    return np.sqrt((chi2 / n) / max(1, min(k - 1, r - 1)))


def main():
    log("Reading persistence common-population layer")
    gdf = gpd.read_file(INPUT_FP)

    if REGIME not in gdf.columns:
        raise ValueError(f"Missing {REGIME}")

    work = gdf[gdf[REGIME].notna()].copy()

    log(f"Rows with persistence regime: {len(work):,}")

    continuous_candidates = [
        "mean_gain_across_buffers_v01",
        "median_gain_across_buffers_v01",
        "share_buffers_relational_better_v01",
        "max_tested_positive_buffer_m_v01",
        "gain_sign_changes_v01",
        "phi_physical_v01",
        "physical_exceptionality_v03",
        "physical_exceptionality_raw_v03",
        "terrain_drama_v03",
        "scenic_coast_v03",
        "relief_norm_v03",
        "slope_norm_v03",
        "coast_proximity_v03",
        "coast_complexity_v03",
        "elevation_m",
        "local_relief_m",
        "slope_deg",
        "coastline_length_3km_m",
        "coastline_complexity_3km",
        "y_observed_v01",
        "observed_recognition_v04",
        "recognition_access_v04",
        "recognition_recreation_v04",
        "recognition_natural_v04",
        "recognition_trails_access_v04",
        "recognition_natural_protected_v04",
        "recognition_total_count_3km_v04",
        "recognition_category_coverage_v04",
        "recognition_cell_confidence_v04",
        "land_area_share",
        "water_area_share",
        "rel_neighbor_count_3km_v01",
        "rel_neighbor_count_10km_v01",
        "rel_neighbor_count_25km_v01",
    ]

    continuous = [
        c for c in continuous_candidates
        if c in work.columns
        and pd.api.types.is_numeric_dtype(work[c])
        and work[c].nunique(dropna=True) > 1
    ]

    categorical_candidates = [
        "spatial_fold_v01",
        "buffered_spatial_fold_v01",
        "operator_spatial_regime_v01",
        "gain_regime_v01",
        "observed_recognition_tier_v04",
    ]

    categorical = [
        c for c in categorical_candidates
        if c in work.columns and work[c].nunique(dropna=True) > 1
    ]

    rows = []

    rel_mask = work[REGIME] == REL
    pt_mask = work[REGIME] == PT

    for col in continuous:
        rel_vals = work.loc[rel_mask, col].dropna().astype(float)
        pt_vals = work.loc[pt_mask, col].dropna().astype(float)

        if len(rel_vals) == 0 or len(pt_vals) == 0:
            continue

        u, p = mannwhitneyu(rel_vals, pt_vals, alternative="two-sided")

        groups = [
            group[col].dropna().astype(float)
            for _, group in work.groupby(REGIME)
            if len(group[col].dropna()) > 0
        ]

        h, hp = kruskal(*groups) if len(groups) >= 2 else (np.nan, np.nan)

        rows.append({
            "variable": col,
            "n_rel": len(rel_vals),
            "n_pt": len(pt_vals),
            "mean_rel": rel_vals.mean(),
            "mean_pt": pt_vals.mean(),
            "median_rel": rel_vals.median(),
            "median_pt": pt_vals.median(),
            "mean_diff_rel_minus_pt": rel_vals.mean() - pt_vals.mean(),
            "median_diff_rel_minus_pt": rel_vals.median() - pt_vals.median(),
            "rank_biserial_rel_vs_pt": rank_biserial(u, len(rel_vals), len(pt_vals)),
            "mannwhitney_p": p,
            "kruskal_h": h,
            "kruskal_p": hp,
        })

    cont_df = pd.DataFrame(rows)

    if len(cont_df):
        cont_df["abs_rank_biserial"] = cont_df["rank_biserial_rel_vs_pt"].abs()
        cont_df = cont_df.sort_values(["abs_rank_biserial", "mannwhitney_p"], ascending=[False, True])

    cat_rows = []

    for col in categorical:
        table = pd.crosstab(work[REGIME], work[col])

        if table.shape[0] < 2 or table.shape[1] < 2:
            continue

        chi2, p, dof, _ = chi2_contingency(table)

        cat_rows.append({
            "variable": col,
            "n": int(table.to_numpy().sum()),
            "n_regime_levels": table.shape[0],
            "n_variable_levels": table.shape[1],
            "chi2": chi2,
            "dof": dof,
            "p_value": p,
            "cramers_v": cramers_v(table.to_numpy()),
        })

    cat_df = pd.DataFrame(cat_rows)

    if len(cat_df):
        cat_df = cat_df.sort_values("cramers_v", ascending=False)

    profile_vars = [
        c for c in continuous
        if c not in [
            "mean_gain_across_buffers_v01",
            "median_gain_across_buffers_v01",
            "share_buffers_relational_better_v01",
            "max_tested_positive_buffer_m_v01",
            "gain_sign_changes_v01",
        ]
    ]

    profile = (
        work.groupby(REGIME)[profile_vars]
        .mean()
        .reset_index()
    )

    qa = []
    qa.append(f"{SCRIPT} QA")
    qa.append("=" * 80)
    qa.append(f"Input: {INPUT_FP}")
    qa.append("")
    qa.append(f"Rows analyzed: {len(work):,}")
    qa.append("")
    qa.append("Persistence regime counts:")
    qa.append(work[REGIME].value_counts().to_string())
    qa.append("")
    qa.append(f"Continuous variables tested: {len(continuous)}")
    qa.append(f"Categorical variables tested: {len(categorical)}")
    qa.append("")
    qa.append("Top continuous differences:")
    qa.append(cont_df.head(40).to_string(index=False) if len(cont_df) else "None")
    qa.append("")
    qa.append("Categorical associations:")
    qa.append(cat_df.to_string(index=False) if len(cat_df) else "None")
    qa.append("")
    qa.append("Mean profile by persistence regime:")
    qa.append(profile.to_string(index=False))
    qa.append("")
    qa.append("Interpretation guardrail:")
    qa.append(
        "This script diagnoses descriptive persistence regimes. It does not prove a final operator-selection theorem "
        "or validate a prospective persistence classifier."
    )

    log("Writing outputs")
    cont_df.to_csv(OUT_CONTINUOUS, index=False)
    cat_df.to_csv(OUT_CATEGORICAL, index=False)
    profile.to_csv(OUT_PROFILE, index=False)
    OUT_QA.write_text("\n".join(qa), encoding="utf-8")

    log(f"Created {OUT_CONTINUOUS}")
    log(f"Created {OUT_CATEGORICAL}")
    log(f"Created {OUT_PROFILE}")
    log(f"Created {OUT_QA}")
    log("Done")


if __name__ == "__main__":
    main()