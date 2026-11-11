from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from scipy.stats import (
    kruskal,
    mannwhitneyu,
    chi2_contingency,
)


SCRIPT = "213_oregon_srs_operator_regime_diagnostic_audit_v01"

BASE = Path("data/processed")

INPUT_FP = (
    BASE
    / "oregon_srs_operator_regime_predictability_v01.gpkg"
)

OUT_GPKG = (
    BASE
    / "oregon_srs_operator_regime_diagnostic_audit_v01.gpkg"
)

OUT_CONTINUOUS_CSV = (
    BASE
    / "oregon_srs_operator_regime_diagnostic_continuous_v01.csv"
)

OUT_CATEGORICAL_CSV = (
    BASE
    / "oregon_srs_operator_regime_diagnostic_categorical_v01.csv"
)

OUT_CORRELATION_CSV = (
    BASE
    / "oregon_srs_operator_regime_gain_correlations_v01.csv"
)

OUT_QA = (
    BASE
    / "oregon_srs_operator_regime_diagnostic_audit_qa_v01.txt"
)


REGIME = "operator_spatial_regime_v01"
GAIN = "relational_gain_v01"

REL_CLASS = "coherent_relational"
PT_CLASS = "coherent_pointwise"
MIXED_CLASS = "mixed_operator_zone"


def log(msg):
    print(f"[{SCRIPT}] {msg}")


def rank_biserial_from_u(u_stat, n1, n2):
    """
    Rank-biserial correlation based on Mann-Whitney U.

    Positive values indicate larger values in group 1
    relative to group 2.
    """

    if n1 == 0 or n2 == 0:
        return np.nan

    return (
        2.0 * u_stat / (n1 * n2)
        - 1.0
    )


def cramers_v(table):
    """
    Bias-corrected Cramer's V.
    """

    if table.size == 0:
        return np.nan

    chi2, _, _, _ = chi2_contingency(table)

    n = table.sum()

    if n <= 1:
        return np.nan

    phi2 = chi2 / n

    r, k = table.shape

    phi2corr = max(
        0,
        phi2
        - ((k - 1) * (r - 1))
        / (n - 1),
    )

    rcorr = (
        r
        - ((r - 1) ** 2)
        / (n - 1)
    )

    kcorr = (
        k
        - ((k - 1) ** 2)
        / (n - 1)
    )

    denom = min(
        kcorr - 1,
        rcorr - 1,
    )

    if denom <= 0:
        return np.nan

    return np.sqrt(
        phi2corr / denom
    )


def main():

    log("Reading operator regime predictability layer")

    gdf = gpd.read_file(INPUT_FP)

    required = [
        "cell_id",
        REGIME,
        GAIN,
    ]

    for col in required:
        if col not in gdf.columns:
            raise ValueError(
                f"Missing required column: {col}"
            )


    # -----------------------------------------------------
    # Analysis population
    # -----------------------------------------------------

    mask = (
        gdf[REGIME].isin(
            [
                REL_CLASS,
                PT_CLASS,
                MIXED_CLASS,
            ]
        )
        & gdf[GAIN].notna()
    )

    work = gdf.loc[mask].copy()

    log(
        f"Rows available for diagnostic audit: "
        f"{len(work):,}"
    )


    # -----------------------------------------------------
    # Recognition component variables
    # -----------------------------------------------------

    recognition_component_candidates = [
        "recognition_access_v04",
        "recognition_recreation_v04",
        "recognition_natural_v04",
        "recognition_trails_access_v04",
        "recognition_natural_protected_v04",

        "trail_path_count_3km",
        "trail_path_score_v04",

        "beach_count_3km",
        "beach_score_v04",

        "park_recreation_count_3km",
        "park_recreation_score_v04",

        "protected_area_count_3km",
        "protected_area_score_v04",

        "parking_count_3km",
        "parking_score_v04",

        "trailhead_count_3km",
        "trailhead_score_v04",

        "visitor_information_count_3km",
        "visitor_information_score_v04",

        "viewpoint_count_3km",
        "viewpoint_score_v04",

        "named_natural_feature_count_3km",
        "named_natural_feature_score_v04",

        "tourism_recreation_count_3km",
        "tourism_recreation_score_v04",

        "recognition_total_count_3km_v04",
        "recognition_category_coverage_v04",
    ]


    # -----------------------------------------------------
    # Recognition confidence / validity diagnostics
    # -----------------------------------------------------

    confidence_candidates = [
        "recognition_dataset_confidence_v04",
        "recognition_cell_confidence_v04",
        "land_area_share",
        "water_area_share",
        "centroid_on_land",
        "is_valid_land_candidate",
        "recognition_valid_land_flag_v04",
    ]


    # -----------------------------------------------------
    # Geographic / structural diagnostics
    # -----------------------------------------------------

    structural_candidates = [
        "rel_neighbor_count_3km_v01",
        "rel_neighbor_count_10km_v01",
        "rel_neighbor_count_25km_v01",

        "operator_neighbor_count_v01",
        "operator_same_sign_share_v01",
        "operator_neighbor_mean_gain_v01",

        "area_sq_km",
        "centroid_x",
        "centroid_y",

        "phi_physical_v01",
        "physical_exceptionality_v03",
        "physical_exceptionality_raw_v03",

        "coast_proximity_v03",
        "coast_complexity_v03",
        "terrain_drama_v03",
        "scenic_coast_v03",
        "relief_norm_v03",
        "slope_norm_v03",

        "y_observed_v01",
        "observed_recognition_v04",
    ]


    continuous_candidates = (
        recognition_component_candidates
        + confidence_candidates
        + structural_candidates
    )

    continuous_features = []

    for col in continuous_candidates:

        if col not in work.columns:
            continue

        if not pd.api.types.is_numeric_dtype(
            work[col]
        ):
            continue

        if work[col].nunique(
            dropna=True
        ) <= 1:
            continue

        continuous_features.append(col)

    continuous_features = list(
        dict.fromkeys(
            continuous_features
        )
    )


    log(
        f"Continuous diagnostic variables: "
        f"{len(continuous_features)}"
    )


    # -----------------------------------------------------
    # Categorical diagnostic candidates
    # -----------------------------------------------------

    categorical_candidates = [
        "observed_recognition_tier_v04",
        "spatial_fold_v01",
        "operator_cluster_id_v01",
    ]

    categorical_features = [
        c
        for c in categorical_candidates
        if c in work.columns
        and work[c].nunique(
            dropna=True
        ) > 1
    ]


    log(
        f"Categorical diagnostic variables: "
        f"{len(categorical_features)}"
    )


    # -----------------------------------------------------
    # Continuous variable regime comparisons
    # -----------------------------------------------------

    continuous_rows = []

    for col in continuous_features:

        rel = (
            work.loc[
                work[REGIME] == REL_CLASS,
                col,
            ]
            .dropna()
            .astype(float)
        )

        pt = (
            work.loc[
                work[REGIME] == PT_CLASS,
                col,
            ]
            .dropna()
            .astype(float)
        )

        mixed = (
            work.loc[
                work[REGIME] == MIXED_CLASS,
                col,
            ]
            .dropna()
            .astype(float)
        )


        if len(rel) == 0 or len(pt) == 0:
            continue


        u_stat, u_p = mannwhitneyu(
            rel,
            pt,
            alternative="two-sided",
        )

        rb = rank_biserial_from_u(
            u_stat,
            len(rel),
            len(pt),
        )


        valid_groups = [
            x
            for x in [
                rel,
                pt,
                mixed,
            ]
            if len(x) > 0
        ]

        if len(valid_groups) >= 2:

            h_stat, h_p = kruskal(
                *valid_groups
            )

        else:

            h_stat = np.nan
            h_p = np.nan


        continuous_rows.append(
            {
                "variable": col,

                "n_relational": len(rel),
                "n_pointwise": len(pt),
                "n_mixed": len(mixed),

                "mean_relational": rel.mean(),
                "mean_pointwise": pt.mean(),
                "mean_mixed": (
                    mixed.mean()
                    if len(mixed)
                    else np.nan
                ),

                "median_relational": rel.median(),
                "median_pointwise": pt.median(),
                "median_mixed": (
                    mixed.median()
                    if len(mixed)
                    else np.nan
                ),

                "mean_difference_rel_minus_pt":
                    rel.mean() - pt.mean(),

                "median_difference_rel_minus_pt":
                    rel.median() - pt.median(),

                "rank_biserial_rel_vs_pt": rb,

                "mannwhitney_u": u_stat,
                "mannwhitney_p": u_p,

                "kruskal_h": h_stat,
                "kruskal_p": h_p,
            }
        )


    continuous_df = pd.DataFrame(
        continuous_rows
    )

    if len(continuous_df):

        continuous_df[
            "abs_rank_biserial"
        ] = (
            continuous_df[
                "rank_biserial_rel_vs_pt"
            ].abs()
        )

        continuous_df = (
            continuous_df
            .sort_values(
                [
                    "abs_rank_biserial",
                    "mannwhitney_p",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
        )


    # -----------------------------------------------------
    # Correlation with continuous relational gain
    # -----------------------------------------------------

    correlation_rows = []

    for col in continuous_features:

        tmp = (
            work[
                [
                    col,
                    GAIN,
                ]
            ]
            .dropna()
        )

        if len(tmp) < 10:
            continue

        pearson = (
            tmp[col]
            .corr(
                tmp[GAIN],
                method="pearson",
            )
        )

        spearman = (
            tmp[col]
            .corr(
                tmp[GAIN],
                method="spearman",
            )
        )

        correlation_rows.append(
            {
                "variable": col,
                "n": len(tmp),
                "pearson_gain_correlation": pearson,
                "spearman_gain_correlation": spearman,
                "abs_spearman_gain_correlation":
                    abs(spearman),
            }
        )


    correlation_df = pd.DataFrame(
        correlation_rows
    )

    if len(correlation_df):

        correlation_df = (
            correlation_df
            .sort_values(
                "abs_spearman_gain_correlation",
                ascending=False,
            )
        )


    # -----------------------------------------------------
    # Categorical association tests
    # -----------------------------------------------------

    categorical_rows = []

    for col in categorical_features:

        tmp = work[
            [
                REGIME,
                col,
            ]
        ].dropna()

        table = pd.crosstab(
            tmp[REGIME],
            tmp[col],
        )

        if (
            table.shape[0] < 2
            or table.shape[1] < 2
        ):
            continue

        chi2, p, dof, _ = (
            chi2_contingency(table)
        )

        v = cramers_v(
            table.to_numpy()
        )

        categorical_rows.append(
            {
                "variable": col,
                "n": int(
                    table.to_numpy().sum()
                ),
                "n_regime_levels":
                    table.shape[0],
                "n_variable_levels":
                    table.shape[1],
                "chi2": chi2,
                "dof": dof,
                "p_value": p,
                "cramers_v": v,
            }
        )


    categorical_df = pd.DataFrame(
        categorical_rows
    )

    if len(categorical_df):

        categorical_df = (
            categorical_df
            .sort_values(
                "cramers_v",
                ascending=False,
            )
        )


    # -----------------------------------------------------
    # Regime composition summary
    # -----------------------------------------------------

    regime_summary = (
        work.groupby(REGIME)
        .agg(
            n=(GAIN, "size"),
            mean_gain=(GAIN, "mean"),
            median_gain=(GAIN, "median"),
            mean_abs_gain=(
                GAIN,
                lambda s: s.abs().mean(),
            ),
        )
        .reset_index()
    )


    # -----------------------------------------------------
    # Recognition-channel profile
    # -----------------------------------------------------

    profile_cols = [
        c
        for c in [
            "recognition_access_v04",
            "recognition_recreation_v04",
            "recognition_natural_v04",
            "recognition_trails_access_v04",
            "recognition_natural_protected_v04",
            "recognition_total_count_3km_v04",
            "recognition_category_coverage_v04",
            "recognition_dataset_confidence_v04",
            "recognition_cell_confidence_v04",
            "land_area_share",
            "water_area_share",
        ]
        if c in work.columns
    ]


    if profile_cols:

        recognition_profile = (
            work.groupby(REGIME)[
                profile_cols
            ]
            .mean()
            .reset_index()
        )

    else:

        recognition_profile = pd.DataFrame()


    # -----------------------------------------------------
    # Add compact diagnostic summaries to spatial output
    # -----------------------------------------------------

    out = gdf.copy()

    out[
        "diagnostic_regime_analyzed_v01"
    ] = out[REGIME].isin(
        [
            REL_CLASS,
            PT_CLASS,
            MIXED_CLASS,
        ]
    )


    # -----------------------------------------------------
    # QA report
    # -----------------------------------------------------

    qa = []

    qa.append(f"{SCRIPT} QA")
    qa.append("=" * 80)

    qa.append(f"Input: {INPUT_FP}")
    qa.append(f"Output: {OUT_GPKG}")

    qa.append("")
    qa.append(
        f"Rows analyzed: {len(work):,}"
    )

    qa.append(
        f"Continuous variables tested: "
        f"{len(continuous_features)}"
    )

    qa.append(
        f"Categorical variables tested: "
        f"{len(categorical_features)}"
    )


    qa.append("")
    qa.append("Regime summary:")

    qa.append(
        regime_summary.to_string(
            index=False
        )
    )


    qa.append("")
    qa.append(
        "Top continuous regime differences "
        "by absolute rank-biserial effect:"
    )

    if len(continuous_df):

        qa.append(
            continuous_df.head(40).to_string(
                index=False
            )
        )

    else:

        qa.append(
            "No continuous comparisons available."
        )


    qa.append("")
    qa.append(
        "Top correlations with continuous "
        "relational gain:"
    )

    if len(correlation_df):

        qa.append(
            correlation_df.head(40).to_string(
                index=False
            )
        )

    else:

        qa.append(
            "No correlation results available."
        )


    qa.append("")
    qa.append(
        "Categorical regime associations:"
    )

    if len(categorical_df):

        qa.append(
            categorical_df.to_string(
                index=False
            )
        )

    else:

        qa.append(
            "No categorical associations available."
        )


    qa.append("")
    qa.append(
        "Mean recognition-channel profile by regime:"
    )

    if len(recognition_profile):

        qa.append(
            recognition_profile.to_string(
                index=False
            )
        )

    else:

        qa.append(
            "No recognition profile available."
        )


    qa.append("")
    qa.append("Interpretation guardrail:")

    qa.append(
        "This script is diagnostic only. "
        "Downstream recognition outcomes, residuals, "
        "gain values, confidence variables, and other "
        "post-outcome diagnostics must not be inserted "
        "into a prospective gating operator without "
        "separate theoretical and validation review."
    )


    # -----------------------------------------------------
    # Write outputs
    # -----------------------------------------------------

    log("Writing outputs")

    out.to_file(
        OUT_GPKG,
        driver="GPKG",
    )

    continuous_df.to_csv(
        OUT_CONTINUOUS_CSV,
        index=False,
    )

    categorical_df.to_csv(
        OUT_CATEGORICAL_CSV,
        index=False,
    )

    correlation_df.to_csv(
        OUT_CORRELATION_CSV,
        index=False,
    )

    OUT_QA.write_text(
        "\n".join(qa),
        encoding="utf-8",
    )


    log(f"Created {OUT_GPKG}")
    log(f"Created {OUT_CONTINUOUS_CSV}")
    log(f"Created {OUT_CATEGORICAL_CSV}")
    log(f"Created {OUT_CORRELATION_CSV}")
    log(f"Created {OUT_QA}")

    log("Done")


if __name__ == "__main__":
    main()