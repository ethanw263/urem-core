from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu


SCRIPT = "219_oregon_srs_fold_controlled_persistence_diagnostics_v01"

BASE = Path("data/processed")

INPUT = (
    BASE
    / "oregon_srs_relational_persistence_common_population_v01.gpkg"
)

OUT_FOLD = (
    BASE
    / "oregon_srs_fold_controlled_persistence_diagnostics_v01.csv"
)

OUT_SUMMARY = (
    BASE
    / "oregon_srs_fold_controlled_persistence_summary_v01.csv"
)

OUT_QA = (
    BASE
    / "oregon_srs_fold_controlled_persistence_diagnostics_qa_v01.txt"
)


REGIME = "persistence_regime_v01"

REL = "persistent_relational"
PT = "persistent_pointwise"

FOLD_CANDIDATES = [
    "buffered_spatial_fold_v01",
    "spatial_fold_v01",
]


FEATURE_CANDIDATES = [
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
    "rel_neighbor_count_3km_v01",
    "rel_neighbor_count_10km_v01",
    "rel_neighbor_count_25km_v01",
]


def log(msg):
    print(f"[{SCRIPT}] {msg}")


def rank_biserial(u, n1, n2):

    if n1 == 0 or n2 == 0:
        return np.nan

    return (
        2.0 * u / (n1 * n2)
    ) - 1.0


def sign_value(x):

    if pd.isna(x):
        return np.nan

    if x > 0:
        return 1

    if x < 0:
        return -1

    return 0


def main():

    log("Reading persistence common-population layer")

    gdf = gpd.read_file(INPUT)


    # --------------------------------------------------
    # Select fold variable
    # --------------------------------------------------

    fold_col = None

    for candidate in FOLD_CANDIDATES:

        if candidate in gdf.columns:

            fold_col = candidate
            break


    if fold_col is None:

        raise ValueError(
            "No valid spatial fold column found."
        )


    log(
        f"Using fold column: {fold_col}"
    )


    # --------------------------------------------------
    # Restrict to primary persistent regimes
    # --------------------------------------------------

    work = gdf[
        gdf[REGIME].isin(
            [
                REL,
                PT,
            ]
        )
    ].copy()


    log(
        f"Primary persistence rows: {len(work):,}"
    )


    features = [
        c
        for c in FEATURE_CANDIDATES
        if c in work.columns
        and pd.api.types.is_numeric_dtype(
            work[c]
        )
        and work[c].nunique(
            dropna=True
        ) > 1
    ]


    log(
        f"Features available: {len(features)}"
    )


    # --------------------------------------------------
    # Global effects
    # --------------------------------------------------

    global_rows = []


    for feature in features:

        rel = (
            work.loc[
                work[REGIME] == REL,
                feature,
            ]
            .dropna()
            .astype(float)
        )

        pt = (
            work.loc[
                work[REGIME] == PT,
                feature,
            ]
            .dropna()
            .astype(float)
        )


        if len(rel) == 0 or len(pt) == 0:
            continue


        u, p = mannwhitneyu(
            rel,
            pt,
            alternative="two-sided",
        )


        global_rows.append(
            {
                "feature":
                    feature,

                "global_n_rel":
                    len(rel),

                "global_n_pt":
                    len(pt),

                "global_mean_diff":
                    rel.mean()
                    - pt.mean(),

                "global_median_diff":
                    rel.median()
                    - pt.median(),

                "global_rank_biserial":
                    rank_biserial(
                        u,
                        len(rel),
                        len(pt),
                    ),

                "global_p":
                    p,
            }
        )


    global_df = pd.DataFrame(
        global_rows
    )


    # --------------------------------------------------
    # Within-fold effects
    # --------------------------------------------------

    fold_rows = []


    for fold in sorted(
        work[fold_col]
        .dropna()
        .unique()
    ):

        fold_data = work[
            work[fold_col] == fold
        ]


        for feature in features:

            rel = (
                fold_data.loc[
                    fold_data[REGIME] == REL,
                    feature,
                ]
                .dropna()
                .astype(float)
            )

            pt = (
                fold_data.loc[
                    fold_data[REGIME] == PT,
                    feature,
                ]
                .dropna()
                .astype(float)
            )


            if len(rel) < 20 or len(pt) < 20:
                continue


            u, p = mannwhitneyu(
                rel,
                pt,
                alternative="two-sided",
            )


            fold_rows.append(
                {
                    "fold":
                        int(fold),

                    "feature":
                        feature,

                    "n_rel":
                        len(rel),

                    "n_pt":
                        len(pt),

                    "mean_rel":
                        rel.mean(),

                    "mean_pt":
                        pt.mean(),

                    "median_rel":
                        rel.median(),

                    "median_pt":
                        pt.median(),

                    "mean_diff_rel_minus_pt":
                        rel.mean()
                        - pt.mean(),

                    "median_diff_rel_minus_pt":
                        rel.median()
                        - pt.median(),

                    "rank_biserial":
                        rank_biserial(
                            u,
                            len(rel),
                            len(pt),
                        ),

                    "mannwhitney_p":
                        p,
                }
            )


    fold_df = pd.DataFrame(
        fold_rows
    )


    # --------------------------------------------------
    # Controlled consistency summary
    # --------------------------------------------------

    summary_rows = []


    for _, global_row in global_df.iterrows():

        feature = global_row["feature"]

        sub = fold_df[
            fold_df["feature"] == feature
        ].copy()


        if len(sub) == 0:
            continue


        global_sign = sign_value(
            global_row[
                "global_rank_biserial"
            ]
        )


        sub["effect_sign"] = (
            sub[
                "rank_biserial"
            ]
            .apply(sign_value)
        )


        if global_sign == 0:

            directional_consistency = np.nan

        else:

            directional_consistency = (
                sub["effect_sign"]
                == global_sign
            ).mean()


        significant_same_direction = (

            (
                sub["effect_sign"]
                == global_sign
            )

            &

            (
                sub["mannwhitney_p"]
                < 0.05
            )

        ).mean()


        weighted_effect = np.average(
            sub["rank_biserial"],
            weights=(
                sub["n_rel"]
                + sub["n_pt"]
            ),
        )


        summary_rows.append(
            {
                "feature":
                    feature,

                "global_rank_biserial":
                    global_row[
                        "global_rank_biserial"
                    ],

                "global_mean_diff":
                    global_row[
                        "global_mean_diff"
                    ],

                "global_median_diff":
                    global_row[
                        "global_median_diff"
                    ],

                "folds_tested":
                    len(sub),

                "directional_consistency":
                    directional_consistency,

                "significant_same_direction_share":
                    significant_same_direction,

                "weighted_within_fold_effect":
                    weighted_effect,

                "median_within_fold_effect":
                    sub[
                        "rank_biserial"
                    ].median(),

                "min_within_fold_effect":
                    sub[
                        "rank_biserial"
                    ].min(),

                "max_within_fold_effect":
                    sub[
                        "rank_biserial"
                    ].max(),

                "effect_sign_reversals":
                    int(
                        (
                            sub[
                                "effect_sign"
                            ]
                            != global_sign
                        ).sum()
                    ),
            }
        )


    summary_df = pd.DataFrame(
        summary_rows
    )


    summary_df[
        "abs_weighted_within_fold_effect"
    ] = summary_df[
        "weighted_within_fold_effect"
    ].abs()


    summary_df = summary_df.sort_values(
        [
            "directional_consistency",
            "abs_weighted_within_fold_effect",
        ],
        ascending=[
            False,
            False,
        ],
    )


    # --------------------------------------------------
    # QA
    # --------------------------------------------------

    qa = []

    qa.append(f"{SCRIPT} QA")
    qa.append("=" * 80)

    qa.append(
        f"Input: {INPUT}"
    )

    qa.append(
        f"Fold variable: {fold_col}"
    )

    qa.append("")

    qa.append(
        f"Primary persistence rows: "
        f"{len(work):,}"
    )

    qa.append(
        f"Features tested: "
        f"{len(features)}"
    )

    qa.append("")

    qa.append(
        "Primary regime counts:"
    )

    qa.append(
        work[
            REGIME
        ]
        .value_counts()
        .to_string()
    )

    qa.append("")

    qa.append(
        "Top fold-controlled effects:"
    )

    qa.append(
        summary_df.head(
            30
        ).to_string(
            index=False
        )
    )

    qa.append("")

    qa.append(
        "Interpretation guide:"
    )

    qa.append(
        "High directional consistency means that the "
        "global relational-versus-pointwise difference "
        "usually retains the same sign within geographic folds."
    )

    qa.append(
        "Large weighted within-fold effect indicates that "
        "the distinction remains meaningful after coarse "
        "geographic control."
    )

    qa.append(
        "Features with large global effects but weak or "
        "reversing within-fold effects should be treated "
        "as potentially geographically confounded."
    )

    qa.append("")

    qa.append(
        "Guardrail:"
    )

    qa.append(
        "This is a controlled descriptive diagnostic. "
        "It does not establish causality or validate a "
        "prospective operator-selection rule."
    )


    # --------------------------------------------------
    # Write outputs
    # --------------------------------------------------

    log("Writing outputs")


    fold_df.to_csv(
        OUT_FOLD,
        index=False,
    )


    summary_df.to_csv(
        OUT_SUMMARY,
        index=False,
    )


    OUT_QA.write_text(
        "\n".join(qa),
        encoding="utf-8",
    )


    log(f"Created {OUT_FOLD}")
    log(f"Created {OUT_SUMMARY}")
    log(f"Created {OUT_QA}")

    log("Done")


if __name__ == "__main__":
    main()