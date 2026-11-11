from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


SCRIPT = "216_oregon_srs_relational_gain_decay_profile_v01"

BASE = Path("data/processed")

INPUT_CONTEXT = (
    BASE
    / "oregon_srs_relational_context_v01.gpkg"
)

INPUT_FOLDS = (
    BASE
    / "oregon_srs_buffered_spatial_validation_design_v01.gpkg"
)

INPUT_215_GPKG = (
    BASE
    / "oregon_srs_buffered_operator_validation_v01.gpkg"
)

INPUT_215_METRICS = (
    BASE
    / "oregon_srs_buffered_operator_validation_metrics_v01.csv"
)

INPUT_215_FOLD_METRICS = (
    BASE
    / "oregon_srs_buffered_operator_validation_fold_metrics_v01.csv"
)

OUT_GPKG = (
    BASE
    / "oregon_srs_relational_gain_decay_profile_v01.gpkg"
)

OUT_OVERALL_CSV = (
    BASE
    / "oregon_srs_relational_gain_decay_overall_v01.csv"
)

OUT_FOLD_CSV = (
    BASE
    / "oregon_srs_relational_gain_decay_folds_v01.csv"
)

OUT_CELL_CSV = (
    BASE
    / "oregon_srs_relational_gain_decay_cells_v01.csv"
)

OUT_QA = (
    BASE
    / "oregon_srs_relational_gain_decay_profile_qa_v01.txt"
)


TARGET = "y_observed_v01"
FOLD_COL = "buffered_spatial_fold_v01"

EXISTING_BUFFERS_M = [
    0,
    3000,
    10000,
    25000,
]

NEW_BUFFERS_M = [
    1000,
    5000,
    15000,
    20000,
    35000,
    50000,
]

ALL_BUFFERS_M = sorted(
    EXISTING_BUFFERS_M
    + NEW_BUFFERS_M
)

RANDOM_SEED = 42


POINTWISE_FEATURES = [
    "phi_physical_v01",
    "physical_exceptionality_raw_v03",
    "base_exceptionality_v02_norm",
    "terrain_drama_v03",
    "scenic_coast_v03",
    "relief_norm_v03",
    "slope_norm_v03",
    "coast_proximity_v03",
    "beach_proximity_v03",
    "cliff_proximity_v03",
    "coast_complexity_v03",
    "flat_coastal_edge_penalty_v03",
    "complex_flat_shoreline_penalty_v03",
    "elevation_m",
    "local_relief_m",
    "slope_deg",
    "coastline_length_3km_m",
    "coastline_complexity_3km",
]


def log(msg):
    print(f"[{SCRIPT}] {msg}")


def suffix_from_buffer(buffer_m):

    if buffer_m == 0:
        return "0km"

    return f"{int(buffer_m / 1000)}km"


def main():

    log("Reading relational context layer")

    gdf = gpd.read_file(
        INPUT_CONTEXT
    )

    log("Reading buffered fold design")

    folds = gpd.read_file(
        INPUT_FOLDS
    )[
        [
            "cell_id",
            FOLD_COL,
        ]
    ]

    gdf = gdf.drop(
        columns=[FOLD_COL],
        errors="ignore",
    )

    gdf = gdf.merge(
        folds,
        on="cell_id",
        how="left",
    )


    # -----------------------------------------------------
    # Discover relational features exactly as Script 215
    # -----------------------------------------------------

    relational_prefixes = [
        "phi_physical_v01_",
        "physical_exceptionality_raw_v03_",
        "terrain_drama_v03_",
        "scenic_coast_v03_",
        "relief_norm_v03_",
        "slope_norm_v03_",
        "coast_proximity_v03_",
        "coast_complexity_v03_",
        "rel_neighbor_count_",
    ]

    relational_features = [
        c
        for c in gdf.columns
        if any(
            c.startswith(prefix)
            for prefix in relational_prefixes
        )
        and c.endswith("_v01")
    ]

    relational_features = sorted(
        set(relational_features)
    )

    all_features = (
        POINTWISE_FEATURES
        + relational_features
    )

    log(
        f"Pointwise features: "
        f"{len(POINTWISE_FEATURES)}"
    )

    log(
        f"Relational features: "
        f"{len(relational_features)}"
    )

    log(
        f"Total relational-model features: "
        f"{len(all_features)}"
    )


    # -----------------------------------------------------
    # Common modeling population
    # -----------------------------------------------------

    model_cols = (
        [
            "cell_id",
            "centroid_x",
            "centroid_y",
            TARGET,
            FOLD_COL,
        ]
        + all_features
    )

    model = (
        gdf[model_cols]
        .dropna()
        .copy()
    )

    log(
        f"Comparable modeling rows: "
        f"{len(model):,}"
    )


    # -----------------------------------------------------
    # Load existing Script 215 predictions
    # -----------------------------------------------------

    log(
        "Reading existing Script 215 predictions"
    )

    existing = gpd.read_file(
        INPUT_215_GPKG
    )

    existing_cell_frames = []

    for buffer_m in EXISTING_BUFFERS_M:

        suffix = suffix_from_buffer(
            buffer_m
        )

        gain_col = (
            f"gain_rel_{suffix}_v01"
        )

        pt_col = (
            f"pred_pt_{suffix}_v01"
        )

        rel_col = (
            f"pred_rel_{suffix}_v01"
        )

        required = [
            "cell_id",
            gain_col,
            pt_col,
            rel_col,
        ]

        missing = [
            c
            for c in required
            if c not in existing.columns
        ]

        if missing:
            raise ValueError(
                f"Missing Script 215 columns "
                f"for {buffer_m} m: {missing}"
            )

        temp = existing[
            required
        ].copy()

        temp["buffer_m"] = buffer_m

        temp = temp.rename(
            columns={
                gain_col:
                    "relational_gain",

                pt_col:
                    "pred_pointwise",

                rel_col:
                    "pred_relational",
            }
        )

        existing_cell_frames.append(
            temp
        )


    existing_cells_long = pd.concat(
        existing_cell_frames,
        ignore_index=True,
    )


    # -----------------------------------------------------
    # Fit only NEW buffer distances
    # -----------------------------------------------------

    new_prediction_frames = []
    new_fold_rows = []


    for buffer_m in NEW_BUFFERS_M:

        log(
            f"Starting new buffer distance: "
            f"{buffer_m:,} m"
        )

        for fold in sorted(
            model[FOLD_COL]
            .dropna()
            .unique()
        ):

            test_mask = (
                model[FOLD_COL]
                .to_numpy()
                == fold
            )

            test = (
                model.loc[
                    test_mask
                ]
                .copy()
            )

            train_candidates = (
                model.loc[
                    ~test_mask
                ]
                .copy()
            )


            # ---------------------------------------------
            # Buffer exclusion
            # ---------------------------------------------

            test_xy = test[
                [
                    "centroid_x",
                    "centroid_y",
                ]
            ].to_numpy()

            train_xy = train_candidates[
                [
                    "centroid_x",
                    "centroid_y",
                ]
            ].to_numpy()

            tree = cKDTree(
                test_xy
            )

            nearest_distance, _ = (
                tree.query(
                    train_xy,
                    k=1,
                )
            )

            keep_train = (
                nearest_distance
                > buffer_m
            )

            train = (
                train_candidates.loc[
                    keep_train
                ]
                .copy()
            )


            log(
                f"Buffer {buffer_m:,} m, "
                f"fold {int(fold)}: "
                f"train={len(train):,}, "
                f"test={len(test):,}"
            )


            # ---------------------------------------------
            # Matrices
            # ---------------------------------------------

            X_train_pt = train[
                POINTWISE_FEATURES
            ].to_numpy()

            X_test_pt = test[
                POINTWISE_FEATURES
            ].to_numpy()

            X_train_rel = train[
                all_features
            ].to_numpy()

            X_test_rel = test[
                all_features
            ].to_numpy()

            y_train = train[
                TARGET
            ].to_numpy()

            y_test = test[
                TARGET
            ].to_numpy()


            # ---------------------------------------------
            # Pointwise model
            # ---------------------------------------------

            pt_model = LinearRegression()

            pt_model.fit(
                X_train_pt,
                y_train,
            )

            pred_pt = pt_model.predict(
                X_test_pt
            )


            # ---------------------------------------------
            # Relational model
            # ---------------------------------------------

            rel_model = RandomForestRegressor(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=5,
                random_state=(
                    RANDOM_SEED
                    + int(fold)
                    + int(buffer_m)
                ),
                n_jobs=-1,
            )

            rel_model.fit(
                X_train_rel,
                y_train,
            )

            pred_rel = rel_model.predict(
                X_test_rel
            )


            # ---------------------------------------------
            # Cell-level gain
            # ---------------------------------------------

            abs_pt = np.abs(
                y_test
                - pred_pt
            )

            abs_rel = np.abs(
                y_test
                - pred_rel
            )

            gain = (
                abs_pt
                - abs_rel
            )


            new_prediction_frames.append(
                pd.DataFrame(
                    {
                        "cell_id":
                            test[
                                "cell_id"
                            ].to_numpy(),

                        "buffer_m":
                            int(buffer_m),

                        "pred_pointwise":
                            pred_pt,

                        "pred_relational":
                            pred_rel,

                        "relational_gain":
                            gain,
                    }
                )
            )


            new_fold_rows.append(
                {
                    "buffer_m":
                        int(buffer_m),

                    "fold":
                        int(fold),

                    "n_train":
                        len(train),

                    "n_test":
                        len(test),

                    "pt_mae":
                        mean_absolute_error(
                            y_test,
                            pred_pt,
                        ),

                    "rel_mae":
                        mean_absolute_error(
                            y_test,
                            pred_rel,
                        ),

                    "mae_gain_pt_minus_rel":
                        (
                            mean_absolute_error(
                                y_test,
                                pred_pt,
                            )
                            -
                            mean_absolute_error(
                                y_test,
                                pred_rel,
                            )
                        ),
                }
            )


    new_cells_long = pd.concat(
        new_prediction_frames,
        ignore_index=True,
    )

    new_fold_df = pd.DataFrame(
        new_fold_rows
    )


    # -----------------------------------------------------
    # Combine existing and new cell predictions
    # -----------------------------------------------------

    cells_long = pd.concat(
        [
            existing_cells_long,
            new_cells_long,
        ],
        ignore_index=True,
    )

    cells_long = cells_long.sort_values(
        [
            "cell_id",
            "buffer_m",
        ]
    )


    # -----------------------------------------------------
    # Combine fold metrics
    # -----------------------------------------------------

    existing_fold_df = pd.read_csv(
        INPUT_215_FOLD_METRICS
    )

    fold_keep = [
        "buffer_m",
        "fold",
        "n_train",
        "n_test",
        "pt_mae",
        "rel_mae",
        "mae_gain_pt_minus_rel",
    ]

    existing_fold_df = existing_fold_df[
        fold_keep
    ].copy()

    fold_df = pd.concat(
        [
            existing_fold_df,
            new_fold_df,
        ],
        ignore_index=True,
    )

    fold_df = fold_df.sort_values(
        [
            "buffer_m",
            "fold",
        ]
    )


    # -----------------------------------------------------
    # Overall buffer profile
    # -----------------------------------------------------

    overall_rows = []

    for buffer_m in ALL_BUFFERS_M:

        sub = cells_long[
            cells_long[
                "buffer_m"
            ] == buffer_m
        ]

        overall_rows.append(
            {
                "buffer_m":
                    int(buffer_m),

                "buffer_km":
                    buffer_m / 1000,

                "n":
                    len(sub),

                "mean_cell_gain":
                    sub[
                        "relational_gain"
                    ].mean(),

                "median_cell_gain":
                    sub[
                        "relational_gain"
                    ].median(),

                "gain_std":
                    sub[
                        "relational_gain"
                    ].std(),

                "share_cells_relational_better":
                    (
                        sub[
                            "relational_gain"
                        ] > 0
                    ).mean(),

                "mean_fold_gain":
                    fold_df.loc[
                        fold_df[
                            "buffer_m"
                        ] == buffer_m,
                        "mae_gain_pt_minus_rel",
                    ].mean(),

                "median_fold_gain":
                    fold_df.loc[
                        fold_df[
                            "buffer_m"
                        ] == buffer_m,
                        "mae_gain_pt_minus_rel",
                    ].median(),

                "share_folds_relational_better":
                    (
                        fold_df.loc[
                            fold_df[
                                "buffer_m"
                            ] == buffer_m,
                            "mae_gain_pt_minus_rel",
                        ] > 0
                    ).mean(),
            }
        )


    overall_df = pd.DataFrame(
        overall_rows
    )


    # -----------------------------------------------------
    # Fold decay diagnostics
    # -----------------------------------------------------

    fold_pivot = fold_df.pivot(
        index="fold",
        columns="buffer_m",
        values="mae_gain_pt_minus_rel",
    )

    fold_decay_rows = []

    for fold, row in fold_pivot.iterrows():

        valid = row.dropna()

        x = (
            valid.index.to_numpy(
                dtype=float
            )
            / 1000.0
        )

        y = valid.to_numpy(
            dtype=float
        )

        if len(valid) >= 2:

            slope = np.polyfit(
                x,
                y,
                1,
            )[0]

        else:

            slope = np.nan


        positive_buffers = (
            valid[
                valid > 0
            ].index.to_numpy()
        )

        if len(positive_buffers):

            max_positive_buffer_m = int(
                positive_buffers.max()
            )

        else:

            max_positive_buffer_m = np.nan


        signs = np.sign(y)

        sign_changes = int(
            np.sum(
                signs[1:]
                != signs[:-1]
            )
        )


        fold_decay_rows.append(
            {
                "fold":
                    int(fold),

                "gain_slope_per_km":
                    slope,

                "gain_at_0km":
                    row.get(
                        0,
                        np.nan,
                    ),

                "gain_at_50km":
                    row.get(
                        50000,
                        np.nan,
                    ),

                "max_tested_positive_buffer_m":
                    max_positive_buffer_m,

                "n_sign_changes":
                    sign_changes,

                "mean_gain":
                    valid.mean(),

                "median_gain":
                    valid.median(),
            }
        )


    fold_decay_df = pd.DataFrame(
        fold_decay_rows
    )


    # -----------------------------------------------------
    # Cell persistence summary
    #
    # Descriptive only.
    # Not yet a formal interaction horizon.
    # -----------------------------------------------------

    cell_pivot = cells_long.pivot(
        index="cell_id",
        columns="buffer_m",
        values="relational_gain",
    )

    cell_rows = []

    for cell_id, row in cell_pivot.iterrows():

        valid = row.dropna()

        positive = (
            valid > 0
        )

        positive_buffers = (
            valid[
                positive
            ].index.to_numpy()
        )

        if len(positive_buffers):

            max_positive = int(
                positive_buffers.max()
            )

        else:

            max_positive = np.nan


        y = valid.to_numpy(
            dtype=float
        )

        signs = np.sign(y)

        sign_changes = int(
            np.sum(
                signs[1:]
                != signs[:-1]
            )
        )


        cell_rows.append(
            {
                "cell_id":
                    cell_id,

                "mean_gain_across_buffers_v01":
                    valid.mean(),

                "median_gain_across_buffers_v01":
                    valid.median(),

                "share_buffers_relational_better_v01":
                    positive.mean(),

                "max_tested_positive_buffer_m_v01":
                    max_positive,

                "gain_sign_changes_v01":
                    sign_changes,
            }
        )


    cell_summary_df = pd.DataFrame(
        cell_rows
    )


    # -----------------------------------------------------
    # Spatial output
    # -----------------------------------------------------

    output = gdf.merge(
        cell_summary_df,
        on="cell_id",
        how="left",
    )


    # -----------------------------------------------------
    # QA report
    # -----------------------------------------------------

    qa = []

    qa.append(f"{SCRIPT} QA")
    qa.append("=" * 80)

    qa.append(
        f"Input context: {INPUT_CONTEXT}"
    )

    qa.append(
        f"Existing Script 215 predictions: "
        f"{INPUT_215_GPKG}"
    )

    qa.append(
        f"Output: {OUT_GPKG}"
    )

    qa.append("")

    qa.append(
        f"Comparable modeling rows: "
        f"{len(model):,}"
    )

    qa.append(
        f"Existing buffers reused: "
        f"{EXISTING_BUFFERS_M}"
    )

    qa.append(
        f"New buffers fitted: "
        f"{NEW_BUFFERS_M}"
    )

    qa.append(
        f"All buffers analyzed: "
        f"{ALL_BUFFERS_M}"
    )


    qa.append("")
    qa.append(
        "Overall relational gain decay profile:"
    )

    qa.append(
        overall_df.to_string(
            index=False
        )
    )


    qa.append("")
    qa.append(
        "Fold-level relational gain profile:"
    )

    qa.append(
        fold_df.pivot(
            index="fold",
            columns="buffer_m",
            values="mae_gain_pt_minus_rel",
        )
        .to_string()
    )


    qa.append("")
    qa.append(
        "Fold decay diagnostics:"
    )

    qa.append(
        fold_decay_df.to_string(
            index=False
        )
    )


    qa.append("")
    qa.append(
        "Cell persistence summary:"
    )

    qa.append(
        cell_summary_df[
            [
                "mean_gain_across_buffers_v01",
                "median_gain_across_buffers_v01",
                "share_buffers_relational_better_v01",
                "max_tested_positive_buffer_m_v01",
                "gain_sign_changes_v01",
            ]
        ]
        .describe()
        .to_string()
    )


    qa.append("")
    qa.append(
        "Interpretation guardrail:"
    )

    qa.append(
        "Maximum tested positive buffer is a descriptive "
        "persistence statistic, not yet a formal relational "
        "interaction horizon. A formal horizon requires "
        "stability analysis, uncertainty treatment, and "
        "a defensible rule for non-monotonic gain profiles."
    )


    # -----------------------------------------------------
    # Write outputs
    # -----------------------------------------------------

    log("Writing outputs")

    output.to_file(
        OUT_GPKG,
        driver="GPKG",
    )

    overall_df.to_csv(
        OUT_OVERALL_CSV,
        index=False,
    )

    fold_df.to_csv(
        OUT_FOLD_CSV,
        index=False,
    )

    cell_summary_df.to_csv(
        OUT_CELL_CSV,
        index=False,
    )

    OUT_QA.write_text(
        "\n".join(qa),
        encoding="utf-8",
    )


    log(f"Created {OUT_GPKG}")
    log(f"Created {OUT_OVERALL_CSV}")
    log(f"Created {OUT_FOLD_CSV}")
    log(f"Created {OUT_CELL_CSV}")
    log(f"Created {OUT_QA}")

    log("Done")


if __name__ == "__main__":
    main()