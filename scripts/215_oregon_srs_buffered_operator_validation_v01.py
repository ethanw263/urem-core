from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


SCRIPT = "215_oregon_srs_buffered_operator_validation_v01"

BASE = Path("data/processed")

INPUT_CONTEXT = (
    BASE
    / "oregon_srs_relational_context_v01.gpkg"
)

INPUT_FOLDS = (
    BASE
    / "oregon_srs_buffered_spatial_validation_design_v01.gpkg"
)

OUT_PREDICTIONS = (
    BASE
    / "oregon_srs_buffered_operator_validation_v01.gpkg"
)

OUT_METRICS = (
    BASE
    / "oregon_srs_buffered_operator_validation_metrics_v01.csv"
)

OUT_FOLD_METRICS = (
    BASE
    / "oregon_srs_buffered_operator_validation_fold_metrics_v01.csv"
)

OUT_QA = (
    BASE
    / "oregon_srs_buffered_operator_validation_qa_v01.txt"
)


TARGET = "y_observed_v01"

FOLD_COL = "buffered_spatial_fold_v01"

BUFFER_DISTANCES_M = [
    0,
    3000,
    10000,
    25000,
]

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


def metric_dict(y_true, y_pred):

    return {
        "n": len(y_true),
        "r2": r2_score(
            y_true,
            y_pred,
        ),
        "mae": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        ),
    }


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
        columns=[
            FOLD_COL
        ],
        errors="ignore",
    )

    gdf = gdf.merge(
        folds,
        on="cell_id",
        how="left",
    )


    # -----------------------------------------------------
    # Relational feature discovery
    #
    # Match Script 208 convention:
    # relational context columns ending in _v01
    # and beginning with context-variable names or
    # rel_neighbor_count.
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
    # Required-column checks
    # -----------------------------------------------------

    required = (
        [
            "cell_id",
            "centroid_x",
            "centroid_y",
            TARGET,
            FOLD_COL,
        ]
        + all_features
    )

    missing = [
        c
        for c in required
        if c not in gdf.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )


    # -----------------------------------------------------
    # Common complete-case modeling population
    #
    # Critical: both operators use identical rows.
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
        gdf[
            model_cols
        ]
        .dropna()
        .copy()
    )


    log(
        f"Comparable modeling rows: "
        f"{len(model):,}"
    )


    # -----------------------------------------------------
    # Storage
    # -----------------------------------------------------

    prediction_frames = []

    fold_metric_rows = []


    # -----------------------------------------------------
    # Loop over buffer distances
    # -----------------------------------------------------

    for buffer_m in BUFFER_DISTANCES_M:

        log(
            f"Starting buffer distance: "
            f"{buffer_m:,} m"
        )

        buffer_predictions = []


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

            train_candidate_mask = (
                ~test_mask
            )


            test = model.loc[
                test_mask
            ].copy()

            train_candidates = model.loc[
                train_candidate_mask
            ].copy()


            # ---------------------------------------------
            # Apply geographic buffer
            # ---------------------------------------------

            if buffer_m > 0:

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
                    train_candidates
                    .loc[keep_train]
                    .copy()
                )

            else:

                train = (
                    train_candidates.copy()
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
            # Pointwise linear comparator
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
            # Relational random forest
            #
            # Match prior RF architecture closely.
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
            # Fold-level metrics
            # ---------------------------------------------

            pt_metrics = metric_dict(
                y_test,
                pred_pt,
            )

            rel_metrics = metric_dict(
                y_test,
                pred_rel,
            )

            gain = (
                pt_metrics["mae"]
                - rel_metrics["mae"]
            )


            fold_metric_rows.append(
                {
                    "buffer_m":
                        int(buffer_m),

                    "fold":
                        int(fold),

                    "n_train":
                        len(train),

                    "n_test":
                        len(test),

                    "pt_r2":
                        pt_metrics["r2"],

                    "pt_mae":
                        pt_metrics["mae"],

                    "pt_rmse":
                        pt_metrics["rmse"],

                    "rel_r2":
                        rel_metrics["r2"],

                    "rel_mae":
                        rel_metrics["mae"],

                    "rel_rmse":
                        rel_metrics["rmse"],

                    "mae_gain_pt_minus_rel":
                        gain,

                    "relational_better":
                        gain > 0,
                }
            )


            # ---------------------------------------------
            # Cell-level predictions
            # ---------------------------------------------

            pred_df = pd.DataFrame(
                {
                    "cell_id":
                        test["cell_id"].to_numpy(),

                    "buffer_m":
                        int(buffer_m),

                    "fold":
                        int(fold),

                    "y_true":
                        y_test,

                    "pred_pointwise":
                        pred_pt,

                    "pred_relational":
                        pred_rel,
                }
            )

            pred_df[
                "abs_error_pointwise"
            ] = np.abs(
                pred_df["y_true"]
                - pred_df["pred_pointwise"]
            )

            pred_df[
                "abs_error_relational"
            ] = np.abs(
                pred_df["y_true"]
                - pred_df["pred_relational"]
            )

            pred_df[
                "relational_gain"
            ] = (
                pred_df[
                    "abs_error_pointwise"
                ]
                - pred_df[
                    "abs_error_relational"
                ]
            )

            buffer_predictions.append(
                pred_df
            )


        buffer_pred = pd.concat(
            buffer_predictions,
            ignore_index=True,
        )

        prediction_frames.append(
            buffer_pred
        )


    # -----------------------------------------------------
    # Combine predictions
    # -----------------------------------------------------

    predictions_long = pd.concat(
        prediction_frames,
        ignore_index=True,
    )


    # -----------------------------------------------------
    # Overall metrics by buffer
    # -----------------------------------------------------

    metric_rows = []

    for buffer_m in BUFFER_DISTANCES_M:

        sub = predictions_long[
            predictions_long[
                "buffer_m"
            ] == buffer_m
        ].copy()

        pt = metric_dict(
            sub["y_true"],
            sub["pred_pointwise"],
        )

        rel = metric_dict(
            sub["y_true"],
            sub["pred_relational"],
        )

        gain = (
            pt["mae"]
            - rel["mae"]
        )

        cell_gain = (
            sub["relational_gain"]
        )

        metric_rows.append(
            {
                "buffer_m":
                    int(buffer_m),

                "n":
                    len(sub),

                "pt_r2":
                    pt["r2"],

                "pt_mae":
                    pt["mae"],

                "pt_rmse":
                    pt["rmse"],

                "rel_r2":
                    rel["r2"],

                "rel_mae":
                    rel["mae"],

                "rel_rmse":
                    rel["rmse"],

                "mean_mae_gain":
                    gain,

                "median_cell_gain":
                    cell_gain.median(),

                "share_cells_relational_better":
                    (
                        cell_gain > 0
                    ).mean(),

                "share_folds_relational_better":
                    np.mean(
                        [
                            r[
                                "relational_better"
                            ]
                            for r in fold_metric_rows
                            if r["buffer_m"]
                            == buffer_m
                        ]
                    ),
            }
        )


    metrics_df = pd.DataFrame(
        metric_rows
    )

    fold_metrics_df = pd.DataFrame(
        fold_metric_rows
    )


    # -----------------------------------------------------
    # Convert long predictions to wide cell table
    # -----------------------------------------------------

    wide_parts = []

    for buffer_m in BUFFER_DISTANCES_M:

        sub = predictions_long[
            predictions_long[
                "buffer_m"
            ] == buffer_m
        ][
            [
                "cell_id",
                "pred_pointwise",
                "pred_relational",
                "relational_gain",
            ]
        ].copy()

        suffix = (
            f"{int(buffer_m / 1000)}km"
            if buffer_m > 0
            else "0km"
        )

        sub = sub.rename(
            columns={
                "pred_pointwise":
                    f"pred_pt_{suffix}_v01",

                "pred_relational":
                    f"pred_rel_{suffix}_v01",

                "relational_gain":
                    f"gain_rel_{suffix}_v01",
            }
        )

        wide_parts.append(sub)


    wide = wide_parts[0]

    for part in wide_parts[1:]:

        wide = wide.merge(
            part,
            on="cell_id",
            how="outer",
        )


    output = gdf.merge(
        wide,
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
        f"Input folds: {INPUT_FOLDS}"
    )

    qa.append(
        f"Output: {OUT_PREDICTIONS}"
    )

    qa.append("")

    qa.append(
        f"Comparable modeling rows: "
        f"{len(model):,}"
    )

    qa.append(
        f"Pointwise features: "
        f"{len(POINTWISE_FEATURES)}"
    )

    qa.append(
        f"Relational context features: "
        f"{len(relational_features)}"
    )

    qa.append(
        f"Total relational-model features: "
        f"{len(all_features)}"
    )


    qa.append("")
    qa.append(
        "Overall buffered operator comparison:"
    )

    qa.append(
        metrics_df.to_string(
            index=False
        )
    )


    qa.append("")
    qa.append(
        "Fold-level metrics:"
    )

    qa.append(
        fold_metrics_df.to_string(
            index=False
        )
    )


    qa.append("")
    qa.append(
        "Mean fold gain by buffer:"
    )

    qa.append(
        fold_metrics_df.groupby(
            "buffer_m"
        )[
            "mae_gain_pt_minus_rel"
        ]
        .agg(
            [
                "mean",
                "median",
                "std",
                "min",
                "max",
            ]
        )
        .to_string()
    )


    qa.append("")
    qa.append(
        "Scientific interpretation rule:"
    )

    qa.append(
        "A positive mean_mae_gain indicates lower MAE "
        "for the relational operator. Robust relational "
        "evidence requires that the advantage remain "
        "positive as spatial buffer distance increases, "
        "rather than appearing only at zero or short "
        "separation distances."
    )


    # -----------------------------------------------------
    # Write
    # -----------------------------------------------------

    log("Writing outputs")

    output.to_file(
        OUT_PREDICTIONS,
        driver="GPKG",
    )

    metrics_df.to_csv(
        OUT_METRICS,
        index=False,
    )

    fold_metrics_df.to_csv(
        OUT_FOLD_METRICS,
        index=False,
    )

    OUT_QA.write_text(
        "\n".join(qa),
        encoding="utf-8",
    )


    log(f"Created {OUT_PREDICTIONS}")
    log(f"Created {OUT_METRICS}")
    log(f"Created {OUT_FOLD_METRICS}")
    log(f"Created {OUT_QA}")

    log("Done")


if __name__ == "__main__":
    main()