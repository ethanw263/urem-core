from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT = "212_oregon_srs_operator_regime_predictability_v01"

BASE = Path("data/processed")

INPUT_FP = (
    BASE
    / "oregon_srs_operator_regime_spatial_coherence_v01.gpkg"
)

OUT_GPKG = (
    BASE
    / "oregon_srs_operator_regime_predictability_v01.gpkg"
)

OUT_CSV = (
    BASE
    / "oregon_srs_operator_regime_predictability_v01.csv"
)

OUT_IMPORTANCE_CSV = (
    BASE
    / "oregon_srs_operator_regime_feature_importance_v01.csv"
)

OUT_FOLD_CSV = (
    BASE
    / "oregon_srs_operator_regime_predictability_folds_v01.csv"
)

OUT_QA = (
    BASE
    / "oregon_srs_operator_regime_predictability_qa_v01.txt"
)


TARGET = "operator_spatial_regime_v01"
FOLD = "spatial_fold_v01"

REL_CLASS = "coherent_relational"
PT_CLASS = "coherent_pointwise"

RANDOM_SEED = 42


# ---------------------------------------------------------
# IMPORTANT:
# Only upstream physical / contextual state variables are
# allowed here.
#
# Do NOT include:
# - y_observed_v01
# - observed_recognition_v04
# - relational_gain_v01
# - any residual / disequilibrium variable
# - operator preference labels
# ---------------------------------------------------------

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


def safe_auc(y_true, prob):
    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, prob)


def main():

    log("Reading spatial coherence layer")

    gdf = gpd.read_file(INPUT_FP)

    required = [
        "cell_id",
        TARGET,
        FOLD,
    ]

    for col in required:
        if col not in gdf.columns:
            raise ValueError(
                f"Missing required column: {col}"
            )


    # -----------------------------------------------------
    # Select relational context variables
    # -----------------------------------------------------

    relational_features = [
        c
        for c in gdf.columns
        if (
            c.startswith("rel_neighbor_count_")
            or "_mean_" in c
            or "_min_" in c
            or "_max_" in c
            or "_contrast_" in c
        )
        and c.endswith("_v01")
    ]


    # Explicitly exclude anything downstream or suspicious

    forbidden_tokens = [
        "gain",
        "observed",
        "prediction",
        "pred_",
        "d_pointwise",
        "d_relational",
        "operator_",
        "residual",
        "error",
    ]

    relational_features = [
        c
        for c in relational_features
        if not any(
            token in c.lower()
            for token in forbidden_tokens
        )
    ]


    candidate_features = (
        POINTWISE_FEATURES
        + relational_features
    )

    features = [
        c
        for c in candidate_features
        if c in gdf.columns
    ]

    features = list(dict.fromkeys(features))


    log(
        f"Pointwise features available: "
        f"{sum(c in gdf.columns for c in POINTWISE_FEATURES)}"
    )

    log(
        f"Relational context features available: "
        f"{len(relational_features)}"
    )

    log(
        f"Total gating features: {len(features)}"
    )


    # -----------------------------------------------------
    # Restrict target to coherent relational vs pointwise
    #
    # Mixed zones are intentionally excluded in v01.
    # -----------------------------------------------------

    mask = (
        gdf[TARGET].isin(
            [
                REL_CLASS,
                PT_CLASS,
            ]
        )
        & gdf[FOLD].notna()
    )

    work = gdf.loc[mask].copy()


    # -----------------------------------------------------
    # Complete-case feature matrix
    # -----------------------------------------------------

    complete = work[features].notna().all(axis=1)

    work = work.loc[complete].copy()

    log(
        f"Rows available for regime classification: "
        f"{len(work):,}"
    )


    work["regime_target_binary_v01"] = (
        work[TARGET] == REL_CLASS
    ).astype(int)


    X = work[features].to_numpy(dtype=float)

    y = work[
        "regime_target_binary_v01"
    ].to_numpy(dtype=int)

    folds = work[FOLD].to_numpy()


    # -----------------------------------------------------
    # OOF prediction containers
    # -----------------------------------------------------

    pred_logit = np.full(len(work), np.nan)
    prob_logit = np.full(len(work), np.nan)

    pred_rf = np.full(len(work), np.nan)
    prob_rf = np.full(len(work), np.nan)


    fold_rows = []


    # -----------------------------------------------------
    # Spatial leave-fold-out evaluation
    # -----------------------------------------------------

    unique_folds = sorted(
        pd.Series(folds).dropna().unique()
    )

    log(
        f"Spatial folds: {len(unique_folds)}"
    )


    for fold in unique_folds:

        test_mask = folds == fold
        train_mask = ~test_mask

        X_train = X[train_mask]
        X_test = X[test_mask]

        y_train = y[train_mask]
        y_test = y[test_mask]


        if len(np.unique(y_train)) < 2:
            log(
                f"Skipping fold {fold}: "
                "training set has one class"
            )
            continue


        log(
            f"Fold {fold}: "
            f"train={train_mask.sum():,}, "
            f"test={test_mask.sum():,}"
        )


        # -------------------------------------------------
        # Logistic gating model
        # -------------------------------------------------

        logit = Pipeline(
            [
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )

        logit.fit(
            X_train,
            y_train,
        )

        pred_logit[test_mask] = (
            logit.predict(X_test)
        )

        prob_logit[test_mask] = (
            logit.predict_proba(X_test)[:, 1]
        )


        # -------------------------------------------------
        # Random forest gating model
        # -------------------------------------------------

        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=16,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )

        rf.fit(
            X_train,
            y_train,
        )

        pred_rf[test_mask] = (
            rf.predict(X_test)
        )

        prob_rf[test_mask] = (
            rf.predict_proba(X_test)[:, 1]
        )


        fold_rows.append(
            {
                "fold": fold,
                "n": int(test_mask.sum()),

                "relational_share": float(
                    y_test.mean()
                ),

                "logit_accuracy": accuracy_score(
                    y_test,
                    pred_logit[test_mask],
                ),

                "logit_balanced_accuracy":
                    balanced_accuracy_score(
                        y_test,
                        pred_logit[test_mask],
                    ),

                "logit_auc": safe_auc(
                    y_test,
                    prob_logit[test_mask],
                ),

                "rf_accuracy": accuracy_score(
                    y_test,
                    pred_rf[test_mask],
                ),

                "rf_balanced_accuracy":
                    balanced_accuracy_score(
                        y_test,
                        pred_rf[test_mask],
                    ),

                "rf_auc": safe_auc(
                    y_test,
                    prob_rf[test_mask],
                ),
            }
        )


    fold_df = pd.DataFrame(fold_rows)


    # -----------------------------------------------------
    # Overall OOF metrics
    # -----------------------------------------------------

    valid_logit = ~np.isnan(pred_logit)
    valid_rf = ~np.isnan(pred_rf)


    overall = {
        "n_total": len(work),

        "relational_share": float(y.mean()),

        "logit_accuracy": accuracy_score(
            y[valid_logit],
            pred_logit[valid_logit],
        ),

        "logit_balanced_accuracy":
            balanced_accuracy_score(
                y[valid_logit],
                pred_logit[valid_logit],
            ),

        "logit_auc": safe_auc(
            y[valid_logit],
            prob_logit[valid_logit],
        ),

        "rf_accuracy": accuracy_score(
            y[valid_rf],
            pred_rf[valid_rf],
        ),

        "rf_balanced_accuracy":
            balanced_accuracy_score(
                y[valid_rf],
                pred_rf[valid_rf],
            ),

        "rf_auc": safe_auc(
            y[valid_rf],
            prob_rf[valid_rf],
        ),
    }


    # -----------------------------------------------------
    # Fit final RF for descriptive feature importance only
    #
    # This is not used for OOF evaluation.
    # -----------------------------------------------------

    log(
        "Fitting full-data RF for descriptive "
        "feature importance"
    )

    final_rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=16,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )

    final_rf.fit(
        X,
        y,
    )


    importance_df = pd.DataFrame(
        {
            "feature": features,
            "importance": final_rf.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )


    importance_df["feature_group"] = np.where(
        importance_df["feature"].isin(
            POINTWISE_FEATURES
        ),
        "pointwise",
        "relational_context",
    )


    # -----------------------------------------------------
    # Add predictions to output
    # -----------------------------------------------------

    work["gate_pred_logit_oof_v01"] = pred_logit
    work["gate_prob_logit_oof_v01"] = prob_logit

    work["gate_pred_rf_oof_v01"] = pred_rf
    work["gate_prob_rf_oof_v01"] = prob_rf


    result_cols = [
        "cell_id",
        "regime_target_binary_v01",
        "gate_pred_logit_oof_v01",
        "gate_prob_logit_oof_v01",
        "gate_pred_rf_oof_v01",
        "gate_prob_rf_oof_v01",
    ]


    result = gdf.merge(
        work[result_cols],
        on="cell_id",
        how="left",
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
        f"Rows classified: {len(work):,}"
    )

    qa.append(
        f"Features used: {len(features)}"
    )

    qa.append("")
    qa.append("Overall spatial OOF metrics:")

    for key, value in overall.items():
        qa.append(
            f"{key}: {value}"
        )


    qa.append("")
    qa.append("Confusion matrix — logistic:")

    qa.append(
        np.array2string(
            confusion_matrix(
                y[valid_logit],
                pred_logit[valid_logit],
            )
        )
    )


    qa.append("")
    qa.append("Confusion matrix — random forest:")

    qa.append(
        np.array2string(
            confusion_matrix(
                y[valid_rf],
                pred_rf[valid_rf],
            )
        )
    )


    qa.append("")
    qa.append("Classification report — logistic:")

    qa.append(
        classification_report(
            y[valid_logit],
            pred_logit[valid_logit],
            digits=4,
        )
    )


    qa.append("")
    qa.append("Classification report — random forest:")

    qa.append(
        classification_report(
            y[valid_rf],
            pred_rf[valid_rf],
            digits=4,
        )
    )


    qa.append("")
    qa.append("Fold metrics:")

    qa.append(
        fold_df.to_string(
            index=False
        )
    )


    qa.append("")
    qa.append(
        "Top 50 descriptive RF feature importances:"
    )

    qa.append(
        importance_df.head(50).to_string(
            index=False
        )
    )


    qa.append("")
    qa.append(
        "Importance by feature group:"
    )

    qa.append(
        importance_df.groupby(
            "feature_group"
        )["importance"]
        .sum()
        .sort_values(
            ascending=False
        )
        .to_string()
    )


    # -----------------------------------------------------
    # Write outputs
    # -----------------------------------------------------

    log("Writing outputs")

    result.to_file(
        OUT_GPKG,
        driver="GPKG",
    )

    result.drop(
        columns="geometry"
    ).to_csv(
        OUT_CSV,
        index=False,
    )

    importance_df.to_csv(
        OUT_IMPORTANCE_CSV,
        index=False,
    )

    fold_df.to_csv(
        OUT_FOLD_CSV,
        index=False,
    )

    OUT_QA.write_text(
        "\n".join(qa),
        encoding="utf-8",
    )


    log(f"Created {OUT_GPKG}")
    log(f"Created {OUT_CSV}")
    log(f"Created {OUT_IMPORTANCE_CSV}")
    log(f"Created {OUT_FOLD_CSV}")
    log(f"Created {OUT_QA}")

    log("Done")


if __name__ == "__main__":
    main()