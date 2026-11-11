#!/usr/bin/env python3
"""
220_oregon_srs_prospective_persistence_classification_v01.py

Prospective classification of persistent SRS operator regimes in Oregon.

Scientific question
-------------------
Can persistent relational versus persistent pointwise transformation regimes
be predicted prospectively from admissible measured state variables under
spatial out-of-fold validation?

Target
------
1 = persistent_relational
0 = persistent_pointwise

Excluded from v01 target population
-----------------------------------
- weak_relational_persistence
- weak_pointwise_persistence
- unstable_multiscale

Feature families
----------------
1. physical_only
2. recognition_state_only
3. physical_plus_recognition
4. full_admissible_state

Models
------
- regularized logistic regression
- random forest classifier

Validation
----------
8-fold spatial OOF using buffered_spatial_fold_v01.

Important scientific guardrail
------------------------------
Recognition-system variables are used here only to test whether the observed
recognition-system state helps identify which transformation family is
appropriate. They must not be interpreted as admissible predictors of
recognition itself in a recognition-prediction model.

Forbidden predictors
--------------------
Any variables derived from:
- operator gain
- residuals/errors
- pointwise or relational predictions
- operator preference labels
- persistence scores or persistence regimes
- target definitions
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_NAME = "220_oregon_srs_prospective_persistence_classification_v01"

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

INPUT_PERSISTENCE = (
    PROCESSED / "oregon_srs_relational_persistence_common_population_v01.gpkg"
)

INPUT_ENGINE = (
    PROCESSED / "oregon_srs_engine_inputs_v01.gpkg"
)

INPUT_CONTEXT = (
    PROCESSED / "oregon_srs_relational_context_v01.gpkg"
)

OUTPUT_GPKG = (
    PROCESSED / "oregon_srs_prospective_persistence_classification_v01.gpkg"
)

OUTPUT_METRICS = (
    PROCESSED / "oregon_srs_prospective_persistence_classification_metrics_v01.csv"
)

OUTPUT_FOLD_METRICS = (
    PROCESSED / "oregon_srs_prospective_persistence_classification_fold_metrics_v01.csv"
)

OUTPUT_CLASSIFICATION_REPORT = (
    PROCESSED / "oregon_srs_prospective_persistence_classification_reports_v01.csv"
)

OUTPUT_CONFUSION = (
    PROCESSED / "oregon_srs_prospective_persistence_classification_confusion_v01.csv"
)

OUTPUT_RF_IMPORTANCE = (
    PROCESSED / "oregon_srs_prospective_persistence_rf_feature_importance_v01.csv"
)

OUTPUT_RF_GROUP_IMPORTANCE = (
    PROCESSED / "oregon_srs_prospective_persistence_rf_group_importance_v01.csv"
)

OUTPUT_FEATURE_MANIFEST = (
    PROCESSED / "oregon_srs_prospective_persistence_feature_manifest_v01.csv"
)

OUTPUT_QA = (
    PROCESSED / "oregon_srs_prospective_persistence_classification_qa_v01.txt"
)


RANDOM_STATE = 42

TARGET_REGIMES = {
    "persistent_pointwise": 0,
    "persistent_relational": 1,
}

TARGET_COL_CANDIDATES = [
    "persistence_regime_v01",
    "persistence_regime",
]

FOLD_COL_CANDIDATES = [
    "buffered_spatial_fold_v01",
    "spatial_fold_v01",
    "fold_v01",
]

ID_COL_CANDIDATES = [
    "cell_id",
    "grid_id",
    "id",
]


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

PHYSICAL_FEATURES_REQUESTED = [
    "phi_physical_v01",
    "physical_exceptionality_raw_v03",
    "physical_exceptionality_v03",
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
]

RECOGNITION_FEATURES_REQUESTED = [
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
    "observed_recognition_tier_v04",

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
]


# =============================================================================
# FORBIDDEN FEATURE LOGIC
# =============================================================================

FORBIDDEN_PATTERNS = [
    r"gain",
    r"residual",
    r"error",
    r"abs_error",
    r"prediction",
    r"predicted",
    r"yhat",
    r"y_hat",
    r"operator_preference",
    r"operator_regime",
    r"preferred_operator",
    r"persistence_regime",
    r"persistence_score",
    r"relational_win",
    r"pointwise_win",
    r"relational_better",
    r"pointwise_better",
    r"target",
    r"class_label",
]


def is_forbidden_feature(name: str) -> bool:
    lower = name.lower()

    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, lower):
            return True

    return False


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging() -> logging.Logger:
    logger = logging.getLogger(SCRIPT_NAME)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f"[{SCRIPT_NAME}] %(levelname)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


LOGGER = setup_logging()


# =============================================================================
# UTILITIES
# =============================================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")


def find_first_existing_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    label: str,
) -> str:
    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        f"Could not identify {label}. Tried: {list(candidates)}"
    )


def existing_features(
    df: pd.DataFrame,
    requested: Sequence[str],
) -> List[str]:
    return [
        col
        for col in requested
        if col in df.columns and not is_forbidden_feature(col)
    ]


def infer_relational_context_features(
    df: pd.DataFrame,
) -> List[str]:
    """
    Identify admissible relational context variables from Script 207.

    We prefer context-like columns carrying multiscale spatial suffixes or
    neighborhood/context naming. Forbidden derived variables are removed.
    """

    context_patterns = [
        r"_3km",
        r"_10km",
        r"_25km",
        r"neighbor",
        r"neighbour",
        r"context",
        r"local_mean",
        r"local_std",
        r"local_median",
        r"spatial_mean",
        r"spatial_std",
    ]

    selected = []

    for col in df.columns:
        lower = col.lower()

        if col == "geometry":
            continue

        if is_forbidden_feature(col):
            continue

        if any(re.search(pattern, lower) for pattern in context_patterns):
            selected.append(col)

    return sorted(set(selected))


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(roc_auc_score(y_true, y_prob))


def make_logistic_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
) -> Pipeline:

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def make_rf_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
) -> Pipeline:

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def split_feature_types(
    df: pd.DataFrame,
    features: Sequence[str],
) -> Tuple[List[str], List[str]]:

    numeric = []
    categorical = []

    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)

    return numeric, categorical


def feature_group_for_name(
    feature_name: str,
    physical_features: Sequence[str],
    recognition_features: Sequence[str],
    context_features: Sequence[str],
) -> str:

    raw = feature_name

    if "__" in raw:
        raw = raw.split("__", 1)[1]

    for feature in recognition_features:
        if raw == feature or raw.startswith(feature + "_"):
            return "recognition_state"

    for feature in physical_features:
        if raw == feature or raw.startswith(feature + "_"):
            return "physical"

    for feature in context_features:
        if raw == feature or raw.startswith(feature + "_"):
            return "relational_context"

    return "other_admissible"


# =============================================================================
# DATA ASSEMBLY
# =============================================================================

def load_and_merge_inputs() -> gpd.GeoDataFrame:

    require_file(INPUT_PERSISTENCE)
    require_file(INPUT_ENGINE)
    require_file(INPUT_CONTEXT)

    LOGGER.info("Reading persistence common population")
    persistence = gpd.read_file(INPUT_PERSISTENCE)

    LOGGER.info("Reading canonical engine inputs")
    engine = gpd.read_file(INPUT_ENGINE)

    LOGGER.info("Reading relational context")
    context = gpd.read_file(INPUT_CONTEXT)

    id_col = find_first_existing_column(
        persistence,
        ID_COL_CANDIDATES,
        "cell ID column",
    )

    if id_col not in engine.columns:
        raise KeyError(
            f"ID column '{id_col}' missing from engine input."
        )

    if id_col not in context.columns:
        raise KeyError(
            f"ID column '{id_col}' missing from relational context."
        )

    # Remove geometry before tabular merges.
    engine_tab = pd.DataFrame(
        engine.drop(columns="geometry", errors="ignore")
    )

    context_tab = pd.DataFrame(
        context.drop(columns="geometry", errors="ignore")
    )

    # Prevent duplicate columns from silently creating suffix confusion.
    engine_add_cols = [
        id_col
    ] + [
        col
        for col in engine_tab.columns
        if col != id_col and col not in persistence.columns
    ]

    merged = persistence.merge(
        engine_tab[engine_add_cols],
        on=id_col,
        how="left",
        validate="one_to_one",
    )

    context_add_cols = [
        id_col
    ] + [
        col
        for col in context_tab.columns
        if col != id_col and col not in merged.columns
    ]

    merged = merged.merge(
        context_tab[context_add_cols],
        on=id_col,
        how="left",
        validate="one_to_one",
    )

    merged = gpd.GeoDataFrame(
        merged,
        geometry="geometry",
        crs=persistence.crs,
    )

    LOGGER.info(f"Merged rows: {len(merged):,}")
    LOGGER.info(f"Merged columns: {len(merged.columns):,}")

    return merged


# =============================================================================
# FEATURE FAMILY CONSTRUCTION
# =============================================================================

def build_feature_families(
    df: pd.DataFrame,
) -> Tuple[
    Dict[str, List[str]],
    List[str],
    List[str],
    List[str],
]:

    physical = existing_features(
        df,
        PHYSICAL_FEATURES_REQUESTED,
    )

    recognition = existing_features(
        df,
        RECOGNITION_FEATURES_REQUESTED,
    )

    context = infer_relational_context_features(df)

    # Remove overlap so context family does not simply duplicate explicitly
    # listed physical or recognition variables.
    context = [
        col
        for col in context
        if col not in physical
        and col not in recognition
    ]

    full = sorted(
        set(
            physical
            + recognition
            + context
        )
    )

    families = {
        "physical_only": physical,
        "recognition_state_only": recognition,
        "physical_plus_recognition": sorted(
            set(physical + recognition)
        ),
        "full_admissible_state": full,
    }

    for family, features in families.items():
        forbidden = [
            col
            for col in features
            if is_forbidden_feature(col)
        ]

        if forbidden:
            raise ValueError(
                f"Forbidden features entered family {family}: "
                f"{forbidden}"
            )

        if len(features) == 0:
            raise ValueError(
                f"Feature family '{family}' has zero available features."
            )

    return families, physical, recognition, context


# =============================================================================
# SPATIAL OOF EVALUATION
# =============================================================================

def evaluate_family_model(
    df: pd.DataFrame,
    features: List[str],
    family_name: str,
    model_name: str,
    fold_col: str,
    target_col: str,
) -> Tuple[
    pd.DataFrame,
    Dict,
    List[Dict],
    pd.DataFrame,
    pd.DataFrame,
]:

    numeric_features, categorical_features = split_feature_types(
        df,
        features,
    )

    LOGGER.info(
        f"{family_name} / {model_name}: "
        f"{len(features)} features "
        f"({len(numeric_features)} numeric, "
        f"{len(categorical_features)} categorical)"
    )

    if model_name == "logistic":
        pipeline_factory = lambda: make_logistic_pipeline(
            numeric_features,
            categorical_features,
        )
    elif model_name == "random_forest":
        pipeline_factory = lambda: make_rf_pipeline(
            numeric_features,
            categorical_features,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    work = df.copy()

    work[f"pred_{family_name}_{model_name}"] = np.nan
    work[f"prob_{family_name}_{model_name}"] = np.nan

    fold_rows = []
    importance_rows = []

    folds = sorted(work[fold_col].dropna().unique().tolist())

    for fold in folds:
        LOGGER.info(
            f"Running {family_name} / {model_name} / fold {fold}"
        )

        train_mask = work[fold_col] != fold
        test_mask = work[fold_col] == fold

        train = work.loc[train_mask].copy()
        test = work.loc[test_mask].copy()

        X_train = train[features]
        y_train = train[target_col].astype(int).to_numpy()

        X_test = test[features]
        y_test = test[target_col].astype(int).to_numpy()

        if len(np.unique(y_train)) < 2:
            raise ValueError(
                f"Training fold complement for fold {fold} "
                f"contains fewer than two classes."
            )

        model = pipeline_factory()
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        work.loc[
            test_mask,
            f"pred_{family_name}_{model_name}",
        ] = pred

        work.loc[
            test_mask,
            f"prob_{family_name}_{model_name}",
        ] = prob

        fold_rows.append(
            {
                "feature_family": family_name,
                "model": model_name,
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
                "test_positive_rate": float(np.mean(y_test)),
                "accuracy": float(
                    accuracy_score(y_test, pred)
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_test, pred)
                ),
                "roc_auc": safe_auc(y_test, prob),
            }
        )

        # Descriptive RF importance only.
        if model_name == "random_forest":
            preprocessor = model.named_steps["preprocessor"]
            rf_model = model.named_steps["model"]

            transformed_names = (
                preprocessor.get_feature_names_out()
            )

            importances = rf_model.feature_importances_

            for feature_name, importance in zip(
                transformed_names,
                importances,
            ):
                importance_rows.append(
                    {
                        "feature_family": family_name,
                        "model": model_name,
                        "fold": fold,
                        "transformed_feature": feature_name,
                        "importance": float(importance),
                    }
                )

    pred_col = f"pred_{family_name}_{model_name}"
    prob_col = f"prob_{family_name}_{model_name}"

    valid = work[pred_col].notna() & work[prob_col].notna()

    y_true = work.loc[valid, target_col].astype(int).to_numpy()
    y_pred = work.loc[valid, pred_col].astype(int).to_numpy()
    y_prob = work.loc[valid, prob_col].astype(float).to_numpy()

    overall = {
        "feature_family": family_name,
        "model": model_name,
        "n": int(valid.sum()),
        "n_features": len(features),
        "n_numeric_features": len(numeric_features),
        "n_categorical_features": len(categorical_features),
        "positive_rate": float(np.mean(y_true)),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "roc_auc": safe_auc(y_true, y_prob),
    }

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=[
            "persistent_pointwise",
            "persistent_relational",
        ],
        output_dict=True,
        zero_division=0,
    )

    report_rows = []

    for label, values in report_dict.items():
        if isinstance(values, dict):
            row = {
                "feature_family": family_name,
                "model": model_name,
                "label": label,
            }
            row.update(values)
            report_rows.append(row)

    report_df = pd.DataFrame(report_rows)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    confusion_df = pd.DataFrame(
        [
            {
                "feature_family": family_name,
                "model": model_name,
                "true_class": "persistent_pointwise",
                "pred_pointwise": int(cm[0, 0]),
                "pred_relational": int(cm[0, 1]),
            },
            {
                "feature_family": family_name,
                "model": model_name,
                "true_class": "persistent_relational",
                "pred_pointwise": int(cm[1, 0]),
                "pred_relational": int(cm[1, 1]),
            },
        ]
    )

    importance_df = pd.DataFrame(importance_rows)

    return (
        work[
            [
                pred_col,
                prob_col,
            ]
        ],
        overall,
        fold_rows,
        report_df,
        confusion_df,
        importance_df,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    LOGGER.info("Starting prospective persistence classification")

    df = load_and_merge_inputs()

    target_regime_col = find_first_existing_column(
        df,
        TARGET_COL_CANDIDATES,
        "persistence regime column",
    )

    fold_col = find_first_existing_column(
        df,
        FOLD_COL_CANDIDATES,
        "buffered spatial fold column",
    )

    id_col = find_first_existing_column(
        df,
        ID_COL_CANDIDATES,
        "cell ID column",
    )

    LOGGER.info(f"Target regime column: {target_regime_col}")
    LOGGER.info(f"Fold column: {fold_col}")
    LOGGER.info(f"ID column: {id_col}")

    # Primary v01 population only.
    model_df = df[
        df[target_regime_col].isin(TARGET_REGIMES.keys())
    ].copy()

    model_df["persistence_target_v01"] = (
        model_df[target_regime_col]
        .map(TARGET_REGIMES)
        .astype(int)
    )

    LOGGER.info(
        f"Primary modeling rows: {len(model_df):,}"
    )

    LOGGER.info(
        "Target counts:\n"
        + model_df[target_regime_col]
        .value_counts()
        .to_string()
    )

    LOGGER.info(
        "Fold counts:\n"
        + model_df[fold_col]
        .value_counts()
        .sort_index()
        .to_string()
    )

    unique_folds = sorted(
        model_df[fold_col].dropna().unique().tolist()
    )

    if len(unique_folds) != 8:
        raise ValueError(
            f"Expected 8 spatial folds; found {len(unique_folds)}: "
            f"{unique_folds}"
        )

    families, physical, recognition, context = (
        build_feature_families(model_df)
    )

    LOGGER.info(
        f"Physical features available: {len(physical)}"
    )
    LOGGER.info(
        f"Recognition-state features available: {len(recognition)}"
    )
    LOGGER.info(
        f"Relational-context features available: {len(context)}"
    )

    # Feature manifest.
    manifest_rows = []

    for family_name, features in families.items():
        for feature in features:
            if feature in physical:
                source_group = "physical"
            elif feature in recognition:
                source_group = "recognition_state"
            elif feature in context:
                source_group = "relational_context"
            else:
                source_group = "other_admissible"

            manifest_rows.append(
                {
                    "feature_family": family_name,
                    "feature": feature,
                    "source_group": source_group,
                    "dtype": str(model_df[feature].dtype),
                    "missing_n": int(
                        model_df[feature].isna().sum()
                    ),
                    "missing_share": float(
                        model_df[feature].isna().mean()
                    ),
                }
            )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(
        OUTPUT_FEATURE_MANIFEST,
        index=False,
    )

    metrics_rows = []
    fold_metric_rows = []
    report_frames = []
    confusion_frames = []
    importance_frames = []

    output_predictions = model_df[
        [
            id_col,
            target_regime_col,
            "persistence_target_v01",
            fold_col,
            "geometry",
        ]
    ].copy()

    for family_name, features in families.items():
        for model_name in [
            "logistic",
            "random_forest",
        ]:

            (
                prediction_cols,
                overall,
                fold_rows,
                report_df,
                confusion_df,
                importance_df,
            ) = evaluate_family_model(
                df=model_df,
                features=features,
                family_name=family_name,
                model_name=model_name,
                fold_col=fold_col,
                target_col="persistence_target_v01",
            )

            output_predictions = pd.concat(
                [
                    output_predictions,
                    prediction_cols,
                ],
                axis=1,
            )

            metrics_rows.append(overall)
            fold_metric_rows.extend(fold_rows)
            report_frames.append(report_df)
            confusion_frames.append(confusion_df)

            if not importance_df.empty:
                importance_frames.append(importance_df)

    metrics_df = pd.DataFrame(metrics_rows)
    fold_metrics_df = pd.DataFrame(fold_metric_rows)

    reports_df = pd.concat(
        report_frames,
        ignore_index=True,
    )

    confusion_df = pd.concat(
        confusion_frames,
        ignore_index=True,
    )

    if importance_frames:
        importance_raw = pd.concat(
            importance_frames,
            ignore_index=True,
        )

        importance_summary = (
            importance_raw
            .groupby(
                [
                    "feature_family",
                    "model",
                    "transformed_feature",
                ],
                as_index=False,
            )
            .agg(
                mean_importance=("importance", "mean"),
                std_importance=("importance", "std"),
                min_importance=("importance", "min"),
                max_importance=("importance", "max"),
            )
        )

        importance_summary["feature_group"] = (
            importance_summary["transformed_feature"]
            .apply(
                lambda x: feature_group_for_name(
                    x,
                    physical,
                    recognition,
                    context,
                )
            )
        )

        group_importance = (
            importance_summary
            .groupby(
                [
                    "feature_family",
                    "model",
                    "feature_group",
                ],
                as_index=False,
            )
            .agg(
                total_mean_importance=(
                    "mean_importance",
                    "sum",
                ),
                n_transformed_features=(
                    "transformed_feature",
                    "count",
                ),
            )
        )

    else:
        importance_summary = pd.DataFrame()
        group_importance = pd.DataFrame()

    # Save outputs.
    output_predictions = gpd.GeoDataFrame(
        output_predictions,
        geometry="geometry",
        crs=model_df.crs,
    )

    LOGGER.info(f"Writing {OUTPUT_GPKG}")
    output_predictions.to_file(
        OUTPUT_GPKG,
        driver="GPKG",
    )

    metrics_df.to_csv(
        OUTPUT_METRICS,
        index=False,
    )

    fold_metrics_df.to_csv(
        OUTPUT_FOLD_METRICS,
        index=False,
    )

    reports_df.to_csv(
        OUTPUT_CLASSIFICATION_REPORT,
        index=False,
    )

    confusion_df.to_csv(
        OUTPUT_CONFUSION,
        index=False,
    )

    importance_summary.to_csv(
        OUTPUT_RF_IMPORTANCE,
        index=False,
    )

    group_importance.to_csv(
        OUTPUT_RF_GROUP_IMPORTANCE,
        index=False,
    )

    # QA summary.
    best_ba_row = metrics_df.loc[
        metrics_df["balanced_accuracy"].idxmax()
    ]

    best_auc_row = metrics_df.loc[
        metrics_df["roc_auc"].idxmax()
    ]

    qa_lines = [
        f"Script: {SCRIPT_NAME}",
        "",
        "PURPOSE",
        "Test prospective identifiability of persistent relational versus",
        "persistent pointwise operator regimes from admissible measured state",
        "variables under 8-fold spatial OOF validation.",
        "",
        "SCIENTIFIC GUARDRAIL",
        "Recognition-system variables are used only to test whether observed",
        "recognition-system state helps identify the appropriate transformation",
        "family. They are not being used here to predict recognition itself.",
        "",
        f"Input persistence: {INPUT_PERSISTENCE}",
        f"Input engine: {INPUT_ENGINE}",
        f"Input context: {INPUT_CONTEXT}",
        "",
        f"Primary rows: {len(model_df):,}",
        f"Folds: {unique_folds}",
        "",
        "TARGET COUNTS",
        model_df[target_regime_col].value_counts().to_string(),
        "",
        "FEATURE COUNTS",
        f"physical: {len(physical)}",
        f"recognition_state: {len(recognition)}",
        f"relational_context: {len(context)}",
        "",
        "OVERALL METRICS",
        metrics_df.sort_values(
            ["roc_auc", "balanced_accuracy"],
            ascending=False,
        ).to_string(index=False),
        "",
        "BEST BALANCED ACCURACY",
        (
            f"{best_ba_row['feature_family']} / "
            f"{best_ba_row['model']}: "
            f"{best_ba_row['balanced_accuracy']:.6f}"
        ),
        "",
        "BEST ROC AUC",
        (
            f"{best_auc_row['feature_family']} / "
            f"{best_auc_row['model']}: "
            f"{best_auc_row['roc_auc']:.6f}"
        ),
        "",
        "INTERPRETATION RULES",
        "1. Physical-only failure with recognition/full success suggests",
        "   persistence is more a recognition-system-state phenomenon than",
        "   a terrain phenomenon.",
        "2. Near-chance performance across all families means persistence is",
        "   not prospectively identifiable from current measured variables.",
        "3. Full-state improvement over recognition-only suggests relational",
        "   context contributes additional operator-selection signal.",
        "4. Strong aggregate performance with unstable fold performance is not",
        "   sufficient justification for an adaptive or gated SRS engine.",
        "",
        "IMPORTANT",
        "No adaptive engine should be built from this script alone.",
    ]

    OUTPUT_QA.write_text(
        "\n".join(qa_lines),
        encoding="utf-8",
    )

    LOGGER.info("Completed Script 220")
    LOGGER.info("")
    LOGGER.info("Overall metrics:")
    LOGGER.info(
        "\n"
        + metrics_df.sort_values(
            ["roc_auc", "balanced_accuracy"],
            ascending=False,
        ).to_string(index=False)
    )

    LOGGER.info("")
    LOGGER.info("Created:")
    for path in [
        OUTPUT_GPKG,
        OUTPUT_METRICS,
        OUTPUT_FOLD_METRICS,
        OUTPUT_CLASSIFICATION_REPORT,
        OUTPUT_CONFUSION,
        OUTPUT_RF_IMPORTANCE,
        OUTPUT_RF_GROUP_IMPORTANCE,
        OUTPUT_FEATURE_MANIFEST,
        OUTPUT_QA,
    ]:
        LOGGER.info(str(path))


if __name__ == "__main__":
    main()