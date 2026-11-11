#!/usr/bin/env python3
"""
221_oregon_srs_persistence_classification_stability_audit_v01.py

Persistence Classification Stability and Leakage Audit
-------------------------------------------------------

Purpose
-------
Audit the prospective persistence-classification result from Script 220.

Primary scientific questions:
1. Is persistence classification signal genuinely carried by relational
   context variables?
2. Does context-only performance approach full-state performance?
3. Do physical or recognition-state variables add incremental signal once
   relational context is present?
4. Is full-state performance dependent on a small number of suspicious or
   unstable variables?
5. Are the same important features useful across spatial folds?
6. Are folds 0 and 5 genuine spatial transfer failures rather than aggregate
   metric artifacts?

Target
------
1 = persistent_relational
0 = persistent_pointwise

Primary population
------------------
persistent_relational
persistent_pointwise

Excluded
--------
weak_relational_persistence
weak_pointwise_persistence
unstable_multiscale

Validation
----------
8-fold spatial OOF using buffered_spatial_fold_v01.

Models
------
Random forest only for the primary audit because Script 220 established that
the nonlinear RF architecture materially outperformed logistic regression.

Feature families
----------------
1. physical_only
2. recognition_state_only
3. relational_context_only
4. physical_plus_context
5. recognition_plus_context
6. physical_plus_recognition
7. full_admissible_state

Scientific guardrail
--------------------
This script audits operator-regime identifiability. Recognition-state variables
are admissible only under the interpretation:

"Given the observed state of the recognition system, can the appropriate
transformation family be identified?"

They are not thereby validated as predictors of recognition itself.

No adaptive operator gate is created by this script.
"""

from __future__ import annotations

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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_NAME = (
    "221_oregon_srs_persistence_classification_stability_audit_v01"
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

INPUT_PERSISTENCE = (
    PROCESSED
    / "oregon_srs_relational_persistence_common_population_v01.gpkg"
)

INPUT_ENGINE = (
    PROCESSED
    / "oregon_srs_engine_inputs_v01.gpkg"
)

INPUT_CONTEXT = (
    PROCESSED
    / "oregon_srs_relational_context_v01.gpkg"
)

OUTPUT_GPKG = (
    PROCESSED
    / "oregon_srs_persistence_classification_stability_audit_v01.gpkg"
)

OUTPUT_METRICS = (
    PROCESSED
    / "oregon_srs_persistence_classification_stability_metrics_v01.csv"
)

OUTPUT_FOLD_METRICS = (
    PROCESSED
    / "oregon_srs_persistence_classification_stability_fold_metrics_v01.csv"
)

OUTPUT_FEATURE_AUDIT = (
    PROCESSED
    / "oregon_srs_persistence_classification_feature_audit_v01.csv"
)

OUTPUT_FEATURE_IMPORTANCE = (
    PROCESSED
    / "oregon_srs_persistence_classification_feature_importance_stability_v01.csv"
)

OUTPUT_GROUP_IMPORTANCE = (
    PROCESSED
    / "oregon_srs_persistence_classification_group_importance_by_fold_v01.csv"
)

OUTPUT_TRANSFERABILITY = (
    PROCESSED
    / "oregon_srs_persistence_classification_fold_transferability_v01.csv"
)

OUTPUT_DISAGREEMENT = (
    PROCESSED
    / "oregon_srs_persistence_classification_model_disagreement_v01.csv"
)

OUTPUT_QA = (
    PROCESSED
    / "oregon_srs_persistence_classification_stability_audit_qa_v01.txt"
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
# DECLARED FEATURE SETS
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
# FORBIDDEN / SUSPICIOUS FEATURE AUDIT
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

SUSPICIOUS_PATTERNS = [
    r"fold",
    r"cluster",
    r"region_id",
    r"regime",
    r"class",
    r"label",
    r"oof",
    r"buffer",
]


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


def matches_any_pattern(
    name: str,
    patterns: Sequence[str],
) -> bool:
    lower = name.lower()

    return any(
        re.search(pattern, lower)
        for pattern in patterns
    )


def is_forbidden_feature(name: str) -> bool:
    return matches_any_pattern(
        name,
        FORBIDDEN_PATTERNS,
    )


def is_suspicious_feature(name: str) -> bool:
    return matches_any_pattern(
        name,
        SUSPICIOUS_PATTERNS,
    )


def existing_features(
    df: pd.DataFrame,
    requested: Sequence[str],
) -> List[str]:
    return [
        col
        for col in requested
        if col in df.columns
        and not is_forbidden_feature(col)
    ]


def safe_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(
        roc_auc_score(y_true, y_prob)
    )


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
            f"ID column '{id_col}' missing from context input."
        )

    engine_tab = pd.DataFrame(
        engine.drop(columns="geometry", errors="ignore")
    )

    context_tab = pd.DataFrame(
        context.drop(columns="geometry", errors="ignore")
    )

    engine_add_cols = [
        id_col
    ] + [
        col
        for col in engine_tab.columns
        if col != id_col
        and col not in persistence.columns
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
        if col != id_col
        and col not in merged.columns
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
# CONTEXT FEATURE IDENTIFICATION
# =============================================================================

def infer_relational_context_features(
    df: pd.DataFrame,
    physical: Sequence[str],
    recognition: Sequence[str],
) -> Tuple[List[str], pd.DataFrame]:

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

    audit_rows = []
    selected = []

    for col in df.columns:

        if col == "geometry":
            continue

        context_match = matches_any_pattern(
            col,
            context_patterns,
        )

        forbidden = is_forbidden_feature(col)
        suspicious = is_suspicious_feature(col)

        declared_physical = col in physical
        declared_recognition = col in recognition

        selected_as_context = (
            context_match
            and not forbidden
            and not declared_physical
            and not declared_recognition
        )

        if context_match or forbidden or suspicious:
            audit_rows.append(
                {
                    "feature": col,
                    "context_pattern_match": context_match,
                    "forbidden_pattern_match": forbidden,
                    "suspicious_pattern_match": suspicious,
                    "declared_physical": declared_physical,
                    "declared_recognition": declared_recognition,
                    "selected_as_context": selected_as_context,
                    "dtype": str(df[col].dtype),
                    "missing_n": int(df[col].isna().sum()),
                    "missing_share": float(df[col].isna().mean()),
                    "n_unique": int(df[col].nunique(dropna=True)),
                }
            )

        if selected_as_context:
            selected.append(col)

    audit_df = pd.DataFrame(audit_rows)

    return sorted(set(selected)), audit_df


# =============================================================================
# FEATURE FAMILIES
# =============================================================================

def build_feature_families(
    df: pd.DataFrame,
) -> Tuple[
    Dict[str, List[str]],
    List[str],
    List[str],
    List[str],
    pd.DataFrame,
]:

    physical = existing_features(
        df,
        PHYSICAL_FEATURES_REQUESTED,
    )

    recognition = existing_features(
        df,
        RECOGNITION_FEATURES_REQUESTED,
    )

    context, audit_df = infer_relational_context_features(
        df,
        physical,
        recognition,
    )

    families = {
        "physical_only": sorted(set(physical)),

        "recognition_state_only": sorted(
            set(recognition)
        ),

        "relational_context_only": sorted(
            set(context)
        ),

        "physical_plus_context": sorted(
            set(physical + context)
        ),

        "recognition_plus_context": sorted(
            set(recognition + context)
        ),

        "physical_plus_recognition": sorted(
            set(physical + recognition)
        ),

        "full_admissible_state": sorted(
            set(
                physical
                + recognition
                + context
            )
        ),
    }

    for family_name, features in families.items():

        if not features:
            raise ValueError(
                f"Feature family '{family_name}' is empty."
            )

        forbidden = [
            feature
            for feature in features
            if is_forbidden_feature(feature)
        ]

        if forbidden:
            raise ValueError(
                f"Forbidden features entered {family_name}: "
                f"{forbidden}"
            )

    return (
        families,
        physical,
        recognition,
        context,
        audit_df,
    )


# =============================================================================
# MODEL PIPELINE
# =============================================================================

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


# =============================================================================
# FEATURE GROUP MAPPING
# =============================================================================

def raw_feature_from_transformed(
    transformed_name: str,
    all_raw_features: Sequence[str],
) -> str:

    name = transformed_name

    if "__" in name:
        name = name.split("__", 1)[1]

    exact_matches = [
        feature
        for feature in all_raw_features
        if name == feature
    ]

    if exact_matches:
        return exact_matches[0]

    prefix_matches = sorted(
        [
            feature
            for feature in all_raw_features
            if name.startswith(feature + "_")
        ],
        key=len,
        reverse=True,
    )

    if prefix_matches:
        return prefix_matches[0]

    return name


def assign_feature_group(
    raw_feature: str,
    physical: Sequence[str],
    recognition: Sequence[str],
    context: Sequence[str],
) -> str:

    if raw_feature in physical:
        return "physical"

    if raw_feature in recognition:
        return "recognition_state"

    if raw_feature in context:
        return "relational_context"

    return "other"


# =============================================================================
# OOF EVALUATION
# =============================================================================

def evaluate_family(
    df: pd.DataFrame,
    features: List[str],
    family_name: str,
    fold_col: str,
    target_col: str,
    physical: Sequence[str],
    recognition: Sequence[str],
    context: Sequence[str],
) -> Tuple[
    pd.DataFrame,
    Dict,
    List[Dict],
    pd.DataFrame,
]:

    numeric_features, categorical_features = split_feature_types(
        df,
        features,
    )

    LOGGER.info(
        f"{family_name}: "
        f"{len(features)} features "
        f"({len(numeric_features)} numeric, "
        f"{len(categorical_features)} categorical)"
    )

    pred_col = f"pred_{family_name}"
    prob_col = f"prob_{family_name}"

    predictions = pd.DataFrame(
        index=df.index
    )

    predictions[pred_col] = np.nan
    predictions[prob_col] = np.nan

    fold_metric_rows = []
    importance_rows = []

    folds = sorted(
        df[fold_col].dropna().unique().tolist()
    )

    for fold in folds:

        LOGGER.info(
            f"Running {family_name} / fold {fold}"
        )

        train_mask = df[fold_col] != fold
        test_mask = df[fold_col] == fold

        train = df.loc[train_mask]
        test = df.loc[test_mask]

        X_train = train[features]
        y_train = (
            train[target_col]
            .astype(int)
            .to_numpy()
        )

        X_test = test[features]
        y_test = (
            test[target_col]
            .astype(int)
            .to_numpy()
        )

        if len(np.unique(y_train)) < 2:
            raise ValueError(
                f"Training complement for fold {fold} "
                "contains fewer than two classes."
            )

        model = make_rf_pipeline(
            numeric_features,
            categorical_features,
        )

        model.fit(
            X_train,
            y_train,
        )

        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        predictions.loc[
            test_mask,
            pred_col,
        ] = pred

        predictions.loc[
            test_mask,
            prob_col,
        ] = prob

        fold_metric_rows.append(
            {
                "feature_family": family_name,
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
                "test_positive_rate": float(
                    np.mean(y_test)
                ),
                "accuracy": float(
                    accuracy_score(
                        y_test,
                        pred,
                    )
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(
                        y_test,
                        pred,
                    )
                ),
                "roc_auc": safe_auc(
                    y_test,
                    prob,
                ),
            }
        )

        preprocessor = model.named_steps["preprocessor"]
        rf_model = model.named_steps["model"]

        transformed_names = (
            preprocessor.get_feature_names_out()
        )

        importances = rf_model.feature_importances_

        for transformed_name, importance in zip(
            transformed_names,
            importances,
        ):

            raw_feature = raw_feature_from_transformed(
                transformed_name,
                features,
            )

            feature_group = assign_feature_group(
                raw_feature,
                physical,
                recognition,
                context,
            )

            importance_rows.append(
                {
                    "feature_family": family_name,
                    "fold": fold,
                    "transformed_feature": transformed_name,
                    "raw_feature": raw_feature,
                    "feature_group": feature_group,
                    "importance": float(importance),
                }
            )

    valid = (
        predictions[pred_col].notna()
        & predictions[prob_col].notna()
    )

    y_true = (
        df.loc[valid, target_col]
        .astype(int)
        .to_numpy()
    )

    y_pred = (
        predictions.loc[valid, pred_col]
        .astype(int)
        .to_numpy()
    )

    y_prob = (
        predictions.loc[valid, prob_col]
        .astype(float)
        .to_numpy()
    )

    overall = {
        "feature_family": family_name,
        "n": int(valid.sum()),
        "n_features": len(features),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "roc_auc": safe_auc(
            y_true,
            y_prob,
        ),
    }

    importance_df = pd.DataFrame(
        importance_rows
    )

    return (
        predictions,
        overall,
        fold_metric_rows,
        importance_df,
    )


# =============================================================================
# TRANSFERABILITY SUMMARY
# =============================================================================

def build_transferability_summary(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for family_name, group in fold_metrics.groupby(
        "feature_family"
    ):

        ba = group["balanced_accuracy"]
        auc = group["roc_auc"]

        rows.append(
            {
                "feature_family": family_name,

                "mean_fold_balanced_accuracy": float(
                    ba.mean()
                ),

                "std_fold_balanced_accuracy": float(
                    ba.std(ddof=1)
                ),

                "min_fold_balanced_accuracy": float(
                    ba.min()
                ),

                "max_fold_balanced_accuracy": float(
                    ba.max()
                ),

                "folds_ba_above_0_50": int(
                    (ba > 0.50).sum()
                ),

                "folds_ba_above_0_60": int(
                    (ba >= 0.60).sum()
                ),

                "folds_ba_above_0_70": int(
                    (ba >= 0.70).sum()
                ),

                "mean_fold_auc": float(
                    auc.mean()
                ),

                "std_fold_auc": float(
                    auc.std(ddof=1)
                ),

                "min_fold_auc": float(
                    auc.min()
                ),

                "max_fold_auc": float(
                    auc.max()
                ),

                "folds_auc_above_0_50": int(
                    (auc > 0.50).sum()
                ),

                "folds_auc_above_0_60": int(
                    (auc >= 0.60).sum()
                ),

                "folds_auc_above_0_70": int(
                    (auc >= 0.70).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# MODEL DISAGREEMENT
# =============================================================================

def build_disagreement_summary(
    output_predictions: pd.DataFrame,
    fold_col: str,
) -> pd.DataFrame:

    comparisons = [
        (
            "relational_context_only",
            "full_admissible_state",
        ),
        (
            "physical_plus_context",
            "full_admissible_state",
        ),
        (
            "recognition_plus_context",
            "full_admissible_state",
        ),
        (
            "physical_plus_recognition",
            "full_admissible_state",
        ),
    ]

    rows = []

    for family_a, family_b in comparisons:

        pred_a = f"pred_{family_a}"
        pred_b = f"pred_{family_b}"

        prob_a = f"prob_{family_a}"
        prob_b = f"prob_{family_b}"

        if (
            pred_a not in output_predictions.columns
            or pred_b not in output_predictions.columns
        ):
            continue

        for fold, group in output_predictions.groupby(
            fold_col
        ):

            disagreement = (
                group[pred_a].astype(int)
                != group[pred_b].astype(int)
            )

            probability_difference = (
                group[prob_a]
                - group[prob_b]
            ).abs()

            rows.append(
                {
                    "family_a": family_a,
                    "family_b": family_b,
                    "fold": fold,
                    "n": len(group),
                    "prediction_disagreement_rate": float(
                        disagreement.mean()
                    ),
                    "mean_absolute_probability_difference": float(
                        probability_difference.mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    LOGGER.info(
        "Starting persistence classification stability audit"
    )

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

    model_df = df[
        df[target_regime_col].isin(
            TARGET_REGIMES.keys()
        )
    ].copy()

    model_df["persistence_target_v01"] = (
        model_df[target_regime_col]
        .map(TARGET_REGIMES)
        .astype(int)
    )

    LOGGER.info(
        f"Primary modeling rows: {len(model_df):,}"
    )

    folds = sorted(
        model_df[fold_col]
        .dropna()
        .unique()
        .tolist()
    )

    if len(folds) != 8:
        raise ValueError(
            f"Expected 8 folds; found {len(folds)}: {folds}"
        )

    (
        families,
        physical,
        recognition,
        context,
        feature_audit_df,
    ) = build_feature_families(model_df)

    LOGGER.info(
        f"Physical features: {len(physical)}"
    )

    LOGGER.info(
        f"Recognition features: {len(recognition)}"
    )

    LOGGER.info(
        f"Context features: {len(context)}"
    )

    suspicious_selected = feature_audit_df[
        feature_audit_df["selected_as_context"]
        & feature_audit_df["suspicious_pattern_match"]
    ]

    forbidden_selected = feature_audit_df[
        feature_audit_df["selected_as_context"]
        & feature_audit_df["forbidden_pattern_match"]
    ]

    if len(forbidden_selected) > 0:
        raise ValueError(
            "Forbidden features entered context set:\n"
            + forbidden_selected["feature"].to_string(index=False)
        )

    LOGGER.info(
        f"Suspicious selected context features: "
        f"{len(suspicious_selected)}"
    )

    output_predictions = model_df[
        [
            id_col,
            target_regime_col,
            "persistence_target_v01",
            fold_col,
            "geometry",
        ]
    ].copy()

    metrics_rows = []
    fold_metric_rows = []
    importance_frames = []

    for family_name, features in families.items():

        (
            predictions,
            overall,
            fold_rows,
            importance_df,
        ) = evaluate_family(
            df=model_df,
            features=features,
            family_name=family_name,
            fold_col=fold_col,
            target_col="persistence_target_v01",
            physical=physical,
            recognition=recognition,
            context=context,
        )

        output_predictions = pd.concat(
            [
                output_predictions,
                predictions,
            ],
            axis=1,
        )

        metrics_rows.append(overall)
        fold_metric_rows.extend(fold_rows)
        importance_frames.append(importance_df)

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    fold_metrics_df = pd.DataFrame(
        fold_metric_rows
    )

    importance_raw_df = pd.concat(
        importance_frames,
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Feature importance stability
    # -------------------------------------------------------------------------

    feature_importance_stability = (
        importance_raw_df
        .groupby(
            [
                "feature_family",
                "raw_feature",
                "feature_group",
            ],
            as_index=False,
        )
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            min_importance=("importance", "min"),
            max_importance=("importance", "max"),
            folds_present=("fold", "nunique"),
        )
    )

    feature_importance_stability[
        "importance_cv"
    ] = np.where(
        feature_importance_stability["mean_importance"] > 0,
        (
            feature_importance_stability["std_importance"]
            / feature_importance_stability["mean_importance"]
        ),
        np.nan,
    )

    # -------------------------------------------------------------------------
    # Group importance by fold
    # -------------------------------------------------------------------------

    group_importance_df = (
        importance_raw_df
        .groupby(
            [
                "feature_family",
                "fold",
                "feature_group",
            ],
            as_index=False,
        )
        .agg(
            total_importance=("importance", "sum"),
            n_transformed_features=(
                "transformed_feature",
                "count",
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Transferability
    # -------------------------------------------------------------------------

    transferability_df = build_transferability_summary(
        fold_metrics_df
    )

    # -------------------------------------------------------------------------
    # Disagreement
    # -------------------------------------------------------------------------

    disagreement_df = build_disagreement_summary(
        output_predictions,
        fold_col,
    )

    # -------------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------------

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

    feature_audit_df.to_csv(
        OUTPUT_FEATURE_AUDIT,
        index=False,
    )

    feature_importance_stability.to_csv(
        OUTPUT_FEATURE_IMPORTANCE,
        index=False,
    )

    group_importance_df.to_csv(
        OUTPUT_GROUP_IMPORTANCE,
        index=False,
    )

    transferability_df.to_csv(
        OUTPUT_TRANSFERABILITY,
        index=False,
    )

    disagreement_df.to_csv(
        OUTPUT_DISAGREEMENT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # QA
    # -------------------------------------------------------------------------

    sorted_metrics = metrics_df.sort_values(
        ["roc_auc", "balanced_accuracy"],
        ascending=False,
    )

    best_row = sorted_metrics.iloc[0]

    full_row = metrics_df[
        metrics_df["feature_family"]
        == "full_admissible_state"
    ].iloc[0]

    context_row = metrics_df[
        metrics_df["feature_family"]
        == "relational_context_only"
    ].iloc[0]

    full_fold = fold_metrics_df[
        fold_metrics_df["feature_family"]
        == "full_admissible_state"
    ].sort_values("fold")

    context_fold = fold_metrics_df[
        fold_metrics_df["feature_family"]
        == "relational_context_only"
    ].sort_values("fold")

    top_full_features = (
        feature_importance_stability[
            feature_importance_stability["feature_family"]
            == "full_admissible_state"
        ]
        .sort_values(
            "mean_importance",
            ascending=False,
        )
        .head(20)
    )

    qa_lines = [
        f"Script: {SCRIPT_NAME}",
        "",
        "PURPOSE",
        "Audit stability, transferability, feature dependence, and possible",
        "leakage in prospective persistence-regime classification.",
        "",
        "PRIMARY POPULATION",
        f"Rows: {len(model_df):,}",
        f"Folds: {folds}",
        "",
        "FEATURE COUNTS",
        f"physical: {len(physical)}",
        f"recognition_state: {len(recognition)}",
        f"relational_context: {len(context)}",
        "",
        "SUSPICIOUS CONTEXT AUDIT",
        f"suspicious selected context features: {len(suspicious_selected)}",
        f"forbidden selected context features: {len(forbidden_selected)}",
        "",
        "OVERALL FAMILY METRICS",
        sorted_metrics.to_string(index=False),
        "",
        "BEST FAMILY",
        (
            f"{best_row['feature_family']}: "
            f"BA={best_row['balanced_accuracy']:.6f}, "
            f"AUC={best_row['roc_auc']:.6f}"
        ),
        "",
        "CONTEXT-ONLY VS FULL",
        (
            f"context-only BA={context_row['balanced_accuracy']:.6f}, "
            f"AUC={context_row['roc_auc']:.6f}"
        ),
        (
            f"full-state BA={full_row['balanced_accuracy']:.6f}, "
            f"AUC={full_row['roc_auc']:.6f}"
        ),
        (
            f"full minus context AUC="
            f"{full_row['roc_auc'] - context_row['roc_auc']:.6f}"
        ),
        "",
        "FULL-STATE FOLD METRICS",
        full_fold.to_string(index=False),
        "",
        "CONTEXT-ONLY FOLD METRICS",
        context_fold.to_string(index=False),
        "",
        "TRANSFERABILITY SUMMARY",
        transferability_df.sort_values(
            "mean_fold_auc",
            ascending=False,
        ).to_string(index=False),
        "",
        "TOP 20 FULL-STATE FEATURES BY MEAN RF IMPORTANCE",
        top_full_features.to_string(index=False),
        "",
        "SCIENTIFIC DECISION RULES",
        "1. If context-only approaches full-state performance, operator",
        "   persistence is primarily encoded in relational state structure.",
        "2. If full state materially exceeds context-only, local state and",
        "   relational context jointly determine operator persistence.",
        "3. If a few suspicious features dominate, classification evidence",
        "   must be treated as potentially contaminated until re-tested.",
        "4. If top features are unstable across folds, aggregate performance",
        "   does not justify a universal operator-selection mechanism.",
        "5. If folds 0 and 5 remain failures across context-bearing families,",
        "   they should be studied as transfer-failure regions rather than",
        "   averaged away.",
        "",
        "IMPORTANT",
        "This audit does not create or validate an adaptive SRS operator gate.",
    ]

    OUTPUT_QA.write_text(
        "\n".join(qa_lines),
        encoding="utf-8",
    )

    LOGGER.info("Completed Script 221")

    LOGGER.info("")
    LOGGER.info("Overall metrics:")
    LOGGER.info(
        "\n"
        + sorted_metrics.to_string(index=False)
    )

    LOGGER.info("")
    LOGGER.info("Transferability:")
    LOGGER.info(
        "\n"
        + transferability_df.sort_values(
            "mean_fold_auc",
            ascending=False,
        ).to_string(index=False)
    )

    LOGGER.info("")
    LOGGER.info("Created:")

    for path in [
        OUTPUT_GPKG,
        OUTPUT_METRICS,
        OUTPUT_FOLD_METRICS,
        OUTPUT_FEATURE_AUDIT,
        OUTPUT_FEATURE_IMPORTANCE,
        OUTPUT_GROUP_IMPORTANCE,
        OUTPUT_TRANSFERABILITY,
        OUTPUT_DISAGREEMENT,
        OUTPUT_QA,
    ]:
        LOGGER.info(str(path))


if __name__ == "__main__":
    main()