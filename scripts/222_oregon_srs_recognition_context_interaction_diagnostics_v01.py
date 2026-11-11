#!/usr/bin/env python3
"""
222_oregon_srs_recognition_context_interaction_diagnostics_v01.py

Recognition–Context Interaction and Transfer-Failure Diagnostics
-----------------------------------------------------------------

Purpose
-------
Investigate the joint recognition-state + relational-context signal found in
Scripts 220–221.

Primary questions
-----------------
1. Which recognition and context variables differ most between persistence
   regimes within each spatial fold?
2. Which effects are directionally stable versus geographically reversing?
3. Are folds 0 and 5 characterized by recognition-state distribution shift,
   context distribution shift, or recognition-context relationship shift?
4. Does nonlinear interaction structure add predictive information beyond
   additive recognition + context effects?
5. Which recognition-context variable pairs show the strongest fold-stable
   interaction associations?

This is a diagnostic script.

It does NOT:
- create an adaptive operator gate;
- claim a universal interaction law;
- define a final interaction horizon;
- use gain, residual, operator prediction, or persistence-derived variables
  as predictors.

Target
------
1 = persistent_relational
0 = persistent_pointwise

Population
----------
persistent_relational and persistent_pointwise only.

Validation
----------
Existing 8 buffered spatial folds.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from scipy.stats import rankdata

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_NAME = (
    "222_oregon_srs_recognition_context_interaction_diagnostics_v01"
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

INPUT_IMPORTANCE = (
    PROCESSED
    / "oregon_srs_persistence_classification_feature_importance_stability_v01.csv"
)

OUTPUT_FOLD_EFFECTS = (
    PROCESSED
    / "oregon_srs_recognition_context_fold_effects_v01.csv"
)

OUTPUT_EFFECT_STABILITY = (
    PROCESSED
    / "oregon_srs_recognition_context_effect_stability_v01.csv"
)

OUTPUT_DISTRIBUTION_SHIFT = (
    PROCESSED
    / "oregon_srs_recognition_context_distribution_shift_v01.csv"
)

OUTPUT_PAIR_INTERACTIONS = (
    PROCESSED
    / "oregon_srs_recognition_context_pair_interactions_v01.csv"
)

OUTPUT_INTERACTION_STABILITY = (
    PROCESSED
    / "oregon_srs_recognition_context_interaction_stability_v01.csv"
)

OUTPUT_MODEL_METRICS = (
    PROCESSED
    / "oregon_srs_recognition_context_interaction_model_metrics_v01.csv"
)

OUTPUT_MODEL_FOLD_METRICS = (
    PROCESSED
    / "oregon_srs_recognition_context_interaction_model_fold_metrics_v01.csv"
)

OUTPUT_QA = (
    PROCESSED
    / "oregon_srs_recognition_context_interaction_diagnostics_qa_v01.txt"
)

RANDOM_STATE = 42

TOP_RECOGNITION_N = 10
TOP_CONTEXT_N = 20
TOP_INTERACTION_PAIRS = 40

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
# DECLARED RECOGNITION FEATURES
# =============================================================================

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
# SAFETY FILTERS
# =============================================================================

FORBIDDEN_PATTERNS = [
    r"gain",
    r"residual",
    r"error",
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


def is_forbidden(name: str) -> bool:
    lower = name.lower()

    return any(
        re.search(pattern, lower)
        for pattern in FORBIDDEN_PATTERNS
    )


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
# BASIC UTILITIES
# =============================================================================

def require_file(path: Path) -> None:

    if not path.exists():
        raise FileNotFoundError(
            f"Required input not found: {path}"
        )


def find_first_existing_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    label: str,
) -> str:

    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        f"Could not identify {label}. Tried {list(candidates)}"
    )


def safe_auc(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> float:

    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y_true,
            probability,
        )
    )


def rank_biserial_binary(
    values: pd.Series,
    target: pd.Series,
) -> float:
    """
    Rank-biserial correlation using Mann–Whitney U relationship.

    Positive:
    larger values tend to occur in persistent_relational cells.

    Negative:
    larger values tend to occur in persistent_pointwise cells.
    """

    valid = values.notna() & target.notna()

    x = values.loc[valid].astype(float).to_numpy()
    y = target.loc[valid].astype(int).to_numpy()

    x1 = x[y == 1]
    x0 = x[y == 0]

    n1 = len(x1)
    n0 = len(x0)

    if n1 == 0 or n0 == 0:
        return np.nan

    combined = np.concatenate([x1, x0])
    ranks = rankdata(combined)

    rank_sum_1 = ranks[:n1].sum()

    u1 = (
        rank_sum_1
        - n1 * (n1 + 1) / 2
    )

    return float(
        2 * u1 / (n1 * n0) - 1
    )


def standardized_mean_difference(
    a: pd.Series,
    b: pd.Series,
) -> float:

    a = a.dropna().astype(float)
    b = b.dropna().astype(float)

    if len(a) < 2 or len(b) < 2:
        return np.nan

    pooled = np.sqrt(
        (
            a.var(ddof=1)
            + b.var(ddof=1)
        )
        / 2
    )

    if pooled == 0 or np.isnan(pooled):
        return 0.0

    return float(
        (a.mean() - b.mean()) / pooled
    )


# =============================================================================
# DATA ASSEMBLY
# =============================================================================

def load_and_merge_inputs() -> gpd.GeoDataFrame:

    for path in [
        INPUT_PERSISTENCE,
        INPUT_ENGINE,
        INPUT_CONTEXT,
        INPUT_IMPORTANCE,
    ]:
        require_file(path)

    LOGGER.info("Reading persistence population")
    persistence = gpd.read_file(
        INPUT_PERSISTENCE
    )

    LOGGER.info("Reading engine inputs")
    engine = gpd.read_file(
        INPUT_ENGINE
    )

    LOGGER.info("Reading relational context")
    context = gpd.read_file(
        INPUT_CONTEXT
    )

    id_col = find_first_existing_column(
        persistence,
        ID_COL_CANDIDATES,
        "cell ID",
    )

    engine_tab = pd.DataFrame(
        engine.drop(
            columns="geometry",
            errors="ignore",
        )
    )

    context_tab = pd.DataFrame(
        context.drop(
            columns="geometry",
            errors="ignore",
        )
    )

    engine_cols = [
        id_col
    ] + [
        col
        for col in engine_tab.columns
        if col != id_col
        and col not in persistence.columns
    ]

    merged = persistence.merge(
        engine_tab[engine_cols],
        on=id_col,
        how="left",
        validate="one_to_one",
    )

    context_cols = [
        id_col
    ] + [
        col
        for col in context_tab.columns
        if col != id_col
        and col not in merged.columns
    ]

    merged = merged.merge(
        context_tab[context_cols],
        on=id_col,
        how="left",
        validate="one_to_one",
    )

    return gpd.GeoDataFrame(
        merged,
        geometry="geometry",
        crs=persistence.crs,
    )


# =============================================================================
# FEATURE SELECTION FROM SCRIPT 221 IMPORTANCE
# =============================================================================

def select_diagnostic_features(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:

    importance = pd.read_csv(
        INPUT_IMPORTANCE
    )

    full = importance[
        importance["feature_family"]
        == "full_admissible_state"
    ].copy()

    recognition = (
        full[
            full["feature_group"]
            == "recognition_state"
        ]
        .sort_values(
            "mean_importance",
            ascending=False,
        )
        ["raw_feature"]
        .drop_duplicates()
        .tolist()
    )

    context = (
        full[
            full["feature_group"]
            == "relational_context"
        ]
        .sort_values(
            "mean_importance",
            ascending=False,
        )
        ["raw_feature"]
        .drop_duplicates()
        .tolist()
    )

    recognition = [
        feature
        for feature in recognition
        if feature in df.columns
        and not is_forbidden(feature)
        and pd.api.types.is_numeric_dtype(df[feature])
    ][:TOP_RECOGNITION_N]

    context = [
        feature
        for feature in context
        if feature in df.columns
        and not is_forbidden(feature)
        and pd.api.types.is_numeric_dtype(df[feature])
    ][:TOP_CONTEXT_N]

    if not recognition:
        raise ValueError(
            "No recognition diagnostic features selected."
        )

    if not context:
        raise ValueError(
            "No context diagnostic features selected."
        )

    return recognition, context


# =============================================================================
# WITHIN-FOLD EFFECTS
# =============================================================================

def compute_fold_effects(
    df: pd.DataFrame,
    features: Sequence[str],
    fold_col: str,
    target_col: str,
    feature_groups: Dict[str, str],
) -> pd.DataFrame:

    rows = []

    folds = sorted(
        df[fold_col]
        .dropna()
        .unique()
        .tolist()
    )

    for fold in folds:

        fold_df = df[
            df[fold_col] == fold
        ]

        for feature in features:

            effect = rank_biserial_binary(
                fold_df[feature],
                fold_df[target_col],
            )

            relational_values = fold_df.loc[
                fold_df[target_col] == 1,
                feature,
            ]

            pointwise_values = fold_df.loc[
                fold_df[target_col] == 0,
                feature,
            ]

            rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "feature_group": feature_groups[feature],
                    "n": int(
                        fold_df[feature].notna().sum()
                    ),
                    "rank_biserial": effect,
                    "abs_rank_biserial": (
                        abs(effect)
                        if pd.notna(effect)
                        else np.nan
                    ),
                    "relational_mean": float(
                        relational_values.mean()
                    ),
                    "pointwise_mean": float(
                        pointwise_values.mean()
                    ),
                    "mean_difference": float(
                        relational_values.mean()
                        - pointwise_values.mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


def summarize_effect_stability(
    fold_effects: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        feature,
        feature_group,
    ), group in fold_effects.groupby(
        [
            "feature",
            "feature_group",
        ]
    ):

        effects = group[
            "rank_biserial"
        ].dropna()

        positive = int(
            (effects > 0).sum()
        )

        negative = int(
            (effects < 0).sum()
        )

        dominant_direction_count = max(
            positive,
            negative,
        )

        rows.append(
            {
                "feature": feature,
                "feature_group": feature_group,
                "mean_rank_biserial": float(
                    effects.mean()
                ),
                "mean_abs_rank_biserial": float(
                    effects.abs().mean()
                ),
                "std_rank_biserial": float(
                    effects.std(ddof=1)
                ),
                "min_rank_biserial": float(
                    effects.min()
                ),
                "max_rank_biserial": float(
                    effects.max()
                ),
                "positive_folds": positive,
                "negative_folds": negative,
                "dominant_direction_folds": dominant_direction_count,
                "direction_consistency": float(
                    dominant_direction_count
                    / len(effects)
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# DISTRIBUTION SHIFT
# =============================================================================

def compute_distribution_shift(
    df: pd.DataFrame,
    features: Sequence[str],
    fold_col: str,
    feature_groups: Dict[str, str],
) -> pd.DataFrame:

    rows = []

    folds = sorted(
        df[fold_col]
        .dropna()
        .unique()
        .tolist()
    )

    for fold in folds:

        test = df[
            df[fold_col] == fold
        ]

        train = df[
            df[fold_col] != fold
        ]

        for feature in features:

            smd = standardized_mean_difference(
                test[feature],
                train[feature],
            )

            rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "feature_group": feature_groups[feature],
                    "test_mean": float(
                        test[feature].mean()
                    ),
                    "train_mean": float(
                        train[feature].mean()
                    ),
                    "test_train_smd": smd,
                    "abs_test_train_smd": (
                        abs(smd)
                        if pd.notna(smd)
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# PAIRWISE INTERACTION DIAGNOSTICS
# =============================================================================

def zscore_global(
    series: pd.Series,
) -> pd.Series:

    mean = series.mean()
    std = series.std(ddof=0)

    if std == 0 or pd.isna(std):
        return pd.Series(
            np.zeros(len(series)),
            index=series.index,
        )

    return (
        series - mean
    ) / std


def compute_pair_interactions(
    df: pd.DataFrame,
    recognition_features: Sequence[str],
    context_features: Sequence[str],
    fold_col: str,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    standardized = {}

    for feature in (
        list(recognition_features)
        + list(context_features)
    ):
        standardized[feature] = zscore_global(
            df[feature].astype(float)
        )

    rows = []

    folds = sorted(
        df[fold_col]
        .dropna()
        .unique()
        .tolist()
    )

    for recognition_feature in recognition_features:

        for context_feature in context_features:

            interaction = (
                standardized[recognition_feature]
                * standardized[context_feature]
            )

            for fold in folds:

                mask = (
                    df[fold_col] == fold
                )

                effect = rank_biserial_binary(
                    interaction.loc[mask],
                    df.loc[mask, target_col],
                )

                rows.append(
                    {
                        "recognition_feature": recognition_feature,
                        "context_feature": context_feature,
                        "fold": fold,
                        "interaction_rank_biserial": effect,
                        "abs_interaction_rank_biserial": (
                            abs(effect)
                            if pd.notna(effect)
                            else np.nan
                        ),
                    }
                )

    pair_df = pd.DataFrame(rows)

    summary_rows = []

    for (
        recognition_feature,
        context_feature,
    ), group in pair_df.groupby(
        [
            "recognition_feature",
            "context_feature",
        ]
    ):

        effects = group[
            "interaction_rank_biserial"
        ].dropna()

        positive = int(
            (effects > 0).sum()
        )

        negative = int(
            (effects < 0).sum()
        )

        dominant = max(
            positive,
            negative,
        )

        summary_rows.append(
            {
                "recognition_feature": recognition_feature,
                "context_feature": context_feature,
                "mean_interaction_effect": float(
                    effects.mean()
                ),
                "mean_abs_interaction_effect": float(
                    effects.abs().mean()
                ),
                "std_interaction_effect": float(
                    effects.std(ddof=1)
                ),
                "positive_folds": positive,
                "negative_folds": negative,
                "dominant_direction_folds": dominant,
                "direction_consistency": float(
                    dominant / len(effects)
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df = summary_df.sort_values(
        [
            "direction_consistency",
            "mean_abs_interaction_effect",
        ],
        ascending=[
            False,
            False,
        ],
    )

    return pair_df, summary_df


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def make_additive_logistic() -> Pipeline:

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_rf() -> Pipeline:

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=600,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_models(
    df: pd.DataFrame,
    recognition_features: Sequence[str],
    context_features: Sequence[str],
    interaction_summary: pd.DataFrame,
    fold_col: str,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    base_features = (
        list(recognition_features)
        + list(context_features)
    )

    top_pairs = (
        interaction_summary
        .head(TOP_INTERACTION_PAIRS)
        [
            [
                "recognition_feature",
                "context_feature",
            ]
        ]
        .to_records(index=False)
        .tolist()
    )

    interaction_columns = []

    model_df = df.copy()

    for i, (
        recognition_feature,
        context_feature,
    ) in enumerate(top_pairs):

        name = f"interaction_{i:03d}"

        rec_z = zscore_global(
            model_df[recognition_feature].astype(float)
        )

        ctx_z = zscore_global(
            model_df[context_feature].astype(float)
        )

        model_df[name] = (
            rec_z * ctx_z
        )

        interaction_columns.append(name)

    model_specs = {
        "additive_logistic": {
            "features": base_features,
            "factory": make_additive_logistic,
        },
        "explicit_interaction_logistic": {
            "features": (
                base_features
                + interaction_columns
            ),
            "factory": make_additive_logistic,
        },
        "nonlinear_random_forest": {
            "features": base_features,
            "factory": make_rf,
        },
    }

    fold_rows = []

    prediction_store = {
        name: {
            "pred": pd.Series(
                np.nan,
                index=model_df.index,
            ),
            "prob": pd.Series(
                np.nan,
                index=model_df.index,
            ),
        }
        for name in model_specs
    }

    folds = sorted(
        model_df[fold_col]
        .dropna()
        .unique()
        .tolist()
    )

    for model_name, spec in model_specs.items():

        LOGGER.info(
            f"Evaluating model: {model_name}"
        )

        features = spec["features"]

        for fold in folds:

            LOGGER.info(
                f"{model_name} / fold {fold}"
            )

            train_mask = (
                model_df[fold_col] != fold
            )

            test_mask = (
                model_df[fold_col] == fold
            )

            X_train = model_df.loc[
                train_mask,
                features,
            ]

            y_train = model_df.loc[
                train_mask,
                target_col,
            ].astype(int)

            X_test = model_df.loc[
                test_mask,
                features,
            ]

            y_test = model_df.loc[
                test_mask,
                target_col,
            ].astype(int)

            model = spec["factory"]()

            model.fit(
                X_train,
                y_train,
            )

            pred = model.predict(
                X_test
            )

            prob = model.predict_proba(
                X_test
            )[:, 1]

            prediction_store[
                model_name
            ]["pred"].loc[
                test_mask
            ] = pred

            prediction_store[
                model_name
            ]["prob"].loc[
                test_mask
            ] = prob

            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "n_test": int(
                        test_mask.sum()
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
                        y_test.to_numpy(),
                        prob,
                    ),
                }
            )

    overall_rows = []

    y_true = model_df[
        target_col
    ].astype(int).to_numpy()

    for model_name in model_specs:

        pred = (
            prediction_store[model_name]["pred"]
            .astype(int)
            .to_numpy()
        )

        prob = (
            prediction_store[model_name]["prob"]
            .astype(float)
            .to_numpy()
        )

        overall_rows.append(
            {
                "model": model_name,
                "n": len(model_df),
                "n_features": len(
                    model_specs[model_name]["features"]
                ),
                "accuracy": float(
                    accuracy_score(
                        y_true,
                        pred,
                    )
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(
                        y_true,
                        pred,
                    )
                ),
                "roc_auc": safe_auc(
                    y_true,
                    prob,
                ),
            }
        )

    return (
        pd.DataFrame(overall_rows),
        pd.DataFrame(fold_rows),
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    LOGGER.info(
        "Starting recognition-context interaction diagnostics"
    )

    df = load_and_merge_inputs()

    target_regime_col = find_first_existing_column(
        df,
        TARGET_COL_CANDIDATES,
        "persistence regime",
    )

    fold_col = find_first_existing_column(
        df,
        FOLD_COL_CANDIDATES,
        "spatial fold",
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
        f"Primary rows: {len(model_df):,}"
    )

    recognition_features, context_features = (
        select_diagnostic_features(
            model_df
        )
    )

    LOGGER.info(
        f"Selected recognition features: "
        f"{len(recognition_features)}"
    )

    LOGGER.info(
        f"Selected context features: "
        f"{len(context_features)}"
    )

    all_features = (
        recognition_features
        + context_features
    )

    feature_groups = {
        feature: "recognition_state"
        for feature in recognition_features
    }

    feature_groups.update(
        {
            feature: "relational_context"
            for feature in context_features
        }
    )

    # -------------------------------------------------------------------------
    # Fold effects
    # -------------------------------------------------------------------------

    LOGGER.info(
        "Computing within-fold regime effects"
    )

    fold_effects = compute_fold_effects(
        model_df,
        all_features,
        fold_col,
        "persistence_target_v01",
        feature_groups,
    )

    effect_stability = summarize_effect_stability(
        fold_effects
    )

    # -------------------------------------------------------------------------
    # Distribution shift
    # -------------------------------------------------------------------------

    LOGGER.info(
        "Computing train-test distribution shift"
    )

    distribution_shift = compute_distribution_shift(
        model_df,
        all_features,
        fold_col,
        feature_groups,
    )

    # -------------------------------------------------------------------------
    # Interaction diagnostics
    # -------------------------------------------------------------------------

    LOGGER.info(
        "Computing recognition-context pair interactions"
    )

    (
        pair_interactions,
        interaction_stability,
    ) = compute_pair_interactions(
        model_df,
        recognition_features,
        context_features,
        fold_col,
        "persistence_target_v01",
    )

    # -------------------------------------------------------------------------
    # Model comparison
    # -------------------------------------------------------------------------

    LOGGER.info(
        "Evaluating additive, explicit-interaction, and nonlinear models"
    )

    (
        model_metrics,
        model_fold_metrics,
    ) = evaluate_models(
        model_df,
        recognition_features,
        context_features,
        interaction_stability,
        fold_col,
        "persistence_target_v01",
    )

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------

    fold_effects.to_csv(
        OUTPUT_FOLD_EFFECTS,
        index=False,
    )

    effect_stability.to_csv(
        OUTPUT_EFFECT_STABILITY,
        index=False,
    )

    distribution_shift.to_csv(
        OUTPUT_DISTRIBUTION_SHIFT,
        index=False,
    )

    pair_interactions.to_csv(
        OUTPUT_PAIR_INTERACTIONS,
        index=False,
    )

    interaction_stability.to_csv(
        OUTPUT_INTERACTION_STABILITY,
        index=False,
    )

    model_metrics.to_csv(
        OUTPUT_MODEL_METRICS,
        index=False,
    )

    model_fold_metrics.to_csv(
        OUTPUT_MODEL_FOLD_METRICS,
        index=False,
    )

    # -------------------------------------------------------------------------
    # QA summaries
    # -------------------------------------------------------------------------

    top_stable_effects = (
        effect_stability
        .sort_values(
            [
                "direction_consistency",
                "mean_abs_rank_biserial",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(20)
    )

    top_interactions = (
        interaction_stability
        .head(25)
    )

    shift_summary = (
        distribution_shift
        .groupby(
            [
                "fold",
                "feature_group",
            ],
            as_index=False,
        )
        .agg(
            mean_abs_smd=(
                "abs_test_train_smd",
                "mean",
            ),
            max_abs_smd=(
                "abs_test_train_smd",
                "max",
            ),
        )
    )

    qa_lines = [
        f"Script: {SCRIPT_NAME}",
        "",
        "PURPOSE",
        "Diagnose recognition-context interaction structure and spatial",
        "transfer failures in prospective persistence classification.",
        "",
        "PRIMARY POPULATION",
        f"rows: {len(model_df):,}",
        "",
        "SELECTED RECOGNITION FEATURES",
        "\n".join(recognition_features),
        "",
        "SELECTED CONTEXT FEATURES",
        "\n".join(context_features),
        "",
        "MODEL COMPARISON",
        model_metrics.sort_values(
            "roc_auc",
            ascending=False,
        ).to_string(index=False),
        "",
        "MODEL FOLD METRICS",
        model_fold_metrics.sort_values(
            [
                "model",
                "fold",
            ]
        ).to_string(index=False),
        "",
        "DISTRIBUTION SHIFT BY FOLD AND GROUP",
        shift_summary.to_string(index=False),
        "",
        "TOP DIRECTIONALLY STABLE WITHIN-FOLD EFFECTS",
        top_stable_effects.to_string(index=False),
        "",
        "TOP RECOGNITION-CONTEXT INTERACTIONS",
        top_interactions.to_string(index=False),
        "",
        "INTERPRETATION RULES",
        "1. If explicit-interaction logistic materially exceeds additive",
        "   logistic, measured recognition-context interactions carry",
        "   prospective classification information.",
        "2. If RF materially exceeds both logistic models, the joint-state",
        "   relationship is nonlinear or structurally more complex than the",
        "   tested pairwise interactions.",
        "3. Large fold-specific distribution shifts support covariate-shift",
        "   explanations for transfer failure.",
        "4. Direction reversals within folds indicate spatial nonstationarity",
        "   and argue against a universal operator-selection rule.",
        "5. Stable interaction effects are candidates for later mathematical",
        "   formalization, not proof of a universal interaction law.",
        "",
        "IMPORTANT",
        "No adaptive operator gate is created or validated here.",
    ]

    OUTPUT_QA.write_text(
        "\n".join(qa_lines),
        encoding="utf-8",
    )

    LOGGER.info(
        "Completed Script 222"
    )

    LOGGER.info("")
    LOGGER.info("Model comparison:")
    LOGGER.info(
        "\n"
        + model_metrics.sort_values(
            "roc_auc",
            ascending=False,
        ).to_string(index=False)
    )

    LOGGER.info("")
    LOGGER.info("Created:")

    for path in [
        OUTPUT_FOLD_EFFECTS,
        OUTPUT_EFFECT_STABILITY,
        OUTPUT_DISTRIBUTION_SHIFT,
        OUTPUT_PAIR_INTERACTIONS,
        OUTPUT_INTERACTION_STABILITY,
        OUTPUT_MODEL_METRICS,
        OUTPUT_MODEL_FOLD_METRICS,
        OUTPUT_QA,
    ]:
        LOGGER.info(str(path))


if __name__ == "__main__":
    main()