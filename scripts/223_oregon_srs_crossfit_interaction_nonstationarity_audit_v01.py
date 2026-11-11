#!/usr/bin/env python3
"""
223_oregon_srs_crossfit_interaction_nonstationarity_audit_v01.py

Cross-fitted recognition-context interaction and spatial nonstationarity audit.

Purpose:
Test whether the nonlinear persistence-regime signal can be compressed into an
interpretable recognition-context mismatch functional, or whether operator
persistence remains spatially nonstationary.

This script does NOT build an adaptive/gated SRS engine.
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_NAME = "223_oregon_srs_crossfit_interaction_nonstationarity_audit_v01"

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

INPUT_PERSISTENCE = PROCESSED / "oregon_srs_relational_persistence_common_population_v01.gpkg"
INPUT_ENGINE = PROCESSED / "oregon_srs_engine_inputs_v01.gpkg"
INPUT_CONTEXT = PROCESSED / "oregon_srs_relational_context_v01.gpkg"
INPUT_IMPORTANCE = PROCESSED / "oregon_srs_persistence_classification_feature_importance_stability_v01.csv"

OUTPUT_METRICS = PROCESSED / "oregon_srs_crossfit_interaction_nonstationarity_metrics_v01.csv"
OUTPUT_FOLD_METRICS = PROCESSED / "oregon_srs_crossfit_interaction_nonstationarity_fold_metrics_v01.csv"
OUTPUT_PAIR_SELECTION = PROCESSED / "oregon_srs_crossfit_interaction_pair_selection_v01.csv"
OUTPUT_MISMATCH_FEATURES = PROCESSED / "oregon_srs_crossfit_mismatch_features_v01.csv"
OUTPUT_NONSTATIONARITY = PROCESSED / "oregon_srs_crossfit_nonstationarity_audit_v01.csv"
OUTPUT_QA = PROCESSED / "oregon_srs_crossfit_interaction_nonstationarity_audit_qa_v01.txt"

RANDOM_STATE = 42
TOP_RECOGNITION_N = 8
TOP_CONTEXT_N = 16
TOP_PAIRS_PER_FOLD = 30

TARGET_REGIMES = {
    "persistent_pointwise": 0,
    "persistent_relational": 1,
}

TARGET_COL_CANDIDATES = ["persistence_regime_v01", "persistence_regime"]
FOLD_COL_CANDIDATES = ["buffered_spatial_fold_v01", "spatial_fold_v01", "fold_v01"]
ID_COL_CANDIDATES = ["cell_id", "grid_id", "id"]

FORBIDDEN_PATTERNS = [
    r"gain", r"residual", r"error", r"prediction", r"predicted",
    r"yhat", r"y_hat", r"operator_preference", r"operator_regime",
    r"preferred_operator", r"persistence_regime", r"persistence_score",
    r"relational_win", r"pointwise_win", r"relational_better",
    r"pointwise_better", r"target", r"class_label",
]


def setup_logging() -> logging.Logger:
    logger = logging.getLogger(SCRIPT_NAME)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(f"[{SCRIPT_NAME}] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
    return logger


LOGGER = setup_logging()


def is_forbidden(name: str) -> bool:
    low = name.lower()
    return any(re.search(p, low) for p in FORBIDDEN_PATTERNS)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def find_col(df: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find {label}. Tried: {list(candidates)}")


def safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, prob))


def rank_biserial_binary(values: pd.Series, target: pd.Series) -> float:
    valid = values.notna() & target.notna()
    x = values.loc[valid].astype(float).to_numpy()
    y = target.loc[valid].astype(int).to_numpy()

    x1 = x[y == 1]
    x0 = x[y == 0]
    n1, n0 = len(x1), len(x0)

    if n1 == 0 or n0 == 0:
        return np.nan

    ranks = rankdata(np.concatenate([x1, x0]))
    rank_sum_1 = ranks[:n1].sum()
    u1 = rank_sum_1 - n1 * (n1 + 1) / 2
    return float(2 * u1 / (n1 * n0) - 1)


def zfit(train: pd.DataFrame, cols: Sequence[str]) -> Tuple[pd.Series, pd.Series]:
    mean = train[list(cols)].mean()
    std = train[list(cols)].std(ddof=0).replace(0, 1.0).fillna(1.0)
    return mean, std


def zapply(df: pd.DataFrame, cols: Sequence[str], mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    return (df[list(cols)] - mean[list(cols)]) / std[list(cols)]


def load_inputs() -> gpd.GeoDataFrame:
    for p in [INPUT_PERSISTENCE, INPUT_ENGINE, INPUT_CONTEXT, INPUT_IMPORTANCE]:
        require_file(p)

    persistence = gpd.read_file(INPUT_PERSISTENCE)
    engine = gpd.read_file(INPUT_ENGINE)
    context = gpd.read_file(INPUT_CONTEXT)

    id_col = find_col(persistence, ID_COL_CANDIDATES, "cell id")

    engine_tab = pd.DataFrame(engine.drop(columns="geometry", errors="ignore"))
    context_tab = pd.DataFrame(context.drop(columns="geometry", errors="ignore"))

    engine_cols = [id_col] + [c for c in engine_tab.columns if c != id_col and c not in persistence.columns]
    merged = persistence.merge(engine_tab[engine_cols], on=id_col, how="left", validate="one_to_one")

    context_cols = [id_col] + [c for c in context_tab.columns if c != id_col and c not in merged.columns]
    merged = merged.merge(context_tab[context_cols], on=id_col, how="left", validate="one_to_one")

    return gpd.GeoDataFrame(merged, geometry="geometry", crs=persistence.crs)


def select_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    imp = pd.read_csv(INPUT_IMPORTANCE)
    full = imp[imp["feature_family"] == "full_admissible_state"].copy()

    rec = (
        full[full["feature_group"] == "recognition_state"]
        .sort_values("mean_importance", ascending=False)["raw_feature"]
        .drop_duplicates()
        .tolist()
    )

    ctx = (
        full[full["feature_group"] == "relational_context"]
        .sort_values("mean_importance", ascending=False)["raw_feature"]
        .drop_duplicates()
        .tolist()
    )

    rec = [
        c for c in rec
        if c in df.columns and not is_forbidden(c) and pd.api.types.is_numeric_dtype(df[c])
    ][:TOP_RECOGNITION_N]

    ctx = [
        c for c in ctx
        if c in df.columns and not is_forbidden(c) and pd.api.types.is_numeric_dtype(df[c])
    ][:TOP_CONTEXT_N]

    if not rec or not ctx:
        raise ValueError("No recognition or context features selected.")

    return rec, ctx


def make_logistic() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        )),
    ])


def make_rf() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=600,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])


def select_pairs_in_training(
    train: pd.DataFrame,
    rec_features: Sequence[str],
    ctx_features: Sequence[str],
    target_col: str,
) -> Tuple[List[Tuple[str, str]], pd.DataFrame]:

    all_cols = list(rec_features) + list(ctx_features)
    mean, std = zfit(train, all_cols)
    ztrain = zapply(train, all_cols, mean, std)

    rows = []
    y = train[target_col]

    for r in rec_features:
        for c in ctx_features:
            interaction = ztrain[r] * ztrain[c]
            effect = rank_biserial_binary(interaction, y)
            rows.append({
                "recognition_feature": r,
                "context_feature": c,
                "train_interaction_effect": effect,
                "train_abs_interaction_effect": abs(effect) if pd.notna(effect) else np.nan,
            })

    pair_df = pd.DataFrame(rows).sort_values("train_abs_interaction_effect", ascending=False)
    pairs = list(pair_df.head(TOP_PAIRS_PER_FOLD)[["recognition_feature", "context_feature"]].itertuples(index=False, name=None))
    return pairs, pair_df


def build_designs_for_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    rec_features: Sequence[str],
    ctx_features: Sequence[str],
    selected_pairs: Sequence[Tuple[str, str]],
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    base_cols = list(rec_features) + list(ctx_features)
    mean, std = zfit(train, base_cols)

    ztrain = zapply(train, base_cols, mean, std)
    ztest = zapply(test, base_cols, mean, std)

    designs = {
        "additive_logistic": (ztrain[base_cols].copy(), ztest[base_cols].copy()),
        "nonlinear_random_forest": (ztrain[base_cols].copy(), ztest[base_cols].copy()),
    }

    inter_train = ztrain[base_cols].copy()
    inter_test = ztest[base_cols].copy()

    for i, (r, c) in enumerate(selected_pairs):
        name = f"ix_{i:03d}"
        inter_train[name] = ztrain[r] * ztrain[c]
        inter_test[name] = ztest[r] * ztest[c]

    designs["crossfit_interaction_logistic"] = (inter_train, inter_test)

    mismatch_train = pd.DataFrame(index=train.index)
    mismatch_test = pd.DataFrame(index=test.index)

    # Compact mismatch functionals by radius group.
    for radius in ["10km", "25km"]:
        radius_ctx = [c for c in ctx_features if radius in c]
        if not radius_ctx:
            continue

        ctx_train = ztrain[radius_ctx].mean(axis=1)
        ctx_test = ztest[radius_ctx].mean(axis=1)

        rec_train = ztrain[list(rec_features)].mean(axis=1)
        rec_test = ztest[list(rec_features)].mean(axis=1)

        mismatch_train[f"context_mean_{radius}"] = ctx_train
        mismatch_test[f"context_mean_{radius}"] = ctx_test

        mismatch_train[f"recognition_mean"] = rec_train
        mismatch_test[f"recognition_mean"] = rec_test

        mismatch_train[f"mismatch_context_minus_recognition_{radius}"] = ctx_train - rec_train
        mismatch_test[f"mismatch_context_minus_recognition_{radius}"] = ctx_test - rec_test

        mismatch_train[f"interaction_context_times_recognition_{radius}"] = ctx_train * rec_train
        mismatch_test[f"interaction_context_times_recognition_{radius}"] = ctx_test * rec_test

    mismatch_train = mismatch_train.loc[:, ~mismatch_train.columns.duplicated()]
    mismatch_test = mismatch_test.loc[:, ~mismatch_test.columns.duplicated()]

    designs["compact_mismatch_logistic"] = (mismatch_train, mismatch_test)

    scale_train = pd.DataFrame(index=train.index)
    scale_test = pd.DataFrame(index=test.index)

    for radius in ["10km", "25km"]:
        radius_ctx = [c for c in ctx_features if radius in c]
        if not radius_ctx:
            continue

        for r in rec_features:
            scale_train[f"{r}__ctxmean_{radius}"] = ztrain[r] * ztrain[radius_ctx].mean(axis=1)
            scale_test[f"{r}__ctxmean_{radius}"] = ztest[r] * ztest[radius_ctx].mean(axis=1)

    designs["scale_interaction_logistic"] = (scale_train, scale_test)

    return designs


def main() -> None:
    LOGGER.info("Starting Script 223")

    df = load_inputs()

    target_regime_col = find_col(df, TARGET_COL_CANDIDATES, "target regime")
    fold_col = find_col(df, FOLD_COL_CANDIDATES, "fold")

    model_df = df[df[target_regime_col].isin(TARGET_REGIMES.keys())].copy()
    model_df["persistence_target_v01"] = model_df[target_regime_col].map(TARGET_REGIMES).astype(int)

    rec_features, ctx_features = select_features(model_df)

    LOGGER.info(f"Rows: {len(model_df):,}")
    LOGGER.info(f"Recognition features: {len(rec_features)}")
    LOGGER.info(f"Context features: {len(ctx_features)}")

    folds = sorted(model_df[fold_col].dropna().unique().tolist())

    model_factories = {
        "additive_logistic": make_logistic,
        "crossfit_interaction_logistic": make_logistic,
        "compact_mismatch_logistic": make_logistic,
        "scale_interaction_logistic": make_logistic,
        "nonlinear_random_forest": make_rf,
    }

    fold_rows = []
    pair_rows = []
    mismatch_rows = []
    nonstationarity_rows = []

    pred_store = {m: pd.Series(np.nan, index=model_df.index) for m in model_factories}
    prob_store = {m: pd.Series(np.nan, index=model_df.index) for m in model_factories}

    for fold in folds:
        LOGGER.info(f"Fold {fold}")

        train_mask = model_df[fold_col] != fold
        test_mask = model_df[fold_col] == fold

        train = model_df.loc[train_mask].copy()
        test = model_df.loc[test_mask].copy()

        y_train = train["persistence_target_v01"].astype(int)
        y_test = test["persistence_target_v01"].astype(int)

        selected_pairs, pair_df = select_pairs_in_training(
            train, rec_features, ctx_features, "persistence_target_v01"
        )

        pair_df.insert(0, "heldout_fold", fold)
        pair_rows.append(pair_df)

        designs = build_designs_for_fold(train, test, rec_features, ctx_features, selected_pairs)

        # relationship reversal audit: train-complement effect vs heldout effect
        for feature in rec_features + ctx_features:
            train_effect = rank_biserial_binary(train[feature], y_train)
            test_effect = rank_biserial_binary(test[feature], y_test)
            nonstationarity_rows.append({
                "heldout_fold": fold,
                "feature": feature,
                "feature_group": "recognition_state" if feature in rec_features else "relational_context",
                "train_effect": train_effect,
                "heldout_effect": test_effect,
                "effect_difference": test_effect - train_effect if pd.notna(train_effect) and pd.notna(test_effect) else np.nan,
                "sign_reversal": bool(np.sign(train_effect) != np.sign(test_effect)) if pd.notna(train_effect) and pd.notna(test_effect) else False,
            })

        for model_name, (X_train, X_test) in designs.items():
            model = model_factories[model_name]()
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, 1]

            pred_store[model_name].loc[test_mask] = pred
            prob_store[model_name].loc[test_mask] = prob

            fold_rows.append({
                "model": model_name,
                "fold": fold,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "n_features": X_train.shape[1],
                "accuracy": float(accuracy_score(y_test, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                "roc_auc": safe_auc(y_test.to_numpy(), prob),
            })

            if model_name == "compact_mismatch_logistic":
                for col in X_train.columns:
                    mismatch_rows.append({
                        "heldout_fold": fold,
                        "mismatch_feature": col,
                        "train_mean": float(X_train[col].mean()),
                        "test_mean": float(X_test[col].mean()),
                        "train_std": float(X_train[col].std(ddof=1)),
                        "test_std": float(X_test[col].std(ddof=1)),
                    })

    fold_metrics = pd.DataFrame(fold_rows)

    y_true = model_df["persistence_target_v01"].astype(int).to_numpy()
    overall_rows = []

    for model_name in model_factories:
        pred = pred_store[model_name].astype(int).to_numpy()
        prob = prob_store[model_name].astype(float).to_numpy()

        overall_rows.append({
            "model": model_name,
            "n": len(model_df),
            "accuracy": float(accuracy_score(y_true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "roc_auc": safe_auc(y_true, prob),
            "mean_fold_auc": float(fold_metrics.loc[fold_metrics["model"] == model_name, "roc_auc"].mean()),
            "std_fold_auc": float(fold_metrics.loc[fold_metrics["model"] == model_name, "roc_auc"].std(ddof=1)),
            "min_fold_auc": float(fold_metrics.loc[fold_metrics["model"] == model_name, "roc_auc"].min()),
            "folds_auc_above_0_60": int((fold_metrics.loc[fold_metrics["model"] == model_name, "roc_auc"] >= 0.60).sum()),
        })

    metrics = pd.DataFrame(overall_rows).sort_values("roc_auc", ascending=False)
    pair_selection = pd.concat(pair_rows, ignore_index=True)
    mismatch_features = pd.DataFrame(mismatch_rows)
    nonstationarity = pd.DataFrame(nonstationarity_rows)

    metrics.to_csv(OUTPUT_METRICS, index=False)
    fold_metrics.to_csv(OUTPUT_FOLD_METRICS, index=False)
    pair_selection.to_csv(OUTPUT_PAIR_SELECTION, index=False)
    mismatch_features.to_csv(OUTPUT_MISMATCH_FEATURES, index=False)
    nonstationarity.to_csv(OUTPUT_NONSTATIONARITY, index=False)

    reversal_summary = (
        nonstationarity.groupby(["heldout_fold", "feature_group"], as_index=False)
        .agg(
            n_features=("feature", "count"),
            sign_reversals=("sign_reversal", "sum"),
            mean_abs_effect_difference=("effect_difference", lambda s: float(s.abs().mean())),
        )
    )

    top_selected_pairs = (
        pair_selection.groupby(["recognition_feature", "context_feature"], as_index=False)
        .agg(
            selected_in_folds=("heldout_fold", "nunique"),
            mean_train_abs_effect=("train_abs_interaction_effect", "mean"),
            mean_train_effect=("train_interaction_effect", "mean"),
        )
        .sort_values(["selected_in_folds", "mean_train_abs_effect"], ascending=False)
        .head(30)
    )

    qa = [
        f"Script: {SCRIPT_NAME}",
        "",
        "PURPOSE",
        "Cross-fitted audit of recognition-context interactions, compact mismatch structure, and spatial nonstationarity.",
        "",
        f"Rows: {len(model_df):,}",
        f"Folds: {folds}",
        "",
        "Recognition features:",
        "\n".join(rec_features),
        "",
        "Context features:",
        "\n".join(ctx_features),
        "",
        "OVERALL MODEL METRICS",
        metrics.to_string(index=False),
        "",
        "FOLD MODEL METRICS",
        fold_metrics.sort_values(["model", "fold"]).to_string(index=False),
        "",
        "NONSTATIONARITY / SIGN REVERSAL SUMMARY",
        reversal_summary.to_string(index=False),
        "",
        "TOP CROSS-FITTED SELECTED INTERACTION PAIRS",
        top_selected_pairs.to_string(index=False),
        "",
        "INTERPRETATION RULES",
        "1. If compact_mismatch_logistic approaches nonlinear_random_forest, formalize a compact mismatch functional.",
        "2. If crossfit_interaction_logistic strongly exceeds additive_logistic but remains below RF, pairwise interactions matter but are incomplete.",
        "3. If RF remains clearly strongest and sign reversals are common, persistence is spatially nonstationary.",
        "4. If 10km/25km mismatch features perform differently by fold, do not claim a single interaction horizon.",
        "5. This script still does not validate an adaptive operator gate.",
    ]

    OUTPUT_QA.write_text("\n".join(qa), encoding="utf-8")

    LOGGER.info("Completed Script 223")
    LOGGER.info("\n" + metrics.to_string(index=False))
    LOGGER.info("Created:")
    for p in [OUTPUT_METRICS, OUTPUT_FOLD_METRICS, OUTPUT_PAIR_SELECTION, OUTPUT_MISMATCH_FEATURES, OUTPUT_NONSTATIONARITY, OUTPUT_QA]:
        LOGGER.info(str(p))


if __name__ == "__main__":
    main()