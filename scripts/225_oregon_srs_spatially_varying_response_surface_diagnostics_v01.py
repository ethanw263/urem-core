#!/usr/bin/env python3
"""
225_oregon_srs_spatially_varying_response_surface_diagnostics_v01.py

Diagnoses where coordinate-conditioned RF improves over global RF.

No adaptive SRS gate is built here.
"""

from pathlib import Path
import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline


SCRIPT = "225_oregon_srs_spatially_varying_response_surface_diagnostics_v01"

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"

INPUT_PERSISTENCE = P / "oregon_srs_relational_persistence_common_population_v01.gpkg"
INPUT_ENGINE = P / "oregon_srs_engine_inputs_v01.gpkg"
INPUT_CONTEXT = P / "oregon_srs_relational_context_v01.gpkg"
INPUT_IMPORTANCE = P / "oregon_srs_persistence_classification_feature_importance_stability_v01.csv"

OUT_GPKG = P / "oregon_srs_spatially_varying_response_surface_diagnostics_v01.gpkg"
OUT_METRICS = P / "oregon_srs_spatially_varying_response_surface_metrics_v01.csv"
OUT_FOLD_METRICS = P / "oregon_srs_spatially_varying_response_surface_fold_metrics_v01.csv"
OUT_IMPROVEMENT = P / "oregon_srs_spatially_varying_response_surface_improvement_summary_v01.csv"
OUT_QA = P / "oregon_srs_spatially_varying_response_surface_diagnostics_qa_v01.txt"

TARGET = {
    "persistent_pointwise": 0,
    "persistent_relational": 1,
}

N_REC = 8
N_CTX = 16
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT)


def auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def find_col(df, opts):
    for c in opts:
        if c in df.columns:
            return c
    raise KeyError(opts)


def load():
    base = gpd.read_file(INPUT_PERSISTENCE)
    eng = gpd.read_file(INPUT_ENGINE).drop(columns="geometry", errors="ignore")
    ctx = gpd.read_file(INPUT_CONTEXT).drop(columns="geometry", errors="ignore")

    idc = find_col(base, ["cell_id", "grid_id", "id"])

    eng_cols = [idc] + [c for c in eng.columns if c != idc and c not in base.columns]
    df = base.merge(eng[eng_cols], on=idc, how="left", validate="one_to_one")

    ctx_cols = [idc] + [c for c in ctx.columns if c != idc and c not in df.columns]
    df = df.merge(ctx[ctx_cols], on=idc, how="left", validate="one_to_one")

    df = df.loc[:, ~df.columns.duplicated()].copy()

    return gpd.GeoDataFrame(df, geometry="geometry", crs=base.crs)


def add_xy(df):
    cent = df.geometry.centroid
    df["x_coord"] = cent.x
    df["y_coord"] = cent.y
    return df


def select_features(df):
    imp = pd.read_csv(INPUT_IMPORTANCE)
    imp = imp[imp["feature_family"] == "full_admissible_state"].copy()

    rec = (
        imp[imp["feature_group"] == "recognition_state"]
        .sort_values("mean_importance", ascending=False)["raw_feature"]
        .drop_duplicates()
        .tolist()
    )

    ctx = (
        imp[imp["feature_group"] == "relational_context"]
        .sort_values("mean_importance", ascending=False)["raw_feature"]
        .drop_duplicates()
        .tolist()
    )

    rec = [c for c in rec if c in df.columns and pd.api.types.is_numeric_dtype(df[c])][:N_REC]
    ctx = [c for c in ctx if c in df.columns and pd.api.types.is_numeric_dtype(df[c])][:N_CTX]

    return rec, ctx


def make_rf():
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


def evaluate_models(df, fold_col, target_col, regime_col, base_features):
    models = {
        "global_rf": base_features,
        "coordinate_rf": base_features + ["x_coord", "y_coord"],
    }

    folds = sorted(df[fold_col].dropna().unique())

    pred_store = {m: pd.Series(np.nan, index=df.index) for m in models}
    prob_store = {m: pd.Series(np.nan, index=df.index) for m in models}

    fold_rows = []

    for fold in folds:
        log.info(f"Fold {fold}")

        train_mask = df[fold_col] != fold
        test_mask = df[fold_col] == fold

        train = df.loc[train_mask]
        test = df.loc[test_mask]

        y_train = train[target_col].astype(int)
        y_test = test[target_col].astype(int)

        for model_name, features in models.items():
            model = make_rf()
            model.fit(train[features], y_train)

            pred = model.predict(test[features])
            prob = model.predict_proba(test[features])[:, 1]

            pred_store[model_name].loc[test_mask] = pred
            prob_store[model_name].loc[test_mask] = prob

            fold_rows.append({
                "model": model_name,
                "fold": fold,
                "n_test": int(test_mask.sum()),
                "accuracy": float(accuracy_score(y_test, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                "roc_auc": auc(y_test.to_numpy(), prob),
            })

    fold_metrics = pd.DataFrame(fold_rows)

    y = df[target_col].astype(int).to_numpy()
    metrics = []

    for model_name in models:
        pred = pred_store[model_name].astype(int).to_numpy()
        prob = prob_store[model_name].astype(float).to_numpy()

        sub = fold_metrics[fold_metrics["model"] == model_name]

        metrics.append({
            "model": model_name,
            "n": len(df),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "roc_auc": auc(y, prob),
            "mean_fold_auc": float(sub["roc_auc"].mean()),
            "std_fold_auc": float(sub["roc_auc"].std(ddof=1)),
            "min_fold_auc": float(sub["roc_auc"].min()),
            "folds_auc_above_0_60": int((sub["roc_auc"] >= 0.60).sum()),
        })

    out_cols = ["geometry", regime_col, fold_col, target_col]
    out_cols = list(dict.fromkeys(out_cols))

    out = df[out_cols].copy()
    out = out.loc[:, ~out.columns.duplicated()].copy()

    if target_col not in out.columns:
        out[target_col] = df[target_col].astype(int)

    for model_name in models:
        out[f"pred_{model_name}"] = pred_store[model_name].astype(int).to_numpy()
        out[f"prob_{model_name}"] = prob_store[model_name].astype(float).to_numpy()

        out[f"abs_error_{model_name}"] = (
            out[target_col].astype(float).to_numpy()
            - out[f"prob_{model_name}"].astype(float).to_numpy()
        )

        out[f"abs_error_{model_name}"] = out[f"abs_error_{model_name}"].abs()

    out["coord_minus_global_prob"] = (
        out["prob_coordinate_rf"].astype(float).to_numpy()
        - out["prob_global_rf"].astype(float).to_numpy()
    )

    out["global_abs_error_minus_coord_abs_error"] = (
        out["abs_error_global_rf"].astype(float).to_numpy()
        - out["abs_error_coordinate_rf"].astype(float).to_numpy()
    )

    out["coordinate_better"] = out["global_abs_error_minus_coord_abs_error"] > 0
    out["coordinate_worse"] = out["global_abs_error_minus_coord_abs_error"] < 0

    out["both_correct"] = (
        (out["pred_global_rf"].astype(int).to_numpy() == out[target_col].astype(int).to_numpy())
        & (out["pred_coordinate_rf"].astype(int).to_numpy() == out[target_col].astype(int).to_numpy())
    )

    out["coordinate_only_correct"] = (
        (out["pred_global_rf"].astype(int).to_numpy() != out[target_col].astype(int).to_numpy())
        & (out["pred_coordinate_rf"].astype(int).to_numpy() == out[target_col].astype(int).to_numpy())
    )

    out["global_only_correct"] = (
        (out["pred_global_rf"].astype(int).to_numpy() == out[target_col].astype(int).to_numpy())
        & (out["pred_coordinate_rf"].astype(int).to_numpy() != out[target_col].astype(int).to_numpy())
    )

    out["both_wrong"] = (
        (out["pred_global_rf"].astype(int).to_numpy() != out[target_col].astype(int).to_numpy())
        & (out["pred_coordinate_rf"].astype(int).to_numpy() != out[target_col].astype(int).to_numpy())
    )

    return pd.DataFrame(metrics), fold_metrics, gpd.GeoDataFrame(out, geometry="geometry", crs=df.crs)


def summarize_improvement(out, fold_col):
    rows = []

    for fold, g in out.groupby(fold_col):
        rows.append({
            "fold": fold,
            "n": len(g),
            "mean_global_abs_error": float(g["abs_error_global_rf"].mean()),
            "mean_coordinate_abs_error": float(g["abs_error_coordinate_rf"].mean()),
            "mean_error_improvement": float(g["global_abs_error_minus_coord_abs_error"].mean()),
            "median_error_improvement": float(g["global_abs_error_minus_coord_abs_error"].median()),
            "coordinate_better_share": float(g["coordinate_better"].mean()),
            "coordinate_worse_share": float(g["coordinate_worse"].mean()),
            "coordinate_only_correct_share": float(g["coordinate_only_correct"].mean()),
            "global_only_correct_share": float(g["global_only_correct"].mean()),
            "both_correct_share": float(g["both_correct"].mean()),
            "both_wrong_share": float(g["both_wrong"].mean()),
        })

    rows.append({
        "fold": "ALL",
        "n": len(out),
        "mean_global_abs_error": float(out["abs_error_global_rf"].mean()),
        "mean_coordinate_abs_error": float(out["abs_error_coordinate_rf"].mean()),
        "mean_error_improvement": float(out["global_abs_error_minus_coord_abs_error"].mean()),
        "median_error_improvement": float(out["global_abs_error_minus_coord_abs_error"].median()),
        "coordinate_better_share": float(out["coordinate_better"].mean()),
        "coordinate_worse_share": float(out["coordinate_worse"].mean()),
        "coordinate_only_correct_share": float(out["coordinate_only_correct"].mean()),
        "global_only_correct_share": float(out["global_only_correct"].mean()),
        "both_correct_share": float(out["both_correct"].mean()),
        "both_wrong_share": float(out["both_wrong"].mean()),
    })

    return pd.DataFrame(rows)


def main():
    log.info("Starting Script 225")

    df = load()
    df = add_xy(df)

    regime_col = find_col(df, ["persistence_regime_v01", "persistence_regime"])
    fold_col = find_col(df, ["buffered_spatial_fold_v01", "spatial_fold_v01", "fold_v01"])

    df = df[df[regime_col].isin(TARGET)].copy()
    df["persistence_target_v01"] = df[regime_col].map(TARGET).astype(int)

    df = df.loc[:, ~df.columns.duplicated()].copy()

    rec, ctx = select_features(df)
    base_features = rec + ctx

    log.info(f"Rows: {len(df):,}")
    log.info(f"Recognition features: {len(rec)}")
    log.info(f"Context features: {len(ctx)}")

    metrics, fold_metrics, out = evaluate_models(
        df=df,
        fold_col=fold_col,
        target_col="persistence_target_v01",
        regime_col=regime_col,
        base_features=base_features,
    )

    improvement = summarize_improvement(out, fold_col)
    metrics = metrics.sort_values("roc_auc", ascending=False)

    out.to_file(OUT_GPKG, driver="GPKG")
    metrics.to_csv(OUT_METRICS, index=False)
    fold_metrics.to_csv(OUT_FOLD_METRICS, index=False)
    improvement.to_csv(OUT_IMPROVEMENT, index=False)

    qa = [
        f"Script: {SCRIPT}",
        "",
        "PURPOSE",
        "Diagnose where coordinate-conditioned RF improves over global RF for SRS operator-persistence classification.",
        "",
        f"Rows: {len(df):,}",
        "",
        "Recognition features:",
        "\n".join(rec),
        "",
        "Context features:",
        "\n".join(ctx),
        "",
        "OVERALL METRICS",
        metrics.to_string(index=False),
        "",
        "FOLD METRICS",
        fold_metrics.sort_values(["model", "fold"]).to_string(index=False),
        "",
        "IMPROVEMENT SUMMARY",
        improvement.to_string(index=False),
        "",
        "INTERPRETATION RULES",
        "1. If coordinate RF improves broadly across folds, spatially varying response geometry is supported.",
        "2. If improvement is concentrated in one or two folds, spatial drift is localized rather than general.",
        "3. If coordinate-only correct cells cluster spatially, later mapping can identify where global SRS response assumptions fail.",
        "4. This script does not build an adaptive SRS operator gate.",
    ]

    OUT_QA.write_text("\n".join(qa), encoding="utf-8")

    log.info("Completed Script 225")
    log.info("\n" + metrics.to_string(index=False))


if __name__ == "__main__":
    main()