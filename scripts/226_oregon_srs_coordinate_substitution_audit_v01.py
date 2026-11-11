#!/usr/bin/env python3
"""
226_oregon_srs_coordinate_substitution_audit_v01.py

Tests whether raw coordinate gains can be explained by interpretable spatial
substitutes: spatial trend terms, 10-25 km contrasts, and coordinate-free
multiscale context contrasts.

No adaptive SRS gate is built.
"""

from pathlib import Path
import logging
import re

import geopandas as gpd
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline


SCRIPT = "226_oregon_srs_coordinate_substitution_audit_v01"

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"

INPUT_PERSISTENCE = P / "oregon_srs_relational_persistence_common_population_v01.gpkg"
INPUT_ENGINE = P / "oregon_srs_engine_inputs_v01.gpkg"
INPUT_CONTEXT = P / "oregon_srs_relational_context_v01.gpkg"
INPUT_IMPORTANCE = P / "oregon_srs_persistence_classification_feature_importance_stability_v01.csv"

OUT_GPKG = P / "oregon_srs_coordinate_substitution_audit_v01.gpkg"
OUT_METRICS = P / "oregon_srs_coordinate_substitution_metrics_v01.csv"
OUT_FOLD_METRICS = P / "oregon_srs_coordinate_substitution_fold_metrics_v01.csv"
OUT_FEATURES = P / "oregon_srs_coordinate_substitution_feature_manifest_v01.csv"
OUT_QA = P / "oregon_srs_coordinate_substitution_audit_qa_v01.txt"

TARGET = {
    "persistent_pointwise": 0,
    "persistent_relational": 1,
}

N_REC = 8
N_CTX = 16
RANDOM_STATE = 42

FORBIDDEN = [
    "gain", "residual", "error", "prediction", "predicted", "yhat",
    "operator", "persistence_score", "target", "class_label",
]

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT)


def auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def bad(c):
    s = c.lower()
    return any(x in s for x in FORBIDDEN)


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


def add_xy_and_trend(df):
    cent = df.geometry.centroid

    x = cent.x.to_numpy()
    y = cent.y.to_numpy()

    xz = (x - np.nanmean(x)) / np.nanstd(x)
    yz = (y - np.nanmean(y)) / np.nanstd(y)

    df["x_coord"] = x
    df["y_coord"] = y
    df["x_z"] = xz
    df["y_z"] = yz
    df["x2_z"] = xz ** 2
    df["y2_z"] = yz ** 2
    df["xy_z"] = xz * yz

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

    rec = [
        c for c in rec
        if c in df.columns and not bad(c) and pd.api.types.is_numeric_dtype(df[c])
    ][:N_REC]

    ctx = [
        c for c in ctx
        if c in df.columns and not bad(c) and pd.api.types.is_numeric_dtype(df[c])
    ][:N_CTX]

    return rec, ctx


def add_context_contrasts(df, ctx_features):
    new_cols = []

    for c25 in ctx_features:
        if "_25km_" not in c25:
            continue

        c10 = c25.replace("_25km_", "_10km_")
        c3 = c25.replace("_25km_", "_3km_")

        if c10 in df.columns and pd.api.types.is_numeric_dtype(df[c10]):
            name = f"contrast_10_minus_25__{c10}"
            df[name] = df[c10] - df[c25]
            new_cols.append(name)

        if c3 in df.columns and pd.api.types.is_numeric_dtype(df[c3]):
            name = f"contrast_3_minus_25__{c3}"
            df[name] = df[c3] - df[c25]
            new_cols.append(name)

    new_cols = list(dict.fromkeys(new_cols))
    return df, new_cols


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


def evaluate(df, fold_col, target_col, feature_sets):
    folds = sorted(df[fold_col].dropna().unique())

    pred_store = {m: pd.Series(np.nan, index=df.index) for m in feature_sets}
    prob_store = {m: pd.Series(np.nan, index=df.index) for m in feature_sets}

    fold_rows = []

    for fold in folds:
        log.info(f"Fold {fold}")

        train_mask = df[fold_col] != fold
        test_mask = df[fold_col] == fold

        train = df.loc[train_mask].copy()
        test = df.loc[test_mask].copy()

        y_train = train[target_col].astype(int)
        y_test = test[target_col].astype(int)

        for model_name, features in feature_sets.items():
            features = list(dict.fromkeys(features))

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
                "n_features": len(features),
                "accuracy": float(accuracy_score(y_test, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                "roc_auc": auc(y_test.to_numpy(), prob),
            })

    fold_metrics = pd.DataFrame(fold_rows)

    y = df[target_col].astype(int).to_numpy()

    metrics = []
    for model_name, features in feature_sets.items():
        pred = pred_store[model_name].astype(int).to_numpy()
        prob = prob_store[model_name].astype(float).to_numpy()

        sub = fold_metrics[fold_metrics["model"] == model_name]

        metrics.append({
            "model": model_name,
            "n": len(df),
            "n_features": len(list(dict.fromkeys(features))),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "roc_auc": auc(y, prob),
            "mean_fold_auc": float(sub["roc_auc"].mean()),
            "std_fold_auc": float(sub["roc_auc"].std(ddof=1)),
            "min_fold_auc": float(sub["roc_auc"].min()),
            "folds_auc_above_0_60": int((sub["roc_auc"] >= 0.60).sum()),
        })

    out = df[["geometry", fold_col, target_col]].copy()

    for model_name in feature_sets:
        out[f"pred_{model_name}"] = pred_store[model_name].astype(int).to_numpy()
        out[f"prob_{model_name}"] = prob_store[model_name].astype(float).to_numpy()
        out[f"abs_error_{model_name}"] = (
            out[target_col].astype(float).to_numpy()
            - out[f"prob_{model_name}"].astype(float).to_numpy()
        )
        out[f"abs_error_{model_name}"] = out[f"abs_error_{model_name}"].abs()

    return (
        pd.DataFrame(metrics).sort_values("roc_auc", ascending=False),
        fold_metrics,
        gpd.GeoDataFrame(out, geometry="geometry", crs=df.crs),
    )


def main():
    log.info("Starting Script 226")

    df = load()
    df = add_xy_and_trend(df)

    regime_col = find_col(df, ["persistence_regime_v01", "persistence_regime"])
    fold_col = find_col(df, ["buffered_spatial_fold_v01", "spatial_fold_v01", "fold_v01"])

    df = df[df[regime_col].isin(TARGET)].copy()
    df["persistence_target_v01"] = df[regime_col].map(TARGET).astype(int)

    rec, ctx = select_features(df)
    df, contrast_features = add_context_contrasts(df, ctx)

    base = rec + ctx
    raw_coord = ["x_coord", "y_coord"]
    trend = ["x_z", "y_z", "x2_z", "y2_z", "xy_z"]

    feature_sets = {
        "global_rf": base,
        "raw_coordinate_rf": base + raw_coord,
        "spatial_trend_rf": base + trend,
        "context_contrast_rf": base + contrast_features,
        "contrast_plus_trend_rf": base + contrast_features + trend,
        "contrast_plus_raw_coord_rf": base + contrast_features + raw_coord,
    }

    manifest = []
    for model, features in feature_sets.items():
        for f in list(dict.fromkeys(features)):
            if f in rec:
                group = "recognition"
            elif f in ctx:
                group = "context"
            elif f in contrast_features:
                group = "context_contrast"
            elif f in raw_coord:
                group = "raw_coordinate"
            elif f in trend:
                group = "spatial_trend"
            else:
                group = "other"

            manifest.append({
                "model": model,
                "feature": f,
                "feature_group": group,
            })

    log.info(f"Rows: {len(df):,}")
    log.info(f"Recognition features: {len(rec)}")
    log.info(f"Context features: {len(ctx)}")
    log.info(f"Contrast features: {len(contrast_features)}")

    metrics, fold_metrics, out = evaluate(
        df=df,
        fold_col=fold_col,
        target_col="persistence_target_v01",
        feature_sets=feature_sets,
    )

    out.to_file(OUT_GPKG, driver="GPKG")
    metrics.to_csv(OUT_METRICS, index=False)
    fold_metrics.to_csv(OUT_FOLD_METRICS, index=False)
    pd.DataFrame(manifest).to_csv(OUT_FEATURES, index=False)

    qa = [
        f"Script: {SCRIPT}",
        "",
        "PURPOSE",
        "Test whether raw coordinate gains can be replaced by interpretable spatial substitutes.",
        "",
        f"Rows: {len(df):,}",
        f"Recognition features: {len(rec)}",
        f"Context features: {len(ctx)}",
        f"Contrast features: {len(contrast_features)}",
        "",
        "OVERALL METRICS",
        metrics.to_string(index=False),
        "",
        "FOLD METRICS",
        fold_metrics.sort_values(["model", "fold"]).to_string(index=False),
        "",
        "INTERPRETATION RULES",
        "1. If spatial_trend_rf approaches raw_coordinate_rf, broad spatial drift explains coordinate gain.",
        "2. If context_contrast_rf approaches raw_coordinate_rf, interpretable multiscale contrasts explain coordinate gain.",
        "3. If contrast_plus_trend_rf approaches or beats raw_coordinate_rf, spatial drift can be partially re-expressed as interpretable SRS geometry.",
        "4. If raw_coordinate_rf remains clearly superior, coordinates are proxying omitted spatial structure.",
        "5. This does not build or validate an adaptive SRS operator gate.",
    ]

    OUT_QA.write_text("\n".join(qa), encoding="utf-8")

    log.info("Completed Script 226")
    log.info("\n" + metrics.to_string(index=False))


if __name__ == "__main__":
    main()