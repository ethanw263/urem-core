#!/usr/bin/env python3
"""
229b_oregon_srs_differential_operator_substitution_test_v02.py

Corrected differential-operator substitution test.

Fix from Script 229:
Script 229 accidentally tested differential/context-only models because the
Script 228 differential layer did not carry recognition variables forward.
This version merges recognition-state variables back from the canonical engine
input layer before modeling.

Question:
Can interpretable spatial differential operators recover the raw-coordinate or
spatial-trend gain when recognition state is included?

No adaptive SRS gate is built.
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


SCRIPT = "229b_oregon_srs_differential_operator_substitution_test_v02"

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"

INPUT_DIFF = P / "oregon_srs_spatial_differential_operators_v01.gpkg"
INPUT_ENGINE = P / "oregon_srs_engine_inputs_v01.gpkg"
INPUT_IMPORTANCE = P / "oregon_srs_persistence_classification_feature_importance_stability_v01.csv"
INPUT_MANIFEST = P / "oregon_srs_spatial_differential_operator_feature_manifest_v01.csv"

OUT_GPKG = P / "oregon_srs_differential_operator_substitution_test_v02.gpkg"
OUT_METRICS = P / "oregon_srs_differential_operator_substitution_metrics_v02.csv"
OUT_FOLD_METRICS = P / "oregon_srs_differential_operator_substitution_fold_metrics_v02.csv"
OUT_FEATURE_MANIFEST = P / "oregon_srs_differential_operator_substitution_feature_manifest_v02.csv"
OUT_QA = P / "oregon_srs_differential_operator_substitution_test_qa_v02.txt"

TARGET = {
    "persistent_pointwise": 0,
    "persistent_relational": 1,
}

N_REC = 8
N_CTX = 16
MAX_DIFF_FEATURES = 180
RANDOM_STATE = 42

KEEP_OPERATOR_GROUPS = {
    "gradient_x",
    "gradient_y",
    "gradient_magnitude",
    "center_neighbor_difference",
    "transition_strength",
    "local_range",
    "neighbor_dispersion",
    "cross_scale_difference",
}

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


def add_xy_trend(df):
    cent = df.geometry.centroid
    x = cent.x.to_numpy()
    y = cent.y.to_numpy()

    x_std = np.nanstd(x) or 1.0
    y_std = np.nanstd(y) or 1.0

    xz = (x - np.nanmean(x)) / x_std
    yz = (y - np.nanmean(y)) / y_std

    df["x_coord"] = x
    df["y_coord"] = y
    df["x_z"] = xz
    df["y_z"] = yz
    df["x2_z"] = xz ** 2
    df["y2_z"] = yz ** 2
    df["xy_z"] = xz * yz

    return df


def merge_engine_features(diff):
    engine = gpd.read_file(INPUT_ENGINE).drop(columns="geometry", errors="ignore")

    idc = find_col(diff, ["cell_id", "grid_id", "id"])

    keep = [idc]
    for c in engine.columns:
        if c != idc and c not in diff.columns:
            keep.append(c)

    merged = diff.merge(
        engine[keep],
        on=idc,
        how="left",
        validate="one_to_one",
    )

    merged = merged.loc[:, ~merged.columns.duplicated()].copy()
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=diff.crs)


def select_base_features(df):
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
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ][:N_REC]

    ctx = [
        c for c in ctx
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ][:N_CTX]

    if len(rec) == 0:
        raise ValueError(
            "Recognition feature count is 0 after merge. "
            "Check INPUT_ENGINE and feature importance names."
        )

    return rec, ctx


def select_diff_features(df):
    manifest = pd.read_csv(INPUT_MANIFEST)

    manifest = manifest[
        manifest["operator_group"].isin(KEEP_OPERATOR_GROUPS)
    ].copy()

    manifest = manifest[
        manifest["created_feature"].isin(df.columns)
    ].copy()

    rows = []

    for _, r in manifest.iterrows():
        f = r["created_feature"]

        if not pd.api.types.is_numeric_dtype(df[f]):
            continue

        missing_share = float(df[f].isna().mean())
        if missing_share > 0.05:
            continue

        std = float(df[f].std(skipna=True))
        if not np.isfinite(std) or std <= 0:
            continue

        rows.append({
            "feature": f,
            "operator_group": r["operator_group"],
            "source_feature": r["source_feature"],
            "radius_m": r.get("radius_m", np.nan),
            "missing_share": missing_share,
            "std": std,
            "selection_score": std * (1 - missing_share),
        })

    score = pd.DataFrame(rows)

    if score.empty:
        raise ValueError("No usable differential operator features selected.")

    selected = (
        score
        .sort_values("selection_score", ascending=False)
        .head(MAX_DIFF_FEATURES)
        .copy()
    )

    return selected["feature"].tolist(), selected


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

        train = df.loc[train_mask]
        test = df.loc[test_mask]

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
    log.info("Starting Script 229b")

    df = gpd.read_file(INPUT_DIFF)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = merge_engine_features(df)
    df = add_xy_trend(df)

    regime_col = find_col(df, ["persistence_regime_v01", "persistence_regime"])
    fold_col = find_col(df, ["buffered_spatial_fold_v01", "spatial_fold_v01", "fold_v01"])

    df = df[df[regime_col].isin(TARGET)].copy()

    if "persistence_target_v01" not in df.columns:
        df["persistence_target_v01"] = df[regime_col].map(TARGET).astype(int)

    rec, ctx = select_base_features(df)
    diff_features, diff_manifest = select_diff_features(df)

    base = rec + ctx
    raw_coord = ["x_coord", "y_coord"]
    trend = ["x_z", "y_z", "x2_z", "y2_z", "xy_z"]

    feature_sets = {
        "global_rf": base,
        "raw_coordinate_rf": base + raw_coord,
        "spatial_trend_rf": base + trend,
        "differential_operator_rf": base + diff_features,
        "differential_plus_trend_rf": base + diff_features + trend,
        "differential_plus_raw_coord_rf": base + diff_features + raw_coord,
    }

    manifest_rows = []

    for model, features in feature_sets.items():
        for f in list(dict.fromkeys(features)):
            if f in rec:
                group = "recognition"
            elif f in ctx:
                group = "context"
            elif f in diff_features:
                group = "spatial_differential_operator"
            elif f in raw_coord:
                group = "raw_coordinate"
            elif f in trend:
                group = "spatial_trend"
            else:
                group = "other"

            manifest_rows.append({
                "model": model,
                "feature": f,
                "feature_group": group,
            })

    log.info(f"Rows: {len(df):,}")
    log.info(f"Recognition features: {len(rec)}")
    log.info(f"Context features: {len(ctx)}")
    log.info(f"Differential features selected: {len(diff_features)}")

    metrics, fold_metrics, out = evaluate(
        df=df,
        fold_col=fold_col,
        target_col="persistence_target_v01",
        feature_sets=feature_sets,
    )

    out.to_file(OUT_GPKG, driver="GPKG")
    metrics.to_csv(OUT_METRICS, index=False)
    fold_metrics.to_csv(OUT_FOLD_METRICS, index=False)

    feature_manifest = pd.DataFrame(manifest_rows)
    feature_manifest = feature_manifest.merge(
        diff_manifest.rename(columns={"feature": "feature"}),
        on="feature",
        how="left",
    )
    feature_manifest.to_csv(OUT_FEATURE_MANIFEST, index=False)

    qa = [
        f"Script: {SCRIPT}",
        "",
        "PURPOSE",
        "Corrected test: recognition + context + spatial differential operators.",
        "",
        f"Rows: {len(df):,}",
        f"Recognition features: {len(rec)}",
        f"Context features: {len(ctx)}",
        f"Differential features selected: {len(diff_features)}",
        "",
        "RECOGNITION FEATURES",
        "\n".join(rec),
        "",
        "OVERALL METRICS",
        metrics.to_string(index=False),
        "",
        "FOLD METRICS",
        fold_metrics.sort_values(["model", "fold"]).to_string(index=False),
        "",
        "DIFFERENTIAL FEATURE GROUP COUNTS",
        diff_manifest["operator_group"].value_counts().to_string(),
        "",
        "INTERPRETATION RULES",
        "1. If differential_operator_rf approaches raw_coordinate_rf, spatial differential operators explain much of S_i.",
        "2. If differential_plus_trend_rf approaches or beats raw_coordinate_rf, S_i is partly decomposable into trend plus differential structure.",
        "3. If differential_plus_raw_coord_rf remains clearly best, unresolved spatial structure remains.",
        "4. If fold 0 remains poor, it remains a structural-transfer boundary.",
        "5. This script does not build an adaptive SRS operator gate.",
    ]

    OUT_QA.write_text("\n".join(qa), encoding="utf-8")

    log.info("Completed Script 229b")
    log.info("\n" + metrics.to_string(index=False))


if __name__ == "__main__":
    main()