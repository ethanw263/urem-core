#!/usr/bin/env python3
"""
224_oregon_srs_spatial_regime_geometry_audit_v01.py

Tests whether SRS operator-persistence nonstationarity is structured as
recurring spatial response regimes rather than random fold failure.

No adaptive gate is built here.
"""

from pathlib import Path
import logging
import re

import geopandas as gpd
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT = "224_oregon_srs_spatial_regime_geometry_audit_v01"

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"

INPUT_PERSISTENCE = P / "oregon_srs_relational_persistence_common_population_v01.gpkg"
INPUT_ENGINE = P / "oregon_srs_engine_inputs_v01.gpkg"
INPUT_CONTEXT = P / "oregon_srs_relational_context_v01.gpkg"
INPUT_IMPORTANCE = P / "oregon_srs_persistence_classification_feature_importance_stability_v01.csv"

OUT_METRICS = P / "oregon_srs_spatial_regime_geometry_metrics_v01.csv"
OUT_FOLD_METRICS = P / "oregon_srs_spatial_regime_geometry_fold_metrics_v01.csv"
OUT_REGIME_PROFILES = P / "oregon_srs_spatial_regime_geometry_profiles_v01.csv"
OUT_ASSIGNMENTS = P / "oregon_srs_spatial_regime_geometry_assignments_v01.gpkg"
OUT_QA = P / "oregon_srs_spatial_regime_geometry_audit_qa_v01.txt"

TARGET = {
    "persistent_pointwise": 0,
    "persistent_relational": 1,
}

FORBIDDEN = [
    "gain",
    "residual",
    "error",
    "prediction",
    "predicted",
    "yhat",
    "operator",
    "persistence_score",
    "target",
    "class_label",
]

N_REC = 8
N_CTX = 16
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT)


def bad(c):
    s = c.lower()
    return any(x in s for x in FORBIDDEN)


def auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def find_col(df, options):
    for c in options:
        if c in df.columns:
            return c
    raise KeyError(options)


def load():
    base = gpd.read_file(INPUT_PERSISTENCE)
    eng = gpd.read_file(INPUT_ENGINE).drop(columns="geometry", errors="ignore")
    ctx = gpd.read_file(INPUT_CONTEXT).drop(columns="geometry", errors="ignore")

    idc = find_col(base, ["cell_id", "grid_id", "id"])

    eng_cols = [
        idc
    ] + [
        c
        for c in eng.columns
        if c != idc and c not in base.columns
    ]

    df = base.merge(
        eng[eng_cols],
        on=idc,
        how="left",
        validate="one_to_one",
    )

    ctx_cols = [
        idc
    ] + [
        c
        for c in ctx.columns
        if c != idc and c not in df.columns
    ]

    df = df.merge(
        ctx[ctx_cols],
        on=idc,
        how="left",
        validate="one_to_one",
    )

    return gpd.GeoDataFrame(
        df,
        geometry="geometry",
        crs=base.crs,
    ), idc


def select_features(df):
    imp = pd.read_csv(INPUT_IMPORTANCE)
    imp = imp[
        imp["feature_family"] == "full_admissible_state"
    ].copy()

    rec = (
        imp[
            imp["feature_group"] == "recognition_state"
        ]
        .sort_values(
            "mean_importance",
            ascending=False,
        )["raw_feature"]
        .drop_duplicates()
        .tolist()
    )

    ctx = (
        imp[
            imp["feature_group"] == "relational_context"
        ]
        .sort_values(
            "mean_importance",
            ascending=False,
        )["raw_feature"]
        .drop_duplicates()
        .tolist()
    )

    rec = [
        c
        for c in rec
        if c in df.columns
        and not bad(c)
        and pd.api.types.is_numeric_dtype(df[c])
    ][:N_REC]

    ctx = [
        c
        for c in ctx
        if c in df.columns
        and not bad(c)
        and pd.api.types.is_numeric_dtype(df[c])
    ][:N_CTX]

    return rec, ctx


def make_rf():
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
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


def make_logit():
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
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


def add_xy(df):
    cent = df.geometry.centroid
    df["x_coord"] = cent.x
    df["y_coord"] = cent.y
    return df


def crossfit_regimes(train, test, features, k):
    scaler = StandardScaler()
    imp = SimpleImputer(strategy="median")

    Xtr = imp.fit_transform(train[features])
    Xte = imp.transform(test[features])

    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    km = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=30,
    )

    train_regime = km.fit_predict(Xtr)
    test_regime = km.predict(Xte)

    return train_regime, test_regime


def onehot_regime(X, regimes, prefix, all_regimes=None):
    out = X.copy()

    if all_regimes is None:
        all_regimes = sorted(np.unique(regimes))

    for r in all_regimes:
        out[f"{prefix}_{r}"] = (
            regimes == r
        ).astype(int)

    return out


def evaluate(df, fold_col, target_col, rec, ctx):
    folds = sorted(
        df[fold_col]
        .dropna()
        .unique()
    )

    base_features = rec + ctx
    spatial_features = base_features + [
        "x_coord",
        "y_coord",
    ]

    model_specs = [
        (
            "global_rf",
            "rf",
            base_features,
            None,
        ),
        (
            "global_logit",
            "logit",
            base_features,
            None,
        ),
        (
            "coordinate_rf",
            "rf",
            spatial_features,
            None,
        ),
        (
            "coordinate_logit",
            "logit",
            spatial_features,
            None,
        ),
    ]

    for k in [
        2,
        3,
        4,
        5,
    ]:
        model_specs.append(
            (
                f"regime{k}_rf",
                "rf",
                base_features,
                k,
            )
        )
        model_specs.append(
            (
                f"regime{k}_logit",
                "logit",
                base_features,
                k,
            )
        )

    preds = {
        name: pd.Series(
            np.nan,
            index=df.index,
        )
        for name, *_ in model_specs
    }

    probs = {
        name: pd.Series(
            np.nan,
            index=df.index,
        )
        for name, *_ in model_specs
    }

    fold_rows = []
    profile_rows = []
    assign_parts = []

    for fold in folds:
        log.info(f"Fold {fold}")

        train_mask = df[fold_col] != fold
        test_mask = df[fold_col] == fold

        train = df.loc[
            train_mask
        ].copy()

        test = df.loc[
            test_mask
        ].copy()

        ytr = train[
            target_col
        ].astype(int)

        yte = test[
            target_col
        ].astype(int)

        regime_cache = {}

        for k in [
            2,
            3,
            4,
            5,
        ]:
            trr, ter = crossfit_regimes(
                train,
                test,
                base_features,
                k,
            )

            regime_cache[k] = (
                trr,
                ter,
            )

            for r in sorted(np.unique(trr)):
                sub = train[
                    trr == r
                ]

                profile_rows.append(
                    {
                        "heldout_fold": fold,
                        "k": k,
                        "regime": r,
                        "train_n": len(sub),
                        "train_positive_rate": float(
                            sub[target_col].mean()
                        ),
                    }
                )

            test_assign = test[
                [
                    "geometry",
                    target_col,
                    fold_col,
                ]
            ].copy()

            test_assign["heldout_fold"] = fold
            test_assign[f"regime_k{k}"] = ter

            assign_parts.append(
                test_assign
            )

        for (
            name,
            kind,
            features,
            k,
        ) in model_specs:

            Xtr = train[
                features
            ].copy()

            Xte = test[
                features
            ].copy()

            if k is not None:
                trr, ter = regime_cache[k]

                all_regimes = sorted(
                    np.unique(trr)
                )

                Xtr = onehot_regime(
                    Xtr,
                    trr,
                    f"regime{k}",
                    all_regimes=all_regimes,
                )

                Xte = onehot_regime(
                    Xte,
                    ter,
                    f"regime{k}",
                    all_regimes=all_regimes,
                )

                # Critical sklearn safety:
                # ensure held-out design has identical columns/order.
                Xte = Xte.reindex(
                    columns=Xtr.columns,
                    fill_value=0,
                )

            model = make_rf() if kind == "rf" else make_logit()

            model.fit(
                Xtr,
                ytr,
            )

            pred = model.predict(
                Xte
            )

            prob = model.predict_proba(
                Xte
            )[:, 1]

            preds[name].loc[
                test_mask
            ] = pred

            probs[name].loc[
                test_mask
            ] = prob

            fold_rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "n_test": int(
                        test_mask.sum()
                    ),
                    "accuracy": float(
                        accuracy_score(
                            yte,
                            pred,
                        )
                    ),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(
                            yte,
                            pred,
                        )
                    ),
                    "roc_auc": auc(
                        yte.to_numpy(),
                        prob,
                    ),
                }
            )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    y = df[
        target_col
    ].astype(int).to_numpy()

    metrics = []

    for name, *_ in model_specs:
        pred = preds[name].astype(int).to_numpy()
        prob = probs[name].astype(float).to_numpy()

        sub = fold_metrics[
            fold_metrics["model"] == name
        ]

        metrics.append(
            {
                "model": name,
                "n": len(df),
                "accuracy": float(
                    accuracy_score(
                        y,
                        pred,
                    )
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(
                        y,
                        pred,
                    )
                ),
                "roc_auc": auc(
                    y,
                    prob,
                ),
                "mean_fold_auc": float(
                    sub["roc_auc"].mean()
                ),
                "std_fold_auc": float(
                    sub["roc_auc"].std(ddof=1)
                ),
                "min_fold_auc": float(
                    sub["roc_auc"].min()
                ),
                "folds_auc_above_0_60": int(
                    (
                        sub["roc_auc"] >= 0.60
                    ).sum()
                ),
            }
        )

    assignments = pd.concat(
        assign_parts,
        ignore_index=True,
    )

    return (
        pd.DataFrame(metrics),
        fold_metrics,
        pd.DataFrame(profile_rows),
        assignments,
    )


def main():
    log.info("Starting Script 224")

    df, idc = load()
    df = add_xy(df)

    target_col = find_col(
        df,
        [
            "persistence_regime_v01",
            "persistence_regime",
        ],
    )

    fold_col = find_col(
        df,
        [
            "buffered_spatial_fold_v01",
            "spatial_fold_v01",
            "fold_v01",
        ],
    )

    df = df[
        df[target_col].isin(TARGET)
    ].copy()

    df["persistence_target_v01"] = (
        df[target_col]
        .map(TARGET)
        .astype(int)
    )

    rec, ctx = select_features(df)

    log.info(f"Rows: {len(df):,}")
    log.info(f"Recognition features: {len(rec)}")
    log.info(f"Context features: {len(ctx)}")

    metrics, fold_metrics, profiles, assignments = evaluate(
        df=df,
        fold_col=fold_col,
        target_col="persistence_target_v01",
        rec=rec,
        ctx=ctx,
    )

    metrics = metrics.sort_values(
        "roc_auc",
        ascending=False,
    )

    metrics.to_csv(
        OUT_METRICS,
        index=False,
    )

    fold_metrics.to_csv(
        OUT_FOLD_METRICS,
        index=False,
    )

    profiles.to_csv(
        OUT_REGIME_PROFILES,
        index=False,
    )

    assignments = gpd.GeoDataFrame(
        assignments,
        geometry="geometry",
        crs=df.crs,
    )

    assignments.to_file(
        OUT_ASSIGNMENTS,
        driver="GPKG",
    )

    regime_summary = (
        profiles
        .groupby(
            [
                "k",
                "regime",
            ],
            as_index=False,
        )
        .agg(
            mean_train_positive_rate=(
                "train_positive_rate",
                "mean",
            ),
            std_train_positive_rate=(
                "train_positive_rate",
                "std",
            ),
            mean_train_n=(
                "train_n",
                "mean",
            ),
        )
    )

    qa = [
        f"Script: {SCRIPT}",
        "",
        "PURPOSE",
        "Test whether spatial nonstationarity is better represented by discrete unsupervised response regimes or continuous spatial drift.",
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
        fold_metrics.sort_values(
            [
                "model",
                "fold",
            ]
        ).to_string(index=False),
        "",
        "REGIME PROFILE SUMMARY",
        regime_summary.to_string(index=False),
        "",
        "INTERPRETATION RULES",
        "1. If regime-conditioned models outperform global and coordinate models, discrete spatial response regimes are supported.",
        "2. If coordinate models outperform regime models, continuous spatial drift is more plausible.",
        "3. If neither improves much over global RF, nonstationarity may be complex but not captured by these simple structures.",
        "4. This does not build or validate an adaptive operator gate.",
    ]

    OUT_QA.write_text(
        "\n".join(qa),
        encoding="utf-8",
    )

    log.info("Completed Script 224")
    log.info(
        "\n" + metrics.to_string(index=False)
    )


if __name__ == "__main__":
    main()