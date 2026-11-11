from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT = "208_oregon_srs_relational_transformation_v01"

BASE = Path("data/processed")
INPUT_FP = BASE / "oregon_srs_relational_context_v01.gpkg"

OUT_GPKG = BASE / "oregon_srs_relational_transformation_v01.gpkg"
OUT_CSV = BASE / "oregon_srs_relational_transformation_v01.csv"
OUT_QA = BASE / "oregon_srs_relational_transformation_qa_v01.txt"

KEY = "cell_id"
TARGET = "y_observed_v01"
VALID = "srs_valid_engine_row_v01"
FOLD = "spatial_fold_v01"

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


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true, y_pred, prefix):
    return {
        f"{prefix}_n": int(len(y_true)),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_rmse": rmse(y_true, y_pred),
    }


def main():
    log("Reading relational context layer")
    gdf = gpd.read_file(INPUT_FP)

    required = [KEY, TARGET, VALID, FOLD]
    for c in required:
        if c not in gdf.columns:
            raise ValueError(f"Missing required column: {c}")

    pointwise_features = [c for c in POINTWISE_FEATURES if c in gdf.columns]

    relational_features = [
        c for c in gdf.columns
        if (
            c.startswith("rel_neighbor_count_")
            or "_mean_3km_v01" in c
            or "_max_3km_v01" in c
            or "_min_3km_v01" in c
            or "_contrast_3km_v01" in c
            or "_mean_10km_v01" in c
            or "_max_10km_v01" in c
            or "_min_10km_v01" in c
            or "_contrast_10km_v01" in c
            or "_mean_25km_v01" in c
            or "_max_25km_v01" in c
            or "_min_25km_v01" in c
            or "_contrast_25km_v01" in c
        )
    ]

    relational_features = [
        c for c in relational_features
        if c not in pointwise_features
    ]

    all_features = pointwise_features + relational_features

    log(f"Pointwise features: {len(pointwise_features)}")
    log(f"Relational features: {len(relational_features)}")
    log(f"Total relational-model features: {len(all_features)}")

    model_df = gdf[gdf[VALID].astype(bool)].copy()
    model_df = model_df.dropna(subset=all_features + [TARGET, FOLD])

    log(f"Rows available for relational OOF modeling: {len(model_df):,}")

    folds = sorted(model_df[FOLD].dropna().unique())

    for col in [
        "pred_relational_linear_oof_v01",
        "pred_relational_rf_oof_v01",
        "d_relational_linear_oof_v01",
        "d_relational_rf_oof_v01",
    ]:
        gdf[col] = np.nan

    metric_rows = []

    for fold in folds:
        test_idx = model_df.index[model_df[FOLD] == fold]
        train_idx = model_df.index[model_df[FOLD] != fold]

        X_train = model_df.loc[train_idx, all_features].astype(float)
        y_train = model_df.loc[train_idx, TARGET].astype(float)
        X_test = model_df.loc[test_idx, all_features].astype(float)
        y_test = model_df.loc[test_idx, TARGET].astype(float)

        log(f"Fold {int(fold)}: train={len(train_idx):,}, test={len(test_idx):,}")

        linear = LinearRegression()
        linear.fit(X_train, y_train)

        rf = RandomForestRegressor(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=10,
            random_state=208 + int(fold),
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        pred_linear = linear.predict(X_test)
        pred_rf = rf.predict(X_test)

        gdf.loc[test_idx, "pred_relational_linear_oof_v01"] = pred_linear
        gdf.loc[test_idx, "pred_relational_rf_oof_v01"] = pred_rf

        row = {"fold": int(fold)}
        row.update(metrics(y_test, pred_linear, "rel_linear"))
        row.update(metrics(y_test, pred_rf, "rel_rf"))
        metric_rows.append(row)

    gdf["d_relational_linear_oof_v01"] = (
        gdf["pred_relational_linear_oof_v01"] - gdf[TARGET]
    )
    gdf["d_relational_rf_oof_v01"] = (
        gdf["pred_relational_rf_oof_v01"] - gdf[TARGET]
    )

    oof_mask = gdf["pred_relational_rf_oof_v01"].notna()

    overall = {}
    overall.update(metrics(
        gdf.loc[oof_mask, TARGET].astype(float),
        gdf.loc[oof_mask, "pred_relational_linear_oof_v01"].astype(float),
        "rel_linear_oof",
    ))
    overall.update(metrics(
        gdf.loc[oof_mask, TARGET].astype(float),
        gdf.loc[oof_mask, "pred_relational_rf_oof_v01"].astype(float),
        "rel_rf_oof",
    ))

    if "pred_pointwise_linear_oof_v01" in gdf.columns:
        mask = oof_mask & gdf["pred_pointwise_linear_oof_v01"].notna()
        overall.update(metrics(
            gdf.loc[mask, TARGET].astype(float),
            gdf.loc[mask, "pred_pointwise_linear_oof_v01"].astype(float),
            "pt_linear_oof",
        ))

    if "pred_pointwise_rf_oof_v01" in gdf.columns:
        mask = oof_mask & gdf["pred_pointwise_rf_oof_v01"].notna()
        overall.update(metrics(
            gdf.loc[mask, TARGET].astype(float),
            gdf.loc[mask, "pred_pointwise_rf_oof_v01"].astype(float),
            "pt_rf_oof",
        ))

    metric_df = pd.DataFrame(metric_rows)

    rf_full = RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=10,
        random_state=208,
        n_jobs=-1,
    )
    rf_full.fit(model_df[all_features].astype(float), model_df[TARGET].astype(float))

    importance = pd.DataFrame({
        "feature": all_features,
        "importance": rf_full.feature_importances_,
        "feature_group": [
            "pointwise" if c in pointwise_features else "relational"
            for c in all_features
        ],
    }).sort_values("importance", ascending=False)

    log("Writing outputs")
    gdf.to_file(OUT_GPKG, driver="GPKG")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    qa = []
    qa.append(f"{SCRIPT} QA")
    qa.append("=" * 80)
    qa.append(f"Input: {INPUT_FP}")
    qa.append(f"Output: {OUT_GPKG}")
    qa.append(f"Rows total: {len(gdf):,}")
    qa.append(f"Rows modeled OOF: {int(oof_mask.sum()):,}")
    qa.append("")
    qa.append(f"Pointwise features: {len(pointwise_features)}")
    qa.append(f"Relational features: {len(relational_features)}")
    qa.append("")
    qa.append("Overall OOF metrics:")
    for k, v in overall.items():
        qa.append(f"{k}: {v}")
    qa.append("")
    qa.append("Fold metrics:")
    qa.append(metric_df.to_string(index=False))
    qa.append("")
    qa.append("Top 40 random forest feature importances:")
    qa.append(importance.head(40).to_string(index=False))
    qa.append("")
    qa.append("Feature importance by group:")
    qa.append(
        importance.groupby("feature_group")["importance"]
        .sum()
        .sort_values(ascending=False)
        .to_string()
    )
    qa.append("")
    qa.append("Relational disequilibrium summary:")
    qa.append(
        gdf.loc[oof_mask, [
            "d_relational_linear_oof_v01",
            "d_relational_rf_oof_v01",
        ]].describe().to_string()
    )

    OUT_QA.write_text("\n".join(qa), encoding="utf-8")

    log(f"Created {OUT_GPKG}")
    log(f"Created {OUT_CSV}")
    log(f"Created {OUT_QA}")
    log("Done")


if __name__ == "__main__":
    main()