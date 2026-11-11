from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT = "206b_oregon_srs_spatial_validation_v01"

BASE = Path("data/processed")
INPUT_FP = BASE / "oregon_srs_engine_inputs_v01.gpkg"

OUT_GPKG = BASE / "oregon_srs_pointwise_spatial_oof_v01.gpkg"
OUT_CSV = BASE / "oregon_srs_pointwise_spatial_oof_v01.csv"
OUT_QA = BASE / "oregon_srs_pointwise_spatial_oof_qa_v01.txt"
OUT_FOLDS = BASE / "oregon_srs_spatial_folds_v01.csv"

KEY = "cell_id"
TARGET = "y_observed_v01"
VALID = "srs_valid_engine_row_v01"

N_FOLDS_X = 5
N_FOLDS_Y = 5

FEATURES = [
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


def make_spatial_folds(df):
    x = df["centroid_x"].astype(float)
    y = df["centroid_y"].astype(float)

    x_bin = pd.qcut(x, q=N_FOLDS_X, labels=False, duplicates="drop")
    y_bin = pd.qcut(y, q=N_FOLDS_Y, labels=False, duplicates="drop")

    fold = (x_bin.astype(int) + N_FOLDS_X * y_bin.astype(int)).astype(int)
    return fold


def main():
    log("Reading canonical SRS engine inputs")
    gdf = gpd.read_file(INPUT_FP)

    features = [c for c in FEATURES if c in gdf.columns]
    missing = [c for c in FEATURES if c not in gdf.columns]
    if missing:
        log(f"Missing optional features: {missing}")

    required = [KEY, TARGET, VALID, "centroid_x", "centroid_y"]
    for c in required:
        if c not in gdf.columns:
            raise ValueError(f"Missing required column: {c}")

    model_mask = gdf[VALID].astype(bool)
    model_df = gdf.loc[model_mask].copy()
    model_df = model_df.dropna(subset=features + [TARGET, "centroid_x", "centroid_y"])

    log(f"Rows available for spatial OOF modeling: {len(model_df):,}")

    model_df["spatial_fold_v01"] = make_spatial_folds(model_df)
    folds = sorted(model_df["spatial_fold_v01"].unique())

    log(f"Spatial folds created: {len(folds)}")

    gdf["spatial_fold_v01"] = np.nan
    gdf.loc[model_df.index, "spatial_fold_v01"] = model_df["spatial_fold_v01"]

    for col in [
        "pred_pointwise_linear_oof_v01",
        "pred_pointwise_rf_oof_v01",
        "d_pointwise_linear_oof_v01",
        "d_pointwise_rf_oof_v01",
    ]:
        gdf[col] = np.nan

    fold_rows = []
    metric_rows = []

    for fold in folds:
        test_idx = model_df.index[model_df["spatial_fold_v01"] == fold]
        train_idx = model_df.index[model_df["spatial_fold_v01"] != fold]

        X_train = model_df.loc[train_idx, features].astype(float)
        y_train = model_df.loc[train_idx, TARGET].astype(float)
        X_test = model_df.loc[test_idx, features].astype(float)
        y_test = model_df.loc[test_idx, TARGET].astype(float)

        log(f"Fold {fold}: train={len(train_idx):,}, test={len(test_idx):,}")

        linear = LinearRegression()
        linear.fit(X_train, y_train)

        rf = RandomForestRegressor(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=10,
            random_state=42 + int(fold),
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        pred_linear = linear.predict(X_test)
        pred_rf = rf.predict(X_test)

        gdf.loc[test_idx, "pred_pointwise_linear_oof_v01"] = pred_linear
        gdf.loc[test_idx, "pred_pointwise_rf_oof_v01"] = pred_rf

        fold_metric = {"fold": int(fold)}
        fold_metric.update(metrics(y_test, pred_linear, "linear"))
        fold_metric.update(metrics(y_test, pred_rf, "rf"))
        metric_rows.append(fold_metric)

        fold_rows.append({
            "fold": int(fold),
            "train_n": int(len(train_idx)),
            "test_n": int(len(test_idx)),
            "x_min": float(model_df.loc[test_idx, "centroid_x"].min()),
            "x_max": float(model_df.loc[test_idx, "centroid_x"].max()),
            "y_min": float(model_df.loc[test_idx, "centroid_y"].min()),
            "y_max": float(model_df.loc[test_idx, "centroid_y"].max()),
        })

    gdf["d_pointwise_linear_oof_v01"] = (
        gdf["pred_pointwise_linear_oof_v01"] - gdf[TARGET]
    )
    gdf["d_pointwise_rf_oof_v01"] = (
        gdf["pred_pointwise_rf_oof_v01"] - gdf[TARGET]
    )

    oof_mask = gdf["pred_pointwise_rf_oof_v01"].notna()

    all_metrics = {}
    all_metrics.update(metrics(
        gdf.loc[oof_mask, TARGET].astype(float),
        gdf.loc[oof_mask, "pred_pointwise_linear_oof_v01"].astype(float),
        "linear_oof",
    ))
    all_metrics.update(metrics(
        gdf.loc[oof_mask, TARGET].astype(float),
        gdf.loc[oof_mask, "pred_pointwise_rf_oof_v01"].astype(float),
        "rf_oof",
    ))

    fold_df = pd.DataFrame(fold_rows)
    metric_df = pd.DataFrame(metric_rows)

    log("Writing outputs")
    gdf.to_file(OUT_GPKG, driver="GPKG")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)
    fold_df.to_csv(OUT_FOLDS, index=False)

    qa = []
    qa.append(f"{SCRIPT} QA")
    qa.append("=" * 80)
    qa.append(f"Input: {INPUT_FP}")
    qa.append(f"Output: {OUT_GPKG}")
    qa.append(f"Rows total: {len(gdf):,}")
    qa.append(f"Rows modeled OOF: {int(oof_mask.sum()):,}")
    qa.append(f"Spatial folds: {len(folds)}")
    qa.append("")
    qa.append("Features used:")
    qa.extend([f"- {c}" for c in features])
    qa.append("")
    qa.append("Overall OOF metrics:")
    for k, v in all_metrics.items():
        qa.append(f"{k}: {v}")
    qa.append("")
    qa.append("Fold metrics:")
    qa.append(metric_df.to_string(index=False))
    qa.append("")
    qa.append("Fold geography summary:")
    qa.append(fold_df.to_string(index=False))
    qa.append("")
    qa.append("OOF disequilibrium summary:")
    qa.append(
        gdf.loc[oof_mask, [
            "d_pointwise_linear_oof_v01",
            "d_pointwise_rf_oof_v01",
        ]].describe().to_string()
    )

    OUT_QA.write_text("\n".join(qa), encoding="utf-8")

    log(f"Created {OUT_GPKG}")
    log(f"Created {OUT_CSV}")
    log(f"Created {OUT_QA}")
    log(f"Created {OUT_FOLDS}")
    log("Done")


if __name__ == "__main__":
    main()
    