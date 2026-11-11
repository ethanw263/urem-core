from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

SCRIPT = "206_oregon_srs_pointwise_transformation_v01"

BASE = Path("data/processed")
INPUT_FP = BASE / "oregon_srs_engine_inputs_v01.gpkg"

OUT_GPKG = BASE / "oregon_srs_pointwise_transformation_v01.gpkg"
OUT_CSV = BASE / "oregon_srs_pointwise_transformation_v01.csv"
OUT_QA = BASE / "oregon_srs_pointwise_transformation_qa_v01.txt"

TARGET = "y_observed_v01"
VALID = "srs_valid_engine_row_v01"

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


def safe_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def summarize_metrics(y_true, y_pred, prefix):
    return {
        f"{prefix}_n": int(len(y_true)),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_rmse": safe_rmse(y_true, y_pred),
    }


def main():
    log("Reading canonical SRS engine inputs")
    gdf = gpd.read_file(INPUT_FP)

    if VALID not in gdf.columns:
        raise ValueError(f"Missing required validity column: {VALID}")
    if TARGET not in gdf.columns:
        raise ValueError(f"Missing required target column: {TARGET}")

    features = [c for c in FEATURES if c in gdf.columns]
    missing_features = [c for c in FEATURES if c not in gdf.columns]

    log(f"Available pointwise features: {len(features)}")
    if missing_features:
        log(f"Missing optional pointwise features: {missing_features}")

    model_df = gdf[gdf[VALID].astype(bool)].copy()
    model_df = model_df.dropna(subset=features + [TARGET])

    log(f"Valid modeling rows after NA drop: {len(model_df):,}")

    X = model_df[features].astype(float)
    y = model_df[TARGET].astype(float)

    train_idx, test_idx = train_test_split(
        model_df.index,
        test_size=0.25,
        random_state=42,
    )

    X_train = model_df.loc[train_idx, features].astype(float)
    y_train = model_df.loc[train_idx, TARGET].astype(float)
    X_test = model_df.loc[test_idx, features].astype(float)
    y_test = model_df.loc[test_idx, TARGET].astype(float)

    log("Fitting linear pointwise baseline")
    linear = LinearRegression()
    linear.fit(X_train, y_train)

    log("Fitting random forest pointwise transformation")
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    model_df["pred_pointwise_linear_v01"] = linear.predict(X)
    model_df["pred_pointwise_rf_v01"] = rf.predict(X)

    model_df["d_pointwise_linear_v01"] = (
        model_df["pred_pointwise_linear_v01"] - model_df[TARGET]
    )
    model_df["d_pointwise_rf_v01"] = (
        model_df["pred_pointwise_rf_v01"] - model_df[TARGET]
    )

    gdf["pred_pointwise_linear_v01"] = np.nan
    gdf["pred_pointwise_rf_v01"] = np.nan
    gdf["d_pointwise_linear_v01"] = np.nan
    gdf["d_pointwise_rf_v01"] = np.nan

    gdf.loc[model_df.index, "pred_pointwise_linear_v01"] = model_df["pred_pointwise_linear_v01"]
    gdf.loc[model_df.index, "pred_pointwise_rf_v01"] = model_df["pred_pointwise_rf_v01"]
    gdf.loc[model_df.index, "d_pointwise_linear_v01"] = model_df["d_pointwise_linear_v01"]
    gdf.loc[model_df.index, "d_pointwise_rf_v01"] = model_df["d_pointwise_rf_v01"]

    rf_importance = pd.DataFrame({
        "feature": features,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    train_pred_linear = linear.predict(X_train)
    test_pred_linear = linear.predict(X_test)
    train_pred_rf = rf.predict(X_train)
    test_pred_rf = rf.predict(X_test)

    metrics = {}
    metrics.update(summarize_metrics(y_train, train_pred_linear, "linear_train"))
    metrics.update(summarize_metrics(y_test, test_pred_linear, "linear_test"))
    metrics.update(summarize_metrics(y_train, train_pred_rf, "rf_train"))
    metrics.update(summarize_metrics(y_test, test_pred_rf, "rf_test"))

    log("Writing outputs")
    gdf.to_file(OUT_GPKG, driver="GPKG")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    qa_lines = []
    qa_lines.append(f"{SCRIPT} QA")
    qa_lines.append("=" * 80)
    qa_lines.append(f"Input: {INPUT_FP}")
    qa_lines.append(f"Output GPKG: {OUT_GPKG}")
    qa_lines.append(f"Rows total: {len(gdf):,}")
    qa_lines.append(f"Rows modeled: {len(model_df):,}")
    qa_lines.append("")
    qa_lines.append("Features used:")
    qa_lines.extend([f"- {c}" for c in features])
    qa_lines.append("")
    qa_lines.append("Metrics:")
    for k, v in metrics.items():
        qa_lines.append(f"{k}: {v}")
    qa_lines.append("")
    qa_lines.append("Random forest feature importance:")
    qa_lines.append(rf_importance.to_string(index=False))
    qa_lines.append("")
    qa_lines.append("Disequilibrium summary:")
    qa_lines.append(
        model_df[
            ["d_pointwise_linear_v01", "d_pointwise_rf_v01"]
        ].describe().to_string()
    )

    OUT_QA.write_text("\n".join(qa_lines), encoding="utf-8")

    log(f"Created {OUT_GPKG}")
    log(f"Created {OUT_CSV}")
    log(f"Created {OUT_QA}")
    log("Done")


if __name__ == "__main__":
    main()