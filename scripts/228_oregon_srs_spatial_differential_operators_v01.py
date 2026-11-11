#!/usr/bin/env python3
"""
228_oregon_srs_spatial_differential_operators_v01.py

Construct interpretable spatial differential / transition operators for the
Oregon SRS operator-persistence branch.

Purpose:
Replace part of the unresolved spatial term S_i with measurable spatial
operators:
- local x/y gradients
- gradient magnitude
- directional anisotropy
- local-to-regional differential structure
- transition/boundary strength

This script only creates features. It does not fit a model and does not build
an adaptive SRS gate.
"""

from pathlib import Path
import logging
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


SCRIPT = "228_oregon_srs_spatial_differential_operators_v01"

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"

INPUT_PERSISTENCE = P / "oregon_srs_relational_persistence_common_population_v01.gpkg"
INPUT_ENGINE = P / "oregon_srs_engine_inputs_v01.gpkg"
INPUT_CONTEXT = P / "oregon_srs_relational_context_v01.gpkg"
INPUT_IMPORTANCE = P / "oregon_srs_persistence_classification_feature_importance_stability_v01.csv"

OUT_GPKG = P / "oregon_srs_spatial_differential_operators_v01.gpkg"
OUT_CSV = P / "oregon_srs_spatial_differential_operators_v01.csv"
OUT_FEATURES = P / "oregon_srs_spatial_differential_operator_feature_manifest_v01.csv"
OUT_QA = P / "oregon_srs_spatial_differential_operators_qa_v01.txt"

TARGET = {
    "persistent_pointwise": 0,
    "persistent_relational": 1,
}

N_BASE_CONTEXT = 16
RADII_M = [3000, 10000, 25000]

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

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT}] %(levelname)s: %(message)s",
)

log = logging.getLogger(SCRIPT)


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

    return gpd.GeoDataFrame(df, geometry="geometry", crs=base.crs), idc


def select_context_features(df):
    imp = pd.read_csv(INPUT_IMPORTANCE)
    imp = imp[imp["feature_family"] == "full_admissible_state"].copy()

    ctx = (
        imp[imp["feature_group"] == "relational_context"]
        .sort_values("mean_importance", ascending=False)["raw_feature"]
        .drop_duplicates()
        .tolist()
    )

    ctx = [
        c for c in ctx
        if c in df.columns
        and not bad(c)
        and pd.api.types.is_numeric_dtype(df[c])
    ][:N_BASE_CONTEXT]

    if not ctx:
        raise ValueError("No context features selected.")

    return ctx


def add_xy(df):
    cent = df.geometry.centroid
    df["x_coord"] = cent.x
    df["y_coord"] = cent.y
    return df


def safe_name(name):
    return (
        name
        .replace("physical_exceptionality_raw_v03", "phys_exc_raw")
        .replace("physical_exceptionality_v03", "phys_exc")
        .replace("terrain_drama_v03", "terrain_drama")
        .replace("relief_norm_v03", "relief_norm")
        .replace("slope_norm_v03", "slope_norm")
        .replace("phi_physical_v01", "phi_phys")
        .replace("coast_proximity_v03", "coast_prox")
        .replace("coast_complexity_v03", "coast_complex")
        .replace("_v01", "")
        .replace("_v03", "")
        .replace("_", "_")
    )


def local_plane_gradient(x, y, z):
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    if valid.sum() < 4:
        return np.nan, np.nan, np.nan, np.nan

    xx = x[valid]
    yy = y[valid]
    zz = z[valid]

    x0 = xx.mean()
    y0 = yy.mean()

    X = np.column_stack([
        xx - x0,
        yy - y0,
        np.ones(len(xx)),
    ])

    try:
        beta, *_ = np.linalg.lstsq(X, zz, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan

    gx = float(beta[0])
    gy = float(beta[1])
    mag = float(np.sqrt(gx ** 2 + gy ** 2))
    angle = float(np.arctan2(gy, gx))

    return gx, gy, mag, angle


def neighbor_stats(values, center_value):
    valid = values[np.isfinite(values)]

    if len(valid) == 0 or not np.isfinite(center_value):
        return {
            "nbr_mean": np.nan,
            "nbr_std": np.nan,
            "center_minus_nbr_mean": np.nan,
            "transition_strength": np.nan,
            "range": np.nan,
        }

    nbr_mean = float(np.mean(valid))
    nbr_std = float(np.std(valid))
    diff = float(center_value - nbr_mean)

    return {
        "nbr_mean": nbr_mean,
        "nbr_std": nbr_std,
        "center_minus_nbr_mean": diff,
        "transition_strength": abs(diff),
        "range": float(np.max(valid) - np.min(valid)),
    }


def build_differential_features(df, context_features):
    coords = df[["x_coord", "y_coord"]].to_numpy()
    tree = BallTree(coords, metric="euclidean")

    out = df[[
        "geometry",
        "x_coord",
        "y_coord",
    ]].copy()

    feature_rows = []

    for feature in context_features:
        log.info(f"Processing feature: {feature}")

        base = safe_name(feature)
        z_all = df[feature].astype(float).to_numpy()

        out[feature] = z_all

        for radius in RADII_M:
            rtag = f"{int(radius / 1000)}km"

            indices = tree.query_radius(coords, r=radius)

            gx = np.full(len(df), np.nan)
            gy = np.full(len(df), np.nan)
            gmag = np.full(len(df), np.nan)
            gang = np.full(len(df), np.nan)

            nbr_mean = np.full(len(df), np.nan)
            nbr_std = np.full(len(df), np.nan)
            center_minus = np.full(len(df), np.nan)
            transition = np.full(len(df), np.nan)
            local_range = np.full(len(df), np.nan)
            n_neighbors = np.full(len(df), np.nan)

            for i, inds in enumerate(indices):
                if len(inds) <= 1:
                    continue

                # Include center cell for local plane; exclude center for neighbor mean.
                x = coords[inds, 0]
                y = coords[inds, 1]
                z = z_all[inds]

                gx_i, gy_i, gm_i, ga_i = local_plane_gradient(x, y, z)

                gx[i] = gx_i
                gy[i] = gy_i
                gmag[i] = gm_i
                gang[i] = ga_i

                neighbor_inds = inds[inds != i]
                stats = neighbor_stats(z_all[neighbor_inds], z_all[i])

                nbr_mean[i] = stats["nbr_mean"]
                nbr_std[i] = stats["nbr_std"]
                center_minus[i] = stats["center_minus_nbr_mean"]
                transition[i] = stats["transition_strength"]
                local_range[i] = stats["range"]
                n_neighbors[i] = len(neighbor_inds)

            cols = {
                f"{base}_grad_x_{rtag}": gx,
                f"{base}_grad_y_{rtag}": gy,
                f"{base}_grad_mag_{rtag}": gmag,
                f"{base}_grad_angle_{rtag}": gang,
                f"{base}_nbr_mean_{rtag}": nbr_mean,
                f"{base}_nbr_std_{rtag}": nbr_std,
                f"{base}_center_minus_nbr_mean_{rtag}": center_minus,
                f"{base}_transition_strength_{rtag}": transition,
                f"{base}_local_range_{rtag}": local_range,
                f"{base}_n_neighbors_{rtag}": n_neighbors,
            }

            for col, arr in cols.items():
                out[col] = arr

                if "grad_x" in col:
                    group = "gradient_x"
                elif "grad_y" in col:
                    group = "gradient_y"
                elif "grad_mag" in col:
                    group = "gradient_magnitude"
                elif "grad_angle" in col:
                    group = "gradient_angle"
                elif "center_minus" in col:
                    group = "center_neighbor_difference"
                elif "transition_strength" in col:
                    group = "transition_strength"
                elif "local_range" in col:
                    group = "local_range"
                elif "nbr_std" in col:
                    group = "neighbor_dispersion"
                elif "nbr_mean" in col:
                    group = "neighbor_mean"
                elif "n_neighbors" in col:
                    group = "neighbor_count"
                else:
                    group = "other"

                feature_rows.append({
                    "source_feature": feature,
                    "created_feature": col,
                    "operator_group": group,
                    "radius_m": radius,
                    "missing_n": int(pd.isna(arr).sum()),
                    "missing_share": float(pd.isna(arr).mean()),
                    "mean": float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan,
                    "std": float(np.nanstd(arr)) if np.isfinite(arr).any() else np.nan,
                    "min": float(np.nanmin(arr)) if np.isfinite(arr).any() else np.nan,
                    "max": float(np.nanmax(arr)) if np.isfinite(arr).any() else np.nan,
                })

    # Add cross-scale derivative summaries for matching operator families.
    created = list(out.columns)

    for feature in context_features:
        base = safe_name(feature)

        for op in [
            "grad_mag",
            "transition_strength",
            "center_minus_nbr_mean",
            "local_range",
            "nbr_std",
        ]:
            c3 = f"{base}_{op}_3km"
            c10 = f"{base}_{op}_10km"
            c25 = f"{base}_{op}_25km"

            pairs = [
                (c3, c10, "3_minus_10"),
                (c10, c25, "10_minus_25"),
                (c3, c25, "3_minus_25"),
            ]

            for a, b, tag in pairs:
                if a in out.columns and b in out.columns:
                    new = f"{base}_{op}_{tag}"
                    out[new] = out[a] - out[b]

                    arr = out[new].to_numpy()

                    feature_rows.append({
                        "source_feature": feature,
                        "created_feature": new,
                        "operator_group": "cross_scale_difference",
                        "radius_m": np.nan,
                        "missing_n": int(pd.isna(arr).sum()),
                        "missing_share": float(pd.isna(arr).mean()),
                        "mean": float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan,
                        "std": float(np.nanstd(arr)) if np.isfinite(arr).any() else np.nan,
                        "min": float(np.nanmin(arr)) if np.isfinite(arr).any() else np.nan,
                        "max": float(np.nanmax(arr)) if np.isfinite(arr).any() else np.nan,
                    })

    manifest = pd.DataFrame(feature_rows)

    return gpd.GeoDataFrame(out, geometry="geometry", crs=df.crs), manifest


def main():
    log.info("Starting Script 228")

    df, idc = load()
    df = add_xy(df)

    regime_col = find_col(df, ["persistence_regime_v01", "persistence_regime"])
    fold_col = find_col(df, ["buffered_spatial_fold_v01", "spatial_fold_v01", "fold_v01"])

    keep_cols = [
        idc,
        regime_col,
        fold_col,
        "geometry",
        "x_coord",
        "y_coord",
    ]

    df = df[df[regime_col].isin(TARGET)].copy()
    df["persistence_target_v01"] = df[regime_col].map(TARGET).astype(int)

    context_features = select_context_features(df)

    base_cols = list(dict.fromkeys(
        keep_cols + ["persistence_target_v01"] + context_features
    ))

    work = df[base_cols].copy()

    log.info(f"Rows: {len(work):,}")
    log.info(f"Selected context source features: {len(context_features)}")

    out, manifest = build_differential_features(work, context_features)

    # Reattach IDs and target metadata.
    metadata = work[[idc, regime_col, fold_col, "persistence_target_v01"]].copy()
    out = pd.concat(
        [
            metadata.reset_index(drop=True),
            out.reset_index(drop=True),
        ],
        axis=1,
    )

    out = gpd.GeoDataFrame(out, geometry="geometry", crs=work.crs)
    out = out.loc[:, ~out.columns.duplicated()].copy()

    out.to_file(OUT_GPKG, driver="GPKG")
    out.drop(columns="geometry").to_csv(OUT_CSV, index=False)
    manifest.to_csv(OUT_FEATURES, index=False)

    group_summary = (
        manifest
        .groupby("operator_group", as_index=False)
        .agg(
            n_features=("created_feature", "count"),
            mean_missing_share=("missing_share", "mean"),
        )
        .sort_values("operator_group")
    )

    qa = [
        f"Script: {SCRIPT}",
        "",
        "PURPOSE",
        "Construct interpretable spatial differential, transition, and cross-scale operators for Oregon SRS.",
        "",
        f"Rows: {len(out):,}",
        f"Source context features: {len(context_features)}",
        f"Created operator features: {len(manifest):,}",
        "",
        "SOURCE CONTEXT FEATURES",
        "\n".join(context_features),
        "",
        "OPERATOR GROUP SUMMARY",
        group_summary.to_string(index=False),
        "",
        "OUTPUTS",
        str(OUT_GPKG),
        str(OUT_CSV),
        str(OUT_FEATURES),
        "",
        "NEXT STEP",
        "Use these spatial differential operators in a controlled coordinate-substitution model.",
        "The next test should ask whether directional/transition features recover the coordinate/trend gain.",
    ]

    OUT_QA.write_text("\n".join(qa), encoding="utf-8")

    log.info("Completed Script 228")
    log.info(f"Created operator features: {len(manifest):,}")
    log.info(f"Wrote: {OUT_GPKG}")
    log.info(f"Wrote: {OUT_FEATURES}")


if __name__ == "__main__":
    main()