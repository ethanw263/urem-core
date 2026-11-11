from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

SCRIPT = "209_oregon_srs_operator_comparison_v01"

BASE = Path("data/processed")
INPUT_FP = BASE / "oregon_srs_relational_transformation_v01.gpkg"

OUT_GPKG = BASE / "oregon_srs_operator_comparison_v01.gpkg"
OUT_CSV = BASE / "oregon_srs_operator_comparison_v01.csv"
OUT_QA = BASE / "oregon_srs_operator_comparison_qa_v01.txt"

TARGET = "y_observed_v01"
VALID = "srs_valid_engine_row_v01"
FOLD = "spatial_fold_v01"

PT_COL = "pred_pointwise_linear_oof_v01"
REL_COL = "pred_relational_rf_oof_v01"


def log(msg):
    print(f"[{SCRIPT}] {msg}")


def main():
    log("Reading relational transformation output")
    gdf = gpd.read_file(INPUT_FP)

    required = [TARGET, VALID, FOLD, PT_COL, REL_COL]
    for c in required:
        if c not in gdf.columns:
            raise ValueError(f"Missing required column: {c}")

    mask = (
        gdf[VALID].astype(bool)
        & gdf[TARGET].notna()
        & gdf[PT_COL].notna()
        & gdf[REL_COL].notna()
    )

    log(f"Comparable rows: {mask.sum():,}")

    y = gdf.loc[mask, TARGET].astype(float)
    pt = gdf.loc[mask, PT_COL].astype(float)
    rel = gdf.loc[mask, REL_COL].astype(float)

    gdf["abs_error_pointwise_v01"] = np.nan
    gdf["abs_error_relational_v01"] = np.nan
    gdf["relational_gain_v01"] = np.nan
    gdf["relational_gain_pct_v01"] = np.nan
    gdf["operator_preference_v01"] = None
    gdf["d_pointwise_best_v01"] = np.nan
    gdf["d_relational_best_v01"] = np.nan
    gdf["d_shift_rel_minus_pt_v01"] = np.nan

    pt_err = (y - pt).abs()
    rel_err = (y - rel).abs()
    gain = pt_err - rel_err

    gdf.loc[mask, "abs_error_pointwise_v01"] = pt_err
    gdf.loc[mask, "abs_error_relational_v01"] = rel_err
    gdf.loc[mask, "relational_gain_v01"] = gain
    gdf.loc[mask, "relational_gain_pct_v01"] = gain / pt_err.replace(0, np.nan)

    gdf.loc[mask & (gdf["relational_gain_v01"] > 0), "operator_preference_v01"] = "relational_better"
    gdf.loc[mask & (gdf["relational_gain_v01"] < 0), "operator_preference_v01"] = "pointwise_better"
    gdf.loc[mask & (gdf["relational_gain_v01"] == 0), "operator_preference_v01"] = "tie"

    gdf.loc[mask, "d_pointwise_best_v01"] = pt - y
    gdf.loc[mask, "d_relational_best_v01"] = rel - y
    gdf.loc[mask, "d_shift_rel_minus_pt_v01"] = (
        gdf.loc[mask, "d_relational_best_v01"]
        - gdf.loc[mask, "d_pointwise_best_v01"]
    )

    comp = gdf.loc[mask].copy()

    overall = {
        "n": int(len(comp)),
        "mean_abs_error_pointwise": float(comp["abs_error_pointwise_v01"].mean()),
        "mean_abs_error_relational": float(comp["abs_error_relational_v01"].mean()),
        "mean_relational_gain": float(comp["relational_gain_v01"].mean()),
        "median_relational_gain": float(comp["relational_gain_v01"].median()),
        "share_relational_better": float((comp["relational_gain_v01"] > 0).mean()),
        "share_pointwise_better": float((comp["relational_gain_v01"] < 0).mean()),
    }

    fold_summary = (
        comp.groupby(FOLD)
        .agg(
            n=("relational_gain_v01", "size"),
            mean_gain=("relational_gain_v01", "mean"),
            median_gain=("relational_gain_v01", "median"),
            share_relational_better=("relational_gain_v01", lambda s: float((s > 0).mean())),
            mae_pointwise=("abs_error_pointwise_v01", "mean"),
            mae_relational=("abs_error_relational_v01", "mean"),
        )
        .reset_index()
        .sort_values(FOLD)
    )

    quantile_cols = [
        "phi_physical_v01",
        "y_observed_v01",
        "coast_proximity_v03",
        "coast_complexity_v03",
        "rel_neighbor_count_3km_v01",
        "rel_neighbor_count_10km_v01",
        "rel_neighbor_count_25km_v01",
    ]

    quantile_summaries = []
    for col in quantile_cols:
        if col not in comp.columns:
            continue
        tmp = comp[[col, "relational_gain_v01"]].dropna().copy()
        if tmp[col].nunique() < 4:
            continue
        tmp["quartile"] = pd.qcut(tmp[col], 4, labels=False, duplicates="drop")
        qsum = (
            tmp.groupby("quartile")
            .agg(
                n=("relational_gain_v01", "size"),
                mean_gain=("relational_gain_v01", "mean"),
                median_gain=("relational_gain_v01", "median"),
                share_relational_better=("relational_gain_v01", lambda s: float((s > 0).mean())),
            )
            .reset_index()
        )
        qsum.insert(0, "variable", col)
        quantile_summaries.append(qsum)

    quantile_summary = (
        pd.concat(quantile_summaries, ignore_index=True)
        if quantile_summaries else pd.DataFrame()
    )

    comp["abs_d_pointwise_rank_v01"] = comp["d_pointwise_best_v01"].abs().rank(
        ascending=False, method="min"
    )
    comp["abs_d_relational_rank_v01"] = comp["d_relational_best_v01"].abs().rank(
        ascending=False, method="min"
    )

    top_n = 500
    top_pt = set(comp.nsmallest(top_n, "abs_d_pointwise_rank_v01")["cell_id"])
    top_rel = set(comp.nsmallest(top_n, "abs_d_relational_rank_v01")["cell_id"])
    overlap = len(top_pt & top_rel)
    jaccard = overlap / len(top_pt | top_rel)

    qa = []
    qa.append(f"{SCRIPT} QA")
    qa.append("=" * 80)
    qa.append(f"Input: {INPUT_FP}")
    qa.append(f"Output: {OUT_GPKG}")
    qa.append("")
    qa.append("Comparator definitions:")
    qa.append(f"Pointwise comparator: {PT_COL}")
    qa.append(f"Relational comparator: {REL_COL}")
    qa.append("")
    qa.append("Overall relational gain:")
    for k, v in overall.items():
        qa.append(f"{k}: {v}")
    qa.append("")
    qa.append(f"Top {top_n} absolute-disequilibrium overlap:")
    qa.append(f"overlap: {overlap}")
    qa.append(f"jaccard: {jaccard}")
    qa.append("")
    qa.append("Fold summary:")
    qa.append(fold_summary.to_string(index=False))
    qa.append("")
    qa.append("Gain by quartile:")
    if len(quantile_summary):
        qa.append(quantile_summary.to_string(index=False))
    else:
        qa.append("No quantile summaries created.")
    qa.append("")
    qa.append("Relational gain distribution:")
    qa.append(comp["relational_gain_v01"].describe().to_string())
    qa.append("")
    qa.append("Operator preference counts:")
    qa.append(comp["operator_preference_v01"].value_counts(dropna=False).to_string())

    log("Writing outputs")
    gdf.to_file(OUT_GPKG, driver="GPKG")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)
    OUT_QA.write_text("\n".join(qa), encoding="utf-8")

    log(f"Created {OUT_GPKG}")
    log(f"Created {OUT_CSV}")
    log(f"Created {OUT_QA}")
    log("Done")


if __name__ == "__main__":
    main()