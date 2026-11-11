from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

SCRIPT = "210_oregon_srs_relational_gain_regimes_v01"

BASE = Path("data/processed")
INPUT_FP = BASE / "oregon_srs_operator_comparison_v01.gpkg"

OUT_GPKG = BASE / "oregon_srs_relational_gain_regimes_v01.gpkg"
OUT_CSV = BASE / "oregon_srs_relational_gain_regimes_v01.csv"
OUT_QA = BASE / "oregon_srs_relational_gain_regimes_qa_v01.txt"

VALID = "srs_valid_engine_row_v01"
GAIN = "relational_gain_v01"


def log(msg):
    print(f"[{SCRIPT}] {msg}")


def classify_gain(x):
    if pd.isna(x):
        return None
    if x >= 0.05:
        return "strong_relational_gain"
    if x >= 0.015:
        return "moderate_relational_gain"
    if x > 0:
        return "weak_relational_gain"
    if x <= -0.05:
        return "strong_pointwise_gain"
    if x <= -0.015:
        return "moderate_pointwise_gain"
    return "weak_pointwise_gain"


def main():
    log("Reading operator comparison layer")
    gdf = gpd.read_file(INPUT_FP)

    if GAIN not in gdf.columns:
        raise ValueError(f"Missing {GAIN}")

    mask = gdf[VALID].astype(bool) & gdf[GAIN].notna()
    out = gdf.copy()

    out["gain_regime_v01"] = None
    out.loc[mask, "gain_regime_v01"] = out.loc[mask, GAIN].apply(classify_gain)

    out["abs_relational_gain_v01"] = out[GAIN].abs()
    out["relational_gain_rank_v01"] = np.nan
    out.loc[mask, "relational_gain_rank_v01"] = (
        out.loc[mask, GAIN].rank(ascending=False, method="min")
    )

    regime_summary = (
        out.loc[mask]
        .groupby("gain_regime_v01")
        .agg(
            n=(GAIN, "size"),
            mean_gain=(GAIN, "mean"),
            median_gain=(GAIN, "median"),
            mean_phi=("phi_physical_v01", "mean"),
            mean_y=("y_observed_v01", "mean"),
            mean_coast_prox=("coast_proximity_v03", "mean"),
            mean_coast_complexity=("coast_complexity_v03", "mean"),
            mean_neighbors_3km=("rel_neighbor_count_3km_v01", "mean"),
            mean_neighbors_10km=("rel_neighbor_count_10km_v01", "mean"),
            mean_neighbors_25km=("rel_neighbor_count_25km_v01", "mean"),
        )
        .reset_index()
        .sort_values("mean_gain", ascending=False)
    )

    fold_summary = (
        out.loc[mask]
        .groupby("spatial_fold_v01")
        .agg(
            n=(GAIN, "size"),
            mean_gain=(GAIN, "mean"),
            median_gain=(GAIN, "median"),
            strong_rel_gain_share=(
                "gain_regime_v01",
                lambda s: float((s == "strong_relational_gain").mean()),
            ),
            strong_pt_gain_share=(
                "gain_regime_v01",
                lambda s: float((s == "strong_pointwise_gain").mean()),
            ),
        )
        .reset_index()
        .sort_values("mean_gain", ascending=False)
    )

    top_rel = out.loc[mask].nlargest(100, GAIN)[
        [
            "cell_id",
            GAIN,
            "gain_regime_v01",
            "phi_physical_v01",
            "y_observed_v01",
            "d_pointwise_best_v01",
            "d_relational_best_v01",
            "spatial_fold_v01",
        ]
    ]

    top_pt = out.loc[mask].nsmallest(100, GAIN)[
        [
            "cell_id",
            GAIN,
            "gain_regime_v01",
            "phi_physical_v01",
            "y_observed_v01",
            "d_pointwise_best_v01",
            "d_relational_best_v01",
            "spatial_fold_v01",
        ]
    ]

    qa = []
    qa.append(f"{SCRIPT} QA")
    qa.append("=" * 80)
    qa.append(f"Input: {INPUT_FP}")
    qa.append(f"Output: {OUT_GPKG}")
    qa.append(f"Rows total: {len(out):,}")
    qa.append(f"Rows analyzed: {int(mask.sum()):,}")
    qa.append("")
    qa.append("Gain regime counts:")
    qa.append(out.loc[mask, "gain_regime_v01"].value_counts().to_string())
    qa.append("")
    qa.append("Regime summary:")
    qa.append(regime_summary.to_string(index=False))
    qa.append("")
    qa.append("Fold summary:")
    qa.append(fold_summary.to_string(index=False))
    qa.append("")
    qa.append("Top 100 relational-gain cells:")
    qa.append(top_rel.to_string(index=False))
    qa.append("")
    qa.append("Top 100 pointwise-gain cells:")
    qa.append(top_pt.to_string(index=False))

    log("Writing outputs")
    out.to_file(OUT_GPKG, driver="GPKG")
    out.drop(columns="geometry").to_csv(OUT_CSV, index=False)
    OUT_QA.write_text("\n".join(qa), encoding="utf-8")

    log(f"Created {OUT_GPKG}")
    log(f"Created {OUT_CSV}")
    log(f"Created {OUT_QA}")
    log("Done")


if __name__ == "__main__":
    main()