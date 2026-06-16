#!/usr/bin/env python3
"""
87_orthogonalized_rde_dimensions_v01.py

Purpose
-------
Create orthogonalized RDE dimensions to reduce redundancy between:

O = Opportunity Structure
T = Recognition Transmission
R = Observed Recognition

Problem found in Script 86
--------------------------
Opportunity and transmission were nearly redundant:
O-T Spearman correlation ≈ 0.978

This makes mechanism classification collapse.

Scientific question
-------------------
Can we separate:
1. physical potential
2. opportunity
3. transmission beyond opportunity
4. recognition beyond opportunity/transmission

Outputs
-------
data/processed/orthogonalized_rde_dimensions_v01.csv
data/processed/orthogonalized_rde_dimensions_v01.gpkg
data/processed/orthogonalized_rde_dimension_summary_v01.csv
data/processed/orthogonalized_rde_correlation_v01.csv
"""

from pathlib import Path
import logging
import itertools
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

INPUT_GPKG = DATA / "recognition_disequilibrium_equation_v01.gpkg"

OUT_CSV = DATA / "orthogonalized_rde_dimensions_v01.csv"
OUT_GPKG = DATA / "orthogonalized_rde_dimensions_v01.gpkg"
OUT_SUMMARY = DATA / "orthogonalized_rde_dimension_summary_v01.csv"
OUT_CORR = DATA / "orthogonalized_rde_correlation_v01.csv"

logging.basicConfig(
    level=logging.INFO,
    format="[87_orthogonalized_rde_dimensions_v01] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


def minmax(s):
    s = pd.to_numeric(s, errors="coerce")
    lo = s.min()
    hi = s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def residualize(y, X):
    mask = y.notna()

    for col in X.columns:
        mask &= X[col].notna()

    out = pd.Series(np.nan, index=y.index)

    if mask.sum() < 10:
        return out

    model = LinearRegression()
    model.fit(X.loc[mask], y.loc[mask])

    pred = model.predict(X.loc[mask])
    resid = y.loc[mask] - pred

    out.loc[mask] = resid
    return out


def main():
    log.info(f"Reading RDE layer: {INPUT_GPKG}")
    gdf = gpd.read_file(INPUT_GPKG)

    required = [
        "cell_id",
        "P_physical_potential_v01",
        "O_opportunity_structure_v01",
        "T_recognition_transmission_v01",
        "R_observed_recognition_v01",
        "rde_v01_composite_score",
        "is_valid_land_candidate",
    ]

    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    gdf = gdf.copy()
    valid = gdf["is_valid_land_candidate"].astype(bool)

    work = gdf[valid].copy()

    # ------------------------------------------------------------
    # Orthogonalized dimensions
    # ------------------------------------------------------------

    # P remains unchanged.
    gdf["P_orthogonal_v01"] = gdf["P_physical_potential_v01"]

    # O remains base opportunity.
    gdf["O_base_opportunity_v01"] = gdf["O_opportunity_structure_v01"]

    # T residual: transmission not explained by opportunity.
    t_resid = residualize(
        work["T_recognition_transmission_v01"],
        work[["O_opportunity_structure_v01"]],
    )

    gdf.loc[valid, "T_net_transmission_residual_v01"] = t_resid

    # Convert residual to 0-1 scale.
    gdf["T_net_transmission_v01"] = minmax(
        gdf["T_net_transmission_residual_v01"]
    )

    # R residual: observed recognition not explained by opportunity/transmission.
    r_resid = residualize(
        work["R_observed_recognition_v01"],
        work[
            [
                "O_opportunity_structure_v01",
                "T_recognition_transmission_v01",
            ]
        ],
    )

    gdf.loc[valid, "R_net_recognition_residual_v01"] = r_resid

    # Low residual recognition means recognition is lower than opportunity/transmission predict.
    gdf["R_net_under_recognition_v01"] = minmax(
        -gdf["R_net_recognition_residual_v01"]
    )

    # ------------------------------------------------------------
    # Orthogonalized RDE
    # ------------------------------------------------------------

    gdf["orthogonalized_rde_v01"] = minmax(
        (
            0.40 * gdf["P_orthogonal_v01"]
            + 0.25 * gdf["O_base_opportunity_v01"]
            + 0.20 * gdf["T_net_transmission_v01"]
            + 0.15 * gdf["R_net_under_recognition_v01"]
        )
    )

    gdf["orthogonalized_rde_v01"] = gdf["orthogonalized_rde_v01"].where(valid, 0)

    gdf["orthogonalized_rde_rank_v01"] = gdf[
        "orthogonalized_rde_v01"
    ].rank(ascending=False, method="min")

    # ------------------------------------------------------------
    # New separability check
    # ------------------------------------------------------------

    dims = {
        "P_orthogonal": "P_orthogonal_v01",
        "O_base_opportunity": "O_base_opportunity_v01",
        "T_net_transmission": "T_net_transmission_v01",
        "R_net_under_recognition": "R_net_under_recognition_v01",
    }

    corr_rows = []

    valid_gdf = gdf[valid].copy()

    for (a_name, a_col), (b_name, b_col) in itertools.combinations(dims.items(), 2):
        pearson = valid_gdf[a_col].corr(valid_gdf[b_col], method="pearson")
        spearman = valid_gdf[a_col].corr(valid_gdf[b_col], method="spearman")

        corr_rows.append(
            {
                "dimension_a": a_name,
                "dimension_b": b_name,
                "pearson_corr": pearson,
                "spearman_corr": spearman,
                "abs_spearman_corr": abs(spearman),
                "interpretation": (
                    "high_overlap"
                    if abs(spearman) >= 0.75
                    else "moderate_overlap"
                    if abs(spearman) >= 0.40
                    else "low_overlap"
                ),
            }
        )

    corr = pd.DataFrame(corr_rows).sort_values(
        "abs_spearman_corr",
        ascending=False,
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    top300 = gdf.sort_values("orthogonalized_rde_v01", ascending=False).head(300)

    summary_rows = []

    for col in [
        "P_orthogonal_v01",
        "O_base_opportunity_v01",
        "T_net_transmission_v01",
        "R_net_under_recognition_v01",
        "orthogonalized_rde_v01",
        "rde_v01_composite_score",
    ]:
        summary_rows.append(
            {
                "variable": col,
                "baseline_mean": valid_gdf[col].mean(),
                "baseline_median": valid_gdf[col].median(),
                "top300_mean": top300[col].mean(),
                "top300_median": top300[col].median(),
                "top300_to_baseline_ratio": (
                    top300[col].mean() / valid_gdf[col].mean()
                    if valid_gdf[col].mean() != 0
                    else np.nan
                ),
                "top300_mean_percentile_vs_baseline": (
                    valid_gdf[col] <= top300[col].mean()
                ).mean(),
            }
        )

    summary = pd.DataFrame(summary_rows)

    # ------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------

    log.info(f"Writing CSV: {OUT_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    log.info(f"Writing GPKG: {OUT_GPKG}")
    gdf.to_file(OUT_GPKG, driver="GPKG")

    log.info(f"Writing summary: {OUT_SUMMARY}")
    summary.to_csv(OUT_SUMMARY, index=False)

    log.info(f"Writing correlation: {OUT_CORR}")
    corr.to_csv(OUT_CORR, index=False)

    print("\nOrthogonalized RDE Dimensions v01")
    print("---------------------------------")

    print("\nNew pairwise correlations:")
    print(corr.to_string(index=False))

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nTop 15 Orthogonalized RDE candidates:")
    display_cols = [
        "cell_id",
        "orthogonalized_rde_rank_v01",
        "orthogonalized_rde_v01",
        "P_orthogonal_v01",
        "O_base_opportunity_v01",
        "T_net_transmission_v01",
        "R_net_under_recognition_v01",
        "rde_v01_composite_score",
    ]

    print(
        gdf.sort_values("orthogonalized_rde_v01", ascending=False)
        [display_cols]
        .head(15)
        .to_string(index=False)
    )

    print("\nInterpretation")
    print("--------------")
    print("This creates a cleaner mechanism space by separating transmission")
    print("from opportunity, and under-recognition from opportunity/transmission.")
    print("If correlations drop materially, this becomes the better foundation")
    print("for future mechanism classification.")


if __name__ == "__main__":
    main()