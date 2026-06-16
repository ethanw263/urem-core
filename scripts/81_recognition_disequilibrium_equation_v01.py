#!/usr/bin/env python3
"""
81_recognition_disequilibrium_equation_v01.py

Purpose
-------
Create the first formal Recognition Disequilibrium Equation (RDE) prototype.

This is the first script where UREM shifts from a score workflow toward a
formal methodology.

Core variables
--------------
P = Physical Potential
O = Opportunity Structure
T = Recognition Transmission
R = Observed Recognition

Core idea
---------
Recognition should be evaluated relative to:
1. latent physical potential
2. opportunity for recognition to accumulate
3. transmission pathways through which recognition spreads
4. actual observed recognition

Outputs
-------
data/processed/recognition_disequilibrium_equation_v01.csv
data/processed/recognition_disequilibrium_equation_v01.gpkg
data/processed/ranked_rde_v01_candidates.csv
data/processed/ranked_rde_v01_candidates.gpkg
data/processed/rde_v01_summary.csv
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = DATA / "recognition_transmission_index_v01.gpkg"

OUT_SCORE_CSV = DATA / "recognition_disequilibrium_equation_v01.csv"
OUT_SCORE_GPKG = DATA / "recognition_disequilibrium_equation_v01.gpkg"
OUT_RANKED_CSV = DATA / "ranked_rde_v01_candidates.csv"
OUT_RANKED_GPKG = DATA / "ranked_rde_v01_candidates.gpkg"
OUT_SUMMARY = DATA / "rde_v01_summary.csv"

TOP_N = 1000

logging.basicConfig(
    level=logging.INFO,
    format="[81_recognition_disequilibrium_equation_v01] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


def minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo = s.min()
    hi = s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def pct_rank(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").rank(pct=True)


def safe_ratio(a, b):
    if b == 0 or pd.isna(b):
        return np.nan
    return a / b


def main():
    log.info(f"Reading input: {INPUT_PATH}")
    gdf = gpd.read_file(INPUT_PATH)

    log.info(f"Rows: {len(gdf):,}")

    required = [
        "cell_id",
        "physical_exceptionality_v03",
        "opportunity_structure_index_v01",
        "recognition_transmission_index_v01",
        "observed_recognition_v04",
        "expected_recognition_v06",
        "recognition_disequilibrium_index_v01",
        "transmission_limited_disequilibrium_v01",
        "is_valid_land_candidate",
    ]

    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    gdf = gdf.copy()

    # ------------------------------------------------------------
    # 1. Normalize core components
    # ------------------------------------------------------------

    gdf["P_physical_potential_v01"] = minmax(gdf["physical_exceptionality_v03"])

    gdf["O_opportunity_structure_v01"] = minmax(gdf["opportunity_structure_index_v01"])

    gdf["T_recognition_transmission_v01"] = minmax(
        gdf["recognition_transmission_index_v01"]
    )

    gdf["R_observed_recognition_v01"] = minmax(gdf["observed_recognition_v04"])

    # Deficit versions.
    # Low opportunity or low transmission can explain low recognition.
    # High expected potential with low observed recognition is disequilibrium.
    gdf["O_opportunity_deficit_v01"] = 1 - gdf["O_opportunity_structure_v01"]
    gdf["T_transmission_deficit_v01"] = 1 - gdf["T_recognition_transmission_v01"]
    gdf["R_recognition_deficit_v01"] = 1 - gdf["R_observed_recognition_v01"]

    # ------------------------------------------------------------
    # 2. Theoretical expected recognition capacity
    # ------------------------------------------------------------
    # This is not observed recognition.
    #
    # It estimates how much recognition could reasonably be expected from
    # the interaction of physical potential, opportunity, and transmission.
    #
    # Multiplicative form:
    # Recognition capacity only becomes high when P, O, and T are all present.
    #
    # Additive diagnostic included for comparison.
    # ------------------------------------------------------------

    gdf["recognition_capacity_multiplicative_v01"] = (
        gdf["P_physical_potential_v01"]
        * gdf["O_opportunity_structure_v01"]
        * gdf["T_recognition_transmission_v01"]
    )

    gdf["recognition_capacity_additive_v01"] = (
        0.50 * gdf["P_physical_potential_v01"]
        + 0.25 * gdf["O_opportunity_structure_v01"]
        + 0.25 * gdf["T_recognition_transmission_v01"]
    )

    # ------------------------------------------------------------
    # 3. Recognition Disequilibrium Equation
    # ------------------------------------------------------------
    # Main disequilibrium:
    #
    # RDE = max(0, Capacity - Observed Recognition)
    #
    # This asks:
    # Given physical potential, opportunity, and transmission, is observed
    # recognition lower than the system would imply?
    # ------------------------------------------------------------

    gdf["rde_v01_raw"] = (
        gdf["recognition_capacity_additive_v01"]
        - gdf["R_observed_recognition_v01"]
    )

    gdf["positive_rde_v01"] = gdf["rde_v01_raw"].clip(lower=0)

    gdf["rde_v01_score"] = minmax(gdf["positive_rde_v01"])

    # ------------------------------------------------------------
    # 4. Transmission-limited disequilibrium subtype
    # ------------------------------------------------------------
    # This is different from main RDE.
    #
    # It detects high physical potential where recognition is low and
    # transmission is also weak.
    #
    # This may represent landscapes isolated from recognition systems.
    # ------------------------------------------------------------

    gdf["rde_transmission_limited_subtype_v01"] = minmax(
        (
            gdf["P_physical_potential_v01"]
            + gdf["T_transmission_deficit_v01"]
            + gdf["R_recognition_deficit_v01"]
        )
        / 3
    )

    # ------------------------------------------------------------
    # 5. Opportunity-failure subtype
    # ------------------------------------------------------------
    # High physical potential + low recognition + low opportunity.
    # ------------------------------------------------------------

    gdf["rde_opportunity_failure_subtype_v01"] = minmax(
        (
            gdf["P_physical_potential_v01"]
            + gdf["O_opportunity_deficit_v01"]
            + gdf["R_recognition_deficit_v01"]
        )
        / 3
    )

    # ------------------------------------------------------------
    # 6. Recognition inefficiency subtype
    # ------------------------------------------------------------
    # High physical potential + high opportunity + high transmission
    # but low observed recognition.
    #
    # This may be the most novel/high-value class.
    # ------------------------------------------------------------

    gdf["rde_recognition_inefficiency_subtype_v01"] = minmax(
        (
            gdf["P_physical_potential_v01"]
            + gdf["O_opportunity_structure_v01"]
            + gdf["T_recognition_transmission_v01"]
            + gdf["R_recognition_deficit_v01"]
        )
        / 4
    )

    # ------------------------------------------------------------
    # 7. Final prototype RDE composite
    # ------------------------------------------------------------
    # This is a reporting composite, not a final theoretical law.
    #
    # We emphasize:
    # - main RDE
    # - recognition inefficiency
    # - transmission-limited disequilibrium
    # ------------------------------------------------------------

    gdf["rde_v01_composite_score"] = minmax(
        (
            0.50 * gdf["rde_v01_score"]
            + 0.30 * gdf["rde_recognition_inefficiency_subtype_v01"]
            + 0.20 * gdf["rde_transmission_limited_subtype_v01"]
        )
    )

    valid = gdf["is_valid_land_candidate"].astype(bool)

    for col in [
        "rde_v01_score",
        "rde_transmission_limited_subtype_v01",
        "rde_opportunity_failure_subtype_v01",
        "rde_recognition_inefficiency_subtype_v01",
        "rde_v01_composite_score",
    ]:
        gdf[col] = gdf[col].where(valid, 0)

    gdf["rde_v01_rank"] = gdf["rde_v01_composite_score"].rank(
        ascending=False,
        method="min",
    )

    # ------------------------------------------------------------
    # 8. Candidate ranking
    # ------------------------------------------------------------

    ranked = (
        gdf.sort_values("rde_v01_composite_score", ascending=False)
        .head(TOP_N)
        .copy()
    )

    ranked["rde_v01_candidate_rank"] = range(1, len(ranked) + 1)

    log.info(f"Ranked RDE candidates: {len(ranked):,}")

    # ------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------

    top300 = ranked.head(300)

    variables = [
        "P_physical_potential_v01",
        "O_opportunity_structure_v01",
        "T_recognition_transmission_v01",
        "R_observed_recognition_v01",
        "O_opportunity_deficit_v01",
        "T_transmission_deficit_v01",
        "R_recognition_deficit_v01",
        "recognition_capacity_multiplicative_v01",
        "recognition_capacity_additive_v01",
        "rde_v01_score",
        "rde_transmission_limited_subtype_v01",
        "rde_opportunity_failure_subtype_v01",
        "rde_recognition_inefficiency_subtype_v01",
        "rde_v01_composite_score",
    ]

    summary_rows = []

    for v in variables:
        summary_rows.append(
            {
                "variable": v,
                "baseline_mean": gdf[v].mean(),
                "baseline_median": gdf[v].median(),
                "top300_mean": top300[v].mean(),
                "top300_median": top300[v].median(),
                "top300_to_baseline_ratio": safe_ratio(top300[v].mean(), gdf[v].mean()),
                "top300_mean_percentile_vs_baseline": (gdf[v] <= top300[v].mean()).mean(),
            }
        )

    summary = pd.DataFrame(summary_rows)

    # ------------------------------------------------------------
    # 10. Outputs
    # ------------------------------------------------------------

    log.info(f"Writing score CSV: {OUT_SCORE_CSV}")
    gdf.drop(columns="geometry").to_csv(OUT_SCORE_CSV, index=False)

    log.info(f"Writing score GPKG: {OUT_SCORE_GPKG}")
    gdf.to_file(OUT_SCORE_GPKG, driver="GPKG")

    log.info(f"Writing ranked CSV: {OUT_RANKED_CSV}")
    ranked.drop(columns="geometry").to_csv(OUT_RANKED_CSV, index=False)

    log.info(f"Writing ranked GPKG: {OUT_RANKED_GPKG}")
    ranked.to_file(OUT_RANKED_GPKG, driver="GPKG")

    log.info(f"Writing summary CSV: {OUT_SUMMARY}")
    summary.to_csv(OUT_SUMMARY, index=False)

    print("\nRecognition Disequilibrium Equation v01")
    print("---------------------------------------")
    print(f"Input cells: {len(gdf):,}")
    print(f"Valid cells: {valid.sum():,}")
    print(f"Ranked candidates: {len(ranked):,}")

    print("\nRDE v01 Summary")
    print("---------------")
    print(summary.to_string(index=False))

    print("\nTop 15 RDE v01 Candidates")
    print("-------------------------")
    display_cols = [
        "rde_v01_candidate_rank",
        "cell_id",
        "rde_v01_composite_score",
        "rde_v01_score",
        "rde_recognition_inefficiency_subtype_v01",
        "rde_transmission_limited_subtype_v01",
        "rde_opportunity_failure_subtype_v01",
        "P_physical_potential_v01",
        "O_opportunity_structure_v01",
        "T_recognition_transmission_v01",
        "R_observed_recognition_v01",
    ]
    print(ranked[display_cols].head(15).to_string(index=False))

    print("\nInterpretation")
    print("--------------")
    print("RDE v01 is the first formal prototype equation using:")
    print("P = physical potential")
    print("O = opportunity structure")
    print("T = recognition transmission")
    print("R = observed recognition")
    print("")
    print("This should be treated as the first candidate mathematical core")
    print("for UREM Methodology Phase II, not as a finalized model.")


if __name__ == "__main__":
    main()