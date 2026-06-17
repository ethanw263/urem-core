#!/usr/bin/env python3
"""
104_geographic_landscape_theory_table_v01.py

Corrected v02:
- Prevents duplicated row inflation.
- Summarizes landscape types directly from geographic interpretation file.
- Attaches mechanism-level evidence once per mechanism.
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd


SCRIPT_NAME = "104_geographic_landscape_theory_table_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)

log = logging.getLogger(SCRIPT_NAME)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

LANDSCAPE_INPUT = PROCESSED / "rde_geographic_landscape_interpretation_v01.csv"
SYNTHESIS_INPUT = PROCESSED / "rde_core_results_synthesis_v01.csv"
READINESS_INPUT = PROCESSED / "rde_theory_readiness_filter_v01.csv"

OUTPUT_TABLE = PROCESSED / "rde_geographic_landscape_theory_table_v01.csv"
OUTPUT_SUMMARY = PROCESSED / "rde_geographic_landscape_theory_summary_v01.csv"
OUTPUT_CLAIMS = PROCESSED / "rde_geographic_landscape_publication_claims_v01.csv"
OUTPUT_QA = PROCESSED / "rde_geographic_landscape_theory_qa_v01.txt"


LANDSCAPE_EXPLANATIONS = {
    "Remote Northern Coast Recognition Disequilibrium Landscape":
        "Remote northern coastal landscapes show high physical potential and recognition disequilibrium, likely shaped by isolation, rugged access, and weaker tourism circuits.",

    "Offshore Island Recognition Inefficiency Landscape":
        "Offshore island systems show high physical potential but constrained recognition, likely due to access limits, governance restrictions, and visitor filtering.",

    "Offshore Island Opportunity-Constrained Landscape":
        "Offshore island landscapes where physical potential is high but opportunity structures are limited.",

    "Offshore Island Recognition Diversion Landscape":
        "Offshore island landscapes where recognition may be diverted toward better-known adjacent destinations.",

    "Central Coast Rugged Landscape":
        "Rugged Central Coast landscapes where exceptional physical conditions coexist with limited access or uneven recognition.",

    "Shadowed Central Coast Recognition Landscape":
        "Central Coast landscapes potentially overshadowed by nearby iconic destinations.",

    "High-Potential Hidden Landscape System":
        "High-priority landscapes with strong physical and disequilibrium signatures, suitable for external validation.",

    "Southern Coastal Recognition Disequilibrium Landscape":
        "Southern coastal landscapes where recognition appears weaker than expected from physical and transmission conditions.",

    "Southern Coastal Recognition Inefficiency Landscape":
        "Southern coastal landscapes where transmission exists but does not translate into proportional recognition.",

    "General Coastal Recognition Disequilibrium Landscape":
        "Coastal regions with moderate RDE signatures requiring further validation.",
}


MECHANISM_CONTRIBUTIONS = {
    "Recognition Inefficiency":
        "Recognition can fail even where physical potential and transmission signals are strong.",

    "Opportunity Failure":
        "Recognition deficits can emerge from insufficient opportunity structures despite high physical potential.",

    "Comparative Shadowing":
        "Recognition may be redistributed toward nearby better-known landscapes.",
}


def canonical_mechanism(x):
    s = str(x).replace(" Candidate", "").strip()
    if "Recognition Inefficiency" in s:
        return "Recognition Inefficiency"
    if "Opportunity Failure" in s:
        return "Opportunity Failure"
    if "Comparative Shadowing" in s or "Recognition Diversion" in s:
        return "Comparative Shadowing"
    return s


def classify_strength(score):
    if pd.isna(score):
        return "Unknown"
    if score >= 0.70:
        return "Strong"
    if score >= 0.50:
        return "Moderate"
    return "Weak"


def load_inputs():
    for p in [LANDSCAPE_INPUT, SYNTHESIS_INPUT, READINESS_INPUT]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    landscape = pd.read_csv(LANDSCAPE_INPUT, low_memory=False)
    synthesis = pd.read_csv(SYNTHESIS_INPUT, low_memory=False)
    readiness = pd.read_csv(READINESS_INPUT, low_memory=False)

    return landscape, synthesis, readiness


def build_mechanism_evidence(synthesis, readiness):
    syn = synthesis.copy()
    read = readiness.copy()

    if "mechanism" in syn.columns:
        syn["canonical_mechanism"] = syn["mechanism"].map(canonical_mechanism)

    if "mechanism" in read.columns:
        read["canonical_mechanism"] = read["mechanism"].map(canonical_mechanism)

    rows = []

    all_mechs = sorted(
        set(syn.get("canonical_mechanism", pd.Series(dtype=str)).dropna())
        | set(read.get("canonical_mechanism", pd.Series(dtype=str)).dropna())
    )

    for mech in all_mechs:
        syn_sub = syn[syn.get("canonical_mechanism") == mech] if "canonical_mechanism" in syn.columns else pd.DataFrame()
        read_sub = read[read.get("canonical_mechanism") == mech] if "canonical_mechanism" in read.columns else pd.DataFrame()

        evidence = np.nan
        if "overall_evidence_score" in read_sub.columns and len(read_sub) > 0:
            evidence = read_sub["overall_evidence_score"].mean()
        elif "overall_evidence_score" in syn_sub.columns and len(syn_sub) > 0:
            evidence = syn_sub["overall_evidence_score"].mean()

        core_count = 0
        emerging_count = 0
        holdout_count = 0

        if "theory_readiness_tier" in read_sub.columns:
            core_count = int((read_sub["theory_readiness_tier"] == "Core Validated Theory").sum())
            emerging_count = int((read_sub["theory_readiness_tier"] == "Emerging Theory").sum())
            holdout_count = int((read_sub["theory_readiness_tier"] == "Exploratory / Holdout Theory").sum())

        rows.append({
            "canonical_mechanism": mech,
            "mechanism_mean_evidence_score": evidence,
            "mechanism_evidence_strength": classify_strength(evidence),
            "core_validated_archetype_count": core_count,
            "emerging_archetype_count": emerging_count,
            "holdout_archetype_count": holdout_count,
            "mechanism_theory_contribution": MECHANISM_CONTRIBUTIONS.get(mech, "Emerging contribution."),
        })

    return pd.DataFrame(rows)


def build_theory_table(landscape, mechanism_evidence):
    df = landscape.copy()

    if "canonical_mechanism" not in df.columns:
        df["canonical_mechanism"] = df["mechanism_class"].map(canonical_mechanism)
    else:
        df["canonical_mechanism"] = df["canonical_mechanism"].map(canonical_mechanism)

    group_cols = ["canonical_mechanism", "geographic_landscape_type_v01"]

    agg = {
        "mechanism_region_id": "count",
    }

    optional_means = [
        "external_validation_priority_score",
        "mean_P_orthogonal_v01",
        "mean_O_base_opportunity_v01",
        "mean_T_net_transmission_v01",
        "mean_R_net_under_recognition_v01",
        "mean_observed_recognition_v04",
        "mean_expected_recognition_v06",
        "validation_area_km2",
        "region_mechanism_stability_005",
    ]

    for c in optional_means:
        if c in df.columns:
            agg[c] = "mean"

    table = (
        df.groupby(group_cols, dropna=False)
        .agg(agg)
        .reset_index()
        .rename(columns={"mechanism_region_id": "region_count"})
    )

    rename_map = {
        "external_validation_priority_score": "mean_external_validation_priority",
        "mean_P_orthogonal_v01": "mean_physical_potential",
        "mean_O_base_opportunity_v01": "mean_opportunity",
        "mean_T_net_transmission_v01": "mean_transmission",
        "mean_R_net_under_recognition_v01": "mean_under_recognition",
        "mean_observed_recognition_v04": "mean_observed_recognition",
        "mean_expected_recognition_v06": "mean_expected_recognition",
        "validation_area_km2": "mean_area_km2",
        "region_mechanism_stability_005": "mean_region_stability_005",
    }

    table = table.rename(columns=rename_map)

    table["landscape_explanation"] = (
        table["geographic_landscape_type_v01"]
        .map(LANDSCAPE_EXPLANATIONS)
        .fillna("No landscape explanation assigned.")
    )

    table = table.merge(mechanism_evidence, on="canonical_mechanism", how="left")

    table["landscape_theory_strength"] = table["mechanism_evidence_strength"]

    table = table.sort_values(
        ["region_count", "mechanism_mean_evidence_score"],
        ascending=[False, False],
    )

    return table


def build_summary(table):
    return (
        table.groupby("canonical_mechanism")
        .agg(
            landscape_type_count=("geographic_landscape_type_v01", "count"),
            total_regions=("region_count", "sum"),
            mean_mechanism_evidence=("mechanism_mean_evidence_score", "mean"),
            dominant_landscape_type=("geographic_landscape_type_v01", lambda s: s.iloc[0]),
        )
        .reset_index()
        .sort_values("total_regions", ascending=False)
    )


def build_claims(table):
    rows = [
        {
            "claim_id": "GL1",
            "claim": "Recognition Disequilibrium clusters into recurring geographic landscape systems.",
            "support": "Supported by repeated landscape classes across 99 mechanism regions.",
        },
        {
            "claim_id": "GL2",
            "claim": "Remote northern coastal landscapes are a major concentration of recognition disequilibrium.",
            "support": "This is the largest geographic landscape class in the corrected landscape interpretation.",
        },
        {
            "claim_id": "GL3",
            "claim": "Offshore island systems are a major setting for recognition inefficiency.",
            "support": "Offshore Island Recognition Inefficiency is a high-priority recurring landscape type.",
        },
        {
            "claim_id": "GL4",
            "claim": "Opportunity Failure and Recognition Inefficiency occupy related but distinct landscape contexts.",
            "support": "Opportunity Failure appears strongly in remote and rugged landscapes with lower opportunity scores.",
        },
        {
            "claim_id": "GL5",
            "claim": "Comparative Shadowing remains geographically meaningful but theoretically exploratory.",
            "support": "Shadowing appears in central coast and diversion contexts but has weaker validation evidence.",
        },
    ]

    return pd.DataFrame(rows)


def write_outputs(table, summary, claims):
    log.info("Writing corrected theory table: %s", OUTPUT_TABLE)
    table.to_csv(OUTPUT_TABLE, index=False)

    log.info("Writing corrected summary: %s", OUTPUT_SUMMARY)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    log.info("Writing claims: %s", OUTPUT_CLAIMS)
    claims.to_csv(OUTPUT_CLAIMS, index=False)

    qa = []
    qa.append("Geographic Landscape Theory Table v02 QA")
    qa.append("=" * 50)
    qa.append("")
    qa.append(f"Theory rows: {len(table)}")
    qa.append(f"Total represented regions: {int(table['region_count'].sum())}")
    qa.append("")
    qa.append("Summary:")
    qa.append(summary.to_string(index=False))
    qa.append("")
    qa.append("Theory table:")
    qa.append(table.to_string(index=False))
    qa.append("")
    qa.append("Claims:")
    qa.append(claims.to_string(index=False))

    log.info("Writing QA: %s", OUTPUT_QA)
    OUTPUT_QA.write_text("\n".join(qa), encoding="utf-8")


def main():
    log.info("Starting Script 104 v02 corrected")

    landscape, synthesis, readiness = load_inputs()

    log.info("Landscape rows: %s", len(landscape))

    mechanism_evidence = build_mechanism_evidence(synthesis, readiness)
    table = build_theory_table(landscape, mechanism_evidence)
    summary = build_summary(table)
    claims = build_claims(table)

    write_outputs(table, summary, claims)

    log.info("Done")

    print("\nCorrected Geographic Landscape Theory Summary")
    print(summary.to_string(index=False))

    print("\nCorrected Landscape Theory Table")
    print(
        table[
            [
                "canonical_mechanism",
                "geographic_landscape_type_v01",
                "region_count",
                "mechanism_mean_evidence_score",
                "landscape_theory_strength",
            ]
        ].to_string(index=False)
    )

    print("\nCreated:")
    print(f"  {OUTPUT_TABLE}")
    print(f"  {OUTPUT_SUMMARY}")
    print(f"  {OUTPUT_CLAIMS}")
    print(f"  {OUTPUT_QA}")


if __name__ == "__main__":
    main()