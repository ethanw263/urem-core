#!/usr/bin/env python3
"""
98_core_results_synthesis_v01.py

Purpose
-------
Create a paper-facing synthesis of the current RDE results.

Combines:
- Theory readiness
- Evidence validation
- Mechanism stability
- Archetype stability

Outputs:
- Core claims
- Evidence level
- Stability level
- Publication role
- Recommended wording

Inputs
------
data/processed/rde_theory_readiness_filter_v01.csv
data/processed/rde_mechanism_stability_v02.csv
data/processed/rde_archetype_stability_v02.csv
data/processed/rde_stability_summary_v02.csv

Outputs
-------
data/processed/rde_core_results_synthesis_v01.csv
data/processed/rde_core_claims_v01.csv
data/processed/rde_research_limitations_v01.csv
data/processed/rde_next_validation_plan_v01.csv
data/processed/rde_core_results_synthesis_qa_v01.txt
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_NAME = "98_core_results_synthesis_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUT_READINESS = PROCESSED / "rde_theory_readiness_filter_v01.csv"
INPUT_MECH_STABILITY = PROCESSED / "rde_mechanism_stability_v02.csv"
INPUT_ARCH_STABILITY = PROCESSED / "rde_archetype_stability_v02.csv"
INPUT_STABILITY_SUMMARY = PROCESSED / "rde_stability_summary_v02.csv"

OUTPUT_SYNTHESIS = PROCESSED / "rde_core_results_synthesis_v01.csv"
OUTPUT_CLAIMS = PROCESSED / "rde_core_claims_v01.csv"
OUTPUT_LIMITATIONS = PROCESSED / "rde_research_limitations_v01.csv"
OUTPUT_NEXT_PLAN = PROCESSED / "rde_next_validation_plan_v01.csv"
OUTPUT_QA = PROCESSED / "rde_core_results_synthesis_qa_v01.txt"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [
        INPUT_READINESS,
        INPUT_MECH_STABILITY,
        INPUT_ARCH_STABILITY,
        INPUT_STABILITY_SUMMARY,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    log.info("Reading readiness: %s", INPUT_READINESS)
    readiness = pd.read_csv(INPUT_READINESS, low_memory=False)

    log.info("Reading mechanism stability: %s", INPUT_MECH_STABILITY)
    mech_stability = pd.read_csv(INPUT_MECH_STABILITY, low_memory=False)

    log.info("Reading archetype stability: %s", INPUT_ARCH_STABILITY)
    arch_stability = pd.read_csv(INPUT_ARCH_STABILITY, low_memory=False)

    log.info("Reading stability summary: %s", INPUT_STABILITY_SUMMARY)
    stability_summary = pd.read_csv(INPUT_STABILITY_SUMMARY, low_memory=False)

    return readiness, mech_stability, arch_stability, stability_summary


def stability_at_level(df: pd.DataFrame, level: float, group_cols: list[str]) -> pd.DataFrame:
    sub = df[np.isclose(df["perturbation_level"], level)].copy()
    keep = group_cols + [
        "mechanism_stability_rate",
        "archetype_stability_rate",
        "mechanism_stability_class",
        "archetype_stability_class",
    ]
    return sub[keep]


def classify_claim_strength(row: pd.Series) -> str:
    tier = row.get("theory_readiness_tier", "")
    evidence = row.get("overall_evidence_score", np.nan)
    mech_stab = row.get("mechanism_stability_rate_005", np.nan)
    arch_stab = row.get("archetype_stability_rate_005", np.nan)

    if (
        tier == "Core Validated Theory"
        and pd.notna(evidence)
        and evidence >= 0.60
        and pd.notna(mech_stab)
        and mech_stab >= 0.75
    ):
        if pd.notna(arch_stab) and arch_stab >= 0.60:
            return "Strong Core Claim"
        return "Mechanism-Level Core Claim"

    if (
        tier == "Emerging Theory"
        and pd.notna(evidence)
        and evidence >= 0.45
        and pd.notna(mech_stab)
        and mech_stab >= 0.60
    ):
        return "Emerging Supported Claim"

    return "Exploratory / Holdout Claim"


def recommended_claim_language(row: pd.Series) -> str:
    mechanism = row.get("mechanism", "")
    archetype = row.get("archetype", "")
    strength = row.get("claim_strength", "")

    if strength == "Strong Core Claim":
        return (
            f"The {archetype} archetype provides strong evidence for {mechanism} "
            "as a stable form of Recognition Disequilibrium."
        )

    if strength == "Mechanism-Level Core Claim":
        return (
            f"The evidence supports {mechanism} as a stable mechanism, while the "
            f"specific '{archetype}' archetype should be treated as a useful but "
            "less stable subdivision."
        )

    if strength == "Emerging Supported Claim":
        return (
            f"The {archetype} archetype appears to be an emerging form of "
            f"{mechanism}, but requires additional external validation."
        )

    return (
        f"The {archetype} archetype should be treated as exploratory. It may be "
        "useful for case-study selection or future validation, but should not be "
        "used as a central claim yet."
    )


def publication_role(row: pd.Series) -> str:
    strength = row.get("claim_strength", "")

    if strength == "Strong Core Claim":
        return "Main Results"

    if strength == "Mechanism-Level Core Claim":
        return "Main Results with Cautious Archetype Framing"

    if strength == "Emerging Supported Claim":
        return "Discussion / Exploratory Results"

    return "Appendix / Future Work"


def build_synthesis(
    readiness: pd.DataFrame,
    mech_stability: pd.DataFrame,
    arch_stability: pd.DataFrame,
) -> pd.DataFrame:
    mech_005 = stability_at_level(
        mech_stability,
        0.05,
        ["mechanism"],
    ).rename(
        columns={
            "mechanism_stability_rate": "mechanism_stability_rate_005",
            "archetype_stability_rate": "mechanism_level_archetype_stability_rate_005",
            "mechanism_stability_class": "mechanism_stability_class_005",
            "archetype_stability_class": "mechanism_level_archetype_stability_class_005",
        }
    )

    arch_005 = stability_at_level(
        arch_stability,
        0.05,
        ["mechanism", "archetype"],
    ).rename(
        columns={
            "mechanism_stability_rate": "archetype_mechanism_stability_rate_005",
            "archetype_stability_rate": "archetype_stability_rate_005",
            "mechanism_stability_class": "archetype_mechanism_stability_class_005",
            "archetype_stability_class": "archetype_stability_class_005",
        }
    )

    synthesis = readiness.merge(mech_005, on="mechanism", how="left")
    synthesis = synthesis.merge(arch_005, on=["mechanism", "archetype"], how="left")

    if "mechanism_stability_rate_005" not in synthesis.columns:
        synthesis["mechanism_stability_rate_005"] = np.nan

    if "archetype_stability_rate_005" not in synthesis.columns:
        synthesis["archetype_stability_rate_005"] = np.nan

    synthesis["claim_strength"] = synthesis.apply(classify_claim_strength, axis=1)
    synthesis["recommended_claim_language"] = synthesis.apply(recommended_claim_language, axis=1)
    synthesis["publication_role_final"] = synthesis.apply(publication_role, axis=1)

    order = {
        "Strong Core Claim": 1,
        "Mechanism-Level Core Claim": 2,
        "Emerging Supported Claim": 3,
        "Exploratory / Holdout Claim": 4,
    }

    synthesis["claim_rank"] = synthesis["claim_strength"].map(order)

    synthesis = synthesis.sort_values(
        [
            "claim_rank",
            "mechanism",
            "overall_evidence_score",
            "region_count",
        ],
        ascending=[True, True, False, False],
    )

    return synthesis


def build_core_claims(synthesis: pd.DataFrame, stability_summary: pd.DataFrame) -> pd.DataFrame:
    mech_stab_005 = float(
        stability_summary.loc[
            np.isclose(stability_summary["perturbation_level"], 0.05),
            "mean_mechanism_stability",
        ].iloc[0]
    )

    arch_stab_005 = float(
        stability_summary.loc[
            np.isclose(stability_summary["perturbation_level"], 0.05),
            "mean_archetype_stability",
        ].iloc[0]
    )

    rows = [
        {
            "claim_id": "C1",
            "claim": "Recognition Disequilibrium mechanisms are empirically separable.",
            "supporting_result": (
                "Orthogonalized mechanism classes produced three mechanism families: "
                "Opportunity Failure, Recognition Inefficiency, and Comparative Shadowing."
            ),
            "evidence_status": "Supported by mechanism taxonomy and orthogonalization",
            "recommended_wording": (
                "The results suggest that under-recognition is not a single phenomenon, "
                "but can be decomposed into distinct mechanism families."
            ),
        },
        {
            "claim_id": "C2",
            "claim": "Mechanism-level RDE classifications are robust under perturbation.",
            "supporting_result": f"Mean mechanism stability at ±5% perturbation = {mech_stab_005:.3f}.",
            "evidence_status": "Strong mechanism-level robustness",
            "recommended_wording": (
                "Mechanism-level classifications remain stable under moderate perturbation, "
                "supporting the robustness of the RDE mechanism framework."
            ),
        },
        {
            "claim_id": "C3",
            "claim": "Archetype-level classifications are less stable than mechanisms.",
            "supporting_result": f"Mean archetype stability at ±5% perturbation = {arch_stab_005:.3f}.",
            "evidence_status": "Caution required",
            "recommended_wording": (
                "Fine-grained archetypes should be interpreted as exploratory subdivisions "
                "rather than fully validated classes."
            ),
        },
        {
            "claim_id": "C4",
            "claim": "Opportunity Failure is currently the most defensible validated mechanism.",
            "supporting_result": (
                "Two Opportunity Failure archetypes reached Core Validated Theory status."
            ),
            "evidence_status": "Core validated finding",
            "recommended_wording": (
                "The strongest evidence currently supports opportunity-based recognition "
                "failure, where physical potential exceeds available opportunity structure."
            ),
        },
        {
            "claim_id": "C5",
            "claim": "Recognition Inefficiency is theoretically important but only partly validated.",
            "supporting_result": (
                "Recognition Inefficiency produced one core validated archetype and three "
                "emerging archetypes."
            ),
            "evidence_status": "Promising but mixed",
            "recommended_wording": (
                "Recognition Inefficiency appears to be a meaningful mechanism, but its "
                "subtypes require additional external validation."
            ),
        },
        {
            "claim_id": "C6",
            "claim": "Comparative Shadowing remains exploratory.",
            "supporting_result": (
                "Comparative Shadowing showed mechanism-level stability but weak validation "
                "and weak archetype evidence."
            ),
            "evidence_status": "Exploratory",
            "recommended_wording": (
                "Comparative Shadowing should be retained as a hypothesis for future testing, "
                "not as a central validated claim."
            ),
        },
    ]

    return pd.DataFrame(rows)


def build_limitations() -> pd.DataFrame:
    rows = [
        {
            "limitation": "External validation is still limited.",
            "why_it_matters": "The model has strong internal consistency but still needs independent recognition indicators.",
            "recommended_fix": "Compare top RDE regions against Google reviews, Wikipedia, AllTrails, Flickr, Instagram, tourism guides, or search-volume proxies.",
        },
        {
            "limitation": "Temporal validation has not yet been performed.",
            "why_it_matters": "A strong theory should show whether under-recognized places later become recognized.",
            "recommended_fix": "Build historical recognition layers and test whether earlier RDE scores predict later recognition growth.",
        },
        {
            "limitation": "Domain transfer is untested.",
            "why_it_matters": "The current model may be specific to the California coastal landscape domain.",
            "recommended_fix": "Replicate on Oregon coast, Washington coast, California interior, or another discovery domain.",
        },
        {
            "limitation": "Ablation testing is incomplete.",
            "why_it_matters": "The contribution of P, O, T, and R components has not yet been isolated.",
            "recommended_fix": "Run P-only, P+R, P+O+R, P+T+R, and full P+O+T+R model comparisons.",
        },
        {
            "limitation": "Comparative Shadowing variables are underdeveloped.",
            "why_it_matters": "Shadowing is theoretically interesting but empirically weak in current validation.",
            "recommended_fix": "Add proximity-to-famous-destination, destination gravity, visitor-flow, and media-diversion features.",
        },
        {
            "limitation": "Archetypes are less robust than mechanisms.",
            "why_it_matters": "Fine classes may be over-specific or sensitive to thresholds.",
            "recommended_fix": "Use archetypes as exploratory labels until supported by external or qualitative validation.",
        },
    ]

    return pd.DataFrame(rows)


def build_next_plan() -> pd.DataFrame:
    rows = [
        {
            "priority": 1,
            "next_phase": "Ablation Testing",
            "goal": "Show that P, O, T, and R each contribute meaningful information.",
            "output": "Component ablation comparison table.",
        },
        {
            "priority": 2,
            "next_phase": "External Recognition Validation",
            "goal": "Test whether predicted under-recognition agrees with independent recognition proxies.",
            "output": "External validation score by region.",
        },
        {
            "priority": 3,
            "next_phase": "Temporal Validation",
            "goal": "Test whether high RDE regions predict future recognition growth.",
            "output": "Historical-to-current recognition growth validation.",
        },
        {
            "priority": 4,
            "next_phase": "Comparative Shadowing Enhancement",
            "goal": "Improve weak shadowing mechanism with stronger diversion variables.",
            "output": "Shadowing v02 mechanism layer.",
        },
        {
            "priority": 5,
            "next_phase": "Theory Report",
            "goal": "Convert current results into paper-style narrative.",
            "output": "Draft methodology/results report.",
        },
    ]

    return pd.DataFrame(rows)


def write_outputs(
    synthesis: pd.DataFrame,
    claims: pd.DataFrame,
    limitations: pd.DataFrame,
    next_plan: pd.DataFrame,
) -> None:
    log.info("Writing synthesis: %s", OUTPUT_SYNTHESIS)
    synthesis.to_csv(OUTPUT_SYNTHESIS, index=False)

    log.info("Writing core claims: %s", OUTPUT_CLAIMS)
    claims.to_csv(OUTPUT_CLAIMS, index=False)

    log.info("Writing limitations: %s", OUTPUT_LIMITATIONS)
    limitations.to_csv(OUTPUT_LIMITATIONS, index=False)

    log.info("Writing next validation plan: %s", OUTPUT_NEXT_PLAN)
    next_plan.to_csv(OUTPUT_NEXT_PLAN, index=False)

    qa = []
    qa.append("RDE Core Results Synthesis v01 QA")
    qa.append("=" * 45)
    qa.append("")
    qa.append("Claim strength counts:")
    qa.append(synthesis["claim_strength"].value_counts().to_string())
    qa.append("")
    qa.append("Publication role counts:")
    qa.append(synthesis["publication_role_final"].value_counts().to_string())
    qa.append("")
    qa.append("Core claims:")
    qa.append(claims.to_string(index=False))
    qa.append("")
    qa.append("Limitations:")
    qa.append(limitations.to_string(index=False))
    qa.append("")
    qa.append("Next validation plan:")
    qa.append(next_plan.to_string(index=False))

    log.info("Writing QA: %s", OUTPUT_QA)
    OUTPUT_QA.write_text("\n".join(qa), encoding="utf-8")


def main() -> None:
    log.info("Starting Script 98: core results synthesis")

    readiness, mech_stability, arch_stability, stability_summary = load_inputs()

    synthesis = build_synthesis(readiness, mech_stability, arch_stability)
    claims = build_core_claims(synthesis, stability_summary)
    limitations = build_limitations()
    next_plan = build_next_plan()

    write_outputs(synthesis, claims, limitations, next_plan)

    log.info("Done")

    print("\nRDE Core Results Synthesis")
    print(synthesis[[
        "mechanism",
        "archetype",
        "region_count",
        "theory_readiness_tier",
        "overall_evidence_score",
        "mechanism_stability_rate_005",
        "archetype_stability_rate_005",
        "claim_strength",
        "publication_role_final",
    ]].to_string(index=False))

    print("\nCore Claims:")
    print(claims[["claim_id", "claim", "evidence_status"]].to_string(index=False))

    print("\nCreated:")
    print(f"  {OUTPUT_SYNTHESIS}")
    print(f"  {OUTPUT_CLAIMS}")
    print(f"  {OUTPUT_LIMITATIONS}")
    print(f"  {OUTPUT_NEXT_PLAN}")
    print(f"  {OUTPUT_QA}")


if __name__ == "__main__":
    main()