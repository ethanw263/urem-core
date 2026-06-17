#!/usr/bin/env python3
"""
113_cross_domain_generalization_feasibility_v01.py

Purpose
-------
Assess how likely the current RDE framework is to generalize beyond the
California coastal study domain, and produce a structured replication plan.

This script does NOT claim cross-domain validation has been completed.

Instead, it answers:

1. Which RDE mechanisms appear structurally generalizable?
2. Which mechanisms may be California/coast-specific?
3. Which feature families are portable to other domains?
4. What external replication domains are most suitable?
5. What minimum data requirements are needed to replicate RDE elsewhere?
6. What should a future Oregon/Washington/Baja/etc. replication test look like?

Inputs
------
Uses available validated outputs from Scripts 95-112 when present.

Primary inputs:
- rde_mechanism_defensibility_rankings_v01.csv
- rde_geographic_holdout_mechanism_summary_v01.csv
- rde_geographic_holdout_summary_v01.csv
- rde_geographic_landscape_theory_table_v01.csv
- rde_negative_control_feature_tests_v01.csv
- rde_wikipedia_wikidata_external_validation_mechanism_summary_v01.csv
- rde_external_proxy_validation_mechanism_summary_v01.csv
- rde_publication_readiness_scorecard_v01.csv
- rde_publication_reviewer_attack_matrix_v01.csv

Outputs
-------
data/processed/rde_cross_domain_generalization_feasibility_v01.csv
data/processed/rde_mechanism_generalization_scores_v01.csv
data/processed/rde_transferability_hypotheses_v01.csv
data/processed/rde_cross_domain_replication_plan_v01.csv
data/processed/rde_cross_domain_minimum_data_requirements_v01.csv
data/processed/rde_cross_domain_generalization_qa_v01.txt

Scientific Caution
------------------
This is a feasibility and planning script. It does not replace actual
cross-domain replication. It should be framed as a structured assessment of
generalization readiness, not empirical proof of global generalizability.
"""

from __future__ import annotations

import logging
from pathlib import Path
import math

import numpy as np
import pandas as pd


SCRIPT_NAME = "113_cross_domain_generalization_feasibility_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUTS = {
    "defensibility": PROCESSED / "rde_mechanism_defensibility_rankings_v01.csv",
    "holdout_mechanism": PROCESSED / "rde_geographic_holdout_mechanism_summary_v01.csv",
    "holdout_zone": PROCESSED / "rde_geographic_holdout_summary_v01.csv",
    "geographic_theory": PROCESSED / "rde_geographic_landscape_theory_table_v01.csv",
    "negative_control_tests": PROCESSED / "rde_negative_control_feature_tests_v01.csv",
    "wiki_validation": PROCESSED / "rde_wikipedia_wikidata_external_validation_mechanism_summary_v01.csv",
    "external_proxy": PROCESSED / "rde_external_proxy_validation_mechanism_summary_v01.csv",
    "publication_scorecard": PROCESSED / "rde_publication_readiness_scorecard_v01.csv",
    "reviewer_attack_matrix": PROCESSED / "rde_publication_reviewer_attack_matrix_v01.csv",
}

OUTPUT_FEASIBILITY = PROCESSED / "rde_cross_domain_generalization_feasibility_v01.csv"
OUTPUT_MECHANISM_SCORES = PROCESSED / "rde_mechanism_generalization_scores_v01.csv"
OUTPUT_HYPOTHESES = PROCESSED / "rde_transferability_hypotheses_v01.csv"
OUTPUT_REPLICATION_PLAN = PROCESSED / "rde_cross_domain_replication_plan_v01.csv"
OUTPUT_DATA_REQUIREMENTS = PROCESSED / "rde_cross_domain_minimum_data_requirements_v01.csv"
OUTPUT_QA = PROCESSED / "rde_cross_domain_generalization_qa_v01.txt"


MECHANISMS = [
    "Opportunity Failure",
    "Recognition Inefficiency",
    "Comparative Shadowing",
]


REPLICATION_DOMAINS = [
    {
        "domain": "Oregon Coast",
        "domain_type": "coastal_replication",
        "expected_similarity_to_california": 0.85,
        "data_feasibility": 0.90,
        "scientific_value": 0.90,
        "rationale": (
            "Closest natural replication domain: rugged coast, tourism gradients, "
            "remote headlands, public lands, and variable recognition."
        ),
        "expected_best_mechanisms": "Opportunity Failure; Recognition Inefficiency",
        "main_risk": "Some coastal recognition structures may be more uniformly public/recreational than California.",
    },
    {
        "domain": "Washington Coast",
        "domain_type": "coastal_replication",
        "expected_similarity_to_california": 0.80,
        "data_feasibility": 0.85,
        "scientific_value": 0.88,
        "rationale": (
            "Strong coastal ruggedness and remoteness gradients; useful for testing "
            "Opportunity Failure and remote recognition disequilibrium."
        ),
        "expected_best_mechanisms": "Opportunity Failure; Remote Recognition Inefficiency",
        "main_risk": "Cloud/rainforest/ecoregion differences may require modified physical-potential features.",
    },
    {
        "domain": "Baja California Pacific Coast",
        "domain_type": "international_coastal_replication",
        "expected_similarity_to_california": 0.72,
        "data_feasibility": 0.60,
        "scientific_value": 0.92,
        "rationale": (
            "High scientific value for testing whether RDE generalizes across national, "
            "institutional, and data-density boundaries."
        ),
        "expected_best_mechanisms": "Opportunity Failure; Recognition Inefficiency",
        "main_risk": "OSM/Wikipedia/Wikidata coverage and recognition proxies may be less complete.",
    },
    {
        "domain": "Mediterranean Coast",
        "domain_type": "international_coastal_replication",
        "expected_similarity_to_california": 0.65,
        "data_feasibility": 0.70,
        "scientific_value": 0.95,
        "rationale": (
            "High tourism and long historical recognition gradient; strong test of Comparative "
            "Shadowing and recognition diversion."
        ),
        "expected_best_mechanisms": "Comparative Shadowing; Recognition Inefficiency",
        "main_risk": "Cultural/historical recognition is much deeper and may require different recognition proxies.",
    },
    {
        "domain": "Hawaiian Islands",
        "domain_type": "island_replication",
        "expected_similarity_to_california": 0.55,
        "data_feasibility": 0.80,
        "scientific_value": 0.88,
        "rationale": (
            "Strong test of island recognition dynamics, visitor filtering, access, and "
            "destination shadowing."
        ),
        "expected_best_mechanisms": "Recognition Inefficiency; Comparative Shadowing",
        "main_risk": "Very high baseline tourism recognition may reduce under-recognition signal.",
    },
    {
        "domain": "Mountain / National Park Systems",
        "domain_type": "non_coastal_exceptional_landscape",
        "expected_similarity_to_california": 0.45,
        "data_feasibility": 0.75,
        "scientific_value": 0.86,
        "rationale": (
            "Tests whether RDE transfers beyond coastality into mountain-based physical "
            "exceptionality and recognition gradients."
        ),
        "expected_best_mechanisms": "Opportunity Failure; Recognition Inefficiency",
        "main_risk": "Physical Potential must be redesigned without coastal proximity dominance.",
    },
    {
        "domain": "Wine Regions",
        "domain_type": "non_landscape_market_recognition",
        "expected_similarity_to_california": 0.35,
        "data_feasibility": 0.60,
        "scientific_value": 0.80,
        "rationale": (
            "Tests RDE as a general recognition disequilibrium framework outside scenic geography."
        ),
        "expected_best_mechanisms": "Recognition Inefficiency; Comparative Shadowing",
        "main_risk": "Requires new definitions of physical/product potential and observed recognition.",
    },
    {
        "domain": "Golf Course / Golf Destination Systems",
        "domain_type": "recreation_market_replication",
        "expected_similarity_to_california": 0.40,
        "data_feasibility": 0.55,
        "scientific_value": 0.78,
        "rationale": (
            "RDE could identify high-quality or high-potential golf landscapes/courses with "
            "lower recognition than expected."
        ),
        "expected_best_mechanisms": "Recognition Inefficiency; Opportunity Failure",
        "main_risk": "Requires reliable golf-quality, access, and recognition data.",
    },
]


DATA_REQUIREMENTS = [
    {
        "requirement": "Physical Potential Layer",
        "minimum_needed": "Domain-specific physical or intrinsic-potential variables.",
        "california_current_equivalent": "coastal proximity, relief, slope, elevation, terrain structure",
        "portability": "High for physical geography; must be redesigned for non-landscape domains.",
        "criticality": "Essential",
    },
    {
        "requirement": "Observed Recognition Layer",
        "minimum_needed": "Independent proxies for public/institutional recognition.",
        "california_current_equivalent": "parks, trails, beaches, tourism, viewpoints, OSM recognition features",
        "portability": "Moderate to high; proxy definitions must be domain-specific.",
        "criticality": "Essential",
    },
    {
        "requirement": "Expected Recognition Model",
        "minimum_needed": "Comparable-place matching or counterfactual expected recognition surface.",
        "california_current_equivalent": "expected_recognition_v06",
        "portability": "High conceptually; implementation depends on comparable universe.",
        "criticality": "Essential",
    },
    {
        "requirement": "Opportunity Structure Layer",
        "minimum_needed": "Variables representing access, infrastructure, exposure, or enabling conditions.",
        "california_current_equivalent": "opportunity_structure_index_v01 / O_base_opportunity_v01",
        "portability": "High conceptually; variables must reflect domain.",
        "criticality": "Essential",
    },
    {
        "requirement": "Recognition Transmission Layer",
        "minimum_needed": "Variables representing pathways through which recognition spreads.",
        "california_current_equivalent": "recognition_transmission_index_v01 / T_net_transmission_v01",
        "portability": "Moderate; digital/media/network pathways should be expanded in future.",
        "criticality": "Essential",
    },
    {
        "requirement": "Orthogonalization Step",
        "minimum_needed": "Procedure to separate P, O, T, and R dimensions.",
        "california_current_equivalent": "P_orthogonal, O_base_opportunity, T_net_transmission, R_net_under_recognition",
        "portability": "High; method should transfer if features are available.",
        "criticality": "Essential",
    },
    {
        "requirement": "External Validation Source",
        "minimum_needed": "At least one independent knowledge/recognition dataset.",
        "california_current_equivalent": "Wikipedia/Wikidata validation",
        "portability": "Moderate; coverage varies strongly by region and country.",
        "criticality": "Strongly recommended",
    },
    {
        "requirement": "Negative Controls",
        "minimum_needed": "Recognized/famous examples in same domain.",
        "california_current_equivalent": "Big Sur, Point Reyes, Yosemite, etc.",
        "portability": "High; every domain should have recognized controls.",
        "criticality": "Strongly recommended",
    },
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.warning("Missing input: %s", path)
        return pd.DataFrame()

    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        log.warning("Could not read %s: %s", path, exc)
        return pd.DataFrame()


def normalize_0_1(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if pd.isna(value):
        return np.nan
    if high <= low:
        return np.nan
    return max(0.0, min(1.0, (float(value) - low) / (high - low)))


def class_to_numeric(cls: object) -> float:
    s = str(cls).lower()

    if "strongly defensible" in s:
        return 1.0
    if "strong transferability" in s:
        return 1.0
    if "strong" in s:
        return 0.9
    if "defensible" in s:
        return 0.78
    if "moderate" in s:
        return 0.65
    if "promising" in s:
        return 0.55
    if "weak" in s:
        return 0.35
    if "insufficient" in s:
        return 0.15

    return np.nan


def get_mechanism_value(df: pd.DataFrame, mechanism: str, col: str) -> float:
    if df.empty or col not in df.columns:
        return np.nan

    mech_col = None
    for c in ["mechanism", "canonical_mechanism"]:
        if c in df.columns:
            mech_col = c
            break

    if mech_col is None:
        return np.nan

    sub = df[df[mech_col].astype(str).str.contains(mechanism, case=False, na=False)]
    if len(sub) == 0:
        return np.nan

    return pd.to_numeric(sub[col], errors="coerce").mean()


def get_mechanism_class(df: pd.DataFrame, mechanism: str, col: str) -> str:
    if df.empty or col not in df.columns:
        return ""

    mech_col = None
    for c in ["mechanism", "canonical_mechanism"]:
        if c in df.columns:
            mech_col = c
            break

    if mech_col is None:
        return ""

    sub = df[df[mech_col].astype(str).str.contains(mechanism, case=False, na=False)]
    if len(sub) == 0:
        return ""

    return str(sub[col].iloc[0])


def mechanism_generalization_assessment(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    defensibility = dfs["defensibility"]
    holdout = dfs["holdout_mechanism"]
    wiki = dfs["wiki_validation"]
    proxy = dfs["external_proxy"]

    rows = []

    for mech in MECHANISMS:
        defensibility_fraction = get_mechanism_value(
            defensibility,
            mech,
            "overall_validation_fraction",
        )
        defensibility_class = get_mechanism_class(
            defensibility,
            mech,
            "overall_defensibility_class",
        )

        transferability = get_mechanism_value(
            holdout,
            mech,
            "mechanism_transferability_score",
        )
        holdout_recall = get_mechanism_value(
            holdout,
            mech,
            "leave_zone_out_recall",
        )

        wiki_score = get_mechanism_value(
            wiki,
            mech,
            "mean_external_under_recognition_score",
        )

        proxy_score = get_mechanism_value(
            proxy,
            mech,
            "mean_external_proxy_validation_score",
        )

        # Theory generalization is not the same as evidence strength.
        # It is a weighted feasibility score based on whether mechanism structure
        # appears likely to transfer outside the current domain.
        components = {
            "defensibility": defensibility_fraction,
            "geographic_transferability": transferability,
            "holdout_recall": holdout_recall,
            "external_knowledge_support": wiki_score,
            "external_proxy_support": proxy_score,
        }

        valid_vals = [v for v in components.values() if not pd.isna(v)]
        base_score = float(np.mean(valid_vals)) if valid_vals else np.nan

        # Mechanism-specific conceptual portability adjustment.
        # Opportunity Failure is structurally very portable.
        # Recognition Inefficiency is portable but depends on recognition proxies.
        # Comparative Shadowing is portable conceptually but more culturally/contextually sensitive.
        if mech == "Opportunity Failure":
            portability_adjustment = 0.05
            conceptual_risk = "Low to moderate"
            likely_portability = "High"
        elif mech == "Recognition Inefficiency":
            portability_adjustment = 0.00
            conceptual_risk = "Moderate"
            likely_portability = "Moderate to high"
        else:
            portability_adjustment = -0.08
            conceptual_risk = "Moderate to high"
            likely_portability = "Moderate"

        score = base_score + portability_adjustment if not pd.isna(base_score) else np.nan
        score = max(0.0, min(1.0, score)) if not pd.isna(score) else np.nan

        if pd.isna(score):
            cls = "Unknown"
        elif score >= 0.80:
            cls = "High Generalization Readiness"
        elif score >= 0.65:
            cls = "Moderate Generalization Readiness"
        elif score >= 0.50:
            cls = "Low / Emerging Generalization Readiness"
        else:
            cls = "Weak Generalization Readiness"

        rows.append(
            {
                "mechanism": mech,
                "overall_defensibility_class": defensibility_class,
                "overall_validation_fraction": defensibility_fraction,
                "geographic_transferability_score": transferability,
                "leave_zone_out_recall": holdout_recall,
                "wiki_wikidata_external_score": wiki_score,
                "external_proxy_score": proxy_score,
                "conceptual_portability_adjustment": portability_adjustment,
                "cross_domain_generalization_readiness_score": score,
                "cross_domain_generalization_readiness_class": cls,
                "likely_portability": likely_portability,
                "conceptual_risk": conceptual_risk,
                "interpretation": mechanism_interpretation(mech, score),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "cross_domain_generalization_readiness_score",
        ascending=False,
    )


def mechanism_interpretation(mech: str, score: float) -> str:
    if mech == "Opportunity Failure":
        return (
            "Most likely mechanism to generalize. Opportunity constraints are structural "
            "and should appear across many domains where high potential lacks enabling conditions."
        )
    if mech == "Recognition Inefficiency":
        return (
            "Likely to generalize where high potential and transmission exist but observed recognition "
            "lags. Requires strong observed-recognition measurement in each new domain."
        )
    if mech == "Comparative Shadowing":
        return (
            "Conceptually portable but context-sensitive. Requires careful modeling of competing "
            "destinations, attention sinks, and recognition diversion."
        )
    return "No interpretation available."


def domain_feasibility_table(mechanism_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []

    mean_mech_readiness = mechanism_scores[
        "cross_domain_generalization_readiness_score"
    ].mean()

    for d in REPLICATION_DOMAINS:
        combined = (
            0.35 * d["expected_similarity_to_california"]
            + 0.25 * d["data_feasibility"]
            + 0.25 * d["scientific_value"]
            + 0.15 * mean_mech_readiness
        )

        if combined >= 0.82:
            tier = "Tier 1 Replication Target"
        elif combined >= 0.70:
            tier = "Tier 2 Replication Target"
        elif combined >= 0.58:
            tier = "Tier 3 Exploratory Target"
        else:
            tier = "Long-Term / Experimental Target"

        rows.append(
            {
                **d,
                "mean_current_mechanism_readiness": mean_mech_readiness,
                "replication_priority_score": combined,
                "replication_priority_tier": tier,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "replication_priority_score",
        ascending=False,
    )


def transferability_hypotheses(mechanism_scores: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "hypothesis_id": "H1",
            "hypothesis": (
                "Opportunity Failure will generalize most strongly across domains because opportunity "
                "constraints are structural and not California-specific."
            ),
            "current_support": get_support_statement(mechanism_scores, "Opportunity Failure"),
            "test": "Replicate RDE in Oregon/Washington and test whether Opportunity Failure remains strongly defensible.",
            "risk": "Opportunity variables may need domain-specific redesign.",
        },
        {
            "hypothesis_id": "H2",
            "hypothesis": (
                "Recognition Inefficiency will generalize where high physical potential and transmission "
                "exist but observed recognition remains low."
            ),
            "current_support": get_support_statement(mechanism_scores, "Recognition Inefficiency"),
            "test": "Test in another coastal domain with strong terrain/coastality and mixed tourism recognition.",
            "risk": "Observed-recognition proxies may differ by region and data coverage.",
        },
        {
            "hypothesis_id": "H3",
            "hypothesis": (
                "Comparative Shadowing will generalize mainly in domains with strong destination hierarchies "
                "and nearby iconic attention sinks."
            ),
            "current_support": get_support_statement(mechanism_scores, "Comparative Shadowing"),
            "test": "Test in tourism-dense regions such as Mediterranean coasts or Hawaiian islands.",
            "risk": "Shadowing requires explicit modeling of competing recognized destinations.",
        },
        {
            "hypothesis_id": "H4",
            "hypothesis": (
                "Physical Potential will remain an entry condition rather than a primary mechanism "
                "differentiator in new domains."
            ),
            "current_support": "Supported by ablation and background validation in the current domain.",
            "test": "Repeat ablation and background entry validation in a replication domain.",
            "risk": "In non-coastal domains, physical potential may require a different variable design.",
        },
        {
            "hypothesis_id": "H5",
            "hypothesis": (
                "RDE mechanisms will transfer better across structurally similar domains than across "
                "semantically different domains."
            ),
            "current_support": "Supported indirectly by strong within-domain geographic holdout validation.",
            "test": "Compare Oregon/Washington replication against wine/golf/non-landscape replications.",
            "risk": "Requires multiple replication studies.",
        },
    ]

    return pd.DataFrame(rows)


def get_support_statement(mechanism_scores: pd.DataFrame, mechanism: str) -> str:
    sub = mechanism_scores[mechanism_scores["mechanism"] == mechanism]
    if len(sub) == 0:
        return "No mechanism-specific score available."
    r = sub.iloc[0]
    return (
        f"Current readiness score {r['cross_domain_generalization_readiness_score']:.3f}; "
        f"classified as {r['cross_domain_generalization_readiness_class']}."
    )


def replication_plan(domain_table: pd.DataFrame) -> pd.DataFrame:
    rows = []

    top_domains = domain_table.head(5).copy()

    step_number = 1
    for _, d in top_domains.iterrows():
        domain = d["domain"]

        domain_steps = [
            (
                "Define study area",
                "Create a study-area boundary analogous to the California coastal domain.",
                "Required before any feature construction.",
            ),
            (
                "Construct physical potential",
                "Build domain-specific physical potential variables.",
                "Keep concept stable but redesign variables as needed.",
            ),
            (
                "Construct observed recognition",
                "Build recognition proxies from OSM, protected areas, tourism features, Wikipedia/Wikidata, or domain-specific sources.",
                "Must avoid simply copying California recognition variables if inappropriate.",
            ),
            (
                "Build expected recognition",
                "Use comparable-geography matching or counterfactual modeling.",
                "This is essential for RDE rather than suitability mapping.",
            ),
            (
                "Build opportunity and transmission layers",
                "Construct O and T proxies, then test collinearity.",
                "Repeat orthogonalization if O/T are correlated.",
            ),
            (
                "Run mechanism classification",
                "Classify Opportunity Failure, Recognition Inefficiency, and Comparative Shadowing.",
                "Do not invent new mechanisms unless evidence requires it.",
            ),
            (
                "Run validation stack",
                "Repeat stability, ablation, background validation, external validation, negative controls, and holdout tests.",
                "Compare mechanism behavior to California baseline.",
            ),
        ]

        for task, description, note in domain_steps:
            rows.append(
                {
                    "replication_domain": domain,
                    "replication_priority_tier": d["replication_priority_tier"],
                    "replication_priority_score": d["replication_priority_score"],
                    "step_number": step_number,
                    "task": task,
                    "description": description,
                    "note": note,
                }
            )
            step_number += 1

    return pd.DataFrame(rows)


def data_requirements_table() -> pd.DataFrame:
    return pd.DataFrame(DATA_REQUIREMENTS)


def write_outputs(
    feasibility: pd.DataFrame,
    mechanism_scores: pd.DataFrame,
    hypotheses: pd.DataFrame,
    plan: pd.DataFrame,
    requirements: pd.DataFrame,
) -> None:
    log.info("Writing feasibility: %s", OUTPUT_FEASIBILITY)
    feasibility.to_csv(OUTPUT_FEASIBILITY, index=False)

    log.info("Writing mechanism scores: %s", OUTPUT_MECHANISM_SCORES)
    mechanism_scores.to_csv(OUTPUT_MECHANISM_SCORES, index=False)

    log.info("Writing hypotheses: %s", OUTPUT_HYPOTHESES)
    hypotheses.to_csv(OUTPUT_HYPOTHESES, index=False)

    log.info("Writing replication plan: %s", OUTPUT_REPLICATION_PLAN)
    plan.to_csv(OUTPUT_REPLICATION_PLAN, index=False)

    log.info("Writing data requirements: %s", OUTPUT_DATA_REQUIREMENTS)
    requirements.to_csv(OUTPUT_DATA_REQUIREMENTS, index=False)

    qa = []
    qa.append("RDE Cross-Domain Generalization Feasibility v01 QA")
    qa.append("=" * 60)
    qa.append("")
    qa.append("Scientific warning:")
    qa.append(
        "This is a feasibility assessment and replication plan. It does not prove cross-domain generalization."
    )
    qa.append("")
    qa.append("Mechanism generalization scores:")
    qa.append(mechanism_scores.to_string(index=False))
    qa.append("")
    qa.append("Replication domain feasibility:")
    qa.append(feasibility.to_string(index=False))
    qa.append("")
    qa.append("Transferability hypotheses:")
    qa.append(hypotheses.to_string(index=False))
    qa.append("")
    qa.append("Minimum data requirements:")
    qa.append(requirements.to_string(index=False))
    qa.append("")
    qa.append("Recommended next scientific step:")
    qa.append(
        "Run an actual replication on the Oregon Coast or Washington Coast. "
        "Until that is done, generalization beyond California should be framed as plausible, not proven."
    )

    log.info("Writing QA: %s", OUTPUT_QA)
    OUTPUT_QA.write_text("\n".join(qa), encoding="utf-8")


def main() -> None:
    log.info("Starting Script 113: cross-domain generalization feasibility")

    dfs = {name: read_csv(path) for name, path in INPUTS.items()}

    mechanism_scores = mechanism_generalization_assessment(dfs)
    feasibility = domain_feasibility_table(mechanism_scores)
    hypotheses = transferability_hypotheses(mechanism_scores)
    requirements = data_requirements_table()
    plan = replication_plan(feasibility)

    write_outputs(
        feasibility=feasibility,
        mechanism_scores=mechanism_scores,
        hypotheses=hypotheses,
        plan=plan,
        requirements=requirements,
    )

    log.info("Done")

    print("\nMechanism Generalization Scores:")
    print(mechanism_scores.to_string(index=False))

    print("\nReplication Domain Feasibility:")
    print(
        feasibility[
            [
                "domain",
                "domain_type",
                "replication_priority_score",
                "replication_priority_tier",
                "expected_best_mechanisms",
                "main_risk",
            ]
        ].to_string(index=False)
    )

    print("\nCreated:")
    print(f"  {OUTPUT_FEASIBILITY}")
    print(f"  {OUTPUT_MECHANISM_SCORES}")
    print(f"  {OUTPUT_HYPOTHESES}")
    print(f"  {OUTPUT_REPLICATION_PLAN}")
    print(f"  {OUTPUT_DATA_REQUIREMENTS}")
    print(f"  {OUTPUT_QA}")


if __name__ == "__main__":
    main()
