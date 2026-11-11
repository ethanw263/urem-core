#!/usr/bin/env python3
"""
232_rde_srs_formal_operator_taxonomy_admissibility_v01.py

Formalize the RDE/SRS operator taxonomy into admissible mathematical objects.

No model is fit.
"""

from pathlib import Path
import logging
import pandas as pd


SCRIPT = "232_rde_srs_formal_operator_taxonomy_admissibility_v01"

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
P.mkdir(parents=True, exist_ok=True)

OUT_OPERATORS = P / "rde_srs_formal_operator_admissibility_v01.csv"
OUT_DISEQUILIBRIUM = P / "rde_srs_operator_dependent_disequilibrium_forms_v01.csv"
OUT_IMPLEMENTATION = P / "rde_srs_operator_implementation_readiness_v01.csv"
OUT_MD = P / "rde_srs_formal_operator_taxonomy_admissibility_v01.md"
OUT_QA = P / "rde_srs_formal_operator_taxonomy_admissibility_qa_v01.txt"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No records."

    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = "" if pd.isna(row[c]) else str(row[c])
            v = v.replace("|", "\\|").replace("\n", " ")
            vals.append(v)
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def build_operator_admissibility() -> pd.DataFrame:
    rows = [
        {
            "operator_id": "O01",
            "operator_family": "pointwise_identity",
            "symbol": "I",
            "formal_definition": "(I R)_i = R_i",
            "acts_on": "recognition",
            "domain": "scalar recognition field R defined on cells i",
            "codomain": "scalar recognition state per cell",
            "operator_type": "pointwise",
            "relational_order": "zero_order",
            "requires_neighbors": False,
            "requires_direction": False,
            "requires_weights": False,
            "requires_flow": False,
            "requires_comparison_set": False,
            "admissibility_conditions": "R_i must be observed or independently imputed for cell i.",
            "supported_disequilibrium_form": "D_i^(I) = E[R_i | C(P)_i] - R_i",
            "current_empirical_status": "baseline formulation",
            "current_status": "baseline_ready",
        },
        {
            "operator_id": "O02",
            "operator_family": "neighborhood_aggregation",
            "symbol": "A_r",
            "formal_definition": "(A_r R)_i = sum_j w_ij(r) R_j",
            "acts_on": "recognition",
            "domain": "recognition field R and neighborhood weight matrix W_r",
            "codomain": "weighted neighborhood recognition state",
            "operator_type": "relational_radial",
            "relational_order": "first_order",
            "requires_neighbors": True,
            "requires_direction": False,
            "requires_weights": True,
            "requires_flow": False,
            "requires_comparison_set": False,
            "admissibility_conditions": "Weights w_ij(r) must be nonnegative and normalized where neighbors exist; radius r and kernel form must be fixed before validation; weights cannot be derived from persistence labels.",
            "supported_disequilibrium_form": "D_i^(A_r) = E[(A_r R)_i | C(P)_i] - (A_r R)_i",
            "current_empirical_status": "conceptually supported but not directly tested as target operator",
            "current_status": "ready_for_targeted_test",
        },
        {
            "operator_id": "O03",
            "operator_family": "context_contrast",
            "symbol": "Delta_(r1,r2)",
            "formal_definition": "Delta C_i = C_r1(P)_i - C_r2(P)_i",
            "acts_on": "physical_context",
            "domain": "multiscale physical context fields C_r(P)",
            "codomain": "cross-scale physical contrast field",
            "operator_type": "multiscale_context",
            "relational_order": "cross_scale",
            "requires_neighbors": True,
            "requires_direction": False,
            "requires_weights": True,
            "requires_flow": False,
            "requires_comparison_set": False,
            "admissibility_conditions": "Scales r1 and r2 must be fixed before validation; both context fields must be constructed without target leakage; scale contrasts must preserve consistent physical meaning.",
            "supported_disequilibrium_form": "D_i = F(E[R | C(P)_i], R_i, Delta C_i)",
            "current_empirical_status": "context contrasts provide partial explanatory value but do not fully replace spatial structure",
            "current_status": "partially_supported_ready",
        },
        {
            "operator_id": "O04",
            "operator_family": "graph_laplacian",
            "symbol": "L",
            "formal_definition": "(L R)_i = sum_j w_ij (R_i - R_j)",
            "acts_on": "recognition",
            "domain": "recognition field R on graph nodes and graph weights W",
            "codomain": "local relational recognition divergence",
            "operator_type": "relational_divergence",
            "relational_order": "first_or_higher_order",
            "requires_neighbors": True,
            "requires_direction": False,
            "requires_weights": True,
            "requires_flow": False,
            "requires_comparison_set": False,
            "admissibility_conditions": "Graph W must be independently constructed from geography, accessibility, physical similarity, or another declared relation; graph construction must not use target labels; weight normalization must be declared before validation.",
            "supported_disequilibrium_form": "D_i^(L) = E[(L R)_i | C(P)_i] - (L R)_i",
            "current_empirical_status": "formal candidate not yet directly tested",
            "current_status": "formal_candidate_next",
        },
        {
            "operator_id": "O05",
            "operator_family": "flow_or_transmission",
            "symbol": "T",
            "formal_definition": "(T R)_i = sum_j f_ji R_j - sum_j f_ij R_i",
            "acts_on": "recognition",
            "domain": "recognition field R and directed transmission matrix F",
            "codomain": "net recognition transmission imbalance",
            "operator_type": "directional_flow",
            "relational_order": "directed_first_order",
            "requires_neighbors": True,
            "requires_direction": True,
            "requires_weights": True,
            "requires_flow": True,
            "requires_comparison_set": False,
            "admissibility_conditions": "Directed weights f_ij must be independently defined from geography, accessibility, movement, network connection, or another defensible transmission proxy; flow weights cannot be inferred from target residuals.",
            "supported_disequilibrium_form": "D_i^(T) = E[(T R)_i | C(P)_i] - (T R)_i",
            "current_empirical_status": "theoretically important but requires defensible flow proxy",
            "current_status": "future_requires_flow_proxy",
        },
        {
            "operator_id": "O06",
            "operator_family": "comparative_shadow",
            "symbol": "S",
            "formal_definition": "(S R)_i = R_i - max_j(w_ij R_j)",
            "acts_on": "recognition",
            "domain": "recognition field and geographically admissible competitor set",
            "codomain": "comparative recognition shadow value",
            "operator_type": "comparative_relational",
            "relational_order": "competitive_first_order",
            "requires_neighbors": True,
            "requires_direction": False,
            "requires_weights": True,
            "requires_flow": False,
            "requires_comparison_set": True,
            "admissibility_conditions": "Competitor set must be fixed by geography, accessibility, physical similarity, or another declared comparison rule before recognition deficit is evaluated.",
            "supported_disequilibrium_form": "D_i^(S) = E[(S R)_i | C(P)_i] - (S R)_i",
            "current_empirical_status": "linked to comparative shadowing theory but requires integration test",
            "current_status": "integration_candidate",
        },
        {
            "operator_id": "O07",
            "operator_family": "local_differential",
            "symbol": "D_local",
            "formal_definition": "D_local(P)_i in {gradient, local range, dispersion, transition strength, center-neighbor difference}",
            "acts_on": "physical_context",
            "domain": "spatial physical and contextual fields",
            "codomain": "local derivative and transition descriptors",
            "operator_type": "local_spatial_derivative",
            "relational_order": "local_differential",
            "requires_neighbors": True,
            "requires_direction": True,
            "requires_weights": True,
            "requires_flow": False,
            "requires_comparison_set": False,
            "admissibility_conditions": "Derivative stencil, neighborhood radius, and orientation must be fixed before validation; no target-derived smoothing or post hoc feature construction.",
            "supported_disequilibrium_form": "Pr(G_i = R) = F(Y_i, C_i, D_local(P)_i)",
            "current_empirical_status": "Scripts 228 and 229b show local differential operators do not replace the stronger compact spatial trend representation",
            "current_status": "tested_not_primary_substitute",
        },
        {
            "operator_id": "O08",
            "operator_family": "broad_spatial_trend",
            "symbol": "T_s",
            "formal_definition": "T_s(i) = basis(x_i, y_i)",
            "acts_on": "spatial_location",
            "domain": "projected spatial coordinates or declared broad spatial basis",
            "codomain": "smooth spatial trend descriptors",
            "operator_type": "spatial_nonstationarity_proxy",
            "relational_order": "broad_scale_proxy",
            "requires_neighbors": False,
            "requires_direction": False,
            "requires_weights": False,
            "requires_flow": False,
            "requires_comparison_set": False,
            "admissibility_conditions": "Trend basis must be declared before validation; coordinates and trend terms must be treated as proxies for unresolved spatial structure, not causal mechanisms.",
            "supported_disequilibrium_form": "Pr(G_i = R) = F(Y_i, C_i, Delta C_i, T_s(i))",
            "current_empirical_status": "best tested compact proxy for broad spatial nonstationarity in current Oregon persistence experiments",
            "current_status": "best_tested_compact_proxy",
        },
        {
            "operator_id": "O09",
            "operator_family": "operator_mixture",
            "symbol": "M_i",
            "formal_definition": "L_i = sum_k alpha_ik L_k, alpha_ik >= 0, sum_k alpha_ik = 1",
            "acts_on": "operator_family",
            "domain": "candidate operators L_k and admissible spatial mixture weights alpha_ik",
            "codomain": "location-specific mixture-operated recognition field",
            "operator_type": "continuous_operator_state",
            "relational_order": "mixture",
            "requires_neighbors": True,
            "requires_direction": "depends_on_components",
            "requires_weights": True,
            "requires_flow": "depends_on_components",
            "requires_comparison_set": "depends_on_components",
            "admissibility_conditions": "Mixture weights must be estimated under spatial cross-validation; weights cannot be directly derived from target labels; component operators must each satisfy their own admissibility conditions.",
            "supported_disequilibrium_form": "D_i^(M) = E[(L_i R)_i | C(P)_i] - (L_i R)_i",
            "current_empirical_status": "future theoretical extension",
            "current_status": "future_theoretical_extension",
        },
    ]

    return pd.DataFrame(rows)


def build_disequilibrium_forms() -> pd.DataFrame:
    rows = [
        {
            "form_id": "D01",
            "operator_symbol": "I",
            "name": "pointwise_disequilibrium",
            "formula": "D_i^(I) = E[R_i | C(P)_i] - R_i",
            "interpretation": "Recognition deficit at the focal cell itself.",
            "mathematical_role": "baseline zero-order disequilibrium",
            "test_readiness": "ready",
        },
        {
            "form_id": "D02",
            "operator_symbol": "A_r",
            "name": "neighborhood_aggregated_disequilibrium",
            "formula": "D_i^(A_r) = E[(A_r R)_i | C(P)_i] - (A_r R)_i",
            "interpretation": "Recognition deficit in the local neighborhood recognition state.",
            "mathematical_role": "first relational extension of pointwise RDE",
            "test_readiness": "ready_next",
        },
        {
            "form_id": "D03",
            "operator_symbol": "L",
            "name": "laplacian_disequilibrium",
            "formula": "D_i^(L) = E[(L R)_i | C(P)_i] - (L R)_i",
            "interpretation": "Disequilibrium in local recognition divergence relative to the recognition graph.",
            "mathematical_role": "graph-relational disequilibrium",
            "test_readiness": "requires_graph_design",
        },
        {
            "form_id": "D04",
            "operator_symbol": "T",
            "name": "transmission_disequilibrium",
            "formula": "D_i^(T) = E[(T R)_i | C(P)_i] - (T R)_i",
            "interpretation": "Disequilibrium in directed recognition transmission or inflow-outflow structure.",
            "mathematical_role": "directed transport disequilibrium",
            "test_readiness": "future",
        },
        {
            "form_id": "D05",
            "operator_symbol": "S",
            "name": "comparative_shadow_disequilibrium",
            "formula": "D_i^(S) = E[(S R)_i | C(P)_i] - (S R)_i",
            "interpretation": "Disequilibrium after accounting for recognition dominance from admissible geographic competitors.",
            "mathematical_role": "competitive relational disequilibrium",
            "test_readiness": "integration_candidate",
        },
        {
            "form_id": "D06",
            "operator_symbol": "M_i",
            "name": "mixture_operator_disequilibrium",
            "formula": "D_i^(M) = E[(L_i R)_i | C(P)_i] - (L_i R)_i",
            "interpretation": "Disequilibrium under a continuously varying mixture of admissible spatial recognition operators.",
            "mathematical_role": "continuous operator-state disequilibrium",
            "test_readiness": "future",
        },
    ]

    return pd.DataFrame(rows)


def build_implementation_readiness(operators: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in operators.iterrows():
        status = r["current_status"]

        if status in {
            "baseline_ready",
            "partially_supported_ready",
            "best_tested_compact_proxy",
        }:
            readiness = "ready"
        elif status in {
            "ready_for_targeted_test",
            "formal_candidate_next",
            "integration_candidate",
        }:
            readiness = "near_term"
        elif status == "tested_not_primary_substitute":
            readiness = "defer_current_form"
        else:
            readiness = "future"

        family = r["operator_family"]

        if family == "pointwise_identity":
            primary_risk = "under-specification of relational recognition structure"
            next_action = "retain as formal baseline and comparison operator"
        elif family == "neighborhood_aggregation":
            primary_risk = "arbitrary radius or kernel choice"
            next_action = "run controlled multiscale A_r operator test"
        elif family == "graph_laplacian":
            primary_risk = "graph definition may dominate inferred disequilibrium"
            next_action = "construct controlled geographic graph and compare against I and A_r"
        elif family == "flow_or_transmission":
            primary_risk = "lack of defensible independent transmission weights"
            next_action = "defer until a defensible movement or transmission proxy exists"
        elif family == "comparative_shadow":
            primary_risk = "post hoc competitor-set definition"
            next_action = "integrate with comparative shadowing mechanism under fixed competitor rules"
        elif family == "local_differential":
            primary_risk = "high-dimensional operator expansion without transferable gain"
            next_action = "retain as diagnostic family but do not prioritize as main operator"
        elif family == "broad_spatial_trend":
            primary_risk = "causal overinterpretation of coordinate-derived proxy"
            next_action = "retain as compact benchmark for unresolved spatial nonstationarity"
        elif family == "operator_mixture":
            primary_risk = "mixture-weight leakage and weak identifiability"
            next_action = "defer until discrete operator family is empirically tested"
        else:
            primary_risk = "target leakage or post hoc operator design"
            next_action = "future theory only"

        rows.append(
            {
                "operator_id": r["operator_id"],
                "operator_family": family,
                "symbol": r["symbol"],
                "readiness": readiness,
                "minimum_inputs": r["domain"],
                "primary_risk": primary_risk,
                "recommended_next_action": next_action,
            }
        )

    return pd.DataFrame(rows)


def run_consistency_checks(
    operators: pd.DataFrame,
    disequilibrium: pd.DataFrame,
    readiness: pd.DataFrame,
) -> list[str]:
    checks = []

    checks.append(f"unique_operator_ids: {operators['operator_id'].is_unique}")
    checks.append(f"unique_disequilibrium_form_ids: {disequilibrium['form_id'].is_unique}")

    readiness_complete = set(readiness["operator_id"]) == set(operators["operator_id"])
    checks.append(f"readiness_matches_operator_set: {readiness_complete}")

    admissibility_complete = (
        operators["admissibility_conditions"].fillna("").str.strip().ne("").all()
    )
    checks.append(f"all_operators_have_admissibility_conditions: {admissibility_complete}")

    formal_definition_complete = (
        operators["formal_definition"].fillna("").str.strip().ne("").all()
    )
    checks.append(f"all_operators_have_formal_definitions: {formal_definition_complete}")

    return checks


def build_markdown(
    operators: pd.DataFrame,
    disequilibrium: pd.DataFrame,
    readiness: pd.DataFrame,
) -> str:
    operator_display_cols = [
        "operator_id",
        "operator_family",
        "symbol",
        "formal_definition",
        "acts_on",
        "operator_type",
        "current_status",
    ]

    diseq_display_cols = [
        "form_id",
        "operator_symbol",
        "name",
        "formula",
        "mathematical_role",
        "test_readiness",
    ]

    readiness_display_cols = [
        "operator_id",
        "operator_family",
        "readiness",
        "primary_risk",
        "recommended_next_action",
    ]

    parts = []

    parts.append("# RDE/SRS Formal Operator Taxonomy and Admissibility v01\n")
    parts.append(f"Generated by `{SCRIPT}`.\n")

    parts.append("## Purpose\n")
    parts.append(
        "This document converts the emerging RDE/SRS operator taxonomy into explicit formal operator objects and admissibility conditions.\n"
    )
    parts.append(
        "The purpose is not to create another predictive model. The purpose is to constrain future mathematical development so that operator definitions, spatial relations, neighborhood structures, and comparison rules are specified before validation.\n"
    )
    parts.append(
        "The central theoretical proposition is that recognition disequilibrium may require an operator-dependent definition when recognition structure is relational.\n"
    )

    parts.append("## Formal operator family\n")
    parts.append(
        "The current candidate operator family is:\n\n"
        "$$\n"
        "\\mathfrak{L} = \\{I, A_r, \\Delta_{r_1,r_2}, L, T, S, D_{local}, T_s, M_i\\}.\n"
        "$$\n"
    )

    parts.append(
        "Some operators act directly on recognition: $I, A_r, L, T, S$.\n\n"
        "Some act on physical or contextual fields: $\\Delta_{r_1,r_2}, D_{local}$.\n\n"
        "One is currently treated as a broad spatial nonstationarity proxy: $T_s$.\n\n"
        "The final object is a possible continuous mixture of admissible operators: $M_i$.\n"
    )

    parts.append("## Formal operator taxonomy\n")
    parts.append(markdown_table(operators[operator_display_cols]))
    parts.append("\n")

    parts.append("## General operator-dependent RDE form\n")
    parts.append(
        "Let $P_i$ denote physical potential or physical exceptionality, "
        "$C(P)_i$ denote multiscale physical context, and $R_i$ denote observed recognition.\n\n"
        "For an admissible operator $\\mathcal{L} \\in \\mathfrak{L}$, define:\n\n"
        "$$\n"
        "R_i^{(\\mathcal{L})} = (\\mathcal{L}R)_i.\n"
        "$$\n\n"
        "Then a general operator-dependent recognition disequilibrium is:\n\n"
        "$$\n"
        "D_i^{(\\mathcal{L})} = \\mathbb{E}[(\\mathcal{L}R)_i \\mid C(P)_i] - (\\mathcal{L}R)_i.\n"
        "$$\n\n"
        "The pointwise RDE formulation is therefore the special case $\\mathcal{L}=I$.\n"
    )

    parts.append("## Candidate disequilibrium forms\n")
    parts.append(markdown_table(disequilibrium[diseq_display_cols]))
    parts.append("\n")

    parts.append("## Admissibility principle\n")
    parts.append(
        "An operator is admissible only if its relational structure is defined independently of the target phenomenon being evaluated.\n\n"
        "For an operator $\\mathcal{L}_\\theta$ with structural parameters $\\theta$, the practical design requirement is that neighborhood radii, graph edges, kernels, comparison sets, flow weights, and operator families are not chosen by inspecting held-out target labels.\n"
    )

    parts.append("## Core admissibility rules\n")
    parts.append(
        "1. Operators must be defined before held-out validation.\n"
        "2. Neighborhood weights, graph edges, transmission flows, and comparison sets cannot be constructed from held-out persistence labels.\n"
        "3. Physical context operators must not incorporate recognition targets during construction.\n"
        "4. Coordinate or spatial-trend terms are admissible as diagnostic proxies for spatial nonstationarity, but not automatically interpretable as mechanisms.\n"
        "5. High-dimensional differential operator families should not be retained merely because they are mathematically rich. They must demonstrate transferable value.\n"
        "6. Operator mixtures must satisfy both component-level admissibility and mixture-weight admissibility.\n"
        "7. A relational operator should be introduced because it represents a geographic recognition process, not simply because it increases predictive performance.\n"
    )

    parts.append("## Current empirical interpretation\n")
    parts.append(
        "The current Oregon evidence supports the following restricted interpretation.\n\n"
        "First, broad spatial nonstationarity matters for recognition persistence.\n\n"
        "Second, compact spatial trend terms currently outperform the large local differential operator expansion as a transferable representation.\n\n"
        "Third, the failure of local differential operators to dominate does not falsify relational RDE. It indicates that the tested physical derivative family is not the correct substitute for the unresolved spatial structure.\n\n"
        "Fourth, the next mathematically disciplined comparison should move from feature expansion to operator comparison.\n\n"
        "The most defensible sequence is $I \\rightarrow A_r \\rightarrow L$.\n"
    )

    parts.append("## Implementation readiness\n")
    parts.append(markdown_table(readiness[readiness_display_cols]))
    parts.append("\n")

    parts.append("## Near-term mathematical research program\n")
    parts.append(
        "### Stage 1: Identity baseline\n\n"
        "Retain $D_i^{(I)} = \\mathbb{E}[R_i \\mid C(P)_i] - R_i$.\n\n"
        "### Stage 2: Neighborhood aggregation operator\n\n"
        "Construct $(A_rR)_i = \\sum_j w_{ij}(r)R_j$ and define "
        "$D_i^{(A_r)} = \\mathbb{E}[(A_rR)_i \\mid C(P)_i] - (A_rR)_i$.\n\n"
        "The critical test is whether persistence structure becomes more transferable when recognition is treated as a neighborhood state rather than a point value.\n\n"
        "### Stage 3: Graph Laplacian operator\n\n"
        "Construct an independently defined spatial graph $G=(V,E,W)$ and "
        "$(LR)_i = \\sum_j w_{ij}(R_i-R_j)$.\n\n"
        "Then define $D_i^{(L)} = \\mathbb{E}[(LR)_i \\mid C(P)_i] - (LR)_i$.\n\n"
        "This asks whether recognition disequilibrium is better represented as a divergence from relational recognition structure.\n"
    )

    parts.append("## Stopping rule\n")
    parts.append(
        "The operator program should not expand indefinitely.\n\n"
        "A new operator family should proceed only when at least one of the following is true:\n\n"
        "1. it represents a distinct geographic recognition process;\n"
        "2. it produces a falsifiable theoretical distinction;\n"
        "3. it improves spatial transfer under controlled validation;\n"
        "4. it explains previously unresolved spatial structure more interpretably than a coordinate proxy.\n\n"
        "If none of these conditions holds, the operator family should be deferred.\n"
    )

    parts.append("## Recommended next script\n")
    parts.append(
        "`233_oregon_srs_operator_dependent_disequilibrium_prototype_v01.py`\n\n"
        "Its purpose should be to construct and compare $I$, $A_r$, and $L$ under the same Oregon spatial folds and the same downstream persistence-validation framework.\n\n"
        "The goal should not be to maximize model accuracy. The goal should be to determine whether the mathematical object called recognition disequilibrium changes meaningfully when recognition is treated as a point state, neighborhood state, or graph-divergence state.\n"
    )

    return "\n".join(parts)


def build_qa_report(
    operators: pd.DataFrame,
    disequilibrium: pd.DataFrame,
    readiness: pd.DataFrame,
    checks: list[str],
) -> str:
    readiness_summary = (
        readiness["readiness"]
        .value_counts()
        .rename_axis("readiness")
        .reset_index(name="n_operators")
    )

    operator_type_summary = (
        operators["operator_type"]
        .value_counts()
        .rename_axis("operator_type")
        .reset_index(name="n_operators")
    )

    lines = [
        f"Script: {SCRIPT}",
        "",
        "PURPOSE",
        "Formalize RDE/SRS operator taxonomy, admissibility conditions, operator-dependent disequilibrium forms, and implementation readiness.",
        "",
        f"Formal operators: {len(operators):,}",
        f"Disequilibrium forms: {len(disequilibrium):,}",
        f"Implementation readiness records: {len(readiness):,}",
        "",
        "FORMAL OPERATOR FAMILY",
        "{I, A_r, Delta_(r1,r2), L, T, S, D_local, T_s, M_i}",
        "",
        "OPERATOR TYPE SUMMARY",
        operator_type_summary.to_string(index=False),
        "",
        "READINESS SUMMARY",
        readiness_summary.to_string(index=False),
        "",
        "CONSISTENCY CHECKS",
        *checks,
        "",
        "CURRENT THEORY STATUS",
        "1. Pointwise RDE remains the baseline formulation.",
        "2. Neighborhood aggregation is the nearest-term relational extension.",
        "3. Graph Laplacian disequilibrium is the next formal relational candidate.",
        "4. Local physical differential operators were tested but did not replace compact broad spatial trend structure.",
        "5. Broad spatial trend remains a proxy for unresolved spatial nonstationarity, not a causal mechanism.",
        "6. Flow, comparative-shadow, and operator-mixture formulations remain later-stage theory objects.",
        "",
        "NEXT RECOMMENDED TEST",
        "Controlled operator-dependent disequilibrium prototype: I versus A_r versus L.",
        "",
        "RECOMMENDED NEXT SCRIPT",
        "233_oregon_srs_operator_dependent_disequilibrium_prototype_v01.py",
        "",
        "OUTPUTS",
        str(OUT_OPERATORS),
        str(OUT_DISEQUILIBRIUM),
        str(OUT_IMPLEMENTATION),
        str(OUT_MD),
        str(OUT_QA),
    ]

    return "\n".join(lines)


def main():
    log.info("Starting Script 232")

    operators = build_operator_admissibility()
    disequilibrium = build_disequilibrium_forms()
    readiness = build_implementation_readiness(operators)

    log.info("Formal operators: %s", f"{len(operators):,}")
    log.info("Disequilibrium forms: %s", f"{len(disequilibrium):,}")

    checks = run_consistency_checks(
        operators=operators,
        disequilibrium=disequilibrium,
        readiness=readiness,
    )

    for check in checks:
        log.info(check)

    operators.to_csv(OUT_OPERATORS, index=False)
    disequilibrium.to_csv(OUT_DISEQUILIBRIUM, index=False)
    readiness.to_csv(OUT_IMPLEMENTATION, index=False)

    markdown = build_markdown(
        operators=operators,
        disequilibrium=disequilibrium,
        readiness=readiness,
    )

    OUT_MD.write_text(markdown, encoding="utf-8")

    qa = build_qa_report(
        operators=operators,
        disequilibrium=disequilibrium,
        readiness=readiness,
        checks=checks,
    )

    OUT_QA.write_text(qa, encoding="utf-8")

    log.info("Completed Script 232")
    log.info("Formal operator family: {I, A_r, Delta_(r1,r2), L, T, S, D_local, T_s, M_i}")
    log.info("Created:")

    for path in [
        OUT_OPERATORS,
        OUT_DISEQUILIBRIUM,
        OUT_IMPLEMENTATION,
        OUT_MD,
        OUT_QA,
    ]:
        log.info("  %s", path)


if __name__ == "__main__":
    main()