#!/usr/bin/env python3
"""
231_rde_srs_theory_integration_mathematical_consolidation_v01.py

PURPOSE
-------
Integrate the Oregon SRS operator-persistence findings into the broader
RDE / UREM theoretical framework.

This script DOES NOT fit a new predictive model.

Instead, it:

1. Reads the frozen Oregon SRS evidence ledger from Script 230.
2. Reads the frozen model comparison and fold synthesis.
3. Constructs a formal hierarchy linking:
       physical state
       multiscale context
       expected recognition
       recognition disequilibrium
       operator regime
       operator persistence
4. Distinguishes:
       empirical findings
       provisional interpretations
       theoretical hypotheses
       unresolved questions
5. Formalizes the operator-dependent disequilibrium hypothesis.
6. Creates an operator taxonomy for the next formal mathematical stage.
7. Creates a falsification ledger so later scripts test theory rather
   than merely add models.
8. Produces a mathematical consolidation document and QA report.

IMPORTANT THEORETICAL CAUTION
-----------------------------
Script 230 supports the statement that broad spatial trend T_s is the
best compact spatial representation AMONG THE TESTED REPRESENTATIONS.

It does NOT establish the identity:

    S_i = T_s

The correct provisional interpretation is:

    T_s is currently the best tested compact proxy for unresolved
    spatially structured variation in operator persistence.

The broader latent spatial mechanism S_i remains unresolved.

OUTPUTS
-------
data/processed/
    rde_srs_theory_integration_hierarchy_v01.csv
    rde_srs_operator_taxonomy_v01.csv
    rde_srs_theoretical_claim_ledger_v01.csv
    rde_srs_falsification_design_v01.csv
    rde_srs_fold_regime_interpretation_v01.csv
    rde_srs_mathematical_consolidation_v01.md
    rde_srs_theory_integration_qa_v01.txt
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT = "231_rde_srs_theory_integration_mathematical_consolidation_v01"

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"

INPUT_EVIDENCE = (
    P / "oregon_srs_branch_decision_evidence_ledger_v02.csv"
)

INPUT_MODEL_COMPARISON = (
    P / "oregon_srs_branch_decision_model_comparison_v02.csv"
)

INPUT_FOLD_SYNTHESIS = (
    P / "oregon_srs_branch_decision_fold_synthesis_v02.csv"
)

OUT_HIERARCHY = (
    P / "rde_srs_theory_integration_hierarchy_v01.csv"
)

OUT_OPERATOR_TAXONOMY = (
    P / "rde_srs_operator_taxonomy_v01.csv"
)

OUT_CLAIM_LEDGER = (
    P / "rde_srs_theoretical_claim_ledger_v01.csv"
)

OUT_FALSIFICATION = (
    P / "rde_srs_falsification_design_v01.csv"
)

OUT_FOLD_INTERPRETATION = (
    P / "rde_srs_fold_regime_interpretation_v01.csv"
)

OUT_CONSOLIDATION = (
    P / "rde_srs_mathematical_consolidation_v01.md"
)

OUT_QA = (
    P / "rde_srs_theory_integration_qa_v01.txt"
)


logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT}] %(levelname)s: %(message)s",
)

log = logging.getLogger(SCRIPT)


# ============================================================
# HELPERS
# ============================================================

def read_csv_required(path: Path) -> pd.DataFrame:
    """
    Read a required CSV and fail clearly if it is missing.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required input does not exist:\n{path}"
        )

    return pd.read_csv(path)


def safe_metric(
    df: pd.DataFrame,
    model: str,
    column: str = "roc_auc",
) -> float:
    """
    Extract a model metric safely.
    """

    if "model" not in df.columns:
        return np.nan

    if column not in df.columns:
        return np.nan

    s = df.loc[df["model"] == model, column]

    if s.empty:
        return np.nan

    return float(s.iloc[-1])


def fmt(x, digits=3):
    """
    Format numeric values safely.
    """

    if pd.isna(x):
        return "NA"

    return f"{float(x):.{digits}f}"


def yes_no(value):
    """
    Convert boolean-like values to readable strings.
    """

    return "yes" if bool(value) else "no"


def markdown_table(df: pd.DataFrame) -> str:
    """
    Convert dataframe to markdown with graceful empty handling.
    """

    if df.empty:
        return "No records available."

    return df.to_markdown(index=False)


# ============================================================
# THEORY HIERARCHY
# ============================================================

def build_theory_hierarchy() -> pd.DataFrame:
    """
    Build the formal RDE -> SRS hierarchy.

    The hierarchy distinguishes physical state, context,
    expectation, disequilibrium, operator regime, and persistence.
    """

    rows = [
        {
            "level": 1,
            "symbol": "P_i",
            "name": "physical_state",
            "formal_role": (
                "Local physical state or physical potential at location i."
            ),
            "mathematical_object": (
                "scalar, vector, or field-valued physical descriptor"
            ),
            "empirical_status": "established_input_layer",
            "rde_role": (
                "Provides the physical basis against which recognition "
                "expectation is constructed."
            ),
            "srs_role": (
                "Supplies the underlying geographic field whose context "
                "may be represented pointwise or relationally."
            ),
        },
        {
            "level": 2,
            "symbol": "C_r(P)_i",
            "name": "multiscale_physical_context",
            "formal_role": (
                "Spatial contextualization of physical state over scale r."
            ),
            "mathematical_object": (
                "multiscale spatial contextual operator"
            ),
            "empirical_status": "supported",
            "rde_role": (
                "Defines comparable geographic context for expected recognition."
            ),
            "srs_role": (
                "Strong predictor family for operator persistence when combined "
                "with recognition state and nonlinear response structure."
            ),
        },
        {
            "level": 3,
            "symbol": "Delta_C_i",
            "name": "cross_scale_context_contrast",
            "formal_role": (
                "Difference or contrast between contextual representations "
                "at two or more spatial scales."
            ),
            "mathematical_object": (
                "cross-scale contrast functional"
            ),
            "empirical_status": "partially_supported",
            "rde_role": (
                "Represents whether local physical conditions differ from "
                "their broader geographic setting."
            ),
            "srs_role": (
                "Improves some persistence models but does not fully replace "
                "spatially varying response geometry."
            ),
        },
        {
            "level": 4,
            "symbol": "E[R_i | C(P)]",
            "name": "expected_recognition",
            "formal_role": (
                "Conditional recognition expectation under comparable "
                "physical and geographic context."
            ),
            "mathematical_object": (
                "conditional expectation or learned counterfactual expectation"
            ),
            "empirical_status": "core_rde_construct",
            "rde_role": (
                "Defines the recognition level expected from geographic context."
            ),
            "srs_role": (
                "Provides the expectation term whose interpretation may depend "
                "on the recognition operator regime."
            ),
        },
        {
            "level": 5,
            "symbol": "D_i",
            "name": "recognition_disequilibrium",
            "formal_role": (
                "Difference between expected and observed recognition state."
            ),
            "mathematical_object": (
                "residual, standardized residual, or operator-dependent residual"
            ),
            "empirical_status": "core_rde_construct_under_refinement",
            "rde_role": (
                "Measures recognition deficit or disequilibrium."
            ),
            "srs_role": (
                "May require operator-specific definition if recognition "
                "structure is relational rather than pointwise."
            ),
        },
        {
            "level": 6,
            "symbol": "G_i",
            "name": "recognition_operator_regime",
            "formal_role": (
                "Latent or classified regime describing how recognition "
                "is spatially organized."
            ),
            "mathematical_object": (
                "discrete regime label or continuous operator-mixture state"
            ),
            "empirical_status": "supported_as_structured_but_not_fully_resolved",
            "rde_role": (
                "Potentially determines the correct spatial interpretation "
                "of disequilibrium."
            ),
            "srs_role": (
                "Distinguishes persistent relational versus pointwise behavior."
            ),
        },
        {
            "level": 7,
            "symbol": "Pi_i",
            "name": "operator_persistence",
            "formal_role": (
                "Stability of recognition operator regime across admissible "
                "perturbations, scales, or state realizations."
            ),
            "mathematical_object": (
                "persistence probability, stability functional, or binary target"
            ),
            "empirical_status": "supported_empirical_target",
            "rde_role": (
                "Determines whether operator interpretation is stable enough "
                "to influence disequilibrium analysis."
            ),
            "srs_role": (
                "Current Oregon branch target."
            ),
        },
        {
            "level": 8,
            "symbol": "T_s",
            "name": "broad_spatial_trend_proxy",
            "formal_role": (
                "Low-order representation of broad spatially structured "
                "variation in operator persistence."
            ),
            "mathematical_object": (
                "coordinate-derived trend basis"
            ),
            "empirical_status": "best_tested_compact_spatial_proxy",
            "rde_role": (
                "Potential control for broad spatial nonstationarity."
            ),
            "srs_role": (
                "Best tested compact representation of unresolved spatial "
                "variation; not established as causal mechanism."
            ),
        },
        {
            "level": 9,
            "symbol": "B_i",
            "name": "structural_transfer_boundary",
            "formal_role": (
                "Region of state-space or geography where learned operator "
                "relationships fail to transfer."
            ),
            "mathematical_object": (
                "support boundary, extrapolation boundary, or latent process boundary"
            ),
            "empirical_status": "unresolved_hypothesis",
            "rde_role": (
                "Potential limit on geographic transfer of recognition laws."
            ),
            "srs_role": (
                "Candidate explanation for persistent Fold 0 failure."
            ),
        },
    ]

    return pd.DataFrame(rows)


# ============================================================
# OPERATOR TAXONOMY
# ============================================================

def build_operator_taxonomy() -> pd.DataFrame:
    """
    Construct initial formal operator taxonomy.

    This is intentionally broader than the tested Script 228 family.
    The taxonomy distinguishes mathematical operator classes from
    empirical implementation status.
    """

    rows = [
        {
            "operator_id": "O01",
            "operator_family": "pointwise_identity",
            "symbol": "I",
            "generic_form": "(I R)_i = R_i",
            "spatial_meaning": (
                "Recognition is interpreted locally without explicit relational coupling."
            ),
            "candidate_rde_form": (
                "D_i^(g) = E[R_i | C(P)] - R_i"
            ),
            "tested_status": "conceptually_present",
            "current_support": "baseline_reference",
            "next_priority": "retain",
        },
        {
            "operator_id": "O02",
            "operator_family": "neighborhood_aggregation",
            "symbol": "A_r",
            "generic_form": (
                "(A_r R)_i = sum_j w_ij(r) R_j"
            ),
            "spatial_meaning": (
                "Recognition state depends on weighted neighboring recognition."
            ),
            "candidate_rde_form": (
                "D_i^(r) = E[A_r R_i | C(P)] - A_r R_i"
            ),
            "tested_status": "partially_implicit_in_prior_srs_work",
            "current_support": "candidate",
            "next_priority": "formalize",
        },
        {
            "operator_id": "O03",
            "operator_family": "context_contrast",
            "symbol": "Delta_(r1,r2)",
            "generic_form": (
                "Delta C_i = C_r1(P)_i - C_r2(P)_i"
            ),
            "spatial_meaning": (
                "Measures local-to-regional geographic contrast."
            ),
            "candidate_rde_form": (
                "D_i = F(E[R|C], R, Delta C_i)"
            ),
            "tested_status": "tested",
            "current_support": "partial_support",
            "next_priority": "retain",
        },
        {
            "operator_id": "O04",
            "operator_family": "graph_laplacian",
            "symbol": "L",
            "generic_form": (
                "(L R)_i = sum_j w_ij (R_i - R_j)"
            ),
            "spatial_meaning": (
                "Measures recognition divergence from a relational neighborhood."
            ),
            "candidate_rde_form": (
                "D_i^(L) = E[L R_i | C(P)] - L R_i"
            ),
            "tested_status": "not_formally_tested_in_current_branch",
            "current_support": "theoretical_candidate",
            "next_priority": "future_targeted_test",
        },
        {
            "operator_id": "O05",
            "operator_family": "flow_or_transmission",
            "symbol": "T",
            "generic_form": (
                "(T R)_i = sum_j f_ji R_j - sum_j f_ij R_i"
            ),
            "spatial_meaning": (
                "Recognition is organized through directional transmission or flow."
            ),
            "candidate_rde_form": (
                "D_i^(T) = E[T R_i | C(P)] - T R_i"
            ),
            "tested_status": "mechanism_level_precursors_exist",
            "current_support": "theoretical_candidate",
            "next_priority": "future",
        },
        {
            "operator_id": "O06",
            "operator_family": "comparative_shadow",
            "symbol": "S",
            "generic_form": (
                "(S R)_i = R_i - max_j(w_ij R_j)"
            ),
            "spatial_meaning": (
                "Recognition depends on comparison with dominant nearby alternatives."
            ),
            "candidate_rde_form": (
                "D_i^(S) = E[S R_i | C(P)] - S R_i"
            ),
            "tested_status": "related_prior_rde_mechanism_work_exists",
            "current_support": "mechanism_candidate",
            "next_priority": "integration_candidate",
        },
        {
            "operator_id": "O07",
            "operator_family": "local_differential",
            "symbol": "D_local",
            "generic_form": (
                "D_local(P) in {gradient, range, dispersion, transition, scale difference}"
            ),
            "spatial_meaning": (
                "Local spatial derivatives and transition structure."
            ),
            "candidate_rde_form": (
                "operator persistence = F(Y, C, D_local)"
            ),
            "tested_status": "tested_scripts_228_229b",
            "current_support": "not_supported_as_primary_spatial_substitute",
            "next_priority": "defer",
        },
        {
            "operator_id": "O08",
            "operator_family": "broad_spatial_trend",
            "symbol": "T_s",
            "generic_form": (
                "T_s = basis(x_i, y_i)"
            ),
            "spatial_meaning": (
                "Broad smooth spatial nonstationarity."
            ),
            "candidate_rde_form": (
                "Pr(G_i=R)=F(Y_i,C_i,Delta C_i,T_s)"
            ),
            "tested_status": "tested",
            "current_support": "best_tested_compact_spatial_proxy",
            "next_priority": "retain_with_causal_caution",
        },
        {
            "operator_id": "O09",
            "operator_family": "operator_mixture",
            "symbol": "M_i",
            "generic_form": (
                "L_i = sum_k alpha_ik L_k, with sum_k alpha_ik = 1"
            ),
            "spatial_meaning": (
                "Recognition organization may vary continuously between operator types."
            ),
            "candidate_rde_form": (
                "D_i^(M) = E[L_i R_i | C(P)] - L_i R_i"
            ),
            "tested_status": "not_tested",
            "current_support": "theoretical_extension",
            "next_priority": "later",
        },
    ]

    return pd.DataFrame(rows)


# ============================================================
# CLAIM LEDGER
# ============================================================

def build_claim_ledger(
    evidence: pd.DataFrame,
    model_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build explicit theory claim ledger.

    The goal is to prevent empirical evidence from silently becoming
    stronger theoretical claims than the data support.
    """

    rows = [
        {
            "claim_id": "C01",
            "claim": (
                "Operator persistence is prospectively structured rather "
                "than purely random."
            ),
            "claim_class": "empirical",
            "status": "supported",
            "basis": (
                "Nonlinear models achieve meaningful prospective discrimination "
                "across most folds."
            ),
            "allowed_inference": (
                "Persistence has learnable geographic and recognition-state structure."
            ),
            "forbidden_overclaim": (
                "The exact causal operator mechanism has been identified."
            ),
        },
        {
            "claim_id": "C02",
            "claim": (
                "Operator persistence is nonlinear in recognition state and "
                "multiscale context."
            ),
            "claim_class": "empirical",
            "status": "supported",
            "basis": (
                "Nonlinear RF substantially exceeds additive logistic and "
                "low-order interaction models."
            ),
            "allowed_inference": (
                "A simple additive law is inadequate."
            ),
            "forbidden_overclaim": (
                "Random forest structure is itself the mathematical law."
            ),
        },
        {
            "claim_id": "C03",
            "claim": (
                "A universal scalar mismatch index is sufficient to determine "
                "operator persistence."
            ),
            "claim_class": "empirical",
            "status": "rejected_current_form",
            "basis": (
                "Compact mismatch model materially underperforms nonlinear benchmark."
            ),
            "allowed_inference": (
                "Current evidence rejects the tested universal scalar mismatch law."
            ),
            "forbidden_overclaim": (
                "No scalar invariant can ever exist."
            ),
        },
        {
            "claim_id": "C04",
            "claim": (
                "Simple discrete geographic regimes fully explain spatial "
                "variation in persistence."
            ),
            "claim_class": "empirical",
            "status": "not_supported",
            "basis": (
                "Tested KMeans regime models underperform coordinate/trend conditioning."
            ),
            "allowed_inference": (
                "The tested simple regime partition is insufficient."
            ),
            "forbidden_overclaim": (
                "All discrete regime theories are impossible."
            ),
        },
        {
            "claim_id": "C05",
            "claim": (
                "Broad spatial trend is the best tested compact proxy for "
                "unresolved spatially structured persistence variation."
            ),
            "claim_class": "empirical_interpretive",
            "status": "supported_provisionally",
            "basis": (
                "Spatial trend approximately matches or slightly exceeds raw "
                "coordinate performance in Script 229b."
            ),
            "allowed_inference": (
                "Use T_s as current compact spatial representation."
            ),
            "forbidden_overclaim": (
                "S_i equals T_s causally or ontologically."
            ),
        },
        {
            "claim_id": "C06",
            "claim": (
                "The tested local differential operator family explains the "
                "main unresolved spatial persistence signal."
            ),
            "claim_class": "empirical",
            "status": "not_supported_current_form",
            "basis": (
                "Differential models underperform clean spatial-trend and "
                "raw-coordinate models."
            ),
            "allowed_inference": (
                "Defer the current Script 228 differential family."
            ),
            "forbidden_overclaim": (
                "All spatial differential operators are irrelevant."
            ),
        },
        {
            "claim_id": "C07",
            "claim": (
                "Recognition disequilibrium may require an operator-dependent definition."
            ),
            "claim_class": "theoretical_hypothesis",
            "status": "motivated_not_yet_validated",
            "basis": (
                "Persistent relational versus pointwise regimes are structured, "
                "suggesting that the observation operator may affect the meaning "
                "of recognition residuals."
            ),
            "allowed_inference": (
                "Develop and falsify operator-specific disequilibrium formulations."
            ),
            "forbidden_overclaim": (
                "Operator-dependent RDE has already been empirically established."
            ),
        },
        {
            "claim_id": "C08",
            "claim": (
                "Fold 0 represents a structural transfer boundary."
            ),
            "claim_class": "unresolved_hypothesis",
            "status": "unresolved",
            "basis": (
                "Coordinates, trends, contrasts, and differential operators "
                "do not repair Fold 0 transfer."
            ),
            "allowed_inference": (
                "Investigate support, extrapolation, and omitted-process explanations."
            ),
            "forbidden_overclaim": (
                "A true geographic phase boundary has been proven."
            ),
        },
        {
            "claim_id": "C09",
            "claim": (
                "The correct RDE hierarchy may include operator regime between "
                "disequilibrium construction and persistence interpretation."
            ),
            "claim_class": "theoretical_hypothesis",
            "status": "proposed",
            "basis": (
                "SRS persistence results motivate explicit separation between "
                "recognition residual and spatial organization of recognition."
            ),
            "allowed_inference": (
                "Formalize alternative hierarchical models."
            ),
            "forbidden_overclaim": (
                "The hierarchy is uniquely identified by Oregon data."
            ),
        },
    ]

    return pd.DataFrame(rows)


# ============================================================
# FALSIFICATION DESIGN
# ============================================================

def build_falsification_design() -> pd.DataFrame:
    """
    Create explicit falsification targets for future scripts.

    This prevents the project from continuing through open-ended
    feature accumulation.
    """

    rows = [
        {
            "test_id": "F01",
            "hypothesis": (
                "Operator-dependent disequilibrium transfers better than a "
                "single pointwise disequilibrium definition."
            ),
            "null_hypothesis": (
                "Operator-dependent residual definitions provide no prospective "
                "transfer improvement."
            ),
            "required_data": (
                "California and Oregon harmonized physical, recognition, and "
                "operator-regime layers."
            ),
            "test_design": (
                "Compare fixed pointwise residual against operator-conditioned "
                "residual under held-out geographic transfer."
            ),
            "success_condition": (
                "Consistent transfer improvement across states and spatial folds."
            ),
            "failure_condition": (
                "No improvement or improvement limited to one geography."
            ),
            "priority": 1,
        },
        {
            "test_id": "F02",
            "hypothesis": (
                "Broad spatial trend is a proxy for omitted geographic mechanism "
                "rather than a transferable law."
            ),
            "null_hypothesis": (
                "The same normalized trend basis transfers across study regions."
            ),
            "required_data": (
                "California/Oregon harmonized branch outputs."
            ),
            "test_design": (
                "Train spatial-response representation in one geography and "
                "test transfer or structural correspondence in the other."
            ),
            "success_condition": (
                "Specific geographic covariates replace trend consistently."
            ),
            "failure_condition": (
                "Trend remains necessary and nontransferable."
            ),
            "priority": 2,
        },
        {
            "test_id": "F03",
            "hypothesis": (
                "Fold 0 is an unsupported-state-space boundary."
            ),
            "null_hypothesis": (
                "Fold 0 is ordinary sampling noise."
            ),
            "required_data": (
                "Fold-level feature support, nearest-neighbor distances, "
                "density ratios, and geographic boundary diagnostics."
            ),
            "test_design": (
                "Quantify train-test support overlap and compare Fold 0 with "
                "successful folds."
            ),
            "success_condition": (
                "Fold 0 exhibits materially lower support overlap or distinct "
                "state-space occupancy."
            ),
            "failure_condition": (
                "Fold 0 support is comparable to successful folds."
            ),
            "priority": 3,
        },
        {
            "test_id": "F04",
            "hypothesis": (
                "Relational recognition regimes require relational residual operators."
            ),
            "null_hypothesis": (
                "Pointwise residuals are equally informative in relational regimes."
            ),
            "required_data": (
                "Stable operator regime labels and relational recognition operators."
            ),
            "test_design": (
                "Compare residual discrimination and mechanism consistency within "
                "persistent relational versus pointwise regimes."
            ),
            "success_condition": (
                "Relational residual improves specifically within persistent "
                "relational regimes."
            ),
            "failure_condition": (
                "No regime-specific advantage."
            ),
            "priority": 4,
        },
        {
            "test_id": "F05",
            "hypothesis": (
                "Operator regime is better represented as a continuous mixture "
                "than a hard binary state."
            ),
            "null_hypothesis": (
                "Hard regime labels are sufficient."
            ),
            "required_data": (
                "Perturbation-level operator stability trajectories."
            ),
            "test_design": (
                "Estimate continuous operator mixture weights and compare "
                "prospective stability and transfer."
            ),
            "success_condition": (
                "Mixture representation improves calibration and transfer."
            ),
            "failure_condition": (
                "Binary regime remains equally or more stable."
            ),
            "priority": 5,
        },
    ]

    return pd.DataFrame(rows)


# ============================================================
# FOLD INTERPRETATION
# ============================================================

def build_fold_interpretation(
    fold: pd.DataFrame,
) -> pd.DataFrame:
    """
    Translate fold-level metrics into cautious structural interpretations.
    """

    required = {
        "fold",
        "auc_229b_global_rf",
        "auc_229b_raw_coordinate_rf",
        "auc_229b_spatial_trend_rf",
        "auc_229b_differential_operator_rf",
        "auc_229b_differential_plus_trend_rf",
    }

    missing = required - set(fold.columns)

    if missing:
        raise ValueError(
            "Fold synthesis is missing required columns:\n"
            + "\n".join(sorted(missing))
        )

    out = fold[
        [
            "fold",
            "auc_229b_global_rf",
            "auc_229b_raw_coordinate_rf",
            "auc_229b_spatial_trend_rf",
            "auc_229b_differential_operator_rf",
            "auc_229b_differential_plus_trend_rf",
        ]
    ].copy()

    out["coordinate_gain"] = (
        out["auc_229b_raw_coordinate_rf"]
        - out["auc_229b_global_rf"]
    )

    out["trend_gain"] = (
        out["auc_229b_spatial_trend_rf"]
        - out["auc_229b_global_rf"]
    )

    out["differential_gain"] = (
        out["auc_229b_differential_operator_rf"]
        - out["auc_229b_global_rf"]
    )

    interpretations = []

    for _, row in out.iterrows():

        fold_id = int(row["fold"])
        trend_gain = row["trend_gain"]
        coord_gain = row["coordinate_gain"]
        diff_gain = row["differential_gain"]

        if fold_id == 0:
            interpretation = (
                "persistent_transfer_failure_candidate"
            )
            theoretical_note = (
                "Candidate support, extrapolation, or omitted-process boundary. "
                "Do not label as proven phase boundary."
            )

        elif trend_gain >= 0.05:
            interpretation = (
                "strong_broad_spatial_nonstationarity"
            )
            theoretical_note = (
                "Broad spatial response geometry materially improves transfer."
            )

        elif coord_gain >= 0.03:
            interpretation = (
                "moderate_spatial_nonstationarity"
            )
            theoretical_note = (
                "Spatial conditioning helps, but mechanism remains unresolved."
            )

        elif diff_gain > trend_gain and diff_gain > 0.02:
            interpretation = (
                "local_transition_structure_candidate"
            )
            theoretical_note = (
                "Local operator structure may matter in this fold despite weak "
                "global differential performance."
            )

        elif max(trend_gain, coord_gain, diff_gain) <= 0:
            interpretation = (
                "no_spatial_conditioning_gain"
            )
            theoretical_note = (
                "Current spatial representations do not improve transfer."
            )

        else:
            interpretation = (
                "weak_or_mixed_spatial_effect"
            )
            theoretical_note = (
                "Spatial effect exists but is small or representation-dependent."
            )

        interpretations.append(
            {
                "fold": fold_id,
                "fold_interpretation": interpretation,
                "theoretical_note": theoretical_note,
            }
        )

    interpretation_df = pd.DataFrame(interpretations)

    out = out.merge(
        interpretation_df,
        on="fold",
        how="left",
    )

    return out


# ============================================================
# MATHEMATICAL CONSOLIDATION DOCUMENT
# ============================================================

def build_consolidation_document(
    hierarchy: pd.DataFrame,
    operator_taxonomy: pd.DataFrame,
    claims: pd.DataFrame,
    falsification: pd.DataFrame,
    fold_interpretation: pd.DataFrame,
    model_comparison: pd.DataFrame,
) -> str:
    """
    Build formal mathematical theory consolidation document.
    """

    # Pull best available Script 229b metrics.
    stage = model_comparison.copy()

    if "analysis_stage" in stage.columns:
        stage_229b = stage[
            stage["analysis_stage"]
            == "229b_differential_substitution"
        ].copy()

        if stage_229b.empty:
            stage_229b = stage
    else:
        stage_229b = stage

    auc_global = safe_metric(
        stage_229b,
        "global_rf",
    )

    auc_coord = safe_metric(
        stage_229b,
        "raw_coordinate_rf",
    )

    auc_trend = safe_metric(
        stage_229b,
        "spatial_trend_rf",
    )

    auc_diff = safe_metric(
        stage_229b,
        "differential_operator_rf",
    )

    supported_claims = claims[
        claims["status"].isin(
            [
                "supported",
                "supported_provisionally",
            ]
        )
    ]

    hypotheses = claims[
        claims["claim_class"].isin(
            [
                "theoretical_hypothesis",
                "unresolved_hypothesis",
            ]
        )
    ]

    document = f"""# RDE / SRS Theory Integration and Mathematical Consolidation v01

Generated by `{SCRIPT}`.

## 1. Purpose

This document integrates the Oregon Spatial Recognition Structure (SRS)
operator-persistence branch into the broader Recognition Disequilibrium
Evolution (RDE) framework.

This is a theory-consolidation checkpoint.

It does not introduce a new predictive model.

Its purpose is to distinguish clearly between:

1. empirical findings;
2. provisional mathematical interpretations;
3. theoretical hypotheses;
4. unresolved transfer problems;
5. future falsification tests.

---

## 2. Current empirical foundation

The Oregon SRS branch supports the following broad result:

> Persistent relational versus pointwise recognition regimes are
> prospectively structured, nonlinear, multiscale, and spatially
> nonstationary.

The corrected final comparison gives approximately:

| Representation | AUC |
|---|---:|
| global nonlinear model | {fmt(auc_global)} |
| raw coordinate conditioning | {fmt(auc_coord)} |
| broad spatial trend | {fmt(auc_trend)} |
| tested local differential family | {fmt(auc_diff)} |

The interpretation is not:

$$
S_i = T_s.
$$

The supported provisional statement is narrower:

$$
T_s
\\in
\\arg\\max_{{Z \\in \\mathcal{{Z}}_{{\\mathrm{{tested}}}}}}
\\operatorname{{TransferSkill}}
\\left[
\\mathcal{{F}}
\\left(
Y_i,
\\mathcal{{C}}_r(P)_i,
\\Delta\\mathcal{{C}}(P)_i,
Z_i
\\right)
\\right].
$$

Thus, broad spatial trend is the best compact spatial representation
among those tested so far.

It remains a proxy.

It is not established as a causal geographic mechanism.

---

## 3. Integrated RDE-SRS hierarchy

The proposed theoretical hierarchy is:

$$
P_i
\\rightarrow
\\mathcal{{C}}_r(P)_i
\\rightarrow
\\Delta\\mathcal{{C}}_i
\\rightarrow
\\mathbb{{E}}[R_i \\mid \\mathcal{{C}}(P)]
\\rightarrow
D_i
\\rightarrow
G_i
\\rightarrow
\\Pi_i.
$$

The hierarchy components are:

{markdown_table(hierarchy)}

The key conceptual distinction is that:

- RDE asks whether recognition differs from geographically expected recognition;
- SRS asks how recognition is spatially organized;
- operator persistence asks whether that spatial organization is stable.

These are related but not identical questions.

---

## 4. Baseline pointwise disequilibrium

The current pointwise form of recognition disequilibrium can be written as:

$$
D_i^{{(g)}}
=
\\mathbb{{E}}
\\left[
R_i
\\mid
\\mathcal{{C}}(P)_i
\\right]
-
R_i.
$$

This form is appropriate when recognition is treated as a local state variable.

The SRS findings motivate the question of whether this residual is always
the correct mathematical object.

---

## 5. Operator-dependent disequilibrium hypothesis

Suppose recognition is organized through a spatial operator
$\\mathcal{{L}}$.

Then define:

$$
D_i^{{(\\mathcal{{L}})}}
=
\\mathbb{{E}}
\\left[
(\\mathcal{{L}}R)_i
\\mid
\\mathcal{{C}}(P)_i
\\right]
-
(\\mathcal{{L}}R)_i.
$$

This produces a family of disequilibrium fields:

$$
\\mathfrak{{D}}
=
\\left\\{{
D^{{(I)}},
D^{{(A_r)}},
D^{{(L)}},
D^{{(T)}},
D^{{(S)}}
\\right\\}}.
$$

where candidate operators may represent:

- pointwise recognition;
- neighborhood aggregation;
- relational divergence;
- transmission or flow;
- comparative shadowing.

This leads to the central theoretical hypothesis:

> Recognition disequilibrium may be operator-dependent.

This hypothesis is motivated by the SRS results.

It is not yet empirically validated.

---

## 6. Recognition operator regime

Let:

$$
G_i \\in \\mathcal{{G}}
$$

denote the recognition operator regime at location $i$.

In the simplest discrete formulation:

$$
\\mathcal{{G}}
=
\\left\\{{
g, r
\\right\\}},
$$

where:

- $g$ denotes pointwise or geographically local recognition;
- $r$ denotes relational recognition.

A more general formulation allows a mixture:

$$
\\mathcal{{L}}_i
=
\\sum_{{k=1}}^K
\\alpha_{{ik}}
\\mathcal{{L}}_k,
$$

subject to:

$$
\\alpha_{{ik}} \\geq 0,
\\qquad
\\sum_{{k=1}}^K
\\alpha_{{ik}} = 1.
$$

The binary formulation is empirically useful.

The mixture formulation is a future theoretical extension.

---

## 7. Operator persistence

Let operator persistence be:

$$
\\Pi_i
=
\\Pr
\\left(
G_i^{{(1)}}
=
G_i^{{(2)}}
=
\\cdots
=
G_i^{{(m)}}
\\right),
$$

under a defined family of admissible perturbations, scales, or state
realizations.

The Oregon branch currently supports a prospective persistence model of
the approximate form:

$$
\\Pr(G_i = R)
=
\\mathcal{{F}}
\\left(
Y_i,
\\mathcal{{C}}_r(P)_i,
\\Delta_{{r_a,r_b}}
\\mathcal{{C}}(P)_i,
T_s
\\right),
$$

with nonlinear $\\mathcal{{F}}$.

The empirical findings imply:

$$
\\mathcal{{F}}_{{\\mathrm{{nonlinear}}}}
>
\\mathcal{{F}}_{{\\mathrm{{pairwise}}}}
>
\\mathcal{{F}}_{{\\mathrm{{additive}}}}
$$

for the tested Oregon branch.

The tested universal scalar mismatch law is not supported.

---

## 8. Operator taxonomy

The initial mathematical operator taxonomy is:

{markdown_table(operator_taxonomy)}

This taxonomy separates:

1. operators already represented empirically;
2. operators partially represented through earlier mechanism work;
3. mathematical candidates that remain untested.

The taxonomy should guide targeted testing.

It should not become a justification for testing every possible spatial operator.

---

## 9. Structural transfer boundaries

The Oregon fold evidence suggests that ordinary spatial nonstationarity
and structural transfer failure may be distinct phenomena.

Define a candidate transfer-boundary indicator:

$$
B_i
=
\\mathbf{{1}}
\\left[
x_i \\notin
\\operatorname{{supp}}
\\left(
\\mathcal{{X}}_{{\\mathrm{{train}}}}
\\right)
\\right],
$$

or more generally a continuous support-distance functional:

$$
b_i
=
d
\\left(
x_i,
\\operatorname{{supp}}
\\left(
\\mathcal{{X}}_{{\\mathrm{{train}}}}
\\right)
\\right).
$$

Fold 0 motivates this hypothesis because its failure is not repaired by:

- raw coordinates;
- broad trend;
- multiscale contrast;
- tested differential operators.

This does not prove a structural boundary.

It defines a falsifiable next-stage question.

---

## 10. Fold-level structural interpretation

{markdown_table(fold_interpretation)}

The fold structure suggests at least three candidate categories:

### A. Broad spatial nonstationarity

The response surface changes gradually over geography and can be
partially represented by $T_s$.

### B. Representation-dependent regional variation

Different spatial representations help in different regions.

This is visible where context contrast outperforms trend or where
local operators show isolated fold-level gains.

### C. Structural transfer boundary

A region remains poorly predicted under all tested spatial
representations.

Fold 0 is the current candidate.

---

## 11. Supported theoretical claims

{markdown_table(supported_claims)}

---

## 12. Hypotheses not yet established

{markdown_table(hypotheses)}

The distinction between supported findings and hypotheses is essential.

The current project has evidence for structured operator persistence.

It does not yet have proof that operator-dependent disequilibrium
generalizes across states or domains.

---

## 13. Formal RDE-SRS synthesis

The integrated conceptual model is:

$$
P
\\xrightarrow{{\\mathcal{{C}}}}
C
\\xrightarrow{{\\mathcal{{E}}}}
\\widehat{{R}}
$$

with recognition disequilibrium:

$$
D^{{(\\mathcal{{L}})}}
=
\\mathcal{{E}}_{{\\mathcal{{L}}}}(P,C)
-
\\mathcal{{L}}R.
$$

The operator regime is:

$$
G_i
=
\\Gamma
\\left(
Y_i,
C_i,
\\Delta C_i,
S_i
\\right),
$$

and persistence is:

$$
\\Pi_i
=
\\Psi
\\left(
G_i^{{(1)}},
\\ldots,
G_i^{{(m)}}
\\right).
$$

Current Oregon evidence suggests the practical approximation:

$$
S_i
\\leadsto
T_s
$$

for present modeling purposes, with the explicit warning that:

$$
T_s
\\neq
\\text{{identified causal mechanism}}.
$$

The full theoretical architecture is therefore:

$$
\\boxed{{
P_i
\\rightarrow
\\mathcal{{C}}_r(P)_i
\\rightarrow
\\widehat{{R}}_i
\\rightarrow
D_i^{{(\\mathcal{{L}})}}
\\rightarrow
G_i
\\rightarrow
\\Pi_i
}}
$$

This is the principal mathematical integration proposed by Script 231.

---

## 14. Falsification program

The next research stage should be governed by explicit falsification
tests rather than unrestricted feature expansion.

{markdown_table(falsification)}

---

## 15. Project decision

The Oregon SRS operator-persistence branch is now considered
methodologically consolidated.

The project should not continue with a sequence of generic Oregon
classification models.

The recommended progression is:

1. formal operator taxonomy;
2. cross-state falsification design;
3. California-Oregon harmonization;
4. prospective cross-state comparison;
5. only then, targeted operator-dependent disequilibrium tests.

The strongest current theoretical proposition is:

> Recognition disequilibrium may not be fully characterized by the
> magnitude of a pointwise recognition residual. Where recognition is
> spatially relational, the relevant disequilibrium object may depend on
> the spatial operator through which recognition is organized.

This proposition is mathematically motivated by the SRS branch.

It remains a falsifiable theory rather than a completed empirical result.
"""

    return document


# ============================================================
# QA REPORT
# ============================================================

def build_qa_report(
    hierarchy: pd.DataFrame,
    taxonomy: pd.DataFrame,
    claims: pd.DataFrame,
    falsification: pd.DataFrame,
    fold_interpretation: pd.DataFrame,
) -> str:
    """
    Build QA summary.
    """

    status_counts = (
        claims["status"]
        .value_counts(dropna=False)
        .rename_axis("status")
        .reset_index(name="n")
    )

    fold_counts = (
        fold_interpretation["fold_interpretation"]
        .value_counts(dropna=False)
        .rename_axis("fold_interpretation")
        .reset_index(name="n")
    )

    lines = [
        f"Script: {SCRIPT}",
        "",
        "PURPOSE",
        "Integrate frozen Oregon SRS findings into broader RDE mathematical theory.",
        "",
        "MODELING STATUS",
        "No new predictive model was fit.",
        "This is a theory integration and falsification-design checkpoint.",
        "",
        "THEORY HIERARCHY",
        f"Hierarchy levels: {len(hierarchy):,}",
        "",
        hierarchy[
            [
                "level",
                "symbol",
                "name",
                "empirical_status",
            ]
        ].to_string(index=False),
        "",
        "OPERATOR TAXONOMY",
        f"Operator families: {len(taxonomy):,}",
        "",
        taxonomy[
            [
                "operator_id",
                "operator_family",
                "tested_status",
                "current_support",
                "next_priority",
            ]
        ].to_string(index=False),
        "",
        "CLAIM STATUS COUNTS",
        status_counts.to_string(index=False),
        "",
        "FALSIFICATION TESTS",
        f"Tests defined: {len(falsification):,}",
        "",
        falsification[
            [
                "test_id",
                "priority",
                "hypothesis",
            ]
        ].sort_values("priority").to_string(index=False),
        "",
        "FOLD INTERPRETATION COUNTS",
        fold_counts.to_string(index=False),
        "",
        "CORE THEORETICAL RESULT",
        (
            "Recognition disequilibrium may require operator-dependent "
            "definition where recognition structure is relational."
        ),
        "",
        "CORE CAUTION",
        (
            "Broad spatial trend T_s is the best tested compact spatial proxy. "
            "It is not an identified causal mechanism and should not be equated "
            "with latent S_i."
        ),
        "",
        "PROJECT DECISION",
        (
            "Do not continue generic Oregon model expansion. Proceed through "
            "formal operator taxonomy and cross-state falsification."
        ),
        "",
        "OUTPUTS",
        str(OUT_HIERARCHY),
        str(OUT_OPERATOR_TAXONOMY),
        str(OUT_CLAIM_LEDGER),
        str(OUT_FALSIFICATION),
        str(OUT_FOLD_INTERPRETATION),
        str(OUT_CONSOLIDATION),
        str(OUT_QA),
    ]

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():

    log.info("Starting Script 231")

    # --------------------------------------------------------
    # Read frozen Script 230 outputs
    # --------------------------------------------------------

    evidence = read_csv_required(INPUT_EVIDENCE)
    model_comparison = read_csv_required(INPUT_MODEL_COMPARISON)
    fold = read_csv_required(INPUT_FOLD_SYNTHESIS)

    log.info(f"Evidence records: {len(evidence):,}")
    log.info(f"Model comparison records: {len(model_comparison):,}")
    log.info(f"Fold records: {len(fold):,}")

    # --------------------------------------------------------
    # Construct theory products
    # --------------------------------------------------------

    hierarchy = build_theory_hierarchy()

    operator_taxonomy = build_operator_taxonomy()

    claim_ledger = build_claim_ledger(
        evidence=evidence,
        model_comparison=model_comparison,
    )

    falsification = build_falsification_design()

    fold_interpretation = build_fold_interpretation(
        fold=fold,
    )

    # --------------------------------------------------------
    # Build mathematical consolidation
    # --------------------------------------------------------

    consolidation = build_consolidation_document(
        hierarchy=hierarchy,
        operator_taxonomy=operator_taxonomy,
        claims=claim_ledger,
        falsification=falsification,
        fold_interpretation=fold_interpretation,
        model_comparison=model_comparison,
    )

    qa = build_qa_report(
        hierarchy=hierarchy,
        taxonomy=operator_taxonomy,
        claims=claim_ledger,
        falsification=falsification,
        fold_interpretation=fold_interpretation,
    )

    # --------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------

    hierarchy.to_csv(
        OUT_HIERARCHY,
        index=False,
    )

    operator_taxonomy.to_csv(
        OUT_OPERATOR_TAXONOMY,
        index=False,
    )

    claim_ledger.to_csv(
        OUT_CLAIM_LEDGER,
        index=False,
    )

    falsification.to_csv(
        OUT_FALSIFICATION,
        index=False,
    )

    fold_interpretation.to_csv(
        OUT_FOLD_INTERPRETATION,
        index=False,
    )

    OUT_CONSOLIDATION.write_text(
        consolidation,
        encoding="utf-8",
    )

    OUT_QA.write_text(
        qa,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Completion log
    # --------------------------------------------------------

    log.info("Completed Script 231")
    log.info("")
    log.info("Theory hierarchy:")
    log.info(
        "P -> C_r(P) -> Delta C -> E[R|C(P)] -> D -> G -> Pi"
    )
    log.info("")
    log.info("Central theoretical hypothesis:")
    log.info(
        "Recognition disequilibrium may be operator-dependent."
    )
    log.info("")
    log.info("Important caution:")
    log.info(
        "T_s is the best tested compact spatial proxy, not an identified causal mechanism."
    )
    log.info("")
    log.info("Created:")

    for path in [
        OUT_HIERARCHY,
        OUT_OPERATOR_TAXONOMY,
        OUT_CLAIM_LEDGER,
        OUT_FALSIFICATION,
        OUT_FOLD_INTERPRETATION,
        OUT_CONSOLIDATION,
        OUT_QA,
    ]:
        log.info(str(path))


if __name__ == "__main__":
    main()