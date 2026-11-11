#!/usr/bin/env python3
"""
227_oregon_srs_operator_persistence_evidence_synthesis_v01.py

Formal evidence synthesis for the Oregon SRS operator-persistence branch.

Purpose
-------
Consolidate the empirical results from the prospective persistence,
interaction, nonstationarity, spatial-regime, response-surface, and
coordinate-substitution analyses into:

1. a machine-readable evidence ledger;
2. a model-performance comparison table;
3. a fold-level synthesis table;
4. a formal markdown theory checkpoint;
5. a QA report.

This script does NOT:
- fit a new predictive model;
- build an adaptive operator gate;
- claim a universal persistence law;
- claim that raw coordinates are causal;
- close the SRS research program.

The purpose is to freeze the current evidence before the next mathematical
development stage.
"""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


SCRIPT = "227_oregon_srs_operator_persistence_evidence_synthesis_v01"

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"


# ---------------------------------------------------------------------
# INPUTS
# ---------------------------------------------------------------------

INPUT_223_METRICS = (
    P / "oregon_srs_crossfit_interaction_nonstationarity_metrics_v01.csv"
)

INPUT_223_FOLD_METRICS = (
    P / "oregon_srs_crossfit_interaction_nonstationarity_fold_metrics_v01.csv"
)

INPUT_223_NONSTATIONARITY = (
    P / "oregon_srs_crossfit_nonstationarity_audit_v01.csv"
)

INPUT_224_METRICS = (
    P / "oregon_srs_spatial_regime_geometry_metrics_v01.csv"
)

INPUT_224_FOLD_METRICS = (
    P / "oregon_srs_spatial_regime_geometry_fold_metrics_v01.csv"
)

INPUT_225_METRICS = (
    P / "oregon_srs_spatially_varying_response_surface_metrics_v01.csv"
)

INPUT_225_FOLD_METRICS = (
    P / "oregon_srs_spatially_varying_response_surface_fold_metrics_v01.csv"
)

INPUT_225_IMPROVEMENT = (
    P / "oregon_srs_spatially_varying_response_surface_improvement_summary_v01.csv"
)

INPUT_226_METRICS = (
    P / "oregon_srs_coordinate_substitution_metrics_v01.csv"
)

INPUT_226_FOLD_METRICS = (
    P / "oregon_srs_coordinate_substitution_fold_metrics_v01.csv"
)


# ---------------------------------------------------------------------
# OUTPUTS
# ---------------------------------------------------------------------

OUT_EVIDENCE = (
    P / "oregon_srs_operator_persistence_evidence_ledger_v01.csv"
)

OUT_MODEL_COMPARISON = (
    P / "oregon_srs_operator_persistence_model_comparison_v01.csv"
)

OUT_FOLD_SYNTHESIS = (
    P / "oregon_srs_operator_persistence_fold_synthesis_v01.csv"
)

OUT_THEORY_NOTE = (
    P / "oregon_srs_operator_persistence_theory_checkpoint_v01.md"
)

OUT_QA = (
    P / "oregon_srs_operator_persistence_evidence_synthesis_qa_v01.txt"
)


logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT}] %(levelname)s: %(message)s",
)

log = logging.getLogger(SCRIPT)


# =====================================================================
# UTILITIES
# =====================================================================

def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input not found:\n{path}"
        )


def read_csv(path: Path) -> pd.DataFrame:
    require(path)
    return pd.read_csv(path)


def get_metric(
    df: pd.DataFrame,
    model: str,
    metric: str,
) -> float:

    row = df.loc[
        df["model"] == model,
        metric,
    ]

    if row.empty:
        return np.nan

    return float(row.iloc[0])


def fmt(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


def sign_reversal_summary(
    nonstationarity: pd.DataFrame,
) -> pd.DataFrame:

    if nonstationarity.empty:
        return pd.DataFrame()

    required = {
        "heldout_fold",
        "feature_group",
        "sign_reversal",
    }

    if not required.issubset(nonstationarity.columns):
        return pd.DataFrame()

    out = (
        nonstationarity
        .groupby(
            [
                "heldout_fold",
                "feature_group",
            ],
            as_index=False,
        )
        .agg(
            n_features=(
                "sign_reversal",
                "size",
            ),
            sign_reversals=(
                "sign_reversal",
                "sum",
            ),
        )
    )

    out["sign_reversal_share"] = (
        out["sign_reversals"]
        / out["n_features"]
    )

    return out


# =====================================================================
# MODEL COMPARISON
# =====================================================================

def build_model_comparison(
    m223: pd.DataFrame,
    m224: pd.DataFrame,
    m225: pd.DataFrame,
    m226: pd.DataFrame,
) -> pd.DataFrame:

    frames = []

    for stage, df in [
        ("223_crossfit_structure", m223),
        ("224_regime_geometry", m224),
        ("225_response_surface", m225),
        ("226_coordinate_substitution", m226),
    ]:

        temp = df.copy()
        temp.insert(0, "analysis_stage", stage)
        frames.append(temp)

    out = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    preferred = [
        "analysis_stage",
        "model",
        "n",
        "n_features",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "mean_fold_auc",
        "std_fold_auc",
        "min_fold_auc",
        "folds_auc_above_0_60",
    ]

    cols = [
        c for c in preferred
        if c in out.columns
    ]

    remaining = [
        c for c in out.columns
        if c not in cols
    ]

    return out[
        cols + remaining
    ].sort_values(
        [
            "analysis_stage",
            "roc_auc",
        ],
        ascending=[
            True,
            False,
        ],
    )


# =====================================================================
# FOLD SYNTHESIS
# =====================================================================

def build_fold_synthesis(
    f223: pd.DataFrame,
    f225: pd.DataFrame,
    f226: pd.DataFrame,
    improvement225: pd.DataFrame,
    reversal_summary: pd.DataFrame,
) -> pd.DataFrame:

    folds = sorted(
        set(
            pd.to_numeric(
                f225["fold"],
                errors="coerce",
            )
            .dropna()
            .tolist()
        )
    )

    rows = []

    for fold in folds:

        row = {
            "fold": fold,
        }

        # Script 223
        for model in [
            "additive_logistic",
            "crossfit_interaction_logistic",
            "compact_mismatch_logistic",
            "nonlinear_random_forest",
        ]:

            sub = f223[
                (f223["model"] == model)
                & (f223["fold"] == fold)
            ]

            if not sub.empty:
                row[
                    f"auc_223_{model}"
                ] = float(
                    sub["roc_auc"].iloc[0]
                )

        # Script 225
        for model in [
            "global_rf",
            "coordinate_rf",
        ]:

            sub = f225[
                (f225["model"] == model)
                & (f225["fold"] == fold)
            ]

            if not sub.empty:
                row[
                    f"auc_225_{model}"
                ] = float(
                    sub["roc_auc"].iloc[0]
                )

        # Script 226
        for model in [
            "global_rf",
            "raw_coordinate_rf",
            "spatial_trend_rf",
            "context_contrast_rf",
            "contrast_plus_trend_rf",
            "contrast_plus_raw_coord_rf",
        ]:

            sub = f226[
                (f226["model"] == model)
                & (f226["fold"] == fold)
            ]

            if not sub.empty:
                row[
                    f"auc_226_{model}"
                ] = float(
                    sub["roc_auc"].iloc[0]
                )

        # Script 225 cell-level improvement
        imp = improvement225[
            pd.to_numeric(
                improvement225["fold"],
                errors="coerce",
            )
            == fold
        ]

        if not imp.empty:

            for c in [
                "mean_error_improvement",
                "median_error_improvement",
                "coordinate_better_share",
                "coordinate_only_correct_share",
                "global_only_correct_share",
            ]:

                if c in imp.columns:
                    row[
                        f"script225_{c}"
                    ] = float(
                        imp[c].iloc[0]
                    )

        # Sign reversals
        if not reversal_summary.empty:

            for group in [
                "recognition_state",
                "relational_context",
            ]:

                sub = reversal_summary[
                    (reversal_summary["heldout_fold"] == fold)
                    & (
                        reversal_summary["feature_group"]
                        == group
                    )
                ]

                if not sub.empty:

                    row[
                        f"{group}_sign_reversals"
                    ] = int(
                        sub["sign_reversals"].iloc[0]
                    )

                    row[
                        f"{group}_sign_reversal_share"
                    ] = float(
                        sub["sign_reversal_share"].iloc[0]
                    )

        rows.append(row)

    out = pd.DataFrame(rows)

    if {
        "auc_225_coordinate_rf",
        "auc_225_global_rf",
    }.issubset(out.columns):

        out[
            "coordinate_auc_gain_v225"
        ] = (
            out["auc_225_coordinate_rf"]
            - out["auc_225_global_rf"]
        )

    if {
        "auc_226_contrast_plus_raw_coord_rf",
        "auc_226_global_rf",
    }.issubset(out.columns):

        out[
            "best_substitution_auc_gain_v226"
        ] = (
            out[
                "auc_226_contrast_plus_raw_coord_rf"
            ]
            - out[
                "auc_226_global_rf"
            ]
        )

    return out


# =====================================================================
# EVIDENCE LEDGER
# =====================================================================

def build_evidence_ledger(
    m223: pd.DataFrame,
    m224: pd.DataFrame,
    m225: pd.DataFrame,
    m226: pd.DataFrame,
    fold_synthesis: pd.DataFrame,
) -> pd.DataFrame:

    rows: List[Dict] = []

    def add(
        evidence_id: str,
        question: str,
        status: str,
        finding: str,
        quantitative_basis: str,
        implication: str,
        caution: str,
    ) -> None:

        rows.append(
            {
                "evidence_id": evidence_id,
                "question": question,
                "status": status,
                "finding": finding,
                "quantitative_basis": quantitative_basis,
                "implication": implication,
                "caution": caution,
            }
        )

    # --------------------------------------------------------------
    # E01: nonlinear structure
    # --------------------------------------------------------------

    rf223 = get_metric(
        m223,
        "nonlinear_random_forest",
        "roc_auc",
    )

    additive223 = get_metric(
        m223,
        "additive_logistic",
        "roc_auc",
    )

    interaction223 = get_metric(
        m223,
        "crossfit_interaction_logistic",
        "roc_auc",
    )

    mismatch223 = get_metric(
        m223,
        "compact_mismatch_logistic",
        "roc_auc",
    )

    add(
        evidence_id="E01",
        question=(
            "Is persistence-regime structure adequately represented "
            "by a simple additive response?"
        ),
        status="rejected_simple_form",
        finding=(
            "The nonlinear model materially outperforms the additive model."
        ),
        quantitative_basis=(
            f"RF AUC={fmt(rf223)}; "
            f"additive AUC={fmt(additive223)}."
        ),
        implication=(
            "Operator persistence contains nonlinear recognition-context structure."
        ),
        caution=(
            "Random-forest performance does not itself identify the mathematical mechanism."
        ),
    )

    # --------------------------------------------------------------
    # E02: pairwise interactions
    # --------------------------------------------------------------

    add(
        evidence_id="E02",
        question=(
            "Do cross-fitted pairwise recognition-context interactions "
            "recover the nonlinear persistence signal?"
        ),
        status="partial_support",
        finding=(
            "Cross-fitted pairwise interactions improve on the additive model "
            "but remain substantially below the nonlinear benchmark."
        ),
        quantitative_basis=(
            f"interaction AUC={fmt(interaction223)}; "
            f"additive AUC={fmt(additive223)}; "
            f"RF AUC={fmt(rf223)}."
        ),
        implication=(
            "Interaction structure matters, but low-order bilinear structure is incomplete."
        ),
        caution=(
            "Stable pair selection should not be interpreted as a universal causal law."
        ),
    )

    # --------------------------------------------------------------
    # E03: compact mismatch
    # --------------------------------------------------------------

    add(
        evidence_id="E03",
        question=(
            "Can persistence be compressed into a universal scalar "
            "recognition-context mismatch functional?"
        ),
        status="not_supported",
        finding=(
            "The compact mismatch model performs poorly relative to the nonlinear model."
        ),
        quantitative_basis=(
            f"compact mismatch AUC={fmt(mismatch223)}; "
            f"RF AUC={fmt(rf223)}."
        ),
        implication=(
            "A universal scalar mismatch law is not supported by the current Oregon evidence."
        ),
        caution=(
            "More specialized mismatch operators could still be useful in restricted settings."
        ),
    )

    # --------------------------------------------------------------
    # E04: discrete regimes vs spatial drift
    # --------------------------------------------------------------

    coord224 = get_metric(
        m224,
        "coordinate_rf",
        "roc_auc",
    )

    regime_models = m224[
        m224["model"].astype(str).str.match(
            r"regime[0-9]+_rf"
        )
    ]

    best_regime_auc = (
        float(regime_models["roc_auc"].max())
        if not regime_models.empty
        else np.nan
    )

    add(
        evidence_id="E04",
        question=(
            "Is nonstationarity better represented by simple discrete "
            "unsupervised state/context regimes?"
        ),
        status="weak_support_for_discrete_regimes",
        finding=(
            "Coordinate-conditioned RF outperforms the tested KMeans regime-conditioned RF models."
        ),
        quantitative_basis=(
            f"coordinate RF AUC={fmt(coord224)}; "
            f"best discrete-regime RF AUC={fmt(best_regime_auc)}."
        ),
        implication=(
            "Continuous or more complex spatial variation is more plausible "
            "than the tested simple discrete regime partition."
        ),
        caution=(
            "This rejects only the tested regime construction, not every possible regime model."
        ),
    )

    # --------------------------------------------------------------
    # E05: coordinate gain
    # --------------------------------------------------------------

    coord225 = get_metric(
        m225,
        "coordinate_rf",
        "roc_auc",
    )

    global225 = get_metric(
        m225,
        "global_rf",
        "roc_auc",
    )

    add(
        evidence_id="E05",
        question=(
            "Does spatial position contain prospective information beyond "
            "recognition state and multiscale context?"
        ),
        status="supported",
        finding=(
            "Coordinate conditioning improves prospective classification."
        ),
        quantitative_basis=(
            f"coordinate RF AUC={fmt(coord225)}; "
            f"global RF AUC={fmt(global225)}; "
            f"gain={fmt(coord225 - global225)}."
        ),
        implication=(
            "The persistence response surface is spatially nonstationary."
        ),
        caution=(
            "Raw coordinates are diagnostic proxies and should not be interpreted causally."
        ),
    )

    # --------------------------------------------------------------
    # E06: spatial trend substitution
    # --------------------------------------------------------------

    raw226 = get_metric(
        m226,
        "raw_coordinate_rf",
        "roc_auc",
    )

    trend226 = get_metric(
        m226,
        "spatial_trend_rf",
        "roc_auc",
    )

    add(
        evidence_id="E06",
        question=(
            "Can broad spatial trend approximate the raw-coordinate gain?"
        ),
        status="substantial_support",
        finding=(
            "A low-order spatial trend basis approaches raw-coordinate performance."
        ),
        quantitative_basis=(
            f"trend AUC={fmt(trend226)}; "
            f"raw-coordinate AUC={fmt(raw226)}."
        ),
        implication=(
            "A substantial part of spatial nonstationarity behaves as broad spatial drift."
        ),
        caution=(
            "The trend basis is descriptive and does not identify the physical process generating drift."
        ),
    )

    # --------------------------------------------------------------
    # E07: context contrasts
    # --------------------------------------------------------------

    contrast226 = get_metric(
        m226,
        "context_contrast_rf",
        "roc_auc",
    )

    global226 = get_metric(
        m226,
        "global_rf",
        "roc_auc",
    )

    add(
        evidence_id="E07",
        question=(
            "Do local-to-regional physical context contrasts contain "
            "operator-persistence information?"
        ),
        status="supported_but_incomplete",
        finding=(
            "Multiscale context contrasts improve the global model, "
            "but do not fully recover the coordinate-conditioned result."
        ),
        quantitative_basis=(
            f"context-contrast AUC={fmt(contrast226)}; "
            f"global AUC={fmt(global226)}; "
            f"raw-coordinate AUC={fmt(raw226)}."
        ),
        implication=(
            "Scale contrast is part of the persistence geometry but is not the whole mechanism."
        ),
        caution=(
            "Contrast construction currently depends on the available 3/10/25 km feature family."
        ),
    )

    # --------------------------------------------------------------
    # E08: combined geometry
    # --------------------------------------------------------------

    combo_trend226 = get_metric(
        m226,
        "contrast_plus_trend_rf",
        "roc_auc",
    )

    combo_raw226 = get_metric(
        m226,
        "contrast_plus_raw_coord_rf",
        "roc_auc",
    )

    add(
        evidence_id="E08",
        question=(
            "Does combining multiscale contrast with spatial variation "
            "produce the strongest current persistence model?"
        ),
        status="supported",
        finding=(
            "The strongest overall model combines multiscale contrasts "
            "with raw spatial position."
        ),
        quantitative_basis=(
            f"contrast+raw-coordinate AUC={fmt(combo_raw226)}; "
            f"contrast+trend AUC={fmt(combo_trend226)}; "
            f"global AUC={fmt(global226)}."
        ),
        implication=(
            "Current evidence supports a multiscale, nonlinear, spatially varying "
            "persistence geometry."
        ),
        caution=(
            "Overall AUC masks fold-level transfer failure, especially fold 0."
        ),
    )

    # --------------------------------------------------------------
    # E09: fold 5
    # --------------------------------------------------------------

    if not fold_synthesis.empty and 5.0 in fold_synthesis["fold"].values:

        f5 = fold_synthesis[
            fold_synthesis["fold"] == 5.0
        ].iloc[0]

        gain5 = f5.get(
            "coordinate_auc_gain_v225",
            np.nan,
        )

        better5 = f5.get(
            "script225_coordinate_better_share",
            np.nan,
        )

        add(
            evidence_id="E09",
            question=(
                "Is the fold-5 relationship reversal compatible with "
                "recoverable spatial nonstationarity?"
            ),
            status="supported",
            finding=(
                "Fold 5 shows strong recovery under spatial conditioning."
            ),
            quantitative_basis=(
                f"coordinate AUC gain={fmt(gain5)}; "
                f"coordinate-better cell share={fmt(better5)}."
            ),
            implication=(
                "At least one major relationship reversal is geographically structured "
                "rather than simple random instability."
            ),
            caution=(
                "The spatial process responsible for the recovery is not yet causally identified."
            ),
        )

    # --------------------------------------------------------------
    # E10: fold 0
    # --------------------------------------------------------------

    if not fold_synthesis.empty and 0.0 in fold_synthesis["fold"].values:

        f0 = fold_synthesis[
            fold_synthesis["fold"] == 0.0
        ].iloc[0]

        gain0 = f0.get(
            "coordinate_auc_gain_v225",
            np.nan,
        )

        best0 = f0.get(
            "auc_226_contrast_plus_raw_coord_rf",
            np.nan,
        )

        add(
            evidence_id="E10",
            question=(
                "Is fold 0 resolved by coordinate conditioning or "
                "multiscale contrast substitution?"
            ),
            status="unresolved_transfer_boundary",
            finding=(
                "Fold 0 remains a major spatial transfer problem."
            ),
            quantitative_basis=(
                f"coordinate AUC gain={fmt(gain0)}; "
                f"contrast+coordinate AUC={fmt(best0)}."
            ),
            implication=(
                "Fold 0 may represent unsupported state space, a structural transfer boundary, "
                "or an omitted process not captured by current SRS state/context variables."
            ),
            caution=(
                "The current evidence does not distinguish among these explanations."
            ),
        )

    return pd.DataFrame(rows)


# =====================================================================
# THEORY NOTE
# =====================================================================

def build_theory_note(
    evidence: pd.DataFrame,
    model_comparison: pd.DataFrame,
    fold_synthesis: pd.DataFrame,
) -> str:

    lines: List[str] = []

    lines.extend([
        "# Oregon SRS Operator-Persistence Theory Checkpoint v01",
        "",
        f"Generated by `{SCRIPT}`.",
        "",
        "## 1. Purpose",
        "",
        (
            "This document freezes the current empirical evidence from the Oregon "
            "SRS operator-persistence branch. It is a theory checkpoint, not a claim "
            "that the SRS framework is complete."
        ),
        "",
        (
            "The branch asks whether the persistence of relational versus pointwise "
            "spatial transformation can be predicted prospectively from admissible "
            "recognition state, multiscale physical context, and spatial structure."
        ),
        "",
        "## 2. Current empirical conclusion",
        "",
        (
            "The current evidence supports a nonlinear, multiscale, and spatially "
            "nonstationary operator-persistence geometry."
        ),
        "",
        "A compact summary is:",
        "",
        "$$",
        r"\Pr(G_i = R)",
        r"=",
        r"\mathcal{F}\left(",
        r"Y_i,",
        r"\mathcal{C}_{r}(P)_i,",
        r"\Delta_{r_a,r_b}\mathcal{C}(P)_i,",
        r"S_i",
        r"\right),",
        "$$",
        "",
        "where:",
        "",
        "- $G_i$ is the persistent operator regime;",
        "- $Y_i$ is admissible recognition state;",
        "- $\\mathcal{C}_{r}(P)_i$ is physical context evaluated across spatial scale $r$;",
        "- $\\Delta_{r_a,r_b}\\mathcal{C}(P)_i$ is local-to-regional context contrast;",
        "- $S_i$ is unresolved spatially structured variation;",
        "- $\\mathcal{F}$ is currently supported as nonlinear.",
        "",
        (
            "The evidence does not support replacing this structure with a universal "
            "scalar mismatch law or a single additive global response."
        ),
        "",
        "## 3. Supported findings",
        "",
    ])

    supported_statuses = {
        "supported",
        "substantial_support",
        "supported_but_incomplete",
        "partial_support",
    }

    for _, row in evidence.iterrows():
        if row["status"] in supported_statuses:
            lines.extend([
                f"### {row['evidence_id']}: {row['question']}",
                "",
                f"**Finding:** {row['finding']}",
                "",
                f"**Quantitative basis:** {row['quantitative_basis']}",
                "",
                f"**Implication:** {row['implication']}",
                "",
                f"**Caution:** {row['caution']}",
                "",
            ])

    lines.extend([
        "## 4. Rejected or unsupported simple hypotheses",
        "",
    ])

    rejected_statuses = {
        "rejected_simple_form",
        "not_supported",
        "weak_support_for_discrete_regimes",
    }

    for _, row in evidence.iterrows():
        if row["status"] in rejected_statuses:
            lines.extend([
                f"### {row['evidence_id']}: {row['question']}",
                "",
                f"**Finding:** {row['finding']}",
                "",
                f"**Quantitative basis:** {row['quantitative_basis']}",
                "",
                f"**Implication:** {row['implication']}",
                "",
            ])

    lines.extend([
        "## 5. Unresolved structure",
        "",
    ])

    unresolved = evidence[
        evidence["status"].astype(str).str.contains(
            "unresolved",
            case=False,
            na=False,
        )
    ]

    if unresolved.empty:
        lines.append(
            "No unresolved evidence rows were explicitly classified."
        )
        lines.append("")
    else:
        for _, row in unresolved.iterrows():
            lines.extend([
                f"### {row['evidence_id']}: {row['question']}",
                "",
                f"**Finding:** {row['finding']}",
                "",
                f"**Quantitative basis:** {row['quantitative_basis']}",
                "",
                f"**Implication:** {row['implication']}",
                "",
                f"**Caution:** {row['caution']}",
                "",
            ])

    lines.extend([
        "## 6. Current mathematical interpretation",
        "",
        (
            "The present evidence suggests that operator persistence should not be "
            "modeled as a fixed function of physical quality or recognition intensity alone."
        ),
        "",
        "A more plausible current structure is:",
        "",
        "$$",
        r"G_i",
        r"\sim",
        r"\mathcal{F}\left(",
        r"Y_i,",
        r"\mathcal{C}_{10}(P)_i,",
        r"\mathcal{C}_{25}(P)_i,",
        r"\mathcal{C}_{10}(P)_i-\mathcal{C}_{25}(P)_i,",
        r"S_i",
        r"\right).",
        "$$",
        "",
        (
            "This representation treats scale contrast as mathematically meaningful "
            "while retaining an unresolved spatial term rather than falsely interpreting "
            "raw coordinates as causal."
        ),
        "",
        "The key theoretical distinction is:",
        "",
        "$$",
        r"\text{location} \neq \text{mechanism},",
        "$$",
        "",
        "but empirically:",
        "",
        "$$",
        r"\text{location} \rightarrow \text{information about omitted spatial structure}.",
        "$$",
        "",
        (
            "The next mathematical development stage should therefore seek operators "
            "that explain or replace $S_i$, rather than merely adding more generic classifiers."
        ),
        "",
        "## 7. Fold-level interpretation",
        "",
        (
            "The folds should not be treated as interchangeable failures or successes. "
            "The evidence indicates heterogeneous transfer behavior."
        ),
        "",
    ])

    if not fold_synthesis.empty:

        display_cols = [
            c for c in [
                "fold",
                "auc_225_global_rf",
                "auc_225_coordinate_rf",
                "coordinate_auc_gain_v225",
                "auc_226_context_contrast_rf",
                "auc_226_contrast_plus_trend_rf",
                "auc_226_contrast_plus_raw_coord_rf",
            ]
            if c in fold_synthesis.columns
        ]

        lines.append(
            fold_synthesis[
                display_cols
            ].to_markdown(
                index=False
            )
        )

        lines.append("")

    lines.extend([
        (
            "Fold 5 is currently consistent with recoverable spatial nonstationarity: "
            "spatial conditioning substantially improves performance."
        ),
        "",
        (
            "Fold 0 is qualitatively different. Current coordinate and contrast formulations "
            "do not resolve the transfer failure. It should be treated as a possible structural "
            "transfer boundary or unsupported state-space problem until tested directly."
        ),
        "",
        "## 8. What the evidence does not justify",
        "",
        "The current branch does not justify the following claims:",
        "",
        "1. that one universal scalar mismatch equation selects the correct operator;",
        "2. that raw coordinates are causal drivers of operator persistence;",
        "3. that simple KMeans state/context clusters define true spatial response regimes;",
        "4. that the current classifier is ready to serve as a universal adaptive SRS gate;",
        "5. that fold 0 has been explained;",
        "6. that Oregon results automatically generalize to California or other domains.",
        "",
        "## 9. Recommended next mathematical direction",
        "",
        (
            "The next stage should investigate whether the unresolved spatial term $S_i$ "
            "can be decomposed into mathematically meaningful spatial operators."
        ),
        "",
        "Priority directions are:",
        "",
        "1. local directional gradients rather than only radial summaries;",
        "2. anisotropic neighborhood structure;",
        "3. local spatial derivatives of recognition and physical context;",
        "4. boundary and transition-zone operators;",
        "5. state-space support and extrapolation diagnostics for fold 0;",
        "6. geographically varying response derivatives rather than raw-coordinate inclusion.",
        "",
        (
            "The strongest immediate research question is therefore not "
            "\"which classifier performs best?\" but:"
        ),
        "",
        "> Can the unresolved spatial term in operator persistence be replaced by a compact set of interpretable spatial differential or transition operators?",
        "",
        "## 10. Current theory status",
        "",
        (
            "The SRS operator-persistence branch has passed the stage of simple existence testing. "
            "There is prospective nonlinear signal, meaningful multiscale structure, and spatial "
            "nonstationarity. The next challenge is explanatory compression: replacing descriptive "
            "spatial position with interpretable spatial mathematics."
        ),
        "",
    ])

    return "\n".join(lines)


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    log.info("Starting Script 227")

    # --------------------------------------------------------------
    # Load
    # --------------------------------------------------------------

    m223 = read_csv(
        INPUT_223_METRICS
    )

    f223 = read_csv(
        INPUT_223_FOLD_METRICS
    )

    n223 = read_csv(
        INPUT_223_NONSTATIONARITY
    )

    m224 = read_csv(
        INPUT_224_METRICS
    )

    f224 = read_csv(
        INPUT_224_FOLD_METRICS
    )

    m225 = read_csv(
        INPUT_225_METRICS
    )

    f225 = read_csv(
        INPUT_225_FOLD_METRICS
    )

    imp225 = read_csv(
        INPUT_225_IMPROVEMENT
    )

    m226 = read_csv(
        INPUT_226_METRICS
    )

    f226 = read_csv(
        INPUT_226_FOLD_METRICS
    )

    log.info("Loaded all synthesis inputs")

    # --------------------------------------------------------------
    # Sign reversal summary
    # --------------------------------------------------------------

    reversal = sign_reversal_summary(
        n223
    )

    # --------------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------------

    model_comparison = build_model_comparison(
        m223=m223,
        m224=m224,
        m225=m225,
        m226=m226,
    )

    # --------------------------------------------------------------
    # Fold synthesis
    # --------------------------------------------------------------

    fold_synthesis = build_fold_synthesis(
        f223=f223,
        f225=f225,
        f226=f226,
        improvement225=imp225,
        reversal_summary=reversal,
    )

    # --------------------------------------------------------------
    # Evidence ledger
    # --------------------------------------------------------------

    evidence = build_evidence_ledger(
        m223=m223,
        m224=m224,
        m225=m225,
        m226=m226,
        fold_synthesis=fold_synthesis,
    )

    # --------------------------------------------------------------
    # Theory checkpoint
    # --------------------------------------------------------------

    theory_note = build_theory_note(
        evidence=evidence,
        model_comparison=model_comparison,
        fold_synthesis=fold_synthesis,
    )

    # --------------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------------

    evidence.to_csv(
        OUT_EVIDENCE,
        index=False,
    )

    model_comparison.to_csv(
        OUT_MODEL_COMPARISON,
        index=False,
    )

    fold_synthesis.to_csv(
        OUT_FOLD_SYNTHESIS,
        index=False,
    )

    OUT_THEORY_NOTE.write_text(
        theory_note,
        encoding="utf-8",
    )

    # --------------------------------------------------------------
    # QA
    # --------------------------------------------------------------

    qa_lines = [
        f"Script: {SCRIPT}",
        "",
        "PURPOSE",
        (
            "Freeze the current empirical evidence from the Oregon SRS "
            "operator-persistence branch before the next mathematical stage."
        ),
        "",
        "OUTPUT COUNTS",
        f"Evidence ledger rows: {len(evidence)}",
        f"Model comparison rows: {len(model_comparison)}",
        f"Fold synthesis rows: {len(fold_synthesis)}",
        "",
        "EVIDENCE LEDGER",
        evidence.to_string(index=False),
        "",
        "MODEL COMPARISON",
        model_comparison.to_string(index=False),
        "",
        "FOLD SYNTHESIS",
        fold_synthesis.to_string(index=False),
        "",
        "CURRENT SYNTHESIS",
        (
            "1. Persistence-regime structure is prospectively predictable."
        ),
        (
            "2. Nonlinear models outperform additive and compact mismatch formulations."
        ),
        (
            "3. Pairwise interactions contain signal but do not recover the full nonlinear structure."
        ),
        (
            "4. Spatial nonstationarity is supported."
        ),
        (
            "5. Simple discrete KMeans regimes do not explain the spatial variation as well as spatial conditioning."
        ),
        (
            "6. Broad spatial trend recovers much of the raw-coordinate gain."
        ),
        (
            "7. Multiscale context contrasts contain additional persistence information."
        ),
        (
            "8. Contrast plus spatial structure is the strongest current model family."
        ),
        (
            "9. Fold 5 appears consistent with recoverable spatial nonstationarity."
        ),
        (
            "10. Fold 0 remains an unresolved transfer boundary."
        ),
        "",
        "NEXT MATHEMATICAL QUESTION",
        (
            "Can unresolved spatial variation be replaced by interpretable "
            "directional, differential, anisotropic, or boundary-transition operators?"
        ),
    ]

    OUT_QA.write_text(
        "\n".join(qa_lines),
        encoding="utf-8",
    )

    log.info("Completed Script 227")

    log.info("Created:")
    for path in [
        OUT_EVIDENCE,
        OUT_MODEL_COMPARISON,
        OUT_FOLD_SYNTHESIS,
        OUT_THEORY_NOTE,
        OUT_QA,
    ]:
        log.info(str(path))

    log.info("")
    log.info("Evidence status counts:")
    log.info(
        "\n"
        + evidence[
            "status"
        ]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()