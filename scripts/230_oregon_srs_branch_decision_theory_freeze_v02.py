#!/usr/bin/env python3
"""
230_oregon_srs_branch_decision_theory_freeze_v02.py

Final decision checkpoint for Oregon SRS operator-persistence branch v02.

Incorporates Script 229b:
- recognition restored
- differential operators tested properly
- spatial trend remains strongest interpretable substitute
- local differential operators are not currently supported as the main S_i replacement

No new model is fit.
"""

from pathlib import Path
import logging
import pandas as pd
import numpy as np

SCRIPT = "230_oregon_srs_branch_decision_theory_freeze_v02"

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"

INPUT_227_EVIDENCE = P / "oregon_srs_operator_persistence_evidence_ledger_v01.csv"
INPUT_227_MODEL_COMPARISON = P / "oregon_srs_operator_persistence_model_comparison_v01.csv"
INPUT_227_FOLD_SYNTHESIS = P / "oregon_srs_operator_persistence_fold_synthesis_v01.csv"

INPUT_229B_METRICS = P / "oregon_srs_differential_operator_substitution_metrics_v02.csv"
INPUT_229B_FOLD_METRICS = P / "oregon_srs_differential_operator_substitution_fold_metrics_v02.csv"

OUT_EVIDENCE = P / "oregon_srs_branch_decision_evidence_ledger_v02.csv"
OUT_MODEL_COMPARISON = P / "oregon_srs_branch_decision_model_comparison_v02.csv"
OUT_FOLD_SYNTHESIS = P / "oregon_srs_branch_decision_fold_synthesis_v02.csv"
OUT_THEORY_FREEZE = P / "oregon_srs_branch_decision_theory_freeze_v02.md"
OUT_QA = P / "oregon_srs_branch_decision_theory_freeze_qa_v02.txt"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT)


def read(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def metric(df, model, col):
    s = df.loc[df["model"] == model, col]
    if s.empty:
        return np.nan
    return float(s.iloc[0])


def fmt(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.3f}"


def add_evidence_row(rows, evidence_id, question, status, finding, quantitative_basis, implication, caution):
    rows.append({
        "evidence_id": evidence_id,
        "question": question,
        "status": status,
        "finding": finding,
        "quantitative_basis": quantitative_basis,
        "implication": implication,
        "caution": caution,
    })


def main():
    log.info("Starting Script 230")

    evidence_227 = read(INPUT_227_EVIDENCE)
    model_227 = read(INPUT_227_MODEL_COMPARISON)
    fold_227 = read(INPUT_227_FOLD_SYNTHESIS)

    m229 = read(INPUT_229B_METRICS)
    f229 = read(INPUT_229B_FOLD_METRICS)

    spatial_trend = metric(m229, "spatial_trend_rf", "roc_auc")
    raw_coord = metric(m229, "raw_coordinate_rf", "roc_auc")
    global_rf = metric(m229, "global_rf", "roc_auc")
    diff = metric(m229, "differential_operator_rf", "roc_auc")
    diff_trend = metric(m229, "differential_plus_trend_rf", "roc_auc")
    diff_coord = metric(m229, "differential_plus_raw_coord_rf", "roc_auc")

    new_rows = evidence_227.to_dict("records")

    add_evidence_row(
        new_rows,
        "E11",
        "Do local spatial differential and transition operators replace the unresolved spatial term S_i?",
        "not_supported_current_form",
        "The corrected recognition-inclusive differential-operator models underperform the clean spatial-trend and raw-coordinate models.",
        f"differential AUC={fmt(diff)}; differential+trend AUC={fmt(diff_trend)}; spatial trend AUC={fmt(spatial_trend)}; raw-coordinate AUC={fmt(raw_coord)}.",
        "The current v01 differential operators do not explain the main spatially varying persistence signal.",
        "This rejects the current operator construction, not every possible differential, anisotropic, or boundary-transition operator."
    )

    add_evidence_row(
        new_rows,
        "E12",
        "What is the best current representation of unresolved spatial variation?",
        "supported",
        "Broad spatial trend currently approximates or slightly exceeds raw-coordinate performance while remaining simpler than differential-operator expansions.",
        f"spatial trend AUC={fmt(spatial_trend)}; raw-coordinate AUC={fmt(raw_coord)}; global AUC={fmt(global_rf)}.",
        "The unresolved spatial term should currently be represented as broad regional trend T_s, not local differential structure.",
        "T_s is descriptive and should not be interpreted as a causal mechanism."
    )

    evidence = pd.DataFrame(new_rows)

    m229_stage = m229.copy()
    m229_stage.insert(0, "analysis_stage", "229b_differential_substitution")

    model_comparison = pd.concat(
        [model_227, m229_stage],
        ignore_index=True,
        sort=False,
    )

    model_comparison = model_comparison.sort_values(
        ["analysis_stage", "roc_auc"],
        ascending=[True, False],
    )

    f229_pivot = (
        f229.pivot_table(
            index="fold",
            columns="model",
            values="roc_auc",
            aggfunc="first",
        )
        .reset_index()
    )

    f229_pivot.columns = [
        "fold" if c == "fold" else f"auc_229b_{c}"
        for c in f229_pivot.columns
    ]

    fold = fold_227.merge(
        f229_pivot,
        on="fold",
        how="left",
    )

    if {
        "auc_229b_spatial_trend_rf",
        "auc_229b_global_rf",
    }.issubset(fold.columns):
        fold["spatial_trend_gain_v229b"] = (
            fold["auc_229b_spatial_trend_rf"]
            - fold["auc_229b_global_rf"]
        )

    if {
        "auc_229b_differential_operator_rf",
        "auc_229b_global_rf",
    }.issubset(fold.columns):
        fold["differential_gain_v229b"] = (
            fold["auc_229b_differential_operator_rf"]
            - fold["auc_229b_global_rf"]
        )

    if {
        "auc_229b_differential_plus_trend_rf",
        "auc_229b_spatial_trend_rf",
    }.issubset(fold.columns):
        fold["diff_plus_trend_minus_trend_v229b"] = (
            fold["auc_229b_differential_plus_trend_rf"]
            - fold["auc_229b_spatial_trend_rf"]
        )

    supported = evidence[evidence["status"].isin([
        "supported",
        "substantial_support",
        "supported_but_incomplete",
        "partial_support",
    ])]

    rejected = evidence[evidence["status"].isin([
        "rejected_simple_form",
        "not_supported",
        "weak_support_for_discrete_regimes",
        "not_supported_current_form",
    ])]

    unresolved = evidence[
        evidence["status"].astype(str).str.contains("unresolved", case=False, na=False)
    ]

    theory = f"""# Oregon SRS Operator-Persistence Branch Decision & Theory Freeze v02

Generated by `{SCRIPT}`.

## 1. Decision status

This checkpoint freezes the current Oregon SRS operator-persistence branch after the corrected differential-operator substitution test.

The branch has produced a coherent empirical result:

> Persistent relational versus persistent pointwise operator regimes are prospectively structured, nonlinear, multiscale, and spatially nonstationary.

The branch should not continue by stacking more generic classifiers. The next stage should move toward theory consolidation, cross-state comparison, or targeted testing of one clearly defined mathematical operator family.

## 2. Current best mathematical statement

The supported current form is:

$$
\\Pr(G_i = R)
=
\\mathcal{{F}}\\left(
Y_i,
\\mathcal{{C}}_r(P)_i,
\\Delta_{{r_a,r_b}}\\mathcal{{C}}(P)_i,
T_s
\\right),
$$

where:

- $G_i$ is the persistent operator regime;
- $Y_i$ is admissible recognition state;
- $\\mathcal{{C}}_r(P)_i$ is multiscale physical context;
- $\\Delta_{{r_a,r_b}}\\mathcal{{C}}(P)_i$ is local-to-regional context contrast;
- $T_s$ is broad regional spatial trend / unresolved spatial drift;
- $\\mathcal{{F}}$ is nonlinear.

The previous unresolved spatial term $S_i$ is now provisionally refined to:

$$
S_i \\approx T_s
$$

because the corrected Script 229b showed that broad spatial trend performs better than the tested local differential operators.

## 3. Key model results after Script 229b

| Model | AUC |
|---|---:|
| spatial_trend_rf | {fmt(spatial_trend)} |
| raw_coordinate_rf | {fmt(raw_coord)} |
| global_rf | {fmt(global_rf)} |
| differential_operator_rf | {fmt(diff)} |
| differential_plus_trend_rf | {fmt(diff_trend)} |
| differential_plus_raw_coord_rf | {fmt(diff_coord)} |

Interpretation:

- Broad spatial trend is currently the strongest clean substitute for raw coordinates.
- Differential/transition operators do not currently recover the spatial trend gain.
- Adding many local differential operators appears to hurt transfer stability.
- Fold 0 remains unresolved.

## 4. Supported findings

{supported[["evidence_id", "finding", "quantitative_basis", "implication"]].to_markdown(index=False)}

## 5. Rejected or unsupported hypotheses

{rejected[["evidence_id", "finding", "quantitative_basis", "implication"]].to_markdown(index=False)}

## 6. Unresolved findings

{unresolved[["evidence_id", "finding", "quantitative_basis", "implication"]].to_markdown(index=False) if not unresolved.empty else "No unresolved findings were recorded."}

## 7. Fold-level decision notes

The fold-level evidence remains heterogeneous.

{fold.to_markdown(index=False)}

Important fold interpretation:

- Fold 5 is recoverable under spatial trend / coordinate conditioning.
- Fold 6 also benefits from spatial trend.
- Fold 0 remains the main structural-transfer boundary.
- Differential operators do not repair fold 0.
- Therefore fold 0 should be studied as a support/extrapolation/boundary problem, not merely as noise.

## 8. What is now frozen

The following claims are frozen as the current Oregon SRS branch conclusion:

1. Operator persistence is not explained by physical context alone.
2. Operator persistence is not explained by recognition state alone.
3. Operator persistence is nonlinear.
4. Pairwise recognition-context interactions help but are incomplete.
5. A universal scalar mismatch law is not supported.
6. Simple unsupervised KMeans regimes are not sufficient.
7. Spatial nonstationarity is supported.
8. Broad spatial trend is currently the best compact representation of unresolved spatial drift.
9. Local differential/transition operators are not supported in their current form as the main substitute for spatial drift.
10. Fold 0 remains unresolved.

## 9. Recommended project move after this checkpoint

Do not keep expanding the Oregon SRS branch indefinitely.

Recommended next move:

1. Use this branch as a formal methods/theory result.
2. Back-port only the stable, useful pieces later if needed.
3. Move toward manuscript/theory consolidation or a California/Oregon comparison checkpoint.
4. Treat differential/aniso/boundary operators as future work unless there is a specific new mathematical hypothesis.

The clean takeaway is:

> SRS operator persistence is governed by nonlinear recognition-context structure plus broad spatially varying response geometry, not by a simple universal mismatch index or the tested local differential operators.
"""

    evidence.to_csv(OUT_EVIDENCE, index=False)
    model_comparison.to_csv(OUT_MODEL_COMPARISON, index=False)
    fold.to_csv(OUT_FOLD_SYNTHESIS, index=False)
    OUT_THEORY_FREEZE.write_text(theory, encoding="utf-8")

    qa = [
        f"Script: {SCRIPT}",
        "",
        "PURPOSE",
        "Freeze Oregon SRS operator-persistence branch after corrected differential-operator test.",
        "",
        "NEW SCRIPT 229B RESULT",
        f"spatial_trend_rf AUC: {fmt(spatial_trend)}",
        f"raw_coordinate_rf AUC: {fmt(raw_coord)}",
        f"global_rf AUC: {fmt(global_rf)}",
        f"differential_operator_rf AUC: {fmt(diff)}",
        f"differential_plus_trend_rf AUC: {fmt(diff_trend)}",
        "",
        "DECISION",
        "1. Spatial trend is currently the best compact representation of unresolved S_i.",
        "2. Local differential operators are not supported as the primary S_i substitute in current form.",
        "3. Fold 0 remains unresolved.",
        "4. Oregon SRS branch should now be frozen as a theory checkpoint rather than extended with more generic models.",
        "",
        "OUTPUTS",
        str(OUT_EVIDENCE),
        str(OUT_MODEL_COMPARISON),
        str(OUT_FOLD_SYNTHESIS),
        str(OUT_THEORY_FREEZE),
    ]

    OUT_QA.write_text("\n".join(qa), encoding="utf-8")

    log.info("Completed Script 230")
    log.info("")
    log.info("Decision:")
    log.info("S_i provisionally refined to broad spatial trend T_s.")
    log.info("Differential operators not supported in current form.")
    log.info("Created:")
    for p in [OUT_EVIDENCE, OUT_MODEL_COMPARISON, OUT_FOLD_SYNTHESIS, OUT_THEORY_FREEZE, OUT_QA]:
        log.info(str(p))


if __name__ == "__main__":
    main()