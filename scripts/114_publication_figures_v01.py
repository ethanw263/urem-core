#!/usr/bin/env python3
"""
114_publication_figures_v01.py

Purpose
-------
Generate publication-ready figure files and supporting figure-input tables for
the RDE / UREM project.

Figures created
---------------
1. Mechanism Evidence Scores
2. Geographic Holdout Transferability by Zone
3. Mechanism Transferability
4. External Validation by Mechanism
5. Negative Control Contrast
6. Cross-Domain Generalization Readiness
7. Geographic Landscape System Counts
8. Publication Readiness Scorecard

Outputs
-------
data/processed/publication_figures/
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_NAME = "114_publication_figures_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTDIR = PROCESSED / "publication_figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

INPUTS = {
    "mechanism_defensibility": PROCESSED / "rde_mechanism_defensibility_rankings_v01.csv",
    "holdout_zone": PROCESSED / "rde_geographic_holdout_summary_v01.csv",
    "holdout_mechanism": PROCESSED / "rde_geographic_holdout_mechanism_summary_v01.csv",
    "wiki_validation": PROCESSED / "rde_wikipedia_wikidata_external_validation_mechanism_summary_v01.csv",
    "external_proxy": PROCESSED / "rde_external_proxy_validation_mechanism_summary_v01.csv",
    "negative_control_tests": PROCESSED / "rde_negative_control_feature_tests_v01.csv",
    "generalization_scores": PROCESSED / "rde_mechanism_generalization_scores_v01.csv",
    "geographic_theory": PROCESSED / "rde_geographic_landscape_theory_table_v01.csv",
    "scorecard": PROCESSED / "rde_publication_readiness_scorecard_v01.csv",
}

FIGURE_DPI = 220


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.warning("Missing input: %s", path)
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        log.warning("Could not read %s: %s", path, exc)
        return pd.DataFrame()


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def fig1_mechanism_evidence(df: pd.DataFrame) -> dict | None:
    if df.empty or "mechanism" not in df.columns or "overall_validation_fraction" not in df.columns:
        return None
    plot = df.copy().sort_values("overall_validation_fraction", ascending=True)
    plt.figure(figsize=(9, 5))
    plt.barh(plot["mechanism"], plot["overall_validation_fraction"])
    plt.xlim(0, 1)
    plt.xlabel("Overall validation fraction")
    plt.ylabel("Mechanism")
    plt.title("RDE Mechanism Evidence Synthesis")
    for i, v in enumerate(plot["overall_validation_fraction"]):
        cls = plot.get("overall_defensibility_class", pd.Series([""] * len(plot))).iloc[i]
        plt.text(min(v + 0.02, 0.98), i, f"{v:.2f}  {cls}", va="center", fontsize=9)
    out = OUTDIR / "figure_01_mechanism_evidence_scores.png"
    save_fig(out)
    return {"figure_id": "Figure 1", "filename": out.name, "title": "Mechanism Evidence Synthesis", "source": str(INPUTS["mechanism_defensibility"]), "interpretation": "Overall defensibility of core RDE mechanisms."}


def fig2_holdout_zone(df: pd.DataFrame) -> dict | None:
    if df.empty or not {"holdout_zone", "transferability_score"}.issubset(df.columns):
        return None
    plot = df.copy().sort_values("transferability_score", ascending=True)
    labels = [str(x).replace(" / ", " /\n") for x in plot["holdout_zone"]]
    plt.figure(figsize=(10, 6))
    plt.barh(labels, plot["transferability_score"])
    plt.xlim(0, 1)
    plt.xlabel("Transferability score")
    plt.ylabel("Held-out geographic zone")
    plt.title("Leave-One-Zone-Out Geographic Transferability")
    for i, v in enumerate(plot["transferability_score"]):
        acc = plot.get("mechanism_accuracy", pd.Series([np.nan] * len(plot))).iloc[i]
        plt.text(min(v + 0.015, 0.98), i, f"{v:.2f} (acc {acc:.2f})", va="center", fontsize=8)
    out = OUTDIR / "figure_02_holdout_transferability_by_zone.png"
    save_fig(out)
    return {"figure_id": "Figure 2", "filename": out.name, "title": "Geographic Holdout Transferability by Zone", "source": str(INPUTS["holdout_zone"]), "interpretation": "Transferability across California coastal macro-zones."}


def fig3_mechanism_transferability(df: pd.DataFrame) -> dict | None:
    if df.empty or not {"mechanism", "mechanism_transferability_score"}.issubset(df.columns):
        return None
    plot = df.copy().sort_values("mechanism_transferability_score", ascending=True)
    plt.figure(figsize=(9, 5))
    plt.barh(plot["mechanism"], plot["mechanism_transferability_score"])
    plt.xlim(0, 1)
    plt.xlabel("Mechanism transferability score")
    plt.ylabel("Mechanism")
    plt.title("Mechanism Transferability Across Geographic Holdouts")
    for i, v in enumerate(plot["mechanism_transferability_score"]):
        recall = plot.get("leave_zone_out_recall", pd.Series([np.nan] * len(plot))).iloc[i]
        plt.text(min(v + 0.02, 0.98), i, f"{v:.2f} (recall {recall:.2f})", va="center", fontsize=9)
    out = OUTDIR / "figure_03_mechanism_transferability.png"
    save_fig(out)
    return {"figure_id": "Figure 3", "filename": out.name, "title": "Mechanism Transferability", "source": str(INPUTS["holdout_mechanism"]), "interpretation": "Leave-zone-out recall and transferability by mechanism."}


def fig4_external_validation(wiki: pd.DataFrame, proxy: pd.DataFrame) -> dict | None:
    if wiki.empty and proxy.empty:
        return None
    w = wiki.copy()
    p = proxy.copy()
    if "canonical_mechanism" in w.columns:
        w = w.rename(columns={"canonical_mechanism": "mechanism"})
    if "canonical_mechanism" in p.columns:
        p = p.rename(columns={"canonical_mechanism": "mechanism"})
    wcols = [c for c in ["mechanism", "mean_external_under_recognition_score"] if c in w.columns]
    pcols = [c for c in ["mechanism", "mean_external_proxy_validation_score"] if c in p.columns]
    w = w[wcols].copy() if wcols else pd.DataFrame()
    p = p[pcols].copy() if pcols else pd.DataFrame()
    merged = w.merge(p, on="mechanism", how="outer") if not w.empty and not p.empty else (w if not w.empty else p)
    if merged.empty or "mechanism" not in merged.columns:
        return None
    mechanisms = merged["mechanism"].tolist()
    x = np.arange(len(mechanisms))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plotted = False
    if "mean_external_under_recognition_score" in merged.columns:
        plt.bar(x - width / 2, merged["mean_external_under_recognition_score"], width, label="Wikipedia/Wikidata")
        plotted = True
    if "mean_external_proxy_validation_score" in merged.columns:
        plt.bar(x + width / 2, merged["mean_external_proxy_validation_score"], width, label="External proxy")
        plotted = True
    if not plotted:
        plt.close()
        return None
    plt.ylim(0, 1)
    plt.xticks(x, mechanisms, rotation=20, ha="right")
    plt.ylabel("Validation score")
    plt.title("External Validation by Mechanism")
    plt.legend()
    out = OUTDIR / "figure_04_external_validation_by_mechanism.png"
    save_fig(out)
    return {"figure_id": "Figure 4", "filename": out.name, "title": "External Validation by Mechanism", "source": f"{INPUTS['wiki_validation']} ; {INPUTS['external_proxy']}", "interpretation": "External proxy and Wikipedia/Wikidata validation by mechanism."}


def fig5_negative_control(tests: pd.DataFrame) -> dict | None:
    if tests.empty or "feature" not in tests.columns:
        return None
    focus = ["physical_potential", "observed_recognition", "under_recognition", "rde_composite", "rde_positive_signature_score", "recognized_control_signature_score", "disequilibrium_vs_recognition_contrast"]
    plot = tests[tests["feature"].isin(focus)].copy()
    if plot.empty:
        return None
    plot["feature_label"] = plot["feature"].str.replace("_", " ").str.title()
    x = np.arange(len(plot))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, plot["rde_mean"], width, label="RDE candidates")
    plt.bar(x + width / 2, plot["negative_control_mean"], width, label="Recognized controls")
    plt.xticks(x, plot["feature_label"], rotation=35, ha="right")
    plt.ylabel("Mean value")
    plt.title("Negative Control Validation: RDE Candidates vs Recognized Places")
    plt.legend()
    out = OUTDIR / "figure_05_negative_control_contrast.png"
    save_fig(out)
    return {"figure_id": "Figure 5", "filename": out.name, "title": "Negative Control Contrast", "source": str(INPUTS["negative_control_tests"]), "interpretation": "RDE candidates compared with famous recognized places."}


def fig6_generalization(df: pd.DataFrame) -> dict | None:
    if df.empty or not {"mechanism", "cross_domain_generalization_readiness_score"}.issubset(df.columns):
        return None
    plot = df.copy().sort_values("cross_domain_generalization_readiness_score", ascending=True)
    plt.figure(figsize=(9, 5))
    plt.barh(plot["mechanism"], plot["cross_domain_generalization_readiness_score"])
    plt.xlim(0, 1)
    plt.xlabel("Generalization readiness score")
    plt.ylabel("Mechanism")
    plt.title("Cross-Domain Generalization Readiness")
    for i, v in enumerate(plot["cross_domain_generalization_readiness_score"]):
        cls = plot.get("cross_domain_generalization_readiness_class", pd.Series([""] * len(plot))).iloc[i]
        plt.text(min(v + 0.02, 0.98), i, f"{v:.2f}  {cls}", va="center", fontsize=8)
    out = OUTDIR / "figure_06_cross_domain_generalization_readiness.png"
    save_fig(out)
    return {"figure_id": "Figure 6", "filename": out.name, "title": "Cross-Domain Generalization Readiness", "source": str(INPUTS["generalization_scores"]), "interpretation": "Feasibility of generalization beyond California coastal domain."}


def fig7_landscape_counts(df: pd.DataFrame) -> dict | None:
    if df.empty or not {"canonical_mechanism", "geographic_landscape_type_v01", "region_count"}.issubset(df.columns):
        return None
    plot = df.copy()
    plot["label"] = plot["canonical_mechanism"] + " | " + plot["geographic_landscape_type_v01"]
    plot = plot.sort_values("region_count", ascending=True).tail(15)
    labels = [str(x).replace(" | ", " |\n").replace(" Recognition ", " Recognition\n") for x in plot["label"]]
    plt.figure(figsize=(12, 8))
    plt.barh(labels, plot["region_count"])
    plt.xlabel("Region count")
    plt.ylabel("Mechanism / landscape system")
    plt.title("Major Geographic Landscape Systems in RDE")
    for i, v in enumerate(plot["region_count"]):
        plt.text(v + 0.2, i, str(int(v)), va="center", fontsize=8)
    out = OUTDIR / "figure_07_geographic_landscape_counts.png"
    save_fig(out)
    return {"figure_id": "Figure 7", "filename": out.name, "title": "Geographic Landscape Counts", "source": str(INPUTS["geographic_theory"]), "interpretation": "Dominant geographic landscape systems across RDE mechanisms."}


def fig8_scorecard(df: pd.DataFrame) -> dict | None:
    if df.empty or not {"dimension", "score_0_10"}.issubset(df.columns):
        return None
    plot = df.copy().sort_values("score_0_10", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(plot["dimension"], plot["score_0_10"])
    plt.xlim(0, 10)
    plt.xlabel("Score (0-10)")
    plt.ylabel("Publication dimension")
    plt.title("RDE Publication Readiness Scorecard")
    for i, v in enumerate(plot["score_0_10"]):
        rating = plot.get("rating", pd.Series([""] * len(plot))).iloc[i]
        plt.text(min(v + 0.15, 9.7), i, f"{v:.1f}  {rating}", va="center", fontsize=8)
    out = OUTDIR / "figure_08_publication_readiness_scorecard.png"
    save_fig(out)
    return {"figure_id": "Figure 8", "filename": out.name, "title": "Publication Readiness Scorecard", "source": str(INPUTS["scorecard"]), "interpretation": "Current readiness dimensions for publication planning."}


def write_manifest(rows: list[dict]) -> None:
    manifest = pd.DataFrame(rows)
    manifest_path = OUTDIR / "figure_manifest_v01.csv"
    log.info("Writing manifest CSV: %s", manifest_path)
    manifest.to_csv(manifest_path, index=False)
    txt_lines = ["RDE Publication Figures Manifest v01", "=" * 45, ""]
    for row in rows:
        txt_lines.append(f"{row['figure_id']}: {row['title']}")
        txt_lines.append(f"  File: {row['filename']}")
        txt_lines.append(f"  Source: {row['source']}")
        txt_lines.append(f"  Interpretation: {row['interpretation']}")
        txt_lines.append("")
    txt_path = OUTDIR / "figure_manifest_v01.txt"
    log.info("Writing manifest TXT: %s", txt_path)
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")


def main() -> None:
    log.info("Starting Script 114: publication figures")
    data = {name: read_csv(path) for name, path in INPUTS.items()}
    figures = []
    funcs = [
        lambda: fig1_mechanism_evidence(data["mechanism_defensibility"]),
        lambda: fig2_holdout_zone(data["holdout_zone"]),
        lambda: fig3_mechanism_transferability(data["holdout_mechanism"]),
        lambda: fig4_external_validation(data["wiki_validation"], data["external_proxy"]),
        lambda: fig5_negative_control(data["negative_control_tests"]),
        lambda: fig6_generalization(data["generalization_scores"]),
        lambda: fig7_landscape_counts(data["geographic_theory"]),
        lambda: fig8_scorecard(data["scorecard"]),
    ]
    for func in funcs:
        try:
            result = func()
            if result is not None:
                figures.append(result)
        except Exception as exc:
            log.warning("Figure generation failed: %s", exc)
    write_manifest(figures)
    log.info("Done")
    print("\nPublication figures created:")
    print(f"  {OUTDIR}")
    for row in figures:
        print(f"  {row['filename']}")
    print("\nManifest:")
    print(f"  {OUTDIR / 'figure_manifest_v01.csv'}")
    print(f"  {OUTDIR / 'figure_manifest_v01.txt'}")


if __name__ == "__main__":
    main()
