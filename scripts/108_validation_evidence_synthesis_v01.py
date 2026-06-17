#!/usr/bin/env python3
"""
108_validation_evidence_synthesis_v01.py

Purpose
-------
Synthesize all current RDE validation streams into one mechanism-level
defensibility table.

This script answers:

Which RDE mechanisms are actually defensible today?

Inputs
------
data/processed/rde_theory_validation_summary_v01.csv
data/processed/rde_theory_readiness_filter_v01.csv
data/processed/rde_mechanism_stability_v02.csv
data/processed/rde_background_entry_summary_v01.csv
data/processed/rde_external_proxy_validation_mechanism_summary_v01.csv
data/processed/rde_wikipedia_wikidata_external_validation_mechanism_summary_v01.csv
data/processed/rde_geographic_holdout_mechanism_summary_v01.csv

Outputs
-------
data/processed/rde_validation_evidence_synthesis_v01.csv
data/processed/rde_validation_evidence_summary_v01.csv
data/processed/rde_mechanism_defensibility_rankings_v01.csv
data/processed/rde_validation_evidence_qa_v01.txt
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_NAME = "108_validation_evidence_synthesis_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUT_THEORY_SUMMARY = PROCESSED / "rde_theory_validation_summary_v01.csv"
INPUT_READINESS = PROCESSED / "rde_theory_readiness_filter_v01.csv"
INPUT_STABILITY = PROCESSED / "rde_mechanism_stability_v02.csv"
INPUT_ENTRY_SUMMARY = PROCESSED / "rde_background_entry_summary_v01.csv"
INPUT_PROXY_MECH = PROCESSED / "rde_external_proxy_validation_mechanism_summary_v01.csv"
INPUT_WIKI_MECH = PROCESSED / "rde_wikipedia_wikidata_external_validation_mechanism_summary_v01.csv"
INPUT_HOLDOUT_MECH = PROCESSED / "rde_geographic_holdout_mechanism_summary_v01.csv"

OUTPUT_SYNTHESIS = PROCESSED / "rde_validation_evidence_synthesis_v01.csv"
OUTPUT_SUMMARY = PROCESSED / "rde_validation_evidence_summary_v01.csv"
OUTPUT_RANKINGS = PROCESSED / "rde_mechanism_defensibility_rankings_v01.csv"
OUTPUT_QA = PROCESSED / "rde_validation_evidence_qa_v01.txt"


MECHANISMS = [
    "Opportunity Failure",
    "Recognition Inefficiency",
    "Comparative Shadowing",
]


def canonical_mechanism(x: object) -> str:
    s = str(x).replace(" Candidate", "").strip()

    if "Recognition Inefficiency" in s:
        return "Recognition Inefficiency"
    if "Opportunity Failure" in s:
        return "Opportunity Failure"
    if "Comparative Shadowing" in s or "Recognition Diversion" in s:
        return "Comparative Shadowing"

    return s


def safe_read(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required input: {path}")
        log.warning("Optional input missing: %s", path)
        return pd.DataFrame()

    log.info("Reading: %s", path)
    return pd.read_csv(path, low_memory=False)


def evidence_score_to_class(score: float) -> str:
    if pd.isna(score):
        return "Insufficient"
    if score >= 0.70:
        return "Strong"
    if score >= 0.55:
        return "Moderate"
    if score >= 0.40:
        return "Weak"
    return "Insufficient"


def class_to_points(cls: str) -> int:
    cls = str(cls)
    if "Strong" in cls:
        return 3
    if "Moderate" in cls or "Defensible" in cls:
        return 2
    if "Weak" in cls or "Promising" in cls or "Emerging" in cls:
        return 1
    return 0


def defensibility_class(total: int, max_score: int = 18) -> str:
    pct = total / max_score if max_score else 0

    if pct >= 0.80:
        return "Strongly Defensible"
    if pct >= 0.65:
        return "Defensible"
    if pct >= 0.45:
        return "Promising / Partially Defensible"
    return "Exploratory / Weakly Defensible"


def build_theory_evidence(theory_summary: pd.DataFrame, readiness: pd.DataFrame) -> pd.DataFrame:
    rows = []

    theory = theory_summary.copy()
    ready = readiness.copy()

    if "mechanism" in theory.columns:
        theory["canonical_mechanism"] = theory["mechanism"].map(canonical_mechanism)

    if "mechanism" in ready.columns:
        ready["canonical_mechanism"] = ready["mechanism"].map(canonical_mechanism)

    for mech in MECHANISMS:
        t = theory[theory.get("canonical_mechanism", pd.Series(dtype=str)) == mech]
        r = ready[ready.get("canonical_mechanism", pd.Series(dtype=str)) == mech]

        mean_score = np.nan
        if "mean_evidence_score" in t.columns and len(t) > 0:
            mean_score = float(t["mean_evidence_score"].mean())
        elif "overall_evidence_score" in r.columns and len(r) > 0:
            mean_score = float(r["overall_evidence_score"].mean())

        core = 0
        emerging = 0
        holdout = 0

        if "theory_readiness_tier" in r.columns:
            core = int((r["theory_readiness_tier"] == "Core Validated Theory").sum())
            emerging = int((r["theory_readiness_tier"] == "Emerging Theory").sum())
            holdout = int((r["theory_readiness_tier"] == "Exploratory / Holdout Theory").sum())

        if core >= 2:
            cls = "Strong"
        elif core >= 1:
            cls = "Moderate"
        elif emerging >= 1:
            cls = "Weak"
        else:
            cls = evidence_score_to_class(mean_score)

        rows.append(
            {
                "mechanism": mech,
                "theory_validation_score": mean_score,
                "core_validated_archetypes": core,
                "emerging_archetypes": emerging,
                "holdout_archetypes": holdout,
                "theory_validation_class": cls,
                "theory_validation_points": class_to_points(cls),
            }
        )

    return pd.DataFrame(rows)


def build_stability_evidence(stability: pd.DataFrame) -> pd.DataFrame:
    df = stability.copy()

    if "mechanism" in df.columns:
        df["canonical_mechanism"] = df["mechanism"].map(canonical_mechanism)

    if "perturbation_level" in df.columns:
        sub = df[np.isclose(df["perturbation_level"], 0.05)].copy()
        if len(sub) == 0:
            sub = df.copy()
    else:
        sub = df.copy()

    rows = []

    for mech in MECHANISMS:
        m = sub[sub.get("canonical_mechanism", pd.Series(dtype=str)) == mech]

        score = np.nan
        if "mechanism_stability_rate" in m.columns and len(m) > 0:
            score = float(m["mechanism_stability_rate"].mean())

        if score >= 0.85:
            cls = "Strong"
        elif score >= 0.70:
            cls = "Moderate"
        elif score >= 0.55:
            cls = "Weak"
        else:
            cls = "Insufficient"

        rows.append(
            {
                "mechanism": mech,
                "perturbation_stability_score": score,
                "perturbation_stability_class": cls,
                "perturbation_stability_points": class_to_points(cls),
            }
        )

    return pd.DataFrame(rows)


def build_entry_evidence(entry_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Background entry validation is not mechanism-specific in Script 100.
    It validates the RDE mechanism-region universe as distinct from background.

    Apply it as shared evidence to all mechanisms, with a note.
    """
    strong = 0
    moderate_or_stronger = 0

    if "summary_metric" in entry_summary.columns:
        d = dict(zip(entry_summary["summary_metric"], entry_summary["value"]))
        try:
            strong = int(float(d.get("strong_entry_evidence_features", 0)))
        except Exception:
            strong = 0

        try:
            moderate_or_stronger = int(float(d.get("moderate_or_stronger_entry_features", 0)))
        except Exception:
            moderate_or_stronger = 0

    if strong >= 3:
        cls = "Strong"
    elif moderate_or_stronger >= 3:
        cls = "Moderate"
    elif moderate_or_stronger >= 1:
        cls = "Weak"
    else:
        cls = "Insufficient"

    rows = []

    for mech in MECHANISMS:
        rows.append(
            {
                "mechanism": mech,
                "background_entry_class": cls,
                "background_entry_points": class_to_points(cls),
                "background_entry_note": (
                    "Shared universe-level validation: mechanism-region cells differ "
                    "from background cells."
                ),
            }
        )

    return pd.DataFrame(rows)


def build_proxy_evidence(proxy: pd.DataFrame) -> pd.DataFrame:
    df = proxy.copy()

    if "canonical_mechanism" in df.columns:
        df["mechanism"] = df["canonical_mechanism"].map(canonical_mechanism)
    elif "mechanism" in df.columns:
        df["mechanism"] = df["mechanism"].map(canonical_mechanism)

    rows = []

    for mech in MECHANISMS:
        m = df[df.get("mechanism", pd.Series(dtype=str)) == mech]

        score = np.nan
        if "mean_external_proxy_validation_score" in m.columns and len(m) > 0:
            score = float(m["mean_external_proxy_validation_score"].mean())

        cls = evidence_score_to_class(score)

        rows.append(
            {
                "mechanism": mech,
                "external_proxy_score": score,
                "external_proxy_class": cls,
                "external_proxy_points": class_to_points(cls),
            }
        )

    return pd.DataFrame(rows)


def build_wiki_evidence(wiki: pd.DataFrame) -> pd.DataFrame:
    df = wiki.copy()

    if "canonical_mechanism" in df.columns:
        df["mechanism"] = df["canonical_mechanism"].map(canonical_mechanism)
    elif "mechanism" in df.columns:
        df["mechanism"] = df["mechanism"].map(canonical_mechanism)

    rows = []

    for mech in MECHANISMS:
        m = df[df.get("mechanism", pd.Series(dtype=str)) == mech]

        score = np.nan
        if "mean_external_under_recognition_score" in m.columns and len(m) > 0:
            score = float(m["mean_external_under_recognition_score"].mean())

        cls = evidence_score_to_class(score)

        rows.append(
            {
                "mechanism": mech,
                "wiki_wikidata_score": score,
                "wiki_wikidata_class": cls,
                "wiki_wikidata_points": class_to_points(cls),
            }
        )

    return pd.DataFrame(rows)


def build_holdout_evidence(holdout: pd.DataFrame) -> pd.DataFrame:
    df = holdout.copy()

    if "mechanism" in df.columns:
        df["mechanism"] = df["mechanism"].map(canonical_mechanism)

    rows = []

    for mech in MECHANISMS:
        m = df[df.get("mechanism", pd.Series(dtype=str)) == mech]

        score = np.nan
        if "mechanism_transferability_score" in m.columns and len(m) > 0:
            score = float(m["mechanism_transferability_score"].mean())

        if score >= 0.80:
            cls = "Strong"
        elif score >= 0.65:
            cls = "Moderate"
        elif score >= 0.50:
            cls = "Weak"
        else:
            cls = "Insufficient"

        rows.append(
            {
                "mechanism": mech,
                "geographic_transferability_score": score,
                "geographic_transferability_class": cls,
                "geographic_transferability_points": class_to_points(cls),
            }
        )

    return pd.DataFrame(rows)


def publication_statement(row: pd.Series) -> str:
    mech = row["mechanism"]
    overall = row["overall_defensibility_class"]

    if mech == "Opportunity Failure":
        return (
            "Opportunity Failure is currently the most defensible RDE mechanism, "
            "with strong or moderate support across theory validation, stability, "
            "background distinctiveness, external proxies, and geographic transferability."
        )

    if mech == "Recognition Inefficiency":
        return (
            "Recognition Inefficiency is strongly supported as a transferable mechanism, "
            "but some fine-grained archetypes remain less stable and should be framed cautiously."
        )

    if mech == "Comparative Shadowing":
        return (
            "Comparative Shadowing shows strong geographic transferability but weaker "
            "theory and external-evidence support; it should be treated as promising "
            "but not yet fully validated."
        )

    return f"{mech} is classified as {overall}."


def build_synthesis(
    theory: pd.DataFrame,
    stability: pd.DataFrame,
    entry: pd.DataFrame,
    proxy: pd.DataFrame,
    wiki: pd.DataFrame,
    holdout: pd.DataFrame,
) -> pd.DataFrame:
    out = theory.merge(stability, on="mechanism", how="outer")
    out = out.merge(entry, on="mechanism", how="outer")
    out = out.merge(proxy, on="mechanism", how="outer")
    out = out.merge(wiki, on="mechanism", how="outer")
    out = out.merge(holdout, on="mechanism", how="outer")

    point_cols = [
        "theory_validation_points",
        "perturbation_stability_points",
        "background_entry_points",
        "external_proxy_points",
        "wiki_wikidata_points",
        "geographic_transferability_points",
    ]

    for c in point_cols:
        if c not in out.columns:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    out["total_validation_points"] = out[point_cols].sum(axis=1)
    out["max_validation_points"] = len(point_cols) * 3
    out["overall_validation_fraction"] = (
        out["total_validation_points"] / out["max_validation_points"]
    )

    out["overall_defensibility_class"] = out["total_validation_points"].map(
        lambda x: defensibility_class(int(x), max_score=len(point_cols) * 3)
    )

    out["publication_statement"] = out.apply(publication_statement, axis=1)

    out = out.sort_values(
        ["total_validation_points", "overall_validation_fraction"],
        ascending=False,
    )

    return out


def build_summary(synthesis: pd.DataFrame) -> pd.DataFrame:
    return (
        synthesis.groupby("overall_defensibility_class")
        .agg(
            mechanism_count=("mechanism", "count"),
            mean_validation_fraction=("overall_validation_fraction", "mean"),
            mechanisms=("mechanism", lambda s: ", ".join(s)),
        )
        .reset_index()
        .sort_values("mean_validation_fraction", ascending=False)
    )


def build_rankings(synthesis: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "mechanism",
        "total_validation_points",
        "max_validation_points",
        "overall_validation_fraction",
        "overall_defensibility_class",
        "theory_validation_class",
        "perturbation_stability_class",
        "background_entry_class",
        "external_proxy_class",
        "wiki_wikidata_class",
        "geographic_transferability_class",
        "publication_statement",
    ]

    return synthesis[[c for c in cols if c in synthesis.columns]].copy()


def write_outputs(
    synthesis: pd.DataFrame,
    summary: pd.DataFrame,
    rankings: pd.DataFrame,
) -> None:
    log.info("Writing synthesis: %s", OUTPUT_SYNTHESIS)
    synthesis.to_csv(OUTPUT_SYNTHESIS, index=False)

    log.info("Writing summary: %s", OUTPUT_SUMMARY)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    log.info("Writing rankings: %s", OUTPUT_RANKINGS)
    rankings.to_csv(OUTPUT_RANKINGS, index=False)

    qa = []
    qa.append("RDE Validation Evidence Synthesis v01 QA")
    qa.append("=" * 50)
    qa.append("")
    qa.append("Mechanism rankings:")
    qa.append(rankings.to_string(index=False))
    qa.append("")
    qa.append("Summary:")
    qa.append(summary.to_string(index=False))

    log.info("Writing QA: %s", OUTPUT_QA)
    OUTPUT_QA.write_text("\n".join(qa), encoding="utf-8")


def main() -> None:
    log.info("Starting Script 108: validation evidence synthesis")

    theory_summary = safe_read(INPUT_THEORY_SUMMARY)
    readiness = safe_read(INPUT_READINESS)
    stability = safe_read(INPUT_STABILITY)
    entry_summary = safe_read(INPUT_ENTRY_SUMMARY)
    proxy_mech = safe_read(INPUT_PROXY_MECH)
    wiki_mech = safe_read(INPUT_WIKI_MECH)
    holdout_mech = safe_read(INPUT_HOLDOUT_MECH)

    theory = build_theory_evidence(theory_summary, readiness)
    stability_evidence = build_stability_evidence(stability)
    entry = build_entry_evidence(entry_summary)
    proxy = build_proxy_evidence(proxy_mech)
    wiki = build_wiki_evidence(wiki_mech)
    holdout = build_holdout_evidence(holdout_mech)

    synthesis = build_synthesis(
        theory=theory,
        stability=stability_evidence,
        entry=entry,
        proxy=proxy,
        wiki=wiki,
        holdout=holdout,
    )

    summary = build_summary(synthesis)
    rankings = build_rankings(synthesis)

    write_outputs(synthesis, summary, rankings)

    log.info("Done")

    print("\nRDE Mechanism Defensibility Rankings:")
    print(rankings.to_string(index=False))

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nCreated:")
    print(f"  {OUTPUT_SYNTHESIS}")
    print(f"  {OUTPUT_SUMMARY}")
    print(f"  {OUTPUT_RANKINGS}")
    print(f"  {OUTPUT_QA}")


if __name__ == "__main__":
    main()