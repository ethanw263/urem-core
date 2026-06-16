#!/usr/bin/env python3
"""
94_typology_consolidation_and_theory_table_v01.py

Purpose
-------
Create the first Recognition Disequilibrium Theory Table.

Inputs
------
mechanism_region_typology_v02.csv
recognition_inefficiency_deep_typology_v01.csv

Outputs
-------
recognition_disequilibrium_theory_table_v01.csv
recognition_disequilibrium_theory_summary_v01.csv
recognition_disequilibrium_theory_qa_v01.txt
"""

from pathlib import Path
import logging
import pandas as pd


SCRIPT_NAME = "94_typology_consolidation_and_theory_table_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s"
)

log = logging.getLogger(SCRIPT_NAME)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUT_92 = PROCESSED / "mechanism_region_typology_v02.csv"
INPUT_93 = PROCESSED / "recognition_inefficiency_deep_typology_v01.csv"

OUTPUT_TABLE = PROCESSED / "recognition_disequilibrium_theory_table_v01.csv"
OUTPUT_SUMMARY = PROCESSED / "recognition_disequilibrium_theory_summary_v01.csv"
OUTPUT_QA = PROCESSED / "recognition_disequilibrium_theory_qa_v01.txt"


THEORY_MAP = {

    "General Opportunity Failure Landscape":
    (
        "Opportunity Failure",
        "Low opportunity relative to physical potential.",
        "Recognition suppression caused primarily by insufficient opportunity structure."
    ),

    "High-Potential Opportunity Gap Landscape":
    (
        "Opportunity Failure",
        "High physical potential but constrained opportunity.",
        "Potential remains unrealized because enabling conditions are absent."
    ),

    "Low-Opportunity Low-Transmission Landscape":
    (
        "Opportunity Failure",
        "Both opportunity and transmission are weak.",
        "Recognition deficit emerges from multiple systemic constraints."
    ),

    "General Comparative Shadowing Landscape":
    (
        "Comparative Shadowing",
        "Recognition diverted toward nearby competitors.",
        "Recognition is redistributed rather than absent."
    ),

    "Coastal Recognition Diversion Landscape":
    (
        "Comparative Shadowing",
        "Coastal competitor landscapes attract attention.",
        "Recognition concentrates in neighboring destinations."
    ),

    "Recognition Sink Landscape":
    (
        "Comparative Shadowing",
        "Recognition fails to accumulate despite favorable conditions.",
        "Spatial recognition sink phenomenon."
    ),

    "Transmission Diverted Landscape":
    (
        "Comparative Shadowing",
        "Recognition pathways exist but are redirected elsewhere.",
        "Transmission does not terminate locally."
    ),

    "Diffuse Recognition Inefficiency":
    (
        "Recognition Inefficiency",
        "No single dominant bottleneck.",
        "Recognition failure emerges from distributed inefficiencies."
    ),

    "Coastal Recognition Inefficiency":
    (
        "Recognition Inefficiency",
        "Coastal geography remains under-recognized.",
        "Coastal advantage alone is insufficient for recognition."
    ),

    "Physical Exceptionality Recognition Lag":
    (
        "Recognition Inefficiency",
        "Exceptional geography exceeds current recognition.",
        "Recognition responds more slowly than physical potential."
    ),

    "Large-Area Latent Recognition Failure":
    (
        "Recognition Inefficiency",
        "Large landscapes remain under-recognized.",
        "Scale does not guarantee recognition."
    ),

    "Coastal Hidden-Gem Recognition Failure":
    (
        "Recognition Inefficiency",
        "High-value coastal landscapes remain overlooked.",
        "Canonical hidden-gem mechanism."
    ),

    "Shadow-Contaminated Recognition Inefficiency":
    (
        "Recognition Inefficiency",
        "Recognition inefficiency amplified by nearby competitors.",
        "Mechanisms may interact rather than operate independently."
    ),

    "Transmission Bottleneck Exceptional Landscape":
    (
        "Recognition Inefficiency",
        "Physical potential exists but transmission pathways remain weak.",
        "Transmission can be the limiting recognition mechanism."
    )
}


def load_data():

    df92 = pd.read_csv(INPUT_92)
    df93 = pd.read_csv(INPUT_93)

    return df92, df93


def build_theory_table(df92, df93):

    rows = []

    # Opportunity Failure + Shadowing
    for archetype, sub in df92.groupby("region_archetype_v02"):

        if archetype not in THEORY_MAP:
            continue

        mechanism, signature, contribution = THEORY_MAP[archetype]

        rows.append({
            "mechanism": mechanism,
            "archetype": archetype,
            "region_count": len(sub),
            "signature": signature,
            "theory_contribution": contribution,
            "mean_strength":
                sub["archetype_strength_v02"].mean()
                if "archetype_strength_v02" in sub.columns else None
        })

    # Deep RI archetypes
    for archetype, sub in df93.groupby("ri_deep_archetype_v01"):

        if archetype not in THEORY_MAP:
            continue

        mechanism, signature, contribution = THEORY_MAP[archetype]

        rows.append({
            "mechanism": mechanism,
            "archetype": archetype,
            "region_count": len(sub),
            "signature": signature,
            "theory_contribution": contribution,
            "mean_strength":
                sub["ri_deep_strength_v01"].mean()
                if "ri_deep_strength_v01" in sub.columns else None
        })

    theory = pd.DataFrame(rows)

    theory = theory.sort_values(
        ["mechanism", "region_count"],
        ascending=[True, False]
    )

    return theory


def build_summary(theory):

    summary = (
        theory
        .groupby("mechanism")
        .agg(
            archetype_count=("archetype", "count"),
            total_regions=("region_count", "sum"),
            mean_strength=("mean_strength", "mean")
        )
        .reset_index()
    )

    return summary


def write_outputs(theory, summary):

    theory.to_csv(OUTPUT_TABLE, index=False)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    qa = []

    qa.append("Recognition Disequilibrium Theory Table v01")
    qa.append("=" * 50)
    qa.append("")

    qa.append(f"Total theory archetypes: {len(theory)}")
    qa.append("")

    qa.append(summary.to_string(index=False))
    qa.append("")
    qa.append(theory.to_string(index=False))

    OUTPUT_QA.write_text(
        "\n".join(qa),
        encoding="utf-8"
    )


def main():

    log.info("Starting Script 94")

    df92, df93 = load_data()

    theory = build_theory_table(df92, df93)
    summary = build_summary(theory)

    write_outputs(theory, summary)

    log.info("Done")

    print("\nRecognition Disequilibrium Theory Summary")
    print(summary.to_string(index=False))

    print("\nTheory Archetypes:")
    print(theory[[
        "mechanism",
        "archetype",
        "region_count"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()