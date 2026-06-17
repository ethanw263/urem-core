#!/usr/bin/env python3
"""
109_publication_results_package_v01.py

Purpose
-------
Assemble paper-ready RDE result tables and figure-input files.

This script does not create the final manuscript.
It creates a clean publication documentation package from the validated pipeline.

Inputs
------
Core outputs from Scripts 95–108.

Outputs
-------
data/processed/publication_package/
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


SCRIPT_NAME = "109_publication_results_package_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTDIR = PROCESSED / "publication_package"
OUTDIR.mkdir(parents=True, exist_ok=True)


INPUTS = {
    "mechanism_defensibility": PROCESSED / "rde_mechanism_defensibility_rankings_v01.csv",
    "validation_synthesis": PROCESSED / "rde_validation_evidence_synthesis_v01.csv",
    "geographic_theory": PROCESSED / "rde_geographic_landscape_theory_table_v01.csv",
    "geographic_summary": PROCESSED / "rde_geographic_landscape_theory_summary_v01.csv",
    "holdout_summary": PROCESSED / "rde_geographic_holdout_summary_v01.csv",
    "holdout_mechanism": PROCESSED / "rde_geographic_holdout_mechanism_summary_v01.csv",
    "wiki_validation": PROCESSED / "rde_wikipedia_wikidata_external_validation_mechanism_summary_v01.csv",
    "external_proxy": PROCESSED / "rde_external_proxy_validation_mechanism_summary_v01.csv",
    "background_entry": PROCESSED / "rde_background_entry_feature_tests_v01.csv",
    "core_claims": PROCESSED / "rde_core_claims_v01.csv",
    "limitations": PROCESSED / "rde_research_limitations_v01.csv",
    "next_plan": PROCESSED / "rde_next_validation_plan_v01.csv",
}


def read_csv_safe(name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        log.warning("Missing input %s: %s", name, path)
        return pd.DataFrame()

    log.info("Reading %s: %s", name, path)
    return pd.read_csv(path, low_memory=False)


def write(df: pd.DataFrame, filename: str) -> None:
    path = OUTDIR / filename
    log.info("Writing %s", path)
    df.to_csv(path, index=False)


def make_table_1_mechanism_evidence(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = data["mechanism_defensibility"].copy()

    keep = [
        "mechanism",
        "overall_defensibility_class",
        "total_validation_points",
        "max_validation_points",
        "overall_validation_fraction",
        "theory_validation_class",
        "perturbation_stability_class",
        "background_entry_class",
        "external_proxy_class",
        "wiki_wikidata_class",
        "geographic_transferability_class",
        "publication_statement",
    ]

    return df[[c for c in keep if c in df.columns]].copy()


def make_table_2_geographic_systems(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = data["geographic_theory"].copy()

    keep = [
        "canonical_mechanism",
        "geographic_landscape_type_v01",
        "region_count",
        "landscape_theory_strength",
        "mechanism_mean_evidence_score",
        "mean_external_validation_priority",
        "mean_physical_potential",
        "mean_opportunity",
        "mean_transmission",
        "mean_under_recognition",
        "mean_area_km2",
        "landscape_explanation",
        "mechanism_theory_contribution",
    ]

    return df[[c for c in keep if c in df.columns]].copy()


def make_table_3_holdout(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = data["holdout_summary"].copy()

    keep = [
        "holdout_zone",
        "test_region_count",
        "mechanism_accuracy",
        "mean_centroid_cosine_similarity",
        "transferability_score",
        "transferability_class",
        "recognition_inefficiency_count",
        "opportunity_failure_count",
        "comparative_shadowing_count",
    ]

    return df[[c for c in keep if c in df.columns]].copy()


def make_table_4_external_validation(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    wiki = data["wiki_validation"].copy()
    proxy = data["external_proxy"].copy()

    if "canonical_mechanism" in wiki.columns:
        wiki = wiki.rename(columns={"canonical_mechanism": "mechanism"})
    if "canonical_mechanism" in proxy.columns:
        proxy = proxy.rename(columns={"canonical_mechanism": "mechanism"})

    cols_wiki = [
        "mechanism",
        "mean_wikipedia_pages",
        "mean_wikidata_entities",
        "mean_external_under_recognition_score",
        "strong_or_moderate_support_count",
    ]
    wiki = wiki[[c for c in cols_wiki if c in wiki.columns]].copy()

    cols_proxy = [
        "mechanism",
        "mean_external_proxy_validation_score",
        "strong_support_count",
        "moderate_or_stronger_count",
        "mean_external_recognition_proxy",
        "mean_external_under_recognition_norm",
    ]
    proxy = proxy[[c for c in cols_proxy if c in proxy.columns]].copy()

    return wiki.merge(proxy, on="mechanism", how="outer")


def make_figure_inputs(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    outputs = {}

    defensibility = data["mechanism_defensibility"].copy()
    outputs["figure_input_mechanism_evidence_bar.csv"] = defensibility[
        [
            c for c in [
                "mechanism",
                "theory_validation_class",
                "perturbation_stability_class",
                "background_entry_class",
                "external_proxy_class",
                "wiki_wikidata_class",
                "geographic_transferability_class",
                "overall_validation_fraction",
            ]
            if c in defensibility.columns
        ]
    ].copy()

    geo = data["geographic_theory"].copy()
    outputs["figure_input_landscape_counts.csv"] = geo[
        [
            c for c in [
                "canonical_mechanism",
                "geographic_landscape_type_v01",
                "region_count",
                "landscape_theory_strength",
            ]
            if c in geo.columns
        ]
    ].copy()

    holdout = data["holdout_summary"].copy()
    outputs["figure_input_holdout_transferability.csv"] = holdout[
        [
            c for c in [
                "holdout_zone",
                "mechanism_accuracy",
                "mean_centroid_cosine_similarity",
                "transferability_score",
                "transferability_class",
            ]
            if c in holdout.columns
        ]
    ].copy()

    mech_holdout = data["holdout_mechanism"].copy()
    outputs["figure_input_mechanism_transferability.csv"] = mech_holdout[
        [
            c for c in [
                "mechanism",
                "leave_zone_out_recall",
                "mean_centroid_cosine_similarity",
                "mechanism_transferability_score",
                "transferability_class",
            ]
            if c in mech_holdout.columns
        ]
    ].copy()

    background = data["background_entry"].copy()
    outputs["figure_input_background_entry_effects.csv"] = background[
        [
            c for c in [
                "feature",
                "mechanism_mean",
                "background_mean",
                "mean_difference",
                "cliffs_delta",
                "entry_evidence_class",
            ]
            if c in background.columns
        ]
    ].copy()

    return outputs


def make_publication_claims(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [
        {
            "claim_id": "P1",
            "claim": "Recognition Disequilibrium can be decomposed into empirically separable mechanisms.",
            "support": "Supported by orthogonalized dimensions, mechanism taxonomy, and validation synthesis.",
            "status": "Defensible",
        },
        {
            "claim_id": "P2",
            "claim": "Opportunity Failure is the strongest current RDE mechanism.",
            "support": "Only mechanism classified as Strongly Defensible in the validation synthesis.",
            "status": "Strongly Defensible",
        },
        {
            "claim_id": "P3",
            "claim": "Recognition Inefficiency is a transferable but more cautiously framed mechanism.",
            "support": "Defensible overall, strong transferability, moderate theory validation.",
            "status": "Defensible",
        },
        {
            "claim_id": "P4",
            "claim": "Comparative Shadowing is structurally transferable but theoretically under-validated.",
            "support": "Strong holdout transferability but insufficient theory validation.",
            "status": "Promising / Cautious",
        },
        {
            "claim_id": "P5",
            "claim": "Physical Potential functions primarily as an entry condition into the RDE universe.",
            "support": "Ablation and background validation show P distinguishes mechanism regions from background more than it differentiates mechanisms internally.",
            "status": "Defensible",
        },
        {
            "claim_id": "P6",
            "claim": "RDE mechanisms generalize across distinct California coastal macro-zones.",
            "support": "Leave-zone-out geographic holdout validation produced strong transferability for all mechanisms.",
            "status": "Strongly Defensible within California coastal domain",
        },
        {
            "claim_id": "P7",
            "claim": "RDE remains unproven outside the California coastal domain.",
            "support": "No cross-state or cross-domain replication has been completed.",
            "status": "Important limitation",
        },
    ]

    return pd.DataFrame(rows)


def make_manifest(data: dict[str, pd.DataFrame]) -> str:
    lines = []
    lines.append("RDE Publication Results Package v01")
    lines.append("=" * 45)
    lines.append("")
    lines.append("Generated outputs:")
    lines.append("- table_1_mechanism_evidence.csv")
    lines.append("- table_2_geographic_landscape_systems.csv")
    lines.append("- table_3_geographic_holdout_validation.csv")
    lines.append("- table_4_external_validation.csv")
    lines.append("- publication_claims_v01.csv")
    lines.append("- figure_input_*.csv")
    lines.append("")
    lines.append("Input availability:")
    for name, df in data.items():
        lines.append(f"- {name}: rows={len(df)}, columns={len(df.columns)}")
    lines.append("")
    lines.append("Recommended paper framing:")
    lines.append(
        "Frame the project as a mechanism-based methodology for explaining "
        "recognition disequilibrium, not as a site-selection or hotspot-ranking tool."
    )
    lines.append("")
    lines.append("Strongest current claim:")
    lines.append(
        "Opportunity Failure is the most defensible RDE mechanism across validation streams."
    )
    lines.append("")
    lines.append("Most important limitation:")
    lines.append(
        "External validation is improved by Wikipedia/Wikidata and proxy checks, "
        "but cross-domain replication outside California coastal geography remains incomplete."
    )

    return "\n".join(lines)


def main() -> None:
    log.info("Starting Script 109: publication results package")

    data = {name: read_csv_safe(name, path) for name, path in INPUTS.items()}

    table_1 = make_table_1_mechanism_evidence(data)
    table_2 = make_table_2_geographic_systems(data)
    table_3 = make_table_3_holdout(data)
    table_4 = make_table_4_external_validation(data)
    claims = make_publication_claims(data)
    figure_inputs = make_figure_inputs(data)

    write(table_1, "table_1_mechanism_evidence.csv")
    write(table_2, "table_2_geographic_landscape_systems.csv")
    write(table_3, "table_3_geographic_holdout_validation.csv")
    write(table_4, "table_4_external_validation.csv")
    write(claims, "publication_claims_v01.csv")

    for filename, df in figure_inputs.items():
        write(df, filename)

    manifest = make_manifest(data)
    manifest_path = OUTDIR / "publication_package_manifest_v01.txt"
    log.info("Writing manifest: %s", manifest_path)
    manifest_path.write_text(manifest, encoding="utf-8")

    log.info("Done")

    print("\nPublication package created:")
    print(f"  {OUTDIR}")

    print("\nCore tables:")
    print("  table_1_mechanism_evidence.csv")
    print("  table_2_geographic_landscape_systems.csv")
    print("  table_3_geographic_holdout_validation.csv")
    print("  table_4_external_validation.csv")
    print("  publication_claims_v01.csv")

    print("\nFigure inputs:")
    for filename in figure_inputs:
        print(f"  {filename}")


if __name__ == "__main__":
    main()