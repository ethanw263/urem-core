#!/usr/bin/env python3
"""
102_manual_external_validation_template_v01.py

Purpose
-------
Create a manual external-validation review template for RDE candidates.

Input
-----
data/processed/rde_external_validation_candidates_v01.csv

Outputs
-------
data/processed/rde_manual_external_validation_template_v01.csv
data/processed/rde_manual_external_validation_scoring_guide_v01.csv
data/processed/rde_manual_external_validation_qa_v01.txt
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


SCRIPT_NAME = "102_manual_external_validation_template_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUT_CANDIDATES = PROCESSED / "rde_external_validation_candidates_v01.csv"

OUTPUT_TEMPLATE = PROCESSED / "rde_manual_external_validation_template_v01.csv"
OUTPUT_GUIDE = PROCESSED / "rde_manual_external_validation_scoring_guide_v01.csv"
OUTPUT_QA = PROCESSED / "rde_manual_external_validation_qa_v01.txt"


REVIEW_COLUMNS = {
    "review_status": "",
    "reviewer": "",
    "review_date": "",

    "nearest_named_place": "",
    "nearest_city_or_area": "",
    "landscape_description": "",

    "wikipedia_presence_score_0_3": "",
    "wikipedia_notes": "",

    "google_maps_place_density_score_0_3": "",
    "google_maps_notes": "",

    "alltrails_activity_score_0_3": "",
    "alltrails_notes": "",

    "tourism_media_mentions_score_0_3": "",
    "tourism_media_notes": "",

    "social_media_visibility_score_0_3": "",
    "social_media_notes": "",

    "visual_physical_plausibility_score_0_3": "",
    "visual_plausibility_notes": "",

    "accessibility_reality_score_0_3": "",
    "accessibility_notes": "",

    "external_recognition_score_0_15": "",
    "external_under_recognition_score_0_15": "",
    "validation_result": "",
    "validation_confidence_0_3": "",
    "validation_notes": "",
}


def load_candidates() -> pd.DataFrame:
    if not INPUT_CANDIDATES.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_CANDIDATES}")

    log.info("Reading candidates: %s", INPUT_CANDIDATES)
    df = pd.read_csv(INPUT_CANDIDATES, low_memory=False)

    log.info("Rows: %s | Columns: %s", len(df), len(df.columns))
    return df


def build_template(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "mechanism_region_id",
        "canonical_mechanism",
        "mechanism_class",
        "external_validation_priority_tier",
        "external_validation_priority_score",
        "validation_centroid_lat",
        "validation_centroid_lon",
        "validation_area_km2",
        "cell_count",
        "mean_orthogonalized_rde_v01",
        "mean_mechanism_expression_score_v01",
        "mean_P_orthogonal_v01",
        "mean_O_base_opportunity_v01",
        "mean_T_net_transmission_v01",
        "mean_R_net_under_recognition_v01",
        "mean_observed_recognition_v04",
        "mean_expected_recognition_v06",
        "representative_validated_archetype",
        "theory_readiness_tier",
        "claim_strength",
        "publication_role_final",
        "region_mechanism_stability_005",
        "validation_query_general",
        "validation_query_maps",
        "validation_query_wikipedia",
        "validation_query_alltrails",
        "validation_query_tourism",
    ]

    cols = [c for c in preferred if c in df.columns]
    template = df[cols].copy()

    template = template.sort_values(
        ["external_validation_priority_tier", "external_validation_priority_score"],
        ascending=[True, False],
    )

    tier_order = {"High": 1, "Medium": 2, "Low": 3}
    if "external_validation_priority_tier" in template.columns:
        template["_tier_order"] = template["external_validation_priority_tier"].map(tier_order)
        template = template.sort_values(
            ["_tier_order", "external_validation_priority_score"],
            ascending=[True, False],
        ).drop(columns="_tier_order")

    for col, default in REVIEW_COLUMNS.items():
        template[col] = default

    template["review_instruction"] = (
        "Manually inspect outside recognition proxies. Low external recognition "
        "combined with high RDE priority supports under-recognition."
    )

    return template


def build_scoring_guide() -> pd.DataFrame:
    rows = [
        {
            "field": "wikipedia_presence_score_0_3",
            "score_0": "No obvious Wikipedia page for nearby named landscape/place.",
            "score_1": "Minor page or indirect nearby mention only.",
            "score_2": "Relevant page exists but limited prominence.",
            "score_3": "Clear, prominent Wikipedia presence.",
        },
        {
            "field": "google_maps_place_density_score_0_3",
            "score_0": "Very few named places, parks, trailheads, viewpoints, or attractions.",
            "score_1": "Some minor mapped places.",
            "score_2": "Several mapped recreation/tourism places.",
            "score_3": "Dense named place / attraction presence.",
        },
        {
            "field": "alltrails_activity_score_0_3",
            "score_0": "No nearby trails or very low activity.",
            "score_1": "Few trails or low review activity.",
            "score_2": "Moderate trail/review activity.",
            "score_3": "High trail density or review activity.",
        },
        {
            "field": "tourism_media_mentions_score_0_3",
            "score_0": "Little/no tourism, blog, guide, or media mention.",
            "score_1": "Sparse mentions.",
            "score_2": "Moderate mentions.",
            "score_3": "Frequent tourism/media recognition.",
        },
        {
            "field": "social_media_visibility_score_0_3",
            "score_0": "Little/no visible social media footprint.",
            "score_1": "Sparse footprint.",
            "score_2": "Moderate footprint.",
            "score_3": "High social visibility.",
        },
        {
            "field": "visual_physical_plausibility_score_0_3",
            "score_0": "Not visually plausible / likely artifact.",
            "score_1": "Some plausible landscape value.",
            "score_2": "Clearly plausible scenic/physical value.",
            "score_3": "Highly plausible exceptional landscape.",
        },
        {
            "field": "accessibility_reality_score_0_3",
            "score_0": "Very inaccessible or likely private/restricted.",
            "score_1": "Limited access.",
            "score_2": "Some realistic access.",
            "score_3": "Clearly accessible.",
        },
        {
            "field": "external_recognition_score_0_15",
            "score_0": "Sum of recognition proxies: Wikipedia + Maps + AllTrails + Tourism + Social.",
            "score_1": "Higher means more externally recognized.",
            "score_2": "",
            "score_3": "",
        },
        {
            "field": "external_under_recognition_score_0_15",
            "score_0": "Suggested calculation: 15 - external_recognition_score_0_15.",
            "score_1": "Higher means stronger external under-recognition evidence.",
            "score_2": "",
            "score_3": "",
        },
        {
            "field": "validation_result",
            "score_0": "Use one of: Supports RDE / Mixed / Does Not Support RDE / Artifact / Needs Review.",
            "score_1": "",
            "score_2": "",
            "score_3": "",
        },
    ]

    return pd.DataFrame(rows)


def write_outputs(template: pd.DataFrame, guide: pd.DataFrame) -> None:
    log.info("Writing template: %s", OUTPUT_TEMPLATE)
    template.to_csv(OUTPUT_TEMPLATE, index=False)

    log.info("Writing scoring guide: %s", OUTPUT_GUIDE)
    guide.to_csv(OUTPUT_GUIDE, index=False)

    qa = []
    qa.append("RDE Manual External Validation Template v01 QA")
    qa.append("=" * 55)
    qa.append("")
    qa.append(f"Input candidates: {INPUT_CANDIDATES}")
    qa.append(f"Template rows: {len(template)}")
    qa.append(f"Template columns: {len(template.columns)}")
    qa.append("")
    qa.append("Priority counts:")
    if "external_validation_priority_tier" in template.columns:
        qa.append(template["external_validation_priority_tier"].value_counts().to_string())
    qa.append("")
    qa.append("Mechanism counts:")
    if "canonical_mechanism" in template.columns:
        qa.append(template["canonical_mechanism"].value_counts().to_string())
    qa.append("")
    qa.append("Review fields added:")
    for col in REVIEW_COLUMNS:
        qa.append(f"- {col}")
    qa.append("")
    qa.append("Recommended next action:")
    qa.append(
        "Open the template in Excel/Numbers, review the High-priority candidates first, "
        "and manually fill external recognition scores."
    )

    log.info("Writing QA: %s", OUTPUT_QA)
    OUTPUT_QA.write_text("\n".join(qa), encoding="utf-8")


def main() -> None:
    log.info("Starting Script 102: manual external validation template")

    candidates = load_candidates()
    template = build_template(candidates)
    guide = build_scoring_guide()

    write_outputs(template, guide)

    log.info("Done")

    print("\nManual External Validation Template Created:")
    print(f"Rows: {len(template)}")
    print(f"Columns: {len(template.columns)}")

    print("\nPriority counts:")
    if "external_validation_priority_tier" in template.columns:
        print(template["external_validation_priority_tier"].value_counts().to_string())

    print("\nCreated:")
    print(f"  {OUTPUT_TEMPLATE}")
    print(f"  {OUTPUT_GUIDE}")
    print(f"  {OUTPUT_QA}")


if __name__ == "__main__":
    main()