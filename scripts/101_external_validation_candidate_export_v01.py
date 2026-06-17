#!/usr/bin/env python3
"""
101_external_validation_candidate_export_v01.py

Purpose
-------
Prepare candidate regions for external recognition validation.

This does not scrape external sources yet. It creates the review/export table
needed to validate RDE claims against outside recognition proxies.

Inputs
------
data/processed/mechanism_regions_v01.gpkg
data/processed/rde_core_results_synthesis_v01.csv
data/processed/rde_region_stability_v02.csv
data/processed/rde_background_entry_feature_tests_v01.csv

Outputs
-------
data/processed/rde_external_validation_candidates_v01.csv
data/processed/rde_external_validation_candidates_v01.gpkg
data/processed/rde_external_validation_protocol_v01.csv
data/processed/rde_external_validation_qa_v01.txt
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


SCRIPT_NAME = "101_external_validation_candidate_export_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUT_REGIONS = PROCESSED / "mechanism_regions_v01.gpkg"
INPUT_SYNTHESIS = PROCESSED / "rde_core_results_synthesis_v01.csv"
INPUT_REGION_STABILITY = PROCESSED / "rde_region_stability_v02.csv"
INPUT_ENTRY_TESTS = PROCESSED / "rde_background_entry_feature_tests_v01.csv"

OUTPUT_CSV = PROCESSED / "rde_external_validation_candidates_v01.csv"
OUTPUT_GPKG = PROCESSED / "rde_external_validation_candidates_v01.gpkg"
OUTPUT_PROTOCOL = PROCESSED / "rde_external_validation_protocol_v01.csv"
OUTPUT_QA = PROCESSED / "rde_external_validation_qa_v01.txt"


TOP_N_PER_TIER = {
    "Core Validated Theory": 20,
    "Emerging Theory": 15,
    "Exploratory / Holdout Theory": 8,
}


def canonical_mechanism(x: object) -> str:
    s = str(x).replace(" Candidate", "").strip()

    if "Recognition Inefficiency" in s:
        return "Recognition Inefficiency"
    if "Opportunity Failure" in s:
        return "Opportunity Failure"
    if "Comparative Shadowing" in s or "Recognition Diversion" in s:
        return "Comparative Shadowing"
    return s


def load_inputs() -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in [INPUT_REGIONS, INPUT_SYNTHESIS, INPUT_REGION_STABILITY, INPUT_ENTRY_TESTS]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    log.info("Reading regions: %s", INPUT_REGIONS)
    regions = gpd.read_file(INPUT_REGIONS)

    log.info("Reading synthesis: %s", INPUT_SYNTHESIS)
    synthesis = pd.read_csv(INPUT_SYNTHESIS, low_memory=False)

    log.info("Reading region stability: %s", INPUT_REGION_STABILITY)
    stability = pd.read_csv(INPUT_REGION_STABILITY, low_memory=False)

    log.info("Reading entry tests: %s", INPUT_ENTRY_TESTS)
    entry = pd.read_csv(INPUT_ENTRY_TESTS, low_memory=False)

    return regions, synthesis, stability, entry


def add_geometry_metrics(regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    regions = regions.copy()

    projected = regions.to_crs(3310)
    regions["validation_area_km2"] = projected.geometry.area / 1_000_000

    centroids = projected.geometry.centroid.to_crs(4326)
    regions["validation_centroid_lon"] = centroids.x
    regions["validation_centroid_lat"] = centroids.y

    return regions


def prepare_stability(stability: pd.DataFrame) -> pd.DataFrame:
    s = stability.copy()

    # Use ±5% perturbation as the main publication-friendly robustness level.
    s = s[np.isclose(s["perturbation_level"], 0.05)].copy()

    s = s.rename(
        columns={
            "baseline_mechanism": "canonical_mechanism",
            "baseline_archetype": "archetype",
            "mechanism_stability_rate": "region_mechanism_stability_005",
            "archetype_stability_rate": "region_archetype_stability_005",
        }
    )

    keep = [
        "mechanism_region_id",
        "region_mechanism_stability_005",
        "region_archetype_stability_005",
        "mechanism_stability_class",
        "archetype_stability_class",
        "modal_sim_mechanism",
        "modal_sim_archetype",
    ]

    keep = [c for c in keep if c in s.columns]
    return s[keep]


def prepare_synthesis(synthesis: pd.DataFrame) -> pd.DataFrame:
    syn = synthesis.copy()

    syn["canonical_mechanism"] = syn["mechanism"].map(canonical_mechanism)

    keep = [
        "canonical_mechanism",
        "archetype",
        "region_count",
        "theory_readiness_tier",
        "overall_evidence_score",
        "claim_strength",
        "publication_role_final",
        "recommended_claim_language",
    ]

    keep = [c for c in keep if c in syn.columns]
    return syn[keep]


def infer_region_archetype(regions: gpd.GeoDataFrame) -> pd.Series:
    """
    mechanism_regions_v01 may not contain region-level archetypes.
    For candidate export, we attach mechanism-level synthesis later.
    """
    if "region_archetype_v02" in regions.columns:
        return regions["region_archetype_v02"]
    if "region_archetype" in regions.columns:
        return regions["region_archetype"]
    return pd.Series("Unassigned Region Archetype", index=regions.index)


def build_candidates(
    regions: gpd.GeoDataFrame,
    synthesis: pd.DataFrame,
    stability: pd.DataFrame,
) -> gpd.GeoDataFrame:
    gdf = add_geometry_metrics(regions)

    gdf["canonical_mechanism"] = gdf["mechanism_class"].map(canonical_mechanism)
    gdf["archetype"] = infer_region_archetype(gdf)

    gdf["mechanism_region_id"] = gdf["mechanism_region_id"].astype(str)
    stability = stability.copy()
    stability["mechanism_region_id"] = stability["mechanism_region_id"].astype(str)

    gdf = gdf.merge(stability, on="mechanism_region_id", how="left")

    # Since mechanism_regions_v01 may not have archetypes, merge best mechanism-level
    # claim info by mechanism and keep the top claim for that mechanism.
    syn = prepare_synthesis(synthesis)
    syn_best = (
        syn.sort_values(
            ["canonical_mechanism", "overall_evidence_score"],
            ascending=[True, False],
        )
        .groupby("canonical_mechanism")
        .head(1)
        .rename(columns={"archetype": "representative_validated_archetype"})
    )

    gdf = gdf.merge(syn_best, on="canonical_mechanism", how="left")

    return gdf


def score_validation_priority(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    score_cols = [
        "mechanism_region_priority_score_v01",
        "mean_orthogonalized_rde_v01",
        "mean_mechanism_expression_score_v01",
        "mean_P_orthogonal_v01",
        "mean_R_net_under_recognition_v01",
        "region_mechanism_stability_005",
        "overall_evidence_score",
    ]

    available = [c for c in score_cols if c in gdf.columns]

    priority = pd.Series(0.0, index=gdf.index)

    for c in available:
        s = pd.to_numeric(gdf[c], errors="coerce")
        mn = s.min(skipna=True)
        mx = s.max(skipna=True)

        if pd.notna(mn) and pd.notna(mx) and abs(mx - mn) > 1e-12:
            priority += (s - mn) / (mx - mn)

    if available:
        priority = priority / len(available)

    gdf["external_validation_priority_score"] = priority

    def tier(v):
        if pd.isna(v):
            return "Medium"
        if v >= 0.67:
            return "High"
        if v >= 0.33:
            return "Medium"
        return "Low"

    gdf["external_validation_priority_tier"] = gdf[
        "external_validation_priority_score"
    ].map(tier)

    return gdf


def make_search_queries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    def query(row, source):
        lat = row.get("validation_centroid_lat", np.nan)
        lon = row.get("validation_centroid_lon", np.nan)
        mech = row.get("canonical_mechanism", "")
        rid = row.get("mechanism_region_id", "")

        if source == "general":
            return f'"{lat:.4f}, {lon:.4f}" California landscape recreation'
        if source == "maps":
            return f'{lat:.4f}, {lon:.4f} nearby parks trails viewpoints'
        if source == "wikipedia":
            return f'{lat:.4f} {lon:.4f} California Wikipedia landmark'
        if source == "alltrails":
            return f'{lat:.4f} {lon:.4f} AllTrails'
        if source == "tourism":
            return f'{lat:.4f} {lon:.4f} California tourism scenic'
        return f"region {rid} {mech}"

    for source in ["general", "maps", "wikipedia", "alltrails", "tourism"]:
        gdf[f"validation_query_{source}"] = gdf.apply(
            lambda r: query(r, source),
            axis=1,
        )

    return gdf


def build_protocol(entry_tests: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "validation_proxy": "Wikipedia presence",
            "measure": "Does a named feature/place near the centroid have a Wikipedia page?",
            "expected_for_under_recognition": "Absent or weak page presence",
            "priority": 1,
        },
        {
            "validation_proxy": "Google Maps / place density",
            "measure": "Count named parks, trailheads, viewpoints, attractions near centroid.",
            "expected_for_under_recognition": "Low place density relative to physical/RDE score",
            "priority": 1,
        },
        {
            "validation_proxy": "AllTrails / recreation footprint",
            "measure": "Nearby trail count, review count, rating count.",
            "expected_for_under_recognition": "Low trail/review activity despite high physical potential",
            "priority": 2,
        },
        {
            "validation_proxy": "Tourism/media mentions",
            "measure": "Search results or guide mentions for nearby named landscape.",
            "expected_for_under_recognition": "Few mentions or overshadowed by nearby destinations",
            "priority": 2,
        },
        {
            "validation_proxy": "Manual satellite/QGIS review",
            "measure": "Visual confirmation that region is physically plausible and not artifact/water/noise.",
            "expected_for_under_recognition": "Physically plausible landscape with low observed recognition",
            "priority": 1,
        },
        {
            "validation_proxy": "Negative control comparison",
            "measure": "Compare against famous/high-recognition coastal regions.",
            "expected_for_under_recognition": "RDE candidates should show lower external recognition than controls",
            "priority": 3,
        },
    ]

    protocol = pd.DataFrame(rows)

    if len(entry_tests) > 0:
        strong_features = entry_tests[
            entry_tests["entry_evidence_class"].astype(str).str.contains("Strong", na=False)
        ]["feature"].tolist()

        protocol["background_entry_validation_context"] = (
            "Strong entry evidence found for: " + ", ".join(strong_features)
        )

    return protocol


def trim_export_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    preferred = [
        "mechanism_class",
        "canonical_mechanism",
        "mechanism_region_id",
        "mechanism_region_area_km2",
        "cell_count",
        "validation_area_km2",
        "validation_centroid_lon",
        "validation_centroid_lat",
        "mechanism_region_priority_score_v01",
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
        "overall_evidence_score",
        "claim_strength",
        "publication_role_final",
        "recommended_claim_language",
        "region_mechanism_stability_005",
        "region_archetype_stability_005",
        "external_validation_priority_score",
        "external_validation_priority_tier",
        "validation_query_general",
        "validation_query_maps",
        "validation_query_wikipedia",
        "validation_query_alltrails",
        "validation_query_tourism",
        "geometry",
    ]

    cols = [c for c in preferred if c in gdf.columns]
    return gdf[cols].copy()


def write_outputs(gdf: gpd.GeoDataFrame, protocol: pd.DataFrame) -> None:
    export = trim_export_columns(gdf)

    log.info("Writing external validation CSV: %s", OUTPUT_CSV)
    pd.DataFrame(export.drop(columns="geometry", errors="ignore")).to_csv(
        OUTPUT_CSV,
        index=False,
    )

    log.info("Writing external validation GPKG: %s", OUTPUT_GPKG)
    export.to_file(
        OUTPUT_GPKG,
        layer="rde_external_validation_candidates_v01",
        driver="GPKG",
    )

    log.info("Writing validation protocol: %s", OUTPUT_PROTOCOL)
    protocol.to_csv(OUTPUT_PROTOCOL, index=False)

    qa = []
    qa.append("RDE External Validation Candidate Export v01 QA")
    qa.append("=" * 55)
    qa.append("")
    qa.append(f"Candidate regions: {len(export)}")
    qa.append("")
    qa.append("Priority counts:")
    qa.append(export["external_validation_priority_tier"].value_counts().to_string())
    qa.append("")
    qa.append("Mechanism counts:")
    qa.append(export["canonical_mechanism"].value_counts().to_string())
    qa.append("")
    qa.append("Top 20 candidates:")
    qa.append(
        export.sort_values("external_validation_priority_score", ascending=False)
        .head(20)
        .drop(columns="geometry", errors="ignore")
        .to_string(index=False)
    )
    qa.append("")
    qa.append("Validation protocol:")
    qa.append(protocol.to_string(index=False))

    log.info("Writing QA: %s", OUTPUT_QA)
    OUTPUT_QA.write_text("\n".join(qa), encoding="utf-8")


def main() -> None:
    log.info("Starting Script 101: external validation candidate export")

    regions, synthesis, stability, entry_tests = load_inputs()

    stability_prepared = prepare_stability(stability)
    candidates = build_candidates(regions, synthesis, stability_prepared)
    candidates = score_validation_priority(candidates)
    candidates = make_search_queries(candidates)

    protocol = build_protocol(entry_tests)

    write_outputs(candidates, protocol)

    log.info("Done")

    export = trim_export_columns(candidates)

    print("\nExternal Validation Candidate Export Summary:")
    print(f"Candidates: {len(export)}")
    print("\nPriority counts:")
    print(export["external_validation_priority_tier"].value_counts().to_string())
    print("\nMechanism counts:")
    print(export["canonical_mechanism"].value_counts().to_string())
    print("\nTop 15 candidates:")
    print(
        export.sort_values("external_validation_priority_score", ascending=False)
        .head(15)[
            [
                "mechanism_region_id",
                "canonical_mechanism",
                "external_validation_priority_score",
                "external_validation_priority_tier",
                "validation_centroid_lat",
                "validation_centroid_lon",
            ]
        ].to_string(index=False)
    )

    print("\nCreated:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_GPKG}")
    print(f"  {OUTPUT_PROTOCOL}")
    print(f"  {OUTPUT_QA}")


if __name__ == "__main__":
    main()