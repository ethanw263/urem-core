#!/usr/bin/env python3
"""
103_geographic_landscape_interpretation_v01.py

Purpose
-------
Assign interpretable geographic landscape contexts to RDE mechanism regions.

This script translates mechanism-region outputs into readable geographic
landscape archetypes such as:

- Offshore Island Recognition Failure
- Remote Northern Coast Recognition Failure
- Lost Coast / Rugged Coastal Mountain Landscape
- Southern Island / Military-Island Recognition Gap
- Remote Headland / Coastal Upland Landscape
- General Coastal Recognition Disequilibrium Landscape

Inputs
------
data/processed/rde_external_validation_candidates_v01.csv
data/processed/mechanism_regions_v01.gpkg

Outputs
-------
data/processed/rde_geographic_landscape_interpretation_v01.csv
data/processed/rde_geographic_landscape_interpretation_v01.gpkg
data/processed/rde_geographic_landscape_summary_v01.csv
data/processed/rde_geographic_landscape_qa_v01.txt
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


SCRIPT_NAME = "103_geographic_landscape_interpretation_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUT_CANDIDATES = PROCESSED / "rde_external_validation_candidates_v01.csv"
INPUT_REGIONS = PROCESSED / "mechanism_regions_v01.gpkg"

OUTPUT_CSV = PROCESSED / "rde_geographic_landscape_interpretation_v01.csv"
OUTPUT_GPKG = PROCESSED / "rde_geographic_landscape_interpretation_v01.gpkg"
OUTPUT_SUMMARY = PROCESSED / "rde_geographic_landscape_summary_v01.csv"
OUTPUT_QA = PROCESSED / "rde_geographic_landscape_qa_v01.txt"


def load_inputs() -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    if not INPUT_CANDIDATES.exists():
        raise FileNotFoundError(f"Missing candidate input: {INPUT_CANDIDATES}")

    if not INPUT_REGIONS.exists():
        raise FileNotFoundError(f"Missing region geometry input: {INPUT_REGIONS}")

    log.info("Reading candidates: %s", INPUT_CANDIDATES)
    candidates = pd.read_csv(INPUT_CANDIDATES, low_memory=False)

    log.info("Reading regions: %s", INPUT_REGIONS)
    regions = gpd.read_file(INPUT_REGIONS)

    candidates["mechanism_region_id"] = candidates["mechanism_region_id"].astype(str)
    regions["mechanism_region_id"] = regions["mechanism_region_id"].astype(str)

    return candidates, regions


def coastal_macro_zone(lat: float, lon: float) -> str:
    if pd.isna(lat) or pd.isna(lon):
        return "Unknown Coastal Zone"

    if lat >= 40.7:
        return "Far Northern California Coast"
    if 39.0 <= lat < 40.7:
        return "Mendocino / Lost Coast Transition"
    if 37.0 <= lat < 39.0:
        return "North-Central Coast"
    if 35.0 <= lat < 37.0:
        return "Big Sur / Central Coast"
    if 33.7 <= lat < 35.0 and lon <= -119.0:
        return "Channel Islands / Offshore Southern California"
    if 32.5 <= lat < 34.2:
        return "Southern California Coast / Offshore Islands"

    return "General California Coastal Zone"


def island_context(lat: float, lon: float) -> str:
    if pd.isna(lat) or pd.isna(lon):
        return "Unknown Island Context"

    # Broad coordinate-based recognition of Channel Islands / offshore systems.
    if 32.75 <= lat <= 33.10 and -118.70 <= lon <= -118.25:
        return "Likely San Clemente Island / Offshore Island Context"

    if 33.20 <= lat <= 33.55 and -118.65 <= lon <= -118.20:
        return "Likely Santa Catalina Island Context"

    if 33.80 <= lat <= 34.15 and -120.05 <= lon <= -119.40:
        return "Likely Northern Channel Islands Context"

    if 33.15 <= lat <= 33.45 and -119.75 <= lon <= -119.20:
        return "Likely San Nicolas / Offshore Island Context"

    if lon <= -119.0 and lat < 34.5:
        return "Possible Offshore Island / Marine-Coastal Context"

    return "Mainland / Non-Island Context"


def remoteness_context(lat: float, lon: float, opportunity: float, transmission: float) -> str:
    if pd.isna(lat) or pd.isna(lon):
        return "Unknown Remoteness Context"

    if lat >= 39.0 and lon <= -123.5:
        return "Remote Northern Coastal System"

    if opportunity <= 0.35 and transmission >= 0.60:
        return "Low-Opportunity but Visible Landscape"

    if opportunity <= 0.35:
        return "Opportunity-Constrained Remote Landscape"

    if transmission >= 0.70 and opportunity >= 0.45:
        return "Transmission-Enabled but Under-Recognized Landscape"

    return "Moderate Accessibility / Mixed Remoteness"


def scale_context(area_km2: float, cell_count: float) -> str:
    if pd.isna(area_km2):
        area_km2 = 0

    if area_km2 >= 150 or cell_count >= 50:
        return "Large Landscape System"
    if area_km2 >= 50 or cell_count >= 15:
        return "Regional Landscape Cluster"
    if area_km2 >= 15 or cell_count >= 5:
        return "Localized Landscape Cluster"

    return "Small / Isolated Landscape Patch"


def physical_context(p: float, t: float, r: float) -> str:
    if p >= 0.85 and r >= 0.65:
        return "High Physical Potential / High Recognition Deficit"
    if p >= 0.75 and t >= 0.65:
        return "High Physical Potential / High Transmission"
    if p >= 0.75:
        return "High Physical Potential Landscape"
    if r >= 0.70:
        return "Recognition Deficit Dominant Landscape"
    return "Moderate Physical-Disequilibrium Landscape"


def assign_geographic_landscape_type(row: pd.Series) -> str:
    lat = row.get("validation_centroid_lat", np.nan)
    lon = row.get("validation_centroid_lon", np.nan)
    mech = str(row.get("canonical_mechanism", ""))
    p = float(row.get("mean_P_orthogonal_v01", np.nan))
    o = float(row.get("mean_O_base_opportunity_v01", np.nan))
    t = float(row.get("mean_T_net_transmission_v01", np.nan))
    r = float(row.get("mean_R_net_under_recognition_v01", np.nan))
    area = float(row.get("validation_area_km2", np.nan))
    cell_count = float(row.get("cell_count", np.nan))

    macro = coastal_macro_zone(lat, lon)
    island = island_context(lat, lon)
    remote = remoteness_context(lat, lon, o, t)
    scale = scale_context(area, cell_count)

    if island != "Mainland / Non-Island Context" and island != "Unknown Island Context":
        if "Recognition Inefficiency" in mech:
            return "Offshore Island Recognition Inefficiency Landscape"
        if "Opportunity Failure" in mech:
            return "Offshore Island Opportunity-Constrained Landscape"
        if "Comparative" in mech:
            return "Offshore Island Recognition Diversion Landscape"
        return "Offshore Island RDE Landscape"

    if "Far Northern" in macro or "Mendocino" in macro:
        if "Remote" in remote:
            return "Remote Northern Coast Recognition Disequilibrium Landscape"
        return "Northern Coastal Upland Recognition Disequilibrium Landscape"

    if "Big Sur" in macro:
        if "Comparative" in mech:
            return "Shadowed Central Coast Recognition Landscape"
        return "Central Coast Rugged Landscape"

    if "Southern California" in macro:
        if "Recognition Inefficiency" in mech:
            return "Southern Coastal Recognition Inefficiency Landscape"
        return "Southern Coastal Recognition Disequilibrium Landscape"

    if scale == "Large Landscape System":
        return "Large-System Coastal Recognition Disequilibrium Landscape"

    if p >= 0.85 and r >= 0.65:
        return "High-Potential Hidden Landscape System"

    return "General Coastal Recognition Disequilibrium Landscape"


def assign_public_interpretation(row: pd.Series) -> str:
    gtype = row["geographic_landscape_type_v01"]
    mech = row.get("canonical_mechanism", "Unknown mechanism")

    if "Offshore Island" in gtype:
        return (
            "The region appears to represent an offshore island or marine-coastal "
            "landscape where physical potential is high but public recognition may be "
            "limited by access, governance, military/restricted land, or destination filtering."
        )

    if "Remote Northern Coast" in gtype:
        return (
            "The region appears to represent a remote northern coastal landscape where "
            "rugged terrain and distance from major tourism circuits may suppress recognition "
            "despite strong physical potential."
        )

    if "Northern Coastal Upland" in gtype:
        return (
            "The region appears to represent a northern coastal upland/headland system with "
            "moderate access but recognition below what physical and transmission conditions suggest."
        )

    if "Shadowed" in gtype:
        return (
            "The region may be influenced by nearby better-known coastal destinations, making it "
            "a candidate for comparative recognition diversion."
        )

    if "High-Potential Hidden" in gtype:
        return (
            "The region has a strong physical/RDE signature and should be prioritized for external "
            "validation as a potential hidden exceptional landscape."
        )

    return (
        f"The region is assigned to a general coastal RDE landscape class under the "
        f"{mech} mechanism and should be reviewed with external recognition proxies."
    )


def build_interpretation(candidates: pd.DataFrame, regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = regions.merge(
        candidates,
        on="mechanism_region_id",
        how="left",
        suffixes=("", "_candidate"),
    )

    gdf["geographic_macro_zone_v01"] = gdf.apply(
        lambda r: coastal_macro_zone(
            r.get("validation_centroid_lat", np.nan),
            r.get("validation_centroid_lon", np.nan),
        ),
        axis=1,
    )

    gdf["island_context_v01"] = gdf.apply(
        lambda r: island_context(
            r.get("validation_centroid_lat", np.nan),
            r.get("validation_centroid_lon", np.nan),
        ),
        axis=1,
    )

    gdf["remoteness_context_v01"] = gdf.apply(
        lambda r: remoteness_context(
            r.get("validation_centroid_lat", np.nan),
            r.get("validation_centroid_lon", np.nan),
            float(r.get("mean_O_base_opportunity_v01", np.nan)),
            float(r.get("mean_T_net_transmission_v01", np.nan)),
        ),
        axis=1,
    )

    gdf["scale_context_v01"] = gdf.apply(
        lambda r: scale_context(
            float(r.get("validation_area_km2", np.nan)),
            float(r.get("cell_count", np.nan)),
        ),
        axis=1,
    )

    gdf["physical_context_v01"] = gdf.apply(
        lambda r: physical_context(
            float(r.get("mean_P_orthogonal_v01", np.nan)),
            float(r.get("mean_T_net_transmission_v01", np.nan)),
            float(r.get("mean_R_net_under_recognition_v01", np.nan)),
        ),
        axis=1,
    )

    gdf["geographic_landscape_type_v01"] = gdf.apply(assign_geographic_landscape_type, axis=1)
    gdf["geographic_interpretation_v01"] = gdf.apply(assign_public_interpretation, axis=1)

    return gdf


def make_summary(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    summary = (
        gdf.groupby(["canonical_mechanism", "geographic_landscape_type_v01"], dropna=False)
        .agg(
            region_count=("mechanism_region_id", "count"),
            mean_priority=("external_validation_priority_score", "mean"),
            mean_physical=("mean_P_orthogonal_v01", "mean"),
            mean_opportunity=("mean_O_base_opportunity_v01", "mean"),
            mean_transmission=("mean_T_net_transmission_v01", "mean"),
            mean_under_recognition=("mean_R_net_under_recognition_v01", "mean"),
            mean_area_km2=("validation_area_km2", "mean"),
        )
        .reset_index()
        .sort_values(
            ["region_count", "mean_priority"],
            ascending=[False, False],
        )
    )

    return summary


def trim_for_export(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    keep = [
        "mechanism_region_id",
        "canonical_mechanism",
        "mechanism_class",
        "geographic_landscape_type_v01",
        "geographic_macro_zone_v01",
        "island_context_v01",
        "remoteness_context_v01",
        "scale_context_v01",
        "physical_context_v01",
        "geographic_interpretation_v01",
        "external_validation_priority_score",
        "external_validation_priority_tier",
        "validation_centroid_lat",
        "validation_centroid_lon",
        "validation_area_km2",
        "cell_count",
        "mean_orthogonalized_rde_v01",
        "mean_P_orthogonal_v01",
        "mean_O_base_opportunity_v01",
        "mean_T_net_transmission_v01",
        "mean_R_net_under_recognition_v01",
        "mean_observed_recognition_v04",
        "mean_expected_recognition_v06",
        "region_mechanism_stability_005",
        "representative_validated_archetype",
        "claim_strength",
        "publication_role_final",
        "geometry",
    ]

    keep = [c for c in keep if c in gdf.columns]
    return gdf[keep].copy()


def write_outputs(gdf: gpd.GeoDataFrame, summary: pd.DataFrame) -> None:
    export = trim_for_export(gdf)

    log.info("Writing CSV: %s", OUTPUT_CSV)
    pd.DataFrame(export.drop(columns="geometry", errors="ignore")).to_csv(OUTPUT_CSV, index=False)

    log.info("Writing GPKG: %s", OUTPUT_GPKG)
    export.to_file(
        OUTPUT_GPKG,
        layer="rde_geographic_landscape_interpretation_v01",
        driver="GPKG",
    )

    log.info("Writing summary: %s", OUTPUT_SUMMARY)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    qa = []
    qa.append("RDE Geographic Landscape Interpretation v01 QA")
    qa.append("=" * 55)
    qa.append("")
    qa.append(f"Regions interpreted: {len(export)}")
    qa.append("")
    qa.append("Landscape type counts:")
    qa.append(export["geographic_landscape_type_v01"].value_counts().to_string())
    qa.append("")
    qa.append("Macro zone counts:")
    qa.append(export["geographic_macro_zone_v01"].value_counts().to_string())
    qa.append("")
    qa.append("Mechanism counts:")
    qa.append(export["canonical_mechanism"].value_counts().to_string())
    qa.append("")
    qa.append("Summary:")
    qa.append(summary.to_string(index=False))

    log.info("Writing QA: %s", OUTPUT_QA)
    OUTPUT_QA.write_text("\n".join(qa), encoding="utf-8")


def main() -> None:
    log.info("Starting Script 103: geographic landscape interpretation")

    candidates, regions = load_inputs()
    interpreted = build_interpretation(candidates, regions)
    summary = make_summary(interpreted)

    write_outputs(interpreted, summary)

    log.info("Done")

    export = trim_for_export(interpreted)

    print("\nGeographic Landscape Interpretation Summary:")
    print("\nLandscape type counts:")
    print(export["geographic_landscape_type_v01"].value_counts().to_string())

    print("\nMacro zone counts:")
    print(export["geographic_macro_zone_v01"].value_counts().to_string())

    print("\nTop summary:")
    print(summary.head(20).to_string(index=False))

    print("\nCreated:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_GPKG}")
    print(f"  {OUTPUT_SUMMARY}")
    print(f"  {OUTPUT_QA}")


if __name__ == "__main__":
    main()