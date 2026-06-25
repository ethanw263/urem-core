#!/usr/bin/env python3

"""
167_build_rde_validation_dataset_registry_v01.py

Phase II RDE Validation Framework.

Creates a reusable validation dataset registry.

This script does NOT download data yet.
It defines standardized metadata for validation datasets that will later be
used by the RDE Validation Engine.

Outputs:
    data/validation/rde_validation_dataset_registry_v01.csv
    data/validation/rde_validation_dataset_registry_v01.md
"""

from pathlib import Path
import pandas as pd


SCRIPT_NAME = "167_build_rde_validation_dataset_registry_v01"

OUTPUT_DIR = Path("data/validation")

OUTPUT_CSV = OUTPUT_DIR / "rde_validation_dataset_registry_v01.csv"
OUTPUT_MD = OUTPUT_DIR / "rde_validation_dataset_registry_v01.md"


REGISTRY = [
    {
        "dataset_key": "protected_areas",
        "dataset_name": "Protected Areas",
        "category": "conservation",
        "preferred_source": "USGS PAD-US",
        "geometry_type": "polygon",
        "independent_of_model": True,
        "recommended_metrics": "intersects;overlap;distance",
        "default_weight": 0.25,
        "priority": "high",
        "oregon_status": "pending",
        "california_status": "pending",
        "notes": "Primary independent conservation validation layer.",
    },
    {
        "dataset_key": "state_parks",
        "dataset_name": "State Parks",
        "category": "recreation_conservation",
        "preferred_source": "Oregon Parks and Recreation Department / CA State Parks",
        "geometry_type": "polygon",
        "independent_of_model": True,
        "recommended_metrics": "intersects;overlap;distance",
        "default_weight": 0.20,
        "priority": "high",
        "oregon_status": "pending",
        "california_status": "pending",
        "notes": "Tests whether hotspots align with recognized public recreation landscapes.",
    },
    {
        "dataset_key": "national_parks",
        "dataset_name": "National Parks and NPS Units",
        "category": "recreation_conservation",
        "preferred_source": "National Park Service",
        "geometry_type": "polygon",
        "independent_of_model": True,
        "recommended_metrics": "intersects;overlap;distance",
        "default_weight": 0.20,
        "priority": "medium",
        "oregon_status": "pending",
        "california_status": "pending",
        "notes": "Useful but may be sparse in Oregon coastal study area.",
    },
    {
        "dataset_key": "beaches",
        "dataset_name": "Beaches",
        "category": "coastal_landform",
        "preferred_source": "OpenStreetMap / state coastal inventories",
        "geometry_type": "point_or_polygon",
        "independent_of_model": True,
        "recommended_metrics": "distance;density;intersects",
        "default_weight": 0.15,
        "priority": "high",
        "oregon_status": "pending",
        "california_status": "pending",
        "notes": "Tests whether hotspots correspond to accessible coastal recreation landforms.",
    },
    {
        "dataset_key": "viewpoints",
        "dataset_name": "Viewpoints",
        "category": "recognition_access",
        "preferred_source": "OpenStreetMap",
        "geometry_type": "point",
        "independent_of_model": True,
        "recommended_metrics": "distance;density",
        "default_weight": 0.10,
        "priority": "medium",
        "oregon_status": "pending",
        "california_status": "pending",
        "notes": "Tests visible recognition infrastructure, but may partially overlap recognition data logic.",
    },
    {
        "dataset_key": "named_natural_features",
        "dataset_name": "Named Natural Features",
        "category": "geographic_recognition",
        "preferred_source": "USGS GNIS",
        "geometry_type": "point",
        "independent_of_model": True,
        "recommended_metrics": "distance;density",
        "default_weight": 0.15,
        "priority": "high",
        "oregon_status": "pending",
        "california_status": "pending",
        "notes": "Independent named-feature validation for geographic salience.",
    },
    {
        "dataset_key": "river_mouths",
        "dataset_name": "River Mouths",
        "category": "hydrologic_transition",
        "preferred_source": "NHD / derived hydrography",
        "geometry_type": "point",
        "independent_of_model": True,
        "recommended_metrics": "distance;density",
        "default_weight": 0.15,
        "priority": "high",
        "oregon_status": "pending",
        "california_status": "pending",
        "notes": "Tests whether hotspots align with hydrologic/coastal transition environments.",
    },
    {
        "dataset_key": "estuaries",
        "dataset_name": "Estuaries",
        "category": "coastal_hydrology",
        "preferred_source": "NOAA / state estuary datasets",
        "geometry_type": "polygon",
        "independent_of_model": True,
        "recommended_metrics": "intersects;overlap;distance",
        "default_weight": 0.20,
        "priority": "high",
        "oregon_status": "pending",
        "california_status": "pending",
        "notes": "Important independent test based on visual validation results.",
    },
    {
        "dataset_key": "trails",
        "dataset_name": "Trails",
        "category": "access_recreation",
        "preferred_source": "OpenStreetMap / state trail datasets",
        "geometry_type": "line",
        "independent_of_model": False,
        "recommended_metrics": "distance;density",
        "default_weight": 0.05,
        "priority": "low",
        "oregon_status": "pending",
        "california_status": "pending",
        "notes": "Useful context but not fully independent because recognition model may use OSM recreation features.",
    },
    {
        "dataset_key": "campgrounds",
        "dataset_name": "Campgrounds",
        "category": "access_recreation",
        "preferred_source": "OpenStreetMap / Recreation.gov / state datasets",
        "geometry_type": "point_or_polygon",
        "independent_of_model": True,
        "recommended_metrics": "distance;density;intersects",
        "default_weight": 0.10,
        "priority": "medium",
        "oregon_status": "pending",
        "california_status": "pending",
        "notes": "Tests whether hotspots correspond to recreational use nodes.",
    },
]


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    lines = []

    lines.append("# RDE Validation Dataset Registry v01")
    lines.append("")
    lines.append(
        "This registry defines candidate independent validation datasets "
        "for Phase II of the RDE / UREM framework."
    )
    lines.append("")
    lines.append(
        "The registry does not download or validate data. It standardizes "
        "metadata so future validation scripts can register datasets "
        "consistently."
    )
    lines.append("")
    lines.append("## Dataset Registry")
    lines.append("")
    lines.append(df.to_markdown(index=False))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- High-priority datasets should be implemented first."
    )
    lines.append(
        "- Datasets marked `independent_of_model = False` should be used cautiously."
    )
    lines.append(
        "- The validation engine should remain study-area agnostic."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(REGISTRY)

    df = df.sort_values(
        ["priority", "dataset_key"],
        ascending=[True, True],
    ).reset_index(drop=True)

    print()
    print(f"[{SCRIPT_NAME}] Registry rows: {len(df):,}")
    print()
    print(df[[
        "dataset_key",
        "dataset_name",
        "category",
        "preferred_source",
        "priority",
        "independent_of_model",
    ]].to_string(index=False))

    print()
    print(f"[{SCRIPT_NAME}] Writing CSV: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Writing Markdown: {OUTPUT_MD}")
    write_markdown(df, OUTPUT_MD)

    print()
    print(f"[{SCRIPT_NAME}] Done")
    print()
    print("Next step:")
    print("  Script 168 should acquire or standardize the first external validation datasets.")
    print("  Recommended first dataset: USGS PAD-US Protected Areas.")


if __name__ == "__main__":
    main()