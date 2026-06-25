#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

SCRIPT_NAME = "180_build_rde_validation_registry_v01"

OUTPUT_DIR = Path("data/validation")
OUTPUT_CSV = OUTPUT_DIR / "rde_validation_experiment_registry_v01.csv"
OUTPUT_MD = OUTPUT_DIR / "rde_validation_experiment_registry_v01.md"

EXPERIMENTS = [
    {
        "experiment_key": "oregon_padus_external_validation",
        "study_region": "Oregon Coast",
        "dataset_key": "protected_areas",
        "dataset_name": "Protected Areas",
        "validation_type": "external_geographic_validation",
        "validation_layer": "data/validation/standardized/protected_areas_oregon_padus_v01.gpkg",
        "result_file": "data/validation/results/oregon_padus_external_validation_summary_v01.csv",
        "status": "complete",
        "phase": "phase_ii_external_validation",
        "scientific_question": "Do Oregon RDE transition hotspots align with independently mapped protected areas?",
        "notes": "Strong descriptive external validation.",
    },
    {
        "experiment_key": "oregon_padus_null_model_validation",
        "study_region": "Oregon Coast",
        "dataset_key": "protected_areas",
        "dataset_name": "Protected Areas",
        "validation_type": "spatial_null_model_validation",
        "validation_layer": "data/validation/standardized/protected_areas_oregon_padus_v01.gpkg",
        "result_file": "data/validation/results/oregon_padus_null_model_comparison_v01.csv",
        "status": "complete",
        "phase": "phase_ii_statistical_validation",
        "scientific_question": "Are Oregon RDE transition hotspots more strongly embedded in protected areas than random hotspot placements?",
        "notes": "Strong support on overlap intensity; simple intersection not significant.",
    },
    {
        "experiment_key": "oregon_estuaries_external_validation",
        "study_region": "Oregon Coast",
        "dataset_key": "estuaries",
        "dataset_name": "Estuaries",
        "validation_type": "external_geographic_validation",
        "validation_layer": "data/validation/standardized/estuaries_oregon_pmep_v01.gpkg",
        "result_file": "data/validation/results/oregon_estuary_external_validation_summary_v01.csv",
        "status": "complete",
        "phase": "phase_ii_external_validation",
        "scientific_question": "Do Oregon RDE transition hotspots align with independently mapped estuary extents?",
        "notes": "Moderate descriptive overlap; many hotspots are near but not inside estuaries.",
    },
    {
        "experiment_key": "oregon_estuaries_null_model_validation",
        "study_region": "Oregon Coast",
        "dataset_key": "estuaries",
        "dataset_name": "Estuaries",
        "validation_type": "spatial_null_model_validation",
        "validation_layer": "data/validation/standardized/estuaries_oregon_pmep_v01.gpkg",
        "result_file": "data/validation/results/oregon_estuary_null_model_comparison_v01.csv",
        "status": "complete",
        "phase": "phase_ii_statistical_validation",
        "scientific_question": "Are Oregon RDE transition hotspots closer to and more embedded in estuaries than random hotspot placements?",
        "notes": "Strong support on mean overlap, mean distance, and within-5km proximity.",
    },
    {
        "experiment_key": "oregon_validation_evidence_synthesis",
        "study_region": "Oregon Coast",
        "dataset_key": "multi_dataset",
        "dataset_name": "Evidence Synthesis",
        "validation_type": "evidence_synthesis",
        "validation_layer": "",
        "result_file": "data/validation/synthesis/oregon_validation_evidence_synthesis_v01.csv",
        "status": "complete",
        "phase": "phase_ii_evidence_synthesis",
        "scientific_question": "What is the accumulated external and statistical evidence supporting Oregon RDE transition hotspots?",
        "notes": "Initial synthesis includes PAD-US protected areas and PMEP estuaries.",
    },
    {
        "experiment_key": "oregon_state_parks_external_validation",
        "study_region": "Oregon Coast",
        "dataset_key": "state_parks",
        "dataset_name": "State Parks",
        "validation_type": "external_geographic_validation",
        "validation_layer": "",
        "result_file": "",
        "status": "planned",
        "phase": "phase_ii_external_validation",
        "scientific_question": "Do Oregon RDE transition hotspots align with state park landscapes?",
        "notes": "Planned next dataset after validation registry.",
    },
    {
        "experiment_key": "oregon_gnis_named_features_external_validation",
        "study_region": "Oregon Coast",
        "dataset_key": "named_natural_features",
        "dataset_name": "Named Natural Features",
        "validation_type": "external_geographic_validation",
        "validation_layer": "",
        "result_file": "",
        "status": "planned",
        "phase": "phase_ii_external_validation",
        "scientific_question": "Do Oregon RDE transition hotspots align with independently named natural features?",
        "notes": "Use USGS GNIS or equivalent authoritative named-feature dataset.",
    },
    {
        "experiment_key": "oregon_river_mouths_external_validation",
        "study_region": "Oregon Coast",
        "dataset_key": "river_mouths",
        "dataset_name": "River Mouths",
        "validation_type": "external_geographic_validation",
        "validation_layer": "",
        "result_file": "",
        "status": "planned",
        "phase": "phase_ii_external_validation",
        "scientific_question": "Do Oregon RDE transition hotspots align with hydrologic/coastal transition points?",
        "notes": "Likely derived from NHD or equivalent hydrography.",
    },
    {
        "experiment_key": "oregon_beaches_external_validation",
        "study_region": "Oregon Coast",
        "dataset_key": "beaches",
        "dataset_name": "Beaches",
        "validation_type": "external_geographic_validation",
        "validation_layer": "",
        "result_file": "",
        "status": "planned",
        "phase": "phase_ii_external_validation",
        "scientific_question": "Do Oregon RDE transition hotspots align with independently mapped beaches?",
        "notes": "Potentially OSM or state coastal inventory; must consider independence carefully.",
    },
    {
        "experiment_key": "california_validation_framework_backport",
        "study_region": "California Coast",
        "dataset_key": "multi_dataset",
        "dataset_name": "California Cross-State Validation",
        "validation_type": "cross_state_validation",
        "validation_layer": "",
        "result_file": "",
        "status": "planned",
        "phase": "phase_ii_generalization",
        "scientific_question": "Does the same validation framework support California RDE transition structures?",
        "notes": "Do after Oregon validation workflow stabilizes.",
    },
]


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    lines = []

    lines.append("# RDE Validation Experiment Registry v01")
    lines.append("")
    lines.append(
        "This registry tracks Phase II validation experiments for the RDE / UREM framework."
    )
    lines.append("")
    lines.append("It is not a dataset registry. It is an experiment registry.")
    lines.append("")
    lines.append("## Status Counts")
    lines.append("")
    lines.append(df["status"].value_counts().reset_index().rename(
        columns={"index": "status", "status": "count"}
    ).to_markdown(index=False))
    lines.append("")
    lines.append("## Validation Type Counts")
    lines.append("")
    lines.append(df["validation_type"].value_counts().reset_index().rename(
        columns={"index": "validation_type", "validation_type": "count"}
    ).to_markdown(index=False))
    lines.append("")
    lines.append("## Experiments")
    lines.append("")
    lines.append(df.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This registry is intended to function as the scientific control center "
        "for Phase II. Completed experiments represent validation evidence already "
        "generated. Planned experiments represent future evidence required before "
        "cross-state generalization and publication synthesis."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(EXPERIMENTS)

    df["result_exists"] = df["result_file"].apply(
        lambda p: Path(p).exists() if isinstance(p, str) and p else False
    )

    df["validation_layer_exists"] = df["validation_layer"].apply(
        lambda p: Path(p).exists() if isinstance(p, str) and p else False
    )

    df = df.sort_values(
        ["status", "phase", "dataset_key", "validation_type"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    print()
    print(f"[{SCRIPT_NAME}] Experiment rows: {len(df):,}")

    print()
    print(f"[{SCRIPT_NAME}] Status counts:")
    print(df["status"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] Validation type counts:")
    print(df["validation_type"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] Registry preview:")
    print(
        df[
            [
                "experiment_key",
                "study_region",
                "dataset_name",
                "validation_type",
                "status",
                "result_exists",
                "validation_layer_exists",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"[{SCRIPT_NAME}] Writing CSV: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Writing Markdown: {OUTPUT_MD}")
    write_markdown(df, OUTPUT_MD)

    print()
    print(f"[{SCRIPT_NAME}] Done")
    print()
    print("Next step:")
    print("  Use this registry to choose the next planned validation experiment.")
    print("  Recommended next dataset: Oregon State Parks or USGS GNIS Named Natural Features.")


if __name__ == "__main__":
    main()