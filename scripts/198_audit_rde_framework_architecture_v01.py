#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

SCRIPT_NAME = "198_audit_rde_framework_architecture_v01"

ROOT = Path(".")
OUTPUT_DIR = Path("data/validation/framework_audit")
OUTPUT_CSV = OUTPUT_DIR / "rde_framework_architecture_audit_v01.csv"
OUTPUT_MD = OUTPUT_DIR / "rde_framework_architecture_audit_v01.md"


SCAN_TARGETS = [
    "src/validation",
    "scripts",
    "data/validation",
]


TEMPORARY_SCRIPT_KEYWORDS = [
    "build_rde_validation_engine",
    "build_rde_null_model",
    "build_rde_experiment_framework",
    "build_rde_experiment_synthesis_engine",
    "build_rde_experiment_discovery_engine",
    "register_existing",
    "update_rde_validation_registry",
    "consolidate_rde_experiment_registry",
    "add_rde_experiment_provenance",
]


def classify_path(path: Path) -> str:
    s = str(path)

    if s.startswith("src/validation"):
        return "framework_module"

    if s.startswith("scripts"):
        name = path.name.lower()

        if any(k in name for k in TEMPORARY_SCRIPT_KEYWORDS):
            return "temporary_builder_or_migration_script"

        if "standardize" in name:
            return "dataset_standardization_script"

        if "validation" in name or "experiment" in name or "synthesis" in name:
            return "workflow_driver_script"

        return "general_script"

    if s.startswith("data/validation/standardized"):
        return "standardized_validation_data"

    if s.startswith("data/validation/results"):
        return "validation_result_output"

    if s.startswith("data/validation/synthesis"):
        return "synthesis_output"

    if s.startswith("data/validation"):
        return "validation_metadata_or_registry"

    return "other"


def recommendation_for(path: Path, category: str) -> str:
    name = path.name

    if category == "temporary_builder_or_migration_script":
        return "Keep for history for now; later archive or replace with stable framework module."

    if category == "framework_module":
        return "Keep; candidate for consolidation into final src architecture."

    if category == "workflow_driver_script":
        return "Eventually replace with unified CLI/config-driven runner."

    if category == "dataset_standardization_script":
        return "Keep until generic dataset standardization engine exists."

    if category == "standardized_validation_data":
        return "Keep if small/useful for reproducibility; review Git storage later."

    if category == "validation_result_output":
        return "Keep current validated outputs; use canonical registry for discovery."

    if category == "synthesis_output":
        return "Keep latest synthesis; older versions may later move to archive."

    if category == "validation_metadata_or_registry":
        return "Keep canonical registry/provenance files; archive superseded registries later."

    return "Review later."


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for target in SCAN_TARGETS:
        base = ROOT / target

        if not base.exists():
            continue

        for path in sorted(base.rglob("*")):
            if path.is_dir():
                continue

            rel = path.relative_to(ROOT)
            category = classify_path(rel)

            rows.append({
                "path": str(rel),
                "filename": path.name,
                "suffix": path.suffix,
                "size_kb": round(path.stat().st_size / 1024, 2),
                "category": category,
                "recommendation": recommendation_for(rel, category),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No files found for architecture audit.")

    category_counts = (
        df["category"]
        .value_counts()
        .reset_index()
        .rename(columns={"category": "count", "index": "category"})
    )

    print()
    print(f"[{SCRIPT_NAME}] Files audited: {len(df):,}")
    print()
    print(f"[{SCRIPT_NAME}] Category counts:")
    print(category_counts.to_string(index=False))

    print()
    print(f"[{SCRIPT_NAME}] Temporary/builder scripts:")
    temp = df[df["category"] == "temporary_builder_or_migration_script"]
    if temp.empty:
        print("None")
    else:
        print(temp[["path", "recommendation"]].to_string(index=False))

    df.to_csv(OUTPUT_CSV, index=False)

    lines = []
    lines.append("# RDE Framework Architecture Audit v01")
    lines.append("")
    lines.append("This audit inventories the current validation/framework files before Phase IX consolidation.")
    lines.append("")
    lines.append("## Category Counts")
    lines.append("")
    lines.append(category_counts.to_markdown(index=False))
    lines.append("")
    lines.append("## Temporary / Builder / Migration Scripts")
    lines.append("")
    if temp.empty:
        lines.append("None found.")
    else:
        lines.append(temp[["path", "recommendation"]].to_markdown(index=False))
    lines.append("")
    lines.append("## Full Audit")
    lines.append("")
    lines.append(df.to_markdown(index=False))
    lines.append("")
    lines.append("## Consolidation Notes")
    lines.append("")
    lines.append("- Do not delete anything yet.")
    lines.append("- Use this audit to identify modules that should become permanent framework components.")
    lines.append("- Builder scripts may later be archived once equivalent src modules are stable.")
    lines.append("- Workflow driver scripts should eventually be replaced by a unified config-driven CLI.")
    lines.append("- Dataset standardization scripts should remain until a generic standardization engine exists.")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"[{SCRIPT_NAME}] Wrote CSV: {OUTPUT_CSV}")
    print(f"[{SCRIPT_NAME}] Wrote Markdown: {OUTPUT_MD}")
    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()