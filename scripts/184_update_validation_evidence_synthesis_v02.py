#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_NAME = "184_update_validation_evidence_synthesis_v02"

RESULTS_DIR = Path("data/validation/results")
OUTPUT_DIR = Path("data/validation/synthesis")

OUTPUT_CSV = OUTPUT_DIR / "oregon_validation_evidence_synthesis_v02.csv"
OUTPUT_MD = OUTPUT_DIR / "oregon_validation_evidence_synthesis_v02.md"

DATASETS = [
    {
        "dataset_key": "padus",
        "dataset_name": "Protected Areas",
        "hypothesis_type": "conservation_landscape_validation",
        "null_comparison_options": [
            RESULTS_DIR / "oregon_padus_null_model_comparison_v01.csv",
            RESULTS_DIR / "oregon_protected_areas_null_model_comparison_v01.csv",
        ],
        "interpretation_note": "Tests whether hotspots are embedded in protected/conservation landscapes.",
    },
    {
        "dataset_key": "estuaries",
        "dataset_name": "Estuaries",
        "hypothesis_type": "coastal_transition_validation",
        "null_comparison_options": [
            RESULTS_DIR / "oregon_estuary_null_model_comparison_v01.csv",
            RESULTS_DIR / "oregon_estuaries_null_model_comparison_v01.csv",
        ],
        "interpretation_note": "Tests whether hotspots are associated with estuarine transition environments.",
    },
    {
        "dataset_key": "gnis_aggregate",
        "dataset_name": "GNIS Named Natural Features",
        "hypothesis_type": "aggregate_named_feature_validation",
        "null_comparison_options": [
            RESULTS_DIR / "oregon_gnis_named_features_null_model_comparison_v01.csv",
        ],
        "interpretation_note": "Aggregate GNIS is heterogeneous and dominated by broad feature classes; subgroup testing is required.",
    },
]

PRIMARY_METRICS = [
    "mean_overlap_pct",
    "median_overlap_pct",
    "mean_nearest_distance_m",
    "pct_within_1km",
    "pct_within_5km",
]


def metric_class(p_value, z_score, direction):
    if pd.isna(p_value):
        return "insufficient_evidence"

    if direction == "lower_is_better":
        supportive = z_score < 0
    else:
        supportive = z_score > 0

    if not supportive:
        if p_value >= 0.95:
            return "inverse_or_contradictory_evidence"
        return "not_supported"

    if p_value <= 0.01:
        return "strong_support"
    if p_value <= 0.05:
        return "moderate_support"
    if p_value <= 0.10:
        return "weak_support"

    return "directionally_supportive_not_significant"


def dataset_class(sub):
    strong = (sub["metric_evidence_class"] == "strong_support").sum()
    moderate = (sub["metric_evidence_class"] == "moderate_support").sum()
    weak = (sub["metric_evidence_class"] == "weak_support").sum()
    inverse = (sub["metric_evidence_class"] == "inverse_or_contradictory_evidence").sum()

    if inverse >= 3:
        return "not_supported_or_inverse"
    if strong >= 2:
        return "strong_overall_support"
    if strong >= 1 and moderate >= 1:
        return "strong_overall_support"
    if strong >= 1:
        return "moderate_overall_support"
    if moderate >= 2:
        return "moderate_overall_support"
    if moderate >= 1 or weak >= 2:
        return "weak_overall_support"
    if inverse >= 1:
        return "mixed_or_heterogeneous"
    return "limited_or_no_support"


def main():
    print(f"[{SCRIPT_NAME}] Starting")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for ds in DATASETS:
        path = None
        for candidate in ds["null_comparison_options"]:
            if candidate.exists():
                path = candidate
                break

        if path is None:
            print(f"[{SCRIPT_NAME}] Missing all options for: {ds['dataset_name']}")
            continue
        null = pd.read_csv(path)

        for _, r in null.iterrows():
            metric = r["metric"]

            if metric not in PRIMARY_METRICS:
                continue

            row = {
                "dataset_key": ds["dataset_key"],
                "dataset_name": ds["dataset_name"],
                "hypothesis_type": ds["hypothesis_type"],
                "metric": metric,
                "observed_value": r["observed_value"],
                "null_mean": r["null_mean"],
                "null_std": r["null_std"],
                "z_score": r["z_score"],
                "monte_carlo_p_value": r["monte_carlo_p_value"],
                "direction": r["direction"],
                "effect_ratio_observed_to_null": r["effect_ratio_observed_to_null"],
                "interpretation_note": ds["interpretation_note"],
            }

            row["metric_evidence_class"] = metric_class(
                row["monte_carlo_p_value"],
                row["z_score"],
                row["direction"],
            )

            rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        raise ValueError("No evidence rows found.")

    class_rows = []

    for dataset_name, sub in out.groupby("dataset_name"):
        class_rows.append({
            "dataset_name": dataset_name,
            "dataset_overall_support_class": dataset_class(sub),
            "strong_support_metrics": (sub["metric_evidence_class"] == "strong_support").sum(),
            "moderate_support_metrics": (sub["metric_evidence_class"] == "moderate_support").sum(),
            "weak_support_metrics": (sub["metric_evidence_class"] == "weak_support").sum(),
            "inverse_or_contradictory_metrics": (sub["metric_evidence_class"] == "inverse_or_contradictory_evidence").sum(),
            "tested_metric_count": len(sub),
            "mean_abs_z_score": sub["z_score"].abs().replace([np.inf, -np.inf], np.nan).mean(),
        })

    classes = pd.DataFrame(class_rows)

    out = out.merge(
        classes[["dataset_name", "dataset_overall_support_class"]],
        on="dataset_name",
        how="left",
    )

    print()
    print(f"[{SCRIPT_NAME}] Dataset support classes:")
    print(classes.to_string(index=False))

    print()
    print(f"[{SCRIPT_NAME}] Evidence table:")
    print(
        out[
            [
                "dataset_name",
                "metric",
                "observed_value",
                "null_mean",
                "z_score",
                "monte_carlo_p_value",
                "metric_evidence_class",
                "dataset_overall_support_class",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"[{SCRIPT_NAME}] Writing CSV: {OUTPUT_CSV}")
    out.to_csv(OUTPUT_CSV, index=False)

    lines = []
    lines.append("# Oregon RDE Validation Evidence Synthesis v02")
    lines.append("")
    lines.append("This synthesis incorporates supportive, mixed, and inverse validation evidence.")
    lines.append("")
    lines.append("## Dataset-Level Support")
    lines.append("")
    lines.append(classes.to_markdown(index=False))
    lines.append("")
    lines.append("## Metric-Level Evidence")
    lines.append("")
    lines.append(
        out[
            [
                "dataset_name",
                "hypothesis_type",
                "metric",
                "observed_value",
                "null_mean",
                "monte_carlo_p_value",
                "metric_evidence_class",
                "dataset_overall_support_class",
            ]
        ].to_markdown(index=False)
    )
    lines.append("")
    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append("- Protected Areas and Estuaries test independent geographic hypotheses.")
    lines.append("- Aggregate GNIS is heterogeneous and should be decomposed into subgroups.")
    lines.append("- Non-supportive evidence is retained because it helps refine the RDE theory.")

    print(f"[{SCRIPT_NAME}] Writing Markdown: {OUTPUT_MD}")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()