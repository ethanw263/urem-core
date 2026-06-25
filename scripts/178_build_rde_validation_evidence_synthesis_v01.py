#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_NAME = "178_build_rde_validation_evidence_synthesis_v01"

RESULTS_DIR = Path("data/validation/results")
OUTPUT_DIR = Path("data/validation/synthesis")

OUTPUT_CSV = OUTPUT_DIR / "oregon_validation_evidence_synthesis_v01.csv"
OUTPUT_MD = OUTPUT_DIR / "oregon_validation_evidence_synthesis_v01.md"


DATASETS = [
    {
        "dataset_key": "padus",
        "dataset_name": "Protected Areas",
        "external_summary": RESULTS_DIR / "oregon_padus_external_validation_summary_v01.csv",
        "null_comparison": RESULTS_DIR / "oregon_padus_null_model_comparison_v01.csv",
    },
    {
        "dataset_key": "estuaries",
        "dataset_name": "Estuaries",
        "external_summary": RESULTS_DIR / "oregon_estuary_external_validation_summary_v01.csv",
        "null_comparison": RESULTS_DIR / "oregon_estuary_null_model_comparison_v01.csv",
    },
]


PRIMARY_METRICS = [
    "mean_overlap_pct",
    "median_overlap_pct",
    "mean_nearest_distance_m",
    "pct_within_1km",
    "pct_within_5km",
]


def evidence_class(p_value, z_score, effect_ratio, direction):
    if pd.isna(p_value):
        return "insufficient_evidence"

    if p_value <= 0.01:
        return "strong_statistical_support"

    if p_value <= 0.05:
        return "moderate_statistical_support"

    if p_value <= 0.10:
        return "weak_statistical_support"

    return "not_statistically_supported"


def dataset_overall_class(sub):
    strong = (sub["metric_evidence_class"] == "strong_statistical_support").sum()
    moderate = (sub["metric_evidence_class"] == "moderate_statistical_support").sum()
    weak = (sub["metric_evidence_class"] == "weak_statistical_support").sum()

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

    return "limited_or_no_support"


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for dataset in DATASETS:
        key = dataset["dataset_key"]
        name = dataset["dataset_name"]
        external_path = dataset["external_summary"]
        null_path = dataset["null_comparison"]

        if not external_path.exists():
            print(f"[{SCRIPT_NAME}] Missing external summary: {external_path}")
            continue

        if not null_path.exists():
            print(f"[{SCRIPT_NAME}] Missing null comparison: {null_path}")
            continue

        external = pd.read_csv(external_path)
        null = pd.read_csv(null_path)

        for _, metric_row in null.iterrows():
            metric = metric_row["metric"]

            if metric not in PRIMARY_METRICS:
                continue

            row = {
                "dataset_key": key,
                "dataset_name": name,
                "metric": metric,
                "observed_value": metric_row["observed_value"],
                "null_mean": metric_row["null_mean"],
                "null_std": metric_row["null_std"],
                "z_score": metric_row["z_score"],
                "monte_carlo_p_value": metric_row["monte_carlo_p_value"],
                "direction": metric_row["direction"],
                "effect_ratio_observed_to_null": metric_row["effect_ratio_observed_to_null"],
            }

            row["metric_evidence_class"] = evidence_class(
                row["monte_carlo_p_value"],
                row["z_score"],
                row["effect_ratio_observed_to_null"],
                row["direction"],
            )

            rows.append(row)

    if not rows:
        raise ValueError("No evidence rows found.")

    out = pd.DataFrame(rows)

    dataset_classes = []

    for dataset_name, sub in out.groupby("dataset_name"):
        dataset_classes.append(
            {
                "dataset_name": dataset_name,
                "dataset_overall_support_class": dataset_overall_class(sub),
                "strong_metric_count": (sub["metric_evidence_class"] == "strong_statistical_support").sum(),
                "moderate_metric_count": (sub["metric_evidence_class"] == "moderate_statistical_support").sum(),
                "weak_metric_count": (sub["metric_evidence_class"] == "weak_statistical_support").sum(),
                "tested_metric_count": len(sub),
                "mean_abs_z_score": sub["z_score"].abs().replace([np.inf, -np.inf], np.nan).mean(),
                "mean_p_value": sub["monte_carlo_p_value"].mean(),
            }
        )

    classes_df = pd.DataFrame(dataset_classes)

    out = out.merge(
        classes_df[["dataset_name", "dataset_overall_support_class"]],
        on="dataset_name",
        how="left",
    )

    print()
    print(f"[{SCRIPT_NAME}] Evidence rows: {len(out):,}")

    print()
    print(f"[{SCRIPT_NAME}] Dataset support classes:")
    print(classes_df.to_string(index=False))

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
                "effect_ratio_observed_to_null",
                "metric_evidence_class",
                "dataset_overall_support_class",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"[{SCRIPT_NAME}] Writing CSV: {OUTPUT_CSV}")
    out.to_csv(OUTPUT_CSV, index=False)

    lines = []
    lines.append("# Oregon RDE Validation Evidence Synthesis v01")
    lines.append("")
    lines.append("This report synthesizes independent validation evidence for Oregon RDE transition hotspots.")
    lines.append("")
    lines.append("## Dataset-Level Support")
    lines.append("")
    lines.append(classes_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Metric-Level Evidence")
    lines.append("")
    lines.append(
        out[
            [
                "dataset_name",
                "metric",
                "observed_value",
                "null_mean",
                "monte_carlo_p_value",
                "effect_ratio_observed_to_null",
                "metric_evidence_class",
                "dataset_overall_support_class",
            ]
        ].to_markdown(index=False)
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This synthesis separates descriptive external validation from statistical "
        "null-model validation. Strong support means RDE hotspots show stronger "
        "alignment with an independent validation dataset than expected under "
        "randomized hotspot placement."
    )

    print(f"[{SCRIPT_NAME}] Writing Markdown: {OUTPUT_MD}")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()