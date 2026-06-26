#!/usr/bin/env python3

from pathlib import Path

SCRIPT_NAME = "190_build_rde_experiment_synthesis_engine_v01"

OUTPUT_PATH = Path("src/validation/experiments/synthesis.py")

CONTENT = r'''from pathlib import Path
import pandas as pd
import numpy as np


PRIMARY_METRICS = [
    "mean_overlap_pct",
    "median_overlap_pct",
    "mean_nearest_distance_m",
    "median_nearest_distance_m",
    "pct_within_1km",
    "pct_within_5km",
]


def classify_metric(row):
    p = row.get("monte_carlo_p_value")
    z = row.get("z_score")
    direction = row.get("direction")

    if pd.isna(p) or pd.isna(z):
        return "insufficient_evidence"

    if direction == "lower_is_better":
        supportive = z < 0
    else:
        supportive = z > 0

    if not supportive:
        if p >= 0.95:
            return "inverse_or_contradictory_evidence"
        return "not_supported"

    if p <= 0.01:
        return "strong_support"
    if p <= 0.05:
        return "moderate_support"
    if p <= 0.10:
        return "weak_support"

    return "directionally_supportive_not_significant"


def classify_experiment(metric_df):
    strong = (metric_df["metric_evidence_class"] == "strong_support").sum()
    moderate = (metric_df["metric_evidence_class"] == "moderate_support").sum()
    weak = (metric_df["metric_evidence_class"] == "weak_support").sum()
    inverse = (metric_df["metric_evidence_class"] == "inverse_or_contradictory_evidence").sum()

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
    if inverse >= 3:
        return "not_supported_or_inverse"
    if inverse >= 1:
        return "mixed_or_heterogeneous"

    return "limited_or_no_support"


def support_score(metric_class):
    scores = {
        "strong_support": 3.0,
        "moderate_support": 2.0,
        "weak_support": 1.0,
        "directionally_supportive_not_significant": 0.5,
        "insufficient_evidence": 0.0,
        "not_supported": -0.5,
        "inverse_or_contradictory_evidence": -2.0,
    }
    return scores.get(metric_class, 0.0)


def build_experiment_synthesis(
    registry_path="data/validation/rde_experiment_framework_registry_v01.csv",
    output_dir="data/validation/synthesis",
    output_prefix="oregon_experiment_synthesis_v01",
):
    registry_path = Path(registry_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not registry_path.exists():
        raise FileNotFoundError(f"Missing experiment registry: {registry_path}")

    registry = pd.read_csv(registry_path)

    rows = []

    for _, exp in registry.iterrows():
        comparison_path = exp.get("null_comparison_csv")

        if not isinstance(comparison_path, str) or not comparison_path:
            continue

        comparison_path = Path(comparison_path)

        if not comparison_path.exists():
            continue

        comp = pd.read_csv(comparison_path)

        for _, r in comp.iterrows():
            metric = r["metric"]

            if metric not in PRIMARY_METRICS:
                continue

            row = {
                "experiment_key": exp.get("experiment_key"),
                "experiment_name": exp.get("experiment_name"),
                "study_name": exp.get("study_name"),
                "dataset_name": exp.get("dataset_name"),
                "dataset_source": exp.get("dataset_source"),
                "dataset_category": exp.get("dataset_category"),
                "metrics_declared": exp.get("metrics"),
                "n_simulations": exp.get("n_simulations"),
                "metric": metric,
                "observed_value": r.get("observed_value"),
                "null_mean": r.get("null_mean"),
                "null_std": r.get("null_std"),
                "z_score": r.get("z_score"),
                "monte_carlo_p_value": r.get("monte_carlo_p_value"),
                "direction": r.get("direction"),
                "effect_ratio_observed_to_null": r.get("effect_ratio_observed_to_null"),
                "comparison_path": str(comparison_path),
            }

            row["metric_evidence_class"] = classify_metric(row)
            row["metric_support_score"] = support_score(row["metric_evidence_class"])

            rows.append(row)

    if not rows:
        raise ValueError("No experiment comparison rows found.")

    metric_df = pd.DataFrame(rows)

    summary_rows = []

    for experiment_key, sub in metric_df.groupby("experiment_key"):
        sub = sub.copy()
        first = sub.iloc[0]

        experiment_class = classify_experiment(sub)

        summary_rows.append({
            "experiment_key": experiment_key,
            "experiment_name": first["experiment_name"],
            "study_name": first["study_name"],
            "dataset_name": first["dataset_name"],
            "dataset_category": first["dataset_category"],
            "experiment_overall_support_class": experiment_class,
            "tested_metric_count": len(sub),
            "strong_support_metrics": (sub["metric_evidence_class"] == "strong_support").sum(),
            "moderate_support_metrics": (sub["metric_evidence_class"] == "moderate_support").sum(),
            "weak_support_metrics": (sub["metric_evidence_class"] == "weak_support").sum(),
            "inverse_or_contradictory_metrics": (sub["metric_evidence_class"] == "inverse_or_contradictory_evidence").sum(),
            "mean_support_score": sub["metric_support_score"].mean(),
            "total_support_score": sub["metric_support_score"].sum(),
            "mean_abs_z_score": sub["z_score"].abs().replace([np.inf, -np.inf], np.nan).mean(),
            "min_p_value": sub["monte_carlo_p_value"].min(),
            "median_p_value": sub["monte_carlo_p_value"].median(),
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        ["total_support_score", "mean_abs_z_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    metric_df = metric_df.merge(
        summary_df[["experiment_key", "experiment_overall_support_class"]],
        on="experiment_key",
        how="left",
    )

    metric_csv = output_dir / f"{output_prefix}_metric_level.csv"
    summary_csv = output_dir / f"{output_prefix}_experiment_level.csv"
    report_md = output_dir / f"{output_prefix}.md"

    metric_df.to_csv(metric_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    lines = []
    lines.append("# RDE Experiment Synthesis v01")
    lines.append("")
    lines.append("This report synthesizes completed validation experiments from the RDE Experiment Framework registry.")
    lines.append("")
    lines.append("## Experiment-Level Evidence")
    lines.append("")
    lines.append(summary_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Metric-Level Evidence")
    lines.append("")
    lines.append(
        metric_df[
            [
                "experiment_name",
                "metric",
                "observed_value",
                "null_mean",
                "z_score",
                "monte_carlo_p_value",
                "metric_evidence_class",
                "experiment_overall_support_class",
            ]
        ].to_markdown(index=False)
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Positive support indicates that observed RDE hotspots align with the validation hypothesis "
        "more strongly than randomized hotspot placements. Inverse evidence indicates that observed "
        "hotspots align less strongly than expected by chance. Mixed or heterogeneous outcomes should "
        "be interpreted as opportunities to refine the scientific hypothesis rather than as simple failures."
    )

    report_md.write_text("\n".join(lines), encoding="utf-8")

    return {
        "metric_csv": metric_csv,
        "summary_csv": summary_csv,
        "report_md": report_md,
        "metric_df": metric_df,
        "summary_df": summary_df,
    }
'''


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(CONTENT, encoding="utf-8")

    print(f"[{SCRIPT_NAME}] Wrote: {OUTPUT_PATH}")
    print()
    print(f"[{SCRIPT_NAME}] Experiment synthesis engine created.")
    print("Next: build a driver script to run the synthesis engine.")
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()