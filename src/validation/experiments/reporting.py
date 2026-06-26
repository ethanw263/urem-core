from pathlib import Path


def write_experiment_summary(
    experiment,
    outputs,
    output_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    lines.append(f"# {experiment.experiment_name}")
    lines.append("")
    lines.append(f"- Experiment key: `{experiment.experiment_key}`")
    lines.append(f"- Study: {experiment.study_name}")
    lines.append(f"- Dataset: {experiment.dataset_name}")
    lines.append(f"- Source: {experiment.dataset_source}")
    lines.append(f"- Category: {experiment.dataset_category}")
    lines.append(f"- Metrics: {', '.join(experiment.metrics)}")
    lines.append(f"- Null model simulations: {experiment.n_simulations}")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")

    for key, value in outputs.items():
        lines.append(f"- `{key}`: `{value}`")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(experiment.notes or "No notes provided.")

    path.write_text("\n".join(lines), encoding="utf-8")
