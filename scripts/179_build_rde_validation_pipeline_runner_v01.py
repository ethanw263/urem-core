#!/usr/bin/env python3

from pathlib import Path

SCRIPT_NAME = "179_build_rde_validation_pipeline_runner_v01"

VALIDATION_DIR = Path("src/validation")
PIPELINE_PY = VALIDATION_DIR / "pipeline.py"

CONTENT = r'''from pathlib import Path
from typing import Dict, Any

import geopandas as gpd
import pandas as pd

from .datasets import ValidationDataset
from .engine import ValidationEngine
from .null_models import (
    NullModelConfig,
    validation_metrics_for_layer,
    run_spatial_null_model,
    compare_observed_to_null,
)
from .reporting import write_markdown_report


class ValidationPipelineRunner:
    """
    Reusable validation pipeline runner.

    This class assumes the validation dataset has already been standardized.

    Responsibilities:
    - run external validation
    - run spatial null-model validation
    - write CSV outputs
    - write Markdown reports

    It does NOT:
    - download data
    - standardize raw datasets
    - mutate model outputs
    """

    def __init__(
        self,
        study_name: str,
        hotspots_path: str,
        study_area_path: str,
        hotspot_id_column: str,
        hotspot_score_column: str,
        output_dir: str = "data/validation/results",
    ):
        self.study_name = study_name
        self.hotspots_path = Path(hotspots_path)
        self.study_area_path = Path(study_area_path)
        self.hotspot_id_column = hotspot_id_column
        self.hotspot_score_column = hotspot_score_column
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _safe_key(self, text: str) -> str:
        return (
            text.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

    def run_external_validation(
        self,
        dataset: ValidationDataset,
    ) -> Dict[str, Path]:

        dataset_key = self._safe_key(dataset.name)
        study_key = self._safe_key(self.study_name)

        results_csv = self.output_dir / f"{study_key}_{dataset_key}_external_validation_results_v01.csv"
        summary_csv = self.output_dir / f"{study_key}_{dataset_key}_external_validation_summary_v01.csv"
        report_md = self.output_dir / f"{study_key}_{dataset_key}_external_validation_report_v01.md"

        engine = ValidationEngine(
            hotspots_path=str(self.hotspots_path),
            hotspot_id_column=self.hotspot_id_column,
            hotspot_score_column=self.hotspot_score_column,
        )

        engine.register_dataset(dataset)

        results = engine.run()
        summary = engine.summarize(results)

        results.to_csv(results_csv, index=False)
        summary.to_csv(summary_csv, index=False)

        write_markdown_report(
            summary=summary,
            output_path=str(report_md),
            title=f"{self.study_name} {dataset.name} External Validation Report",
        )

        return {
            "external_results_csv": results_csv,
            "external_summary_csv": summary_csv,
            "external_report_md": report_md,
        }

    def run_null_model_validation(
        self,
        dataset: ValidationDataset,
        n_simulations: int = 500,
        random_seed: int = 42,
        progress_interval: int = 50,
    ) -> Dict[str, Path]:

        dataset_key = self._safe_key(dataset.name)
        study_key = self._safe_key(self.study_name)

        observed_csv = self.output_dir / f"{study_key}_{dataset_key}_observed_validation_metrics_v01.csv"
        null_csv = self.output_dir / f"{study_key}_{dataset_key}_null_model_results_v01.csv"
        comparison_csv = self.output_dir / f"{study_key}_{dataset_key}_null_model_comparison_v01.csv"
        report_md = self.output_dir / f"{study_key}_{dataset_key}_null_model_report_v01.md"

        hotspots = gpd.read_file(self.hotspots_path)
        study_area = gpd.read_file(self.study_area_path)
        validation_layer = gpd.read_file(dataset.path)

        if study_area.crs != hotspots.crs:
            study_area = study_area.to_crs(hotspots.crs)

        if validation_layer.crs != hotspots.crs:
            validation_layer = validation_layer.to_crs(hotspots.crs)

        observed = validation_metrics_for_layer(
            hotspots=hotspots,
            validation_layer=validation_layer,
        )

        config = NullModelConfig(
            n_simulations=n_simulations,
            random_seed=random_seed,
            null_model_type="random_translation",
            max_attempts_per_feature=250,
            progress_interval=progress_interval,
        )

        null_results = run_spatial_null_model(
            hotspots=hotspots,
            study_area=study_area,
            validation_layer=validation_layer,
            config=config,
        )

        comparison = compare_observed_to_null(
            observed_metrics=observed,
            null_results=null_results,
        )

        pd.DataFrame([observed]).to_csv(observed_csv, index=False)
        null_results.to_csv(null_csv, index=False)
        comparison.to_csv(comparison_csv, index=False)

        self._write_null_report(
            dataset=dataset,
            observed=observed,
            comparison=comparison,
            output_path=report_md,
            n_simulations=n_simulations,
        )

        return {
            "observed_csv": observed_csv,
            "null_results_csv": null_csv,
            "null_comparison_csv": comparison_csv,
            "null_report_md": report_md,
        }

    def run_full_validation(
        self,
        dataset: ValidationDataset,
        n_simulations: int = 500,
        random_seed: int = 42,
        progress_interval: int = 50,
    ) -> Dict[str, Path]:

        outputs = {}

        outputs.update(
            self.run_external_validation(dataset)
        )

        outputs.update(
            self.run_null_model_validation(
                dataset=dataset,
                n_simulations=n_simulations,
                random_seed=random_seed,
                progress_interval=progress_interval,
            )
        )

        return outputs

    def _write_null_report(
        self,
        dataset: ValidationDataset,
        observed: Dict[str, Any],
        comparison: pd.DataFrame,
        output_path: Path,
        n_simulations: int,
    ) -> None:

        lines = []

        lines.append(f"# {self.study_name} {dataset.name} Null Model Validation Report")
        lines.append("")
        lines.append(f"Simulations: {n_simulations:,}")
        lines.append("")
        lines.append("## Dataset Metadata")
        lines.append("")
        lines.append(f"- Dataset: {dataset.name}")
        lines.append(f"- Source: {dataset.source}")
        lines.append(f"- Category: {dataset.category}")
        lines.append(f"- Independent of model: {dataset.independent_of_model}")
        lines.append(f"- Weight: {dataset.weight}")
        lines.append("")
        lines.append("## Observed Metrics")
        lines.append("")
        lines.append(pd.DataFrame([observed]).to_markdown(index=False))
        lines.append("")
        lines.append("## Observed vs Null")
        lines.append("")
        lines.append(comparison.to_markdown(index=False))
        lines.append("")
        lines.append("## Interpretation")
        lines.append("")
        lines.append(
            "Low Monte Carlo p-values indicate that observed RDE hotspots "
            "align with this independent validation dataset more strongly than "
            "expected under randomized hotspot placement."
        )

        output_path.write_text("\n".join(lines), encoding="utf-8")
'''


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    PIPELINE_PY.write_text(CONTENT, encoding="utf-8")

    print(f"[{SCRIPT_NAME}] Wrote: {PIPELINE_PY}")
    print()
    print(f"[{SCRIPT_NAME}] Validation pipeline runner created.")
    print()
    print("Next step:")
    print("  Build a small driver script to test the runner on Oregon estuaries or PAD-US.")
    print()
    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()