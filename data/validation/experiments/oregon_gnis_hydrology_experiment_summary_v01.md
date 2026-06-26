# Oregon GNIS Hydrology Validation

- Experiment key: `oregon_gnis_hydrology`
- Study: Oregon Coast
- Dataset: GNIS Hydrology
- Source: USGS GNIS FullModel / Gazetteer OR
- Category: geographic_recognition_hydrology
- Metrics: distance, density
- Null model simulations: 500

## Outputs

- `external_results_csv`: `data/validation/results/oregon_coast_gnis_hydrology_external_validation_results_v01.csv`
- `external_summary_csv`: `data/validation/results/oregon_coast_gnis_hydrology_external_validation_summary_v01.csv`
- `external_report_md`: `data/validation/results/oregon_coast_gnis_hydrology_external_validation_report_v01.md`
- `observed_csv`: `data/validation/results/oregon_coast_gnis_hydrology_observed_validation_metrics_v01.csv`
- `null_results_csv`: `data/validation/results/oregon_coast_gnis_hydrology_null_model_results_v01.csv`
- `null_comparison_csv`: `data/validation/results/oregon_coast_gnis_hydrology_null_model_comparison_v01.csv`
- `null_report_md`: `data/validation/results/oregon_coast_gnis_hydrology_null_model_report_v01.md`

## Notes

Tests whether Oregon RDE transition hotspots are closer to GNIS hydrologic named features than expected under randomized hotspot placement. Includes Stream, Lake, Reservoir, Rapids, Spring, Falls, and Swamp.