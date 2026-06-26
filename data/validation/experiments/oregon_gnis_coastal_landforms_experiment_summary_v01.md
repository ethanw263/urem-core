# Oregon GNIS Coastal Landforms Validation

- Experiment key: `oregon_gnis_coastal_landforms`
- Study: Oregon Coast
- Dataset: GNIS Coastal Landforms
- Source: USGS GNIS FullModel / Gazetteer OR
- Category: geographic_recognition
- Metrics: distance, density
- Null model simulations: 500

## Outputs

- `external_results_csv`: `data/validation/results/oregon_coast_gnis_coastal_landforms_external_validation_results_v01.csv`
- `external_summary_csv`: `data/validation/results/oregon_coast_gnis_coastal_landforms_external_validation_summary_v01.csv`
- `external_report_md`: `data/validation/results/oregon_coast_gnis_coastal_landforms_external_validation_report_v01.md`
- `observed_csv`: `data/validation/results/oregon_coast_gnis_coastal_landforms_observed_validation_metrics_v01.csv`
- `null_results_csv`: `data/validation/results/oregon_coast_gnis_coastal_landforms_null_model_results_v01.csv`
- `null_comparison_csv`: `data/validation/results/oregon_coast_gnis_coastal_landforms_null_model_comparison_v01.csv`
- `null_report_md`: `data/validation/results/oregon_coast_gnis_coastal_landforms_null_model_report_v01.md`

## Notes

Tests whether Oregon RDE transition hotspots are closer to officially named coastal landforms than expected under randomized hotspot placement. Coastal landforms include Island, Cape, Bay, Channel, Bar, Beach, and Gut.