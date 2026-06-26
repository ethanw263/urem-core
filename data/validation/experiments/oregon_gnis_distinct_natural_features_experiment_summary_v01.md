# Oregon GNIS Distinct Natural Features Validation

- Experiment key: `oregon_gnis_distinct_natural_features`
- Study: Oregon Coast
- Dataset: GNIS Distinct Natural Features
- Source: USGS GNIS FullModel / Gazetteer OR
- Category: geographic_recognition_distinct_natural_feature
- Metrics: distance, density
- Null model simulations: 500

## Outputs

- `external_results_csv`: `data/validation/results/oregon_coast_gnis_distinct_natural_features_external_validation_results_v01.csv`
- `external_summary_csv`: `data/validation/results/oregon_coast_gnis_distinct_natural_features_external_validation_summary_v01.csv`
- `external_report_md`: `data/validation/results/oregon_coast_gnis_distinct_natural_features_external_validation_report_v01.md`
- `observed_csv`: `data/validation/results/oregon_coast_gnis_distinct_natural_features_observed_validation_metrics_v01.csv`
- `null_results_csv`: `data/validation/results/oregon_coast_gnis_distinct_natural_features_null_model_results_v01.csv`
- `null_comparison_csv`: `data/validation/results/oregon_coast_gnis_distinct_natural_features_null_model_comparison_v01.csv`
- `null_report_md`: `data/validation/results/oregon_coast_gnis_distinct_natural_features_null_model_report_v01.md`

## Notes

Tests whether Oregon RDE transition hotspots are closer to GNIS distinct natural features than expected under randomized hotspot placement. Includes Pillar, Cliff, and Arch.