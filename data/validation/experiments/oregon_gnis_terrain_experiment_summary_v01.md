# Oregon GNIS Terrain Validation

- Experiment key: `oregon_gnis_terrain`
- Study: Oregon Coast
- Dataset: GNIS Terrain
- Source: USGS GNIS FullModel / Gazetteer OR
- Category: geographic_recognition_terrain
- Metrics: distance, density
- Null model simulations: 500

## Outputs

- `external_results_csv`: `data/validation/results/oregon_coast_gnis_terrain_external_validation_results_v01.csv`
- `external_summary_csv`: `data/validation/results/oregon_coast_gnis_terrain_external_validation_summary_v01.csv`
- `external_report_md`: `data/validation/results/oregon_coast_gnis_terrain_external_validation_report_v01.md`
- `observed_csv`: `data/validation/results/oregon_coast_gnis_terrain_observed_validation_metrics_v01.csv`
- `null_results_csv`: `data/validation/results/oregon_coast_gnis_terrain_null_model_results_v01.csv`
- `null_comparison_csv`: `data/validation/results/oregon_coast_gnis_terrain_null_model_comparison_v01.csv`
- `null_report_md`: `data/validation/results/oregon_coast_gnis_terrain_null_model_report_v01.md`

## Notes

Tests whether Oregon RDE transition hotspots are closer to GNIS terrain named features than expected under randomized hotspot placement. Includes Summit, Ridge, Valley, Flat, Gap, Basin, Slope, and Bench.