# Oregon RDE Validation Evidence Synthesis v01

This report synthesizes independent validation evidence for Oregon RDE transition hotspots.

## Dataset-Level Support

| dataset_name    | dataset_overall_support_class   |   strong_metric_count |   moderate_metric_count |   weak_metric_count |   tested_metric_count |   mean_abs_z_score |   mean_p_value |
|:----------------|:--------------------------------|----------------------:|------------------------:|--------------------:|----------------------:|-------------------:|---------------:|
| Estuaries       | strong_overall_support          |                     3 |                       1 |                   0 |                     5 |            3.9271  |         0.2088 |
| Protected Areas | strong_overall_support          |                     2 |                       2 |                   0 |                     5 |            1.89945 |         0.1524 |

## Metric-Level Evidence

| dataset_name    | metric                  |   observed_value |    null_mean |   monte_carlo_p_value |   effect_ratio_observed_to_null | metric_evidence_class        | dataset_overall_support_class   |
|:----------------|:------------------------|-----------------:|-------------:|----------------------:|--------------------------------:|:-----------------------------|:--------------------------------|
| Protected Areas | mean_overlap_pct        |         0.724789 |    0.514443  |                 0     |                        1.40888  | strong_statistical_support   | strong_overall_support          |
| Protected Areas | median_overlap_pct      |         0.983914 |    0.52728   |                 0     |                        1.86602  | strong_statistical_support   | strong_overall_support          |
| Protected Areas | mean_nearest_distance_m |       350.209    |  628.228     |                 0.05  |                        0.557456 | moderate_statistical_support | strong_overall_support          |
| Protected Areas | pct_within_1km          |         0.911765 |    0.779     |                 0.04  |                        1.17043  | moderate_statistical_support | strong_overall_support          |
| Protected Areas | pct_within_5km          |         1        |    0.988235  |                 0.672 |                        1.0119   | not_statistically_supported  | strong_overall_support          |
| Estuaries       | mean_overlap_pct        |         0.212664 |    0.0430289 |                 0     |                        4.94235  | strong_statistical_support   | strong_overall_support          |
| Estuaries       | median_overlap_pct      |         0        |    0         |                 1     |                      nan        | not_statistically_supported  | strong_overall_support          |
| Estuaries       | mean_nearest_distance_m |      2934.26     | 6118.47      |                 0     |                        0.479574 | strong_statistical_support   | strong_overall_support          |
| Estuaries       | pct_within_1km          |         0.294118 |    0.163353  |                 0.044 |                        1.8005   | moderate_statistical_support | strong_overall_support          |
| Estuaries       | pct_within_5km          |         0.794118 |    0.526941  |                 0     |                        1.50703  | strong_statistical_support   | strong_overall_support          |

## Interpretation

This synthesis separates descriptive external validation from statistical null-model validation. Strong support means RDE hotspots show stronger alignment with an independent validation dataset than expected under randomized hotspot placement.