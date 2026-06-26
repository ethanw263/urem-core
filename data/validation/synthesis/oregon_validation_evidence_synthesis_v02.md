# Oregon RDE Validation Evidence Synthesis v02

This synthesis incorporates supportive, mixed, and inverse validation evidence.

## Dataset-Level Support

| dataset_name                | dataset_overall_support_class   |   strong_support_metrics |   moderate_support_metrics |   weak_support_metrics |   inverse_or_contradictory_metrics |   tested_metric_count |   mean_abs_z_score |
|:----------------------------|:--------------------------------|-------------------------:|---------------------------:|-----------------------:|-----------------------------------:|----------------------:|-------------------:|
| Estuaries                   | strong_overall_support          |                        3 |                          1 |                      0 |                                  1 |                     5 |            3.9271  |
| GNIS Named Natural Features | not_supported_or_inverse        |                        0 |                          0 |                      0 |                                  4 |                     5 |            5.22473 |
| Protected Areas             | strong_overall_support          |                        2 |                          2 |                      0 |                                  0 |                     5 |            1.89945 |

## Metric-Level Evidence

| dataset_name                | hypothesis_type                    | metric                  |   observed_value |    null_mean |   monte_carlo_p_value | metric_evidence_class                    | dataset_overall_support_class   |
|:----------------------------|:-----------------------------------|:------------------------|-----------------:|-------------:|----------------------:|:-----------------------------------------|:--------------------------------|
| Protected Areas             | conservation_landscape_validation  | mean_overlap_pct        |         0.724789 |    0.514443  |                 0     | strong_support                           | strong_overall_support          |
| Protected Areas             | conservation_landscape_validation  | median_overlap_pct      |         0.983914 |    0.52728   |                 0     | strong_support                           | strong_overall_support          |
| Protected Areas             | conservation_landscape_validation  | mean_nearest_distance_m |       350.209    |  628.228     |                 0.05  | moderate_support                         | strong_overall_support          |
| Protected Areas             | conservation_landscape_validation  | pct_within_1km          |         0.911765 |    0.779     |                 0.04  | moderate_support                         | strong_overall_support          |
| Protected Areas             | conservation_landscape_validation  | pct_within_5km          |         1        |    0.988235  |                 0.672 | directionally_supportive_not_significant | strong_overall_support          |
| Estuaries                   | coastal_transition_validation      | mean_overlap_pct        |         0.212664 |    0.0430289 |                 0     | strong_support                           | strong_overall_support          |
| Estuaries                   | coastal_transition_validation      | median_overlap_pct      |         0        |    0         |                 1     | inverse_or_contradictory_evidence        | strong_overall_support          |
| Estuaries                   | coastal_transition_validation      | mean_nearest_distance_m |      2934.26     | 6118.47      |                 0     | strong_support                           | strong_overall_support          |
| Estuaries                   | coastal_transition_validation      | pct_within_1km          |         0.294118 |    0.163353  |                 0.044 | moderate_support                         | strong_overall_support          |
| Estuaries                   | coastal_transition_validation      | pct_within_5km          |         0.794118 |    0.526941  |                 0     | strong_support                           | strong_overall_support          |
| GNIS Named Natural Features | aggregate_named_feature_validation | mean_overlap_pct        |         0        |    0         |                 1     | inverse_or_contradictory_evidence        | not_supported_or_inverse        |
| GNIS Named Natural Features | aggregate_named_feature_validation | median_overlap_pct      |         0        |    0         |                 1     | inverse_or_contradictory_evidence        | not_supported_or_inverse        |
| GNIS Named Natural Features | aggregate_named_feature_validation | mean_nearest_distance_m |      2520.71     | 1488.85      |                 1     | inverse_or_contradictory_evidence        | not_supported_or_inverse        |
| GNIS Named Natural Features | aggregate_named_feature_validation | pct_within_1km          |         0.264706 |    0.364118  |                 0.924 | not_supported                            | not_supported_or_inverse        |
| GNIS Named Natural Features | aggregate_named_feature_validation | pct_within_5km          |         0.911765 |    0.996471  |                 1     | inverse_or_contradictory_evidence        | not_supported_or_inverse        |

## Interpretation Notes

- Protected Areas and Estuaries test independent geographic hypotheses.
- Aggregate GNIS is heterogeneous and should be decomposed into subgroups.
- Non-supportive evidence is retained because it helps refine the RDE theory.