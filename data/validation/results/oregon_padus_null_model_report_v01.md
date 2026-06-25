# Oregon PAD-US Null Model Validation Report v01

This report compares observed Oregon RDE transition hotspots against randomized hotspot placements.

## Observed Metrics

|   n_hotspots |   pct_intersects_any |   mean_overlap_pct |   median_overlap_pct |   pct_any_overlap |   mean_nearest_distance_m |   median_nearest_distance_m |   pct_within_1km |   pct_within_5km |
|-------------:|---------------------:|-------------------:|---------------------:|------------------:|--------------------------:|----------------------------:|-----------------:|-----------------:|
|           34 |             0.882353 |           0.724789 |             0.983914 |          0.882353 |                   350.209 |                           0 |         0.911765 |                1 |

## Observed vs Null

| metric                    |   observed_value |   null_mean |    null_std |   z_score |   monte_carlo_p_value | direction        |   effect_ratio_observed_to_null |
|:--------------------------|-----------------:|------------:|------------:|----------:|----------------------:|:-----------------|--------------------------------:|
| pct_intersects_any        |         0.882353 |    0.832706 |   0.0616908 |  0.804772 |                 0.308 | higher_is_better |                        1.05962  |
| mean_overlap_pct          |         0.724789 |    0.514443 |   0.0699997 |  3.00495  |                 0     | higher_is_better |                        1.40888  |
| median_overlap_pct        |         0.983914 |    0.52728  |   0.181247  |  2.5194   |                 0     | higher_is_better |                        1.86602  |
| pct_any_overlap           |         0.882353 |    0.832706 |   0.0616908 |  0.804772 |                 0.308 | higher_is_better |                        1.05962  |
| mean_nearest_distance_m   |       350.209    |  628.228    | 186.852     | -1.4879   |                 0.05  | lower_is_better  |                        0.557456 |
| median_nearest_distance_m |         0        |   76.2784   | 127.974     | -0.596046 |                 0.522 | lower_is_better  |                        0        |
| pct_within_1km            |         0.911765 |    0.779    |   0.0717654 |  1.84998  |                 0.04  | higher_is_better |                        1.17043  |
| pct_within_5km            |         1        |    0.988235 |   0.0185269 |  0.635006 |                 0.672 | higher_is_better |                        1.0119   |

## Interpretation

Low Monte Carlo p-values indicate that observed hotspots align with PAD-US protected areas more strongly than expected under the spatial null model.