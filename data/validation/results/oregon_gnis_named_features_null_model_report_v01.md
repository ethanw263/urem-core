# Oregon GNIS Named Natural Features Null Model Validation Report v01

This report compares observed Oregon RDE transition hotspots against randomized hotspot placements using USGS GNIS named natural features.

## Observed Metrics

|   n_hotspots |   pct_intersects_any |   mean_overlap_pct |   median_overlap_pct |   pct_any_overlap |   mean_nearest_distance_m |   median_nearest_distance_m |   pct_within_1km |   pct_within_5km |
|-------------:|---------------------:|-------------------:|---------------------:|------------------:|--------------------------:|----------------------------:|-----------------:|-----------------:|
|           34 |             0.264706 |                  0 |                    0 |                 0 |                   2520.71 |                     2102.83 |         0.264706 |         0.911765 |

## Observed vs Null

| metric                    |   observed_value |   null_mean |     null_std |   z_score |   monte_carlo_p_value | direction        |   effect_ratio_observed_to_null |
|:--------------------------|-----------------:|------------:|-------------:|----------:|----------------------:|:-----------------|--------------------------------:|
| pct_intersects_any        |         0.264706 |    0.498294 |   0.0777306  |  -3.0051  |                 1     | higher_is_better |                        0.531224 |
| mean_overlap_pct          |         0        |    0        |   0          | nan       |                 1     | higher_is_better |                      nan        |
| median_overlap_pct        |         0        |    0        |   0          | nan       |                 1     | higher_is_better |                      nan        |
| pct_any_overlap           |         0        |    0        |   0          | nan       |                 1     | higher_is_better |                      nan        |
| mean_nearest_distance_m   |      2520.71     | 1488.85     | 173.27       |   5.95526 |                 1     | lower_is_better  |                        1.69306  |
| median_nearest_distance_m |      2102.83     | 1308.31     | 199.797      |   3.97659 |                 0.998 | lower_is_better  |                        1.60728  |
| pct_within_1km            |         0.264706 |    0.364118 |   0.0840576  |  -1.18266 |                 0.924 | higher_is_better |                        0.726979 |
| pct_within_5km            |         0.911765 |    0.996471 |   0.00992305 |  -8.53628 |                 1     | higher_is_better |                        0.914994 |

## Interpretation

Low Monte Carlo p-values indicate that observed RDE hotspots align with GNIS named natural features more strongly than expected under randomized hotspot placement.