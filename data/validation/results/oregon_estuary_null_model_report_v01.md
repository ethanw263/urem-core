# Oregon Estuary Null Model Validation Report v01

This report compares observed Oregon RDE transition hotspots against randomized hotspot placements using PMEP estuary extents.

## Observed Metrics

|   n_hotspots |   pct_intersects_any |   mean_overlap_pct |   median_overlap_pct |   pct_any_overlap |   mean_nearest_distance_m |   median_nearest_distance_m |   pct_within_1km |   pct_within_5km |
|-------------:|---------------------:|-------------------:|---------------------:|------------------:|--------------------------:|----------------------------:|-----------------:|-----------------:|
|           34 |             0.294118 |           0.212664 |                    0 |          0.294118 |                   2934.26 |                      3372.6 |         0.294118 |         0.794118 |

## Observed vs Null

| metric                    |   observed_value |    null_mean |     null_std |    z_score |   monte_carlo_p_value | direction        |   effect_ratio_observed_to_null |
|:--------------------------|-----------------:|-------------:|-------------:|-----------:|----------------------:|:-----------------|--------------------------------:|
| pct_intersects_any        |         0.294118 |    0.225882  |    0.0683747 |   0.997962 |                 0.208 | higher_is_better |                        1.30208  |
| mean_overlap_pct          |         0.212664 |    0.0430289 |    0.0254417 |   6.66759  |                 0     | higher_is_better |                        4.94235  |
| median_overlap_pct        |         0        |    0         |    0         | nan        |                 1     | higher_is_better |                      nan        |
| pct_any_overlap           |         0.294118 |    0.225882  |    0.0683747 |   0.997962 |                 0.208 | higher_is_better |                        1.30208  |
| mean_nearest_distance_m   |      2934.26     | 6118.47      |  882.248     |  -3.6092   |                 0     | lower_is_better  |                        0.479574 |
| median_nearest_distance_m |      3372.6      | 4731.37      | 1083.42      |  -1.25415  |                 0.08  | lower_is_better  |                        0.712816 |
| pct_within_1km            |         0.294118 |    0.163353  |    0.0624201 |   2.09491  |                 0.044 | higher_is_better |                        1.8005   |
| pct_within_5km            |         0.794118 |    0.526941  |    0.0800722 |   3.33669  |                 0     | higher_is_better |                        1.50703  |

## Interpretation

Low Monte Carlo p-values indicate that observed hotspots align with estuaries more strongly than expected under the spatial null model.