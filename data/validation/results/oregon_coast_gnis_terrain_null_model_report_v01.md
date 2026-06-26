# Oregon Coast GNIS Terrain Null Model Validation Report

Simulations: 500

## Dataset Metadata

- Dataset: GNIS Terrain
- Source: USGS GNIS FullModel / Gazetteer OR
- Category: geographic_recognition_terrain
- Independent of model: True
- Weight: 0.15

## Observed Metrics

|   n_hotspots |   pct_intersects_any |   mean_overlap_pct |   median_overlap_pct |   pct_any_overlap |   mean_nearest_distance_m |   median_nearest_distance_m |   pct_within_1km |   pct_within_5km |
|-------------:|---------------------:|-------------------:|---------------------:|------------------:|--------------------------:|----------------------------:|-----------------:|-----------------:|
|           34 |            0.0588235 |                  0 |                    0 |                 0 |                   7146.94 |                     7028.04 |        0.0588235 |         0.323529 |

## Observed vs Null

| metric                    |   observed_value |   null_mean |    null_std |    z_score |   monte_carlo_p_value | direction        |   effect_ratio_observed_to_null |
|:--------------------------|-----------------:|------------:|------------:|-----------:|----------------------:|:-----------------|--------------------------------:|
| pct_intersects_any        |        0.0588235 |    0.173176 |   0.0565965 |  -2.02049  |                 0.996 | higher_is_better |                        0.339674 |
| mean_overlap_pct          |        0         |    0        |   0         | nan        |                 1     | higher_is_better |                      nan        |
| median_overlap_pct        |        0         |    0        |   0         | nan        |                 1     | higher_is_better |                      nan        |
| pct_any_overlap           |        0         |    0        |   0         | nan        |                 1     | higher_is_better |                      nan        |
| mean_nearest_distance_m   |     7146.94      | 3626.23     | 430.168     |   8.18451  |                 1     | lower_is_better  |                        1.9709   |
| median_nearest_distance_m |     7028.04      | 3053.74     | 437.975     |   9.07426  |                 1     | lower_is_better  |                        2.30145  |
| pct_within_1km            |        0.0588235 |    0.086    |   0.0483968 |  -0.561535 |                 0.802 | higher_is_better |                        0.683995 |
| pct_within_5km            |        0.323529  |    0.783588 |   0.0696821 |  -6.60226  |                 1     | higher_is_better |                        0.412882 |

## Interpretation

Low Monte Carlo p-values indicate that observed RDE hotspots align with this independent validation dataset more strongly than expected under randomized hotspot placement.