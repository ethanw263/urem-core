# Oregon Coast GNIS Coastal Landforms Null Model Validation Report

Simulations: 500

## Dataset Metadata

- Dataset: GNIS Coastal Landforms
- Source: USGS GNIS FullModel / Gazetteer OR
- Category: geographic_recognition
- Independent of model: True
- Weight: 0.15

## Observed Metrics

|   n_hotspots |   pct_intersects_any |   mean_overlap_pct |   median_overlap_pct |   pct_any_overlap |   mean_nearest_distance_m |   median_nearest_distance_m |   pct_within_1km |   pct_within_5km |
|-------------:|---------------------:|-------------------:|---------------------:|------------------:|--------------------------:|----------------------------:|-----------------:|-----------------:|
|           34 |             0.205882 |                  0 |                    0 |                 0 |                   3226.48 |                     2713.08 |         0.117647 |         0.794118 |

## Observed vs Null

| metric                    |   observed_value |    null_mean |     null_std |   z_score |   monte_carlo_p_value | direction        |   effect_ratio_observed_to_null |
|:--------------------------|-----------------:|-------------:|-------------:|----------:|----------------------:|:-----------------|--------------------------------:|
| pct_intersects_any        |         0.205882 |    0.112529  |    0.0522009 |   1.78834 |                 0.066 | higher_is_better |                        1.82959  |
| mean_overlap_pct          |         0        |    0         |    0         | nan       |                 1     | higher_is_better |                      nan        |
| median_overlap_pct        |         0        |    0         |    0         | nan       |                 1     | higher_is_better |                      nan        |
| pct_any_overlap           |         0        |    0         |    0         | nan       |                 1     | higher_is_better |                      nan        |
| mean_nearest_distance_m   |      3226.48     | 7444.55      |  967.463     |  -4.35992 |                 0     | lower_is_better  |                        0.433402 |
| median_nearest_distance_m |      2713.08     | 6131.53      | 1417.03      |  -2.4124  |                 0     | lower_is_better  |                        0.442481 |
| pct_within_1km            |         0.117647 |    0.0645294 |    0.0413475 |   1.28466 |                 0.176 | higher_is_better |                        1.82315  |
| pct_within_5km            |         0.794118 |    0.434647  |    0.0847412 |   4.24198 |                 0     | higher_is_better |                        1.82704  |

## Interpretation

Low Monte Carlo p-values indicate that observed RDE hotspots align with this independent validation dataset more strongly than expected under randomized hotspot placement.