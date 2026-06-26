# Oregon Coast GNIS Distinct Natural Features Null Model Validation Report

Simulations: 500

## Dataset Metadata

- Dataset: GNIS Distinct Natural Features
- Source: USGS GNIS FullModel / Gazetteer OR
- Category: geographic_recognition_distinct_natural_feature
- Independent of model: True
- Weight: 0.15

## Observed Metrics

|   n_hotspots |   pct_intersects_any |   mean_overlap_pct |   median_overlap_pct |   pct_any_overlap |   mean_nearest_distance_m |   median_nearest_distance_m |   pct_within_1km |   pct_within_5km |
|-------------:|---------------------:|-------------------:|---------------------:|------------------:|--------------------------:|----------------------------:|-----------------:|-----------------:|
|           34 |                    0 |                  0 |                    0 |                 0 |                   14645.1 |                     11707.1 |                0 |         0.176471 |

## Observed vs Null

| metric                    |   observed_value |      null_mean |     null_std |    z_score |   monte_carlo_p_value | direction        |   effect_ratio_observed_to_null |
|:--------------------------|-----------------:|---------------:|-------------:|-----------:|----------------------:|:-----------------|--------------------------------:|
| pct_intersects_any        |         0        |     0.0182353  |    0.0225682 |  -0.808009 |                 1     | higher_is_better |                         0       |
| mean_overlap_pct          |         0        |     0          |    0         | nan        |                 1     | higher_is_better |                       nan       |
| median_overlap_pct        |         0        |     0          |    0         | nan        |                 1     | higher_is_better |                       nan       |
| pct_any_overlap           |         0        |     0          |    0         | nan        |                 1     | higher_is_better |                       nan       |
| mean_nearest_distance_m   |     14645.1      | 12682.7        | 1369.01      |   1.43346  |                 0.918 | lower_is_better  |                         1.15473 |
| median_nearest_distance_m |     11707.1      | 11446.9        | 1626.09      |   0.160055 |                 0.554 | lower_is_better  |                         1.02274 |
| pct_within_1km            |         0        |     0.00711765 |    0.0142854 |  -0.498246 |                 1     | higher_is_better |                         0       |
| pct_within_5km            |         0.176471 |     0.155412   |    0.0646591 |   0.32569  |                 0.434 | higher_is_better |                         1.1355  |

## Interpretation

Low Monte Carlo p-values indicate that observed RDE hotspots align with this independent validation dataset more strongly than expected under randomized hotspot placement.