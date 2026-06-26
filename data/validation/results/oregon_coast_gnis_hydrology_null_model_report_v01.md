# Oregon Coast GNIS Hydrology Null Model Validation Report

Simulations: 500

## Dataset Metadata

- Dataset: GNIS Hydrology
- Source: USGS GNIS FullModel / Gazetteer OR
- Category: geographic_recognition_hydrology
- Independent of model: True
- Weight: 0.15

## Observed Metrics

|   n_hotspots |   pct_intersects_any |   mean_overlap_pct |   median_overlap_pct |   pct_any_overlap |   mean_nearest_distance_m |   median_nearest_distance_m |   pct_within_1km |   pct_within_5km |
|-------------:|---------------------:|-------------------:|---------------------:|------------------:|--------------------------:|----------------------------:|-----------------:|-----------------:|
|           34 |             0.117647 |                  0 |                    0 |                 0 |                   3252.18 |                     3162.55 |         0.176471 |         0.705882 |

## Observed vs Null

| metric                    |   observed_value |   null_mean |    null_std |   z_score |   monte_carlo_p_value | direction        |   effect_ratio_observed_to_null |
|:--------------------------|-----------------:|------------:|------------:|----------:|----------------------:|:-----------------|--------------------------------:|
| pct_intersects_any        |         0.117647 |    0.404    |   0.0764242 |  -3.74689 |                 1     | higher_is_better |                        0.291206 |
| mean_overlap_pct          |         0        |    0        |   0         | nan       |                 1     | higher_is_better |                      nan        |
| median_overlap_pct        |         0        |    0        |   0         | nan       |                 1     | higher_is_better |                      nan        |
| pct_any_overlap           |         0        |    0        |   0         | nan       |                 1     | higher_is_better |                      nan        |
| mean_nearest_distance_m   |      3252.18     | 1916.28     | 230.309     |   5.80049 |                 1     | lower_is_better  |                        1.69713  |
| median_nearest_distance_m |      3162.55     | 1641.18     | 249.034     |   6.10907 |                 1     | lower_is_better  |                        1.927    |
| pct_within_1km            |         0.176471 |    0.268471 |   0.0757711 |  -1.21418 |                 0.938 | higher_is_better |                        0.657318 |
| pct_within_5km            |         0.705882 |    0.974647 |   0.0273495 |  -9.82703 |                 1     | higher_is_better |                        0.724244 |

## Interpretation

Low Monte Carlo p-values indicate that observed RDE hotspots align with this independent validation dataset more strongly than expected under randomized hotspot placement.