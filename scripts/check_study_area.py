# scripts/check_study_area.py

import geopandas as gpd

gdf = gpd.read_file("data/processed/study_area_25km.gpkg")

print(gdf.head())
print()
print("CRS:", gdf.crs)
print("Features:", len(gdf))
print("Area km²:", gdf.geometry.area.sum() / 1_000_000)
print("Bounds:", gdf.total_bounds)