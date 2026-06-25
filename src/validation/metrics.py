import numpy as np
import geopandas as gpd


def safe_union(gdf: gpd.GeoDataFrame):
    """
    Compatibility wrapper for union_all / unary_union.
    """

    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()

    return gdf.geometry.unary_union


def intersection_count(geom, layer: gpd.GeoDataFrame) -> int:
    if layer.empty:
        return 0

    return int(layer.intersects(geom).sum())


def overlap_area_pct(geom, layer: gpd.GeoDataFrame) -> float:
    if layer.empty or geom is None or geom.area == 0:
        return 0.0

    hits = layer[layer.intersects(geom)]

    if hits.empty:
        return 0.0

    inter_area = hits.geometry.intersection(geom).area.sum()

    return float(inter_area / geom.area)


def nearest_distance(geom, layer: gpd.GeoDataFrame) -> float:
    if layer.empty:
        return np.nan

    centroid = geom.centroid
    distances = layer.geometry.distance(centroid)

    return float(distances.min())


def density_within_distance(
    geom,
    layer: gpd.GeoDataFrame,
    distance_m: float,
) -> int:
    if layer.empty:
        return 0

    centroid = geom.centroid
    distances = layer.geometry.distance(centroid)

    return int((distances <= distance_m).sum())


def support_from_distance(distance_m: float, thresholds=(1000, 5000, 10000)) -> str:
    if np.isnan(distance_m):
        return "no_data"

    if distance_m <= thresholds[0]:
        return "very_near"

    if distance_m <= thresholds[1]:
        return "near"

    if distance_m <= thresholds[2]:
        return "moderate_distance"

    return "far"


def support_from_overlap(overlap_pct: float) -> str:
    if overlap_pct >= 0.50:
        return "strong_overlap"

    if overlap_pct >= 0.25:
        return "moderate_overlap"

    if overlap_pct > 0:
        return "weak_overlap"

    return "no_overlap"
