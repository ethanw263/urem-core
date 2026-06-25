from pathlib import Path
import geopandas as gpd


def load_validation_layer(path: str, target_crs=None) -> gpd.GeoDataFrame:
    """
    Load and lightly clean a validation layer.

    Supports files readable by GeoPandas/pyogrio:
    - GPKG
    - SHP
    - GeoJSON
    - FileGDB layers when path points to a readable layer/file

    Assumption:
    Standardization scripts should already have converted complex raw data
    into clean GeoPackages before validation.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Validation dataset not found: {path}")

    gdf = gpd.read_file(path)

    if target_crs is not None and gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    return gdf
