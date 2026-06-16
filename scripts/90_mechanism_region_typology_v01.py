#!/usr/bin/env python3
"""
90_mechanism_region_typology_v01.py

Purpose
-------
Create higher-order typologies / archetypes from mechanism regions produced by
Script 89.

This script continues the RDE / mechanism geography pipeline:

Script 86: mechanism separability diagnostics
Script 87: orthogonalized RDE dimensions
Script 88: orthogonalized mechanism taxonomy
Script 89: mechanism region clustering
Script 90: mechanism region typology

Core idea
---------
Mechanism classes explain *what kind* of recognition disequilibrium exists.

This script asks:

    Within each mechanism class, what recurring regional archetypes exist?

Examples:
- Recognition Inefficiency:
    Coastal Hidden Gem Landscapes
    Rugged Scenic Interior Landscapes
    Recognition Bottleneck Landscapes
    Shadowed Destination-Adjacent Landscapes

- Opportunity Failure:
    Access-Limited Landscapes
    Infrastructure-Limited Landscapes
    Terrain-Limited Landscapes

- Comparative Shadowing:
    Tourism Diversion Landscapes
    Adjacent-to-Famous-Place Landscapes
    Recognition Sink Landscapes

Outputs
-------
data/processed/mechanism_region_typology_v01.csv
data/processed/mechanism_region_typology_v01.gpkg
data/processed/mechanism_region_typology_summary_v01.csv
data/processed/mechanism_region_typology_feature_profiles_v01.csv
data/processed/mechanism_region_typology_qa_v01.txt
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


SCRIPT_NAME = "90_mechanism_region_typology_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

OUTPUT_CSV = PROCESSED / "mechanism_region_typology_v01.csv"
OUTPUT_GPKG = PROCESSED / "mechanism_region_typology_v01.gpkg"
OUTPUT_SUMMARY_CSV = PROCESSED / "mechanism_region_typology_summary_v01.csv"
OUTPUT_PROFILE_CSV = PROCESSED / "mechanism_region_typology_feature_profiles_v01.csv"
OUTPUT_QA_TXT = PROCESSED / "mechanism_region_typology_qa_v01.txt"


INPUT_CANDIDATES = [
    PROCESSED / "mechanism_region_clustering_v01.gpkg",
    PROCESSED / "orthogonalized_mechanism_regions_v01.gpkg",
    PROCESSED / "rde_mechanism_regions_v01.gpkg",
    PROCESSED / "mechanism_regions_v01.gpkg",
    PROCESSED / "mechanism_region_clustering_v01.csv",
    PROCESSED / "orthogonalized_mechanism_regions_v01.csv",
    PROCESSED / "rde_mechanism_regions_v01.csv",
    PROCESSED / "mechanism_regions_v01.csv",
]


MECHANISM_PRIORITY = [
    "Recognition Inefficiency",
    "Opportunity Failure",
    "Comparative Shadowing / Recognition Diversion",
    "Comparative Shadowing",
    "Recognition Diversion",
    "Recognized Exceptional Landscapes",
    "Background / Mixed",
    "Background",
    "Mixed",
]


FEATURE_ALIASES = {
    "physical": [
        "p_orthogonal",
        "physical_potential",
        "physical_exceptionality",
        "physical_exceptionality_v03",
        "physical_potential_v01",
        "p",
    ],
    "opportunity": [
        "o_base_opportunity",
        "opportunity_structure",
        "opportunity_structure_index",
        "opportunity_index",
        "o",
    ],
    "transmission": [
        "t_net_transmission",
        "recognition_transmission",
        "transmission_index",
        "t",
    ],
    "under_recognition": [
        "r_net_under_recognition",
        "under_recognition",
        "under_recognition_deficit",
        "recognition_deficit",
        "urem_score",
        "urem_v05",
        "rde_deficit",
    ],
    "observed_recognition": [
        "observed_recognition",
        "observed_recognition_v04",
        "recognition_score_v04",
        "recognition_score",
        "r",
    ],
    "coastal": [
        "coastal_exposure",
        "coastal_proximity",
        "distance_to_coast_inverse",
        "fp_coastal_proximity",
        "coast",
    ],
    "terrain": [
        "relief",
        "slope",
        "terrain_drama",
        "elevation",
        "fp_relief",
        "fp_slope",
        "fp_elevation",
    ],
    "access": [
        "accessibility_opportunity",
        "accessibility",
        "access",
        "roads",
        "parking",
        "trail",
        "trails",
    ],
    "infrastructure": [
        "infrastructure_opportunity",
        "infrastructure",
        "urban",
        "settlement",
        "poi",
    ],
    "exposure": [
        "exposure_opportunity",
        "exposure",
        "visibility",
        "viewshed",
        "tourism_exposure",
    ],
    "institutional": [
        "institutional_transmission",
        "institutional",
        "park",
        "protected",
        "recreation",
    ],
    "shadow": [
        "shadow",
        "diversion",
        "comparative_shadowing",
        "recognized_neighbor",
        "nearby_recognition",
        "destination_pressure",
    ],
}


def read_input() -> gpd.GeoDataFrame:
    found = None
    for path in INPUT_CANDIDATES:
        if path.exists():
            found = path
            break

    if found is None:
        tried = "\n".join(str(p) for p in INPUT_CANDIDATES)
        raise FileNotFoundError(
            "Could not find Script 89 mechanism region output. Tried:\n" + tried
        )

    log.info("Reading input: %s", found)

    if found.suffix.lower() == ".gpkg":
        gdf = gpd.read_file(found)
    else:
        df = pd.read_csv(found)
        gdf = gpd.GeoDataFrame(df, geometry=None)

    if len(gdf) == 0:
        raise ValueError("Input file exists but contains zero rows.")

    return gdf


def normalize_colname(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")


def build_column_lookup(df: pd.DataFrame) -> Dict[str, str]:
    return {normalize_colname(c): c for c in df.columns}


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lookup = build_column_lookup(df)

    for cand in candidates:
        nc = normalize_colname(cand)
        if nc in lookup:
            return lookup[nc]

    for col in df.columns:
        ncol = normalize_colname(col)
        for cand in candidates:
            nc = normalize_colname(cand)
            if nc and nc in ncol:
                return col

    return None


def find_mechanism_col(df: pd.DataFrame) -> str:
    candidates = [
        "mechanism_class",
        "mechanism",
        "orthogonalized_mechanism_class",
        "rde_mechanism_class",
        "class",
        "taxonomy_class",
    ]
    col = find_col(df, candidates)
    if col is None:
        raise ValueError(
            "Could not identify mechanism class column. Expected something like "
            "'mechanism_class', 'mechanism', or 'taxonomy_class'."
        )
    return col


def find_region_col(df: pd.DataFrame) -> str:
    candidates = [
        "mechanism_region_id",
        "region_id",
        "cluster_id",
        "region",
        "component_id",
        "mechanism_cluster_id",
    ]
    col = find_col(df, candidates)
    if col is not None:
        return col

    df["mechanism_region_id"] = np.arange(1, len(df) + 1)
    return "mechanism_region_id"


def canonical_mechanism(value: object) -> str:
    text = str(value).strip()

    low = text.lower()

    if "recognition ineff" in low:
        return "Recognition Inefficiency"
    if "opportunity failure" in low:
        return "Opportunity Failure"
    if "comparative" in low or "shadow" in low or "diversion" in low:
        return "Comparative Shadowing / Recognition Diversion"
    if "recognized exceptional" in low:
        return "Recognized Exceptional Landscapes"
    if "background" in low or "mixed" in low:
        return "Background / Mixed"

    return text


def numeric_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return np.nan
    return float(values.mean())


def numeric_std(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() <= 1:
        return 0.0
    return float(values.std())


def prepare_region_level(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    mechanism_col = find_mechanism_col(gdf)
    region_col = find_region_col(gdf)

    gdf = gdf.copy()
    gdf["mechanism_class"] = gdf[mechanism_col].map(canonical_mechanism)
    gdf["mechanism_region_id"] = gdf[region_col].astype(str)

    has_geometry = isinstance(gdf, gpd.GeoDataFrame) and gdf.geometry.name in gdf.columns

    if not has_geometry:
        log.info("No geometry found. Treating input as already region-level CSV.")
        return gpd.GeoDataFrame(gdf, geometry=None)

    geom_types = set(gdf.geometry.geom_type.dropna().unique())

    duplicated_regions = gdf["mechanism_region_id"].duplicated().any()

    if not duplicated_regions:
        log.info("Input appears to already be region-level.")
        region_gdf = gdf.copy()
    else:
        log.info("Input appears cell-level. Dissolving to region-level.")

        numeric_cols = [
            c
            for c in gdf.columns
            if c not in ["geometry", "mechanism_region_id", "mechanism_class"]
            and pd.api.types.is_numeric_dtype(gdf[c])
        ]

        agg = {c: ["mean", "min", "max", "std"] for c in numeric_cols}
        agg["mechanism_class"] = "first"

        dissolved = gdf.dissolve(by="mechanism_region_id", aggfunc=agg)

        dissolved.columns = [
            "_".join([str(x) for x in col if str(x) != ""]).strip("_")
            if isinstance(col, tuple)
            else col
            for col in dissolved.columns
        ]

        dissolved = dissolved.reset_index()
        if "mechanism_class_first" in dissolved.columns:
            dissolved["mechanism_class"] = dissolved["mechanism_class_first"]
            dissolved = dissolved.drop(columns=["mechanism_class_first"])

        region_gdf = dissolved

    if region_gdf.crs is not None:
        try:
            projected = region_gdf.to_crs(3310)
            area_m2 = projected.geometry.area
            perimeter_m = projected.geometry.length
            region_gdf["region_area_km2"] = area_m2 / 1_000_000
            region_gdf["region_perimeter_km"] = perimeter_m / 1_000
            region_gdf["region_compactness"] = (
                4 * math.pi * area_m2 / np.maximum(perimeter_m ** 2, 1)
            )
            centroids = projected.geometry.centroid.to_crs(region_gdf.crs)
            region_gdf["centroid_lon"] = centroids.x
            region_gdf["centroid_lat"] = centroids.y
        except Exception as exc:
            log.warning("Geometry metrics failed: %s", exc)

    return region_gdf


def feature_score(df: pd.DataFrame, aliases: List[str]) -> pd.Series:
    cols = []
    for alias in aliases:
        col = find_col(df, [alias])
        if col is not None and pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)

    cols = list(dict.fromkeys(cols))

    if not cols:
        return pd.Series(np.nan, index=df.index)

    values = df[cols].apply(pd.to_numeric, errors="coerce")
    return values.mean(axis=1)


def add_feature_bundles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for bundle, aliases in FEATURE_ALIASES.items():
        df[f"bundle_{bundle}"] = feature_score(df, aliases)

    if "region_area_km2" in df.columns:
        area = pd.to_numeric(df["region_area_km2"], errors="coerce")
        df["bundle_scale"] = (area - area.min()) / max(area.max() - area.min(), 1e-9)
    else:
        df["bundle_scale"] = np.nan

    return df


def percentile_within_mechanism(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("mechanism_class")[col].rank(pct=True, method="average")


def add_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bundle_cols = [c for c in df.columns if c.startswith("bundle_")]

    for col in bundle_cols:
        df[f"{col}_pct_mech"] = percentile_within_mechanism(df, col)

    return df


def choose_k(x: np.ndarray, n: int) -> int:
    if n < 6:
        return 1

    max_k = min(5, n - 1)
    best_k = 2
    best_score = -999

    for k in range(2, max_k + 1):
        try:
            labels = KMeans(n_clusters=k, random_state=42, n_init=25).fit_predict(x)
            score = silhouette_score(x, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    return best_k


def cluster_within_mechanisms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["typology_cluster"] = -1

    feature_cols = [
        "bundle_physical",
        "bundle_opportunity",
        "bundle_transmission",
        "bundle_under_recognition",
        "bundle_observed_recognition",
        "bundle_coastal",
        "bundle_terrain",
        "bundle_access",
        "bundle_infrastructure",
        "bundle_exposure",
        "bundle_institutional",
        "bundle_shadow",
        "bundle_scale",
    ]

    existing = [c for c in feature_cols if c in df.columns]
    usable = [
        c for c in existing if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 3
    ]

    if len(usable) < 3:
        log.warning("Limited usable features for clustering. Archetypes will be rule-based.")
        return df

    for mechanism, sub in df.groupby("mechanism_class"):
        idx = sub.index

        xdf = sub[usable].apply(pd.to_numeric, errors="coerce")
        xdf = xdf.fillna(xdf.median()).fillna(0)

        if len(xdf) < 4:
            df.loc[idx, "typology_cluster"] = 0
            continue

        x = StandardScaler().fit_transform(xdf)
        k = choose_k(x, len(xdf))

        if k <= 1:
            labels = np.zeros(len(xdf), dtype=int)
        else:
            labels = KMeans(n_clusters=k, random_state=42, n_init=50).fit_predict(x)

        df.loc[idx, "typology_cluster"] = labels

        log.info("Mechanism '%s': n=%s, k=%s", mechanism, len(xdf), k)

    return df


def high(row: pd.Series, name: str, threshold: float = 0.67) -> bool:
    return pd.notna(row.get(name)) and row.get(name) >= threshold


def low(row: pd.Series, name: str, threshold: float = 0.33) -> bool:
    return pd.notna(row.get(name)) and row.get(name) <= threshold


def assign_archetype(row: pd.Series) -> str:
    mech = row["mechanism_class"]

    physical_hi = high(row, "bundle_physical_pct_mech")
    opportunity_hi = high(row, "bundle_opportunity_pct_mech")
    opportunity_lo = low(row, "bundle_opportunity_pct_mech")
    transmission_hi = high(row, "bundle_transmission_pct_mech")
    transmission_lo = low(row, "bundle_transmission_pct_mech")
    underrec_hi = high(row, "bundle_under_recognition_pct_mech")
    coastal_hi = high(row, "bundle_coastal_pct_mech")
    terrain_hi = high(row, "bundle_terrain_pct_mech")
    access_lo = low(row, "bundle_access_pct_mech")
    infrastructure_lo = low(row, "bundle_infrastructure_pct_mech")
    exposure_lo = low(row, "bundle_exposure_pct_mech")
    shadow_hi = high(row, "bundle_shadow_pct_mech")
    scale_hi = high(row, "bundle_scale_pct_mech")

    if mech == "Recognition Inefficiency":
        if coastal_hi and physical_hi and transmission_hi:
            return "Coastal Hidden Gem Landscape"
        if terrain_hi and physical_hi and not coastal_hi:
            return "Rugged Scenic Interior Landscape"
        if transmission_hi and exposure_lo:
            return "Exposure Bottleneck Landscape"
        if transmission_hi and underrec_hi and shadow_hi:
            return "Shadowed Destination-Adjacent Landscape"
        if physical_hi and opportunity_hi and transmission_hi:
            return "High-Potential Recognition Failure Landscape"
        if scale_hi:
            return "Large-Scale Latent Recognition Landscape"
        return "General Recognition Inefficiency Landscape"

    if mech == "Opportunity Failure":
        if access_lo:
            return "Access-Limited Landscape"
        if infrastructure_lo:
            return "Infrastructure-Limited Landscape"
        if terrain_hi and opportunity_lo:
            return "Terrain-Constrained Opportunity Landscape"
        if coastal_hi and opportunity_lo:
            return "Coastal Opportunity Gap Landscape"
        if physical_hi and opportunity_lo:
            return "High-Potential Opportunity Failure Landscape"
        return "General Opportunity Failure Landscape"

    if mech == "Comparative Shadowing / Recognition Diversion":
        if shadow_hi and coastal_hi:
            return "Coastal Recognition Diversion Landscape"
        if shadow_hi and terrain_hi:
            return "Scenic Shadow Zone Landscape"
        if transmission_hi and underrec_hi:
            return "Recognition Sink Landscape"
        if exposure_lo:
            return "Tourism Diversion Landscape"
        return "General Comparative Shadowing Landscape"

    if mech == "Recognized Exceptional Landscapes":
        if coastal_hi:
            return "Recognized Coastal Exceptional Landscape"
        if terrain_hi:
            return "Recognized Rugged Exceptional Landscape"
        return "Recognized Exceptional Landscape"

    return "Background / Mixed Recognition Landscape"


def add_archetypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["region_archetype"] = df.apply(assign_archetype, axis=1)

    df["typology_code"] = (
        df["mechanism_class"]
        .str.replace(r"[^A-Za-z0-9]+", "_", regex=True)
        .str.strip("_")
        + "__"
        + df["region_archetype"]
        .str.replace(r"[^A-Za-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    return df


def archetype_strength(row: pd.Series) -> float:
    vals = []

    for col in [
        "bundle_physical_pct_mech",
        "bundle_opportunity_pct_mech",
        "bundle_transmission_pct_mech",
        "bundle_under_recognition_pct_mech",
    ]:
        v = row.get(col)
        if pd.notna(v):
            vals.append(float(v))

    if not vals:
        return np.nan

    mech = row["mechanism_class"]

    if mech == "Recognition Inefficiency":
        return float(np.nanmean([
            row.get("bundle_physical_pct_mech", np.nan),
            row.get("bundle_transmission_pct_mech", np.nan),
            row.get("bundle_under_recognition_pct_mech", np.nan),
        ]))

    if mech == "Opportunity Failure":
        opportunity_inverse = (
            1 - row.get("bundle_opportunity_pct_mech")
            if pd.notna(row.get("bundle_opportunity_pct_mech"))
            else np.nan
        )
        return float(np.nanmean([
            row.get("bundle_physical_pct_mech", np.nan),
            opportunity_inverse,
            row.get("bundle_under_recognition_pct_mech", np.nan),
        ]))

    if mech == "Comparative Shadowing / Recognition Diversion":
        return float(np.nanmean([
            row.get("bundle_shadow_pct_mech", np.nan),
            row.get("bundle_under_recognition_pct_mech", np.nan),
            row.get("bundle_transmission_pct_mech", np.nan),
        ]))

    return float(np.nanmean(vals))


def add_strength(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["archetype_strength"] = df.apply(archetype_strength, axis=1)
    return df


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["mechanism_class", "region_archetype"]

    agg = {
        "mechanism_region_id": "count",
        "archetype_strength": "mean",
    }

    for c in [
        "bundle_physical",
        "bundle_opportunity",
        "bundle_transmission",
        "bundle_under_recognition",
        "bundle_observed_recognition",
        "bundle_coastal",
        "bundle_terrain",
        "bundle_access",
        "bundle_infrastructure",
        "bundle_exposure",
        "bundle_institutional",
        "bundle_shadow",
        "region_area_km2",
    ]:
        if c in df.columns:
            agg[c] = "mean"

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(agg)
        .reset_index()
        .rename(columns={"mechanism_region_id": "region_count"})
    )

    summary = summary.sort_values(
        ["mechanism_class", "region_count", "archetype_strength"],
        ascending=[True, False, False],
    )

    return summary


def make_profiles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    bundle_cols = [c for c in df.columns if c.startswith("bundle_") and not c.endswith("_pct_mech")]

    for archetype, sub in df.groupby("region_archetype"):
        row = {
            "region_archetype": archetype,
            "region_count": len(sub),
            "dominant_mechanism": sub["mechanism_class"].mode().iloc[0],
            "mean_archetype_strength": numeric_mean(sub["archetype_strength"]),
        }

        for c in bundle_cols:
            row[f"mean_{c}"] = numeric_mean(sub[c])
            row[f"std_{c}"] = numeric_std(sub[c])

        rows.append(row)

    profiles = pd.DataFrame(rows)
    profiles = profiles.sort_values(
        ["dominant_mechanism", "region_count", "mean_archetype_strength"],
        ascending=[True, False, False],
    )

    return profiles


def write_outputs(gdf: gpd.GeoDataFrame, summary: pd.DataFrame, profiles: pd.DataFrame) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    log.info("Writing CSV: %s", OUTPUT_CSV)
    pd.DataFrame(gdf.drop(columns="geometry", errors="ignore")).to_csv(OUTPUT_CSV, index=False)

    if isinstance(gdf, gpd.GeoDataFrame) and gdf.geometry.name in gdf.columns:
        log.info("Writing GPKG: %s", OUTPUT_GPKG)
        gdf.to_file(OUTPUT_GPKG, layer="mechanism_region_typology_v01", driver="GPKG")

    log.info("Writing summary CSV: %s", OUTPUT_SUMMARY_CSV)
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    log.info("Writing profile CSV: %s", OUTPUT_PROFILE_CSV)
    profiles.to_csv(OUTPUT_PROFILE_CSV, index=False)


def write_qa(gdf: gpd.GeoDataFrame, summary: pd.DataFrame, profiles: pd.DataFrame) -> None:
    mechanism_counts = gdf["mechanism_class"].value_counts(dropna=False).to_dict()
    archetype_counts = gdf["region_archetype"].value_counts(dropna=False).to_dict()

    qa = {
        "script": SCRIPT_NAME,
        "region_count": int(len(gdf)),
        "mechanism_counts": mechanism_counts,
        "archetype_count": int(gdf["region_archetype"].nunique()),
        "archetype_counts": archetype_counts,
        "mean_archetype_strength": float(pd.to_numeric(gdf["archetype_strength"], errors="coerce").mean()),
        "outputs": {
            "csv": str(OUTPUT_CSV),
            "gpkg": str(OUTPUT_GPKG),
            "summary_csv": str(OUTPUT_SUMMARY_CSV),
            "profiles_csv": str(OUTPUT_PROFILE_CSV),
            "qa_txt": str(OUTPUT_QA_TXT),
        },
    }

    text = []
    text.append("Mechanism Region Typology v01 QA")
    text.append("=" * 40)
    text.append("")
    text.append(json.dumps(qa, indent=2))
    text.append("")
    text.append("Top archetypes by region count:")
    text.append(summary.head(20).to_string(index=False))
    text.append("")
    text.append("Feature profile preview:")
    text.append(profiles.head(20).to_string(index=False))

    log.info("Writing QA TXT: %s", OUTPUT_QA_TXT)
    OUTPUT_QA_TXT.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    log.info("Starting Script 90: mechanism region typology v01")

    raw = read_input()
    regions = prepare_region_level(raw)

    log.info("Input region rows: %s", len(regions))

    regions = add_feature_bundles(regions)
    regions = add_percentiles(regions)
    regions = cluster_within_mechanisms(regions)
    regions = add_archetypes(regions)
    regions = add_strength(regions)

    summary = make_summary(regions)
    profiles = make_profiles(regions)

    write_outputs(regions, summary, profiles)
    write_qa(regions, summary, profiles)

    log.info("Done")
    print("\nTypology summary:")
    print(summary[["mechanism_class", "region_archetype", "region_count", "archetype_strength"]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()