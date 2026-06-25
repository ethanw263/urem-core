#!/usr/bin/env python3

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

SCRIPT_NAME = "164_oregon_transition_hotspot_validation_summary_v01"

HOTSPOTS = Path("data/processed/oregon_transition_hotspot_review_package_v01.gpkg")
HOTSPOT_CELLS = Path("data/processed/oregon_recognition_transition_hotspot_cells_v01.gpkg")
DIVERGENCE_ZONES = Path("data/processed/oregon_divergence_source_sink_zones_v02.gpkg")
DISCOVERY_REGIONS = Path("data/processed/oregon_discovery_regions_v06.gpkg")
FLOW_ZONES = Path("data/processed/oregon_recognition_flow_zones_v01.gpkg")

OUTPUT_HOTSPOTS_GPKG = Path("data/processed/oregon_transition_hotspot_validation_summary_v01.gpkg")
OUTPUT_HOTSPOTS_CSV = Path("data/processed/oregon_transition_hotspot_validation_summary_v01.csv")
OUTPUT_SUMMARY_CSV = Path("data/processed/oregon_transition_hotspot_validation_metrics_v01.csv")


def overlap_pct(geom, layer):
    hits = layer[layer.intersects(geom)]
    if hits.empty:
        return 0.0, 0

    inter_area = hits.geometry.intersection(geom).area.sum()
    pct = inter_area / geom.area if geom.area > 0 else 0.0

    return pct, len(hits)


def classify_validation(row):
    discovery = row["validation_discovery_overlap_pct_v01"]
    divergence = row["validation_divergence_overlap_pct_v01"]
    flow = row["validation_flow_overlap_pct_v01"]

    if discovery >= 0.25 and divergence >= 0.75 and flow >= 0.25:
        return "strong_multilayer_validation"

    if discovery >= 0.25 and divergence >= 0.75:
        return "strong_discovery_divergence_validation"

    if divergence >= 0.75 and flow >= 0.25:
        return "strong_divergence_flow_validation"

    if divergence >= 0.75:
        return "divergence_only_validation"

    if discovery >= 0.25:
        return "discovery_only_validation"

    if flow >= 0.25:
        return "flow_only_validation"

    return "weak_validation"


def main():
    print(f"[{SCRIPT_NAME}] Starting")

    hotspots = gpd.read_file(HOTSPOTS)
    cells = gpd.read_file(HOTSPOT_CELLS)
    divz = gpd.read_file(DIVERGENCE_ZONES)
    disc = gpd.read_file(DISCOVERY_REGIONS)
    flow = gpd.read_file(FLOW_ZONES)

    for name, layer in [
        ("hotspot cells", cells),
        ("divergence zones", divz),
        ("discovery regions", disc),
        ("flow zones", flow),
    ]:
        if layer.crs != hotspots.crs:
            print(f"[{SCRIPT_NAME}] Reprojecting {name}")
            if name == "hotspot cells":
                cells = layer.to_crs(hotspots.crs)
            elif name == "divergence zones":
                divz = layer.to_crs(hotspots.crs)
            elif name == "discovery regions":
                disc = layer.to_crs(hotspots.crs)
            elif name == "flow zones":
                flow = layer.to_crs(hotspots.crs)

    print(f"[{SCRIPT_NAME}] Hotspots: {len(hotspots):,}")
    print(f"[{SCRIPT_NAME}] Hotspot cells: {len(cells):,}")
    print(f"[{SCRIPT_NAME}] Divergence zones: {len(divz):,}")
    print(f"[{SCRIPT_NAME}] Discovery regions: {len(disc):,}")
    print(f"[{SCRIPT_NAME}] Flow zones: {len(flow):,}")

    records = []

    for _, row in hotspots.iterrows():
        geom = row.geometry

        discovery_pct, discovery_count = overlap_pct(geom, disc)
        divergence_pct, divergence_count = overlap_pct(geom, divz)
        flow_pct, flow_count = overlap_pct(geom, flow)

        rec = row.drop(labels="geometry").to_dict()

        rec.update(
            {
                "validation_discovery_overlap_pct_v01": discovery_pct,
                "validation_discovery_count_v01": discovery_count,
                "validation_divergence_overlap_pct_v01": divergence_pct,
                "validation_divergence_count_v01": divergence_count,
                "validation_flow_overlap_pct_v01": flow_pct,
                "validation_flow_count_v01": flow_count,
                "geometry": geom,
            }
        )

        records.append(rec)

    out = gpd.GeoDataFrame(records, geometry="geometry", crs=hotspots.crs)

    out["hotspot_validation_class_v01"] = out.apply(classify_validation, axis=1)

    out["hotspot_validation_score_v01"] = (
        0.35 * out["validation_divergence_overlap_pct_v01"].clip(0, 1)
        + 0.25 * out["validation_discovery_overlap_pct_v01"].clip(0, 1)
        + 0.20 * out["validation_flow_overlap_pct_v01"].clip(0, 1)
        + 0.20 * out["mean_transition_score_v01"].clip(0, 1)
    )

    out = out.sort_values(
        ["hotspot_validation_score_v01", "mean_transition_score_v01"],
        ascending=False,
    ).reset_index(drop=True)

    out["hotspot_validation_rank_v01"] = np.arange(1, len(out) + 1)

    summary_rows = []

    summary_rows.append({
        "metric": "total_hotspots",
        "value": len(out),
    })

    summary_rows.append({
        "metric": "mean_validation_score",
        "value": out["hotspot_validation_score_v01"].mean(),
    })

    summary_rows.append({
        "metric": "mean_discovery_overlap_pct",
        "value": out["validation_discovery_overlap_pct_v01"].mean(),
    })

    summary_rows.append({
        "metric": "mean_divergence_overlap_pct",
        "value": out["validation_divergence_overlap_pct_v01"].mean(),
    })

    summary_rows.append({
        "metric": "mean_flow_overlap_pct",
        "value": out["validation_flow_overlap_pct_v01"].mean(),
    })

    for cls, count in out["hotspot_validation_class_v01"].value_counts().items():
        summary_rows.append({
            "metric": f"class_count__{cls}",
            "value": count,
        })

    summary_df = pd.DataFrame(summary_rows)

    print()
    print(f"[{SCRIPT_NAME}] Validation classes:")
    print(out["hotspot_validation_class_v01"].value_counts())

    print()
    print(f"[{SCRIPT_NAME}] Mean overlaps:")
    print(
        out[
            [
                "validation_discovery_overlap_pct_v01",
                "validation_divergence_overlap_pct_v01",
                "validation_flow_overlap_pct_v01",
                "hotspot_validation_score_v01",
            ]
        ].mean()
    )

    print()
    print(f"[{SCRIPT_NAME}] Top validated hotspots:")
    print(
        out[
            [
                "hotspot_validation_rank_v01",
                "transition_hotspot_rank_v01",
                "transition_hotspot_id_v01",
                "mean_transition_score_v01",
                "validation_discovery_overlap_pct_v01",
                "validation_divergence_overlap_pct_v01",
                "validation_flow_overlap_pct_v01",
                "hotspot_validation_class_v01",
                "hotspot_validation_score_v01",
            ]
        ]
        .head(34)
        .to_string(index=False)
    )

    print()
    print(f"[{SCRIPT_NAME}] Writing GPKG: {OUTPUT_HOTSPOTS_GPKG}")
    out.to_file(OUTPUT_HOTSPOTS_GPKG, driver="GPKG")

    print(f"[{SCRIPT_NAME}] Writing CSV: {OUTPUT_HOTSPOTS_CSV}")
    out.drop(columns="geometry").to_csv(OUTPUT_HOTSPOTS_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Writing summary CSV: {OUTPUT_SUMMARY_CSV}")
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print(f"[{SCRIPT_NAME}] Done")


if __name__ == "__main__":
    main()