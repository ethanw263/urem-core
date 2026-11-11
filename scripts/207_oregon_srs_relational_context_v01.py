from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

SCRIPT = "207_oregon_srs_relational_context_v01"

BASE = Path("data/processed")
INPUT_FP = BASE / "oregon_srs_pointwise_spatial_oof_v01.gpkg"

OUT_GPKG = BASE / "oregon_srs_relational_context_v01.gpkg"
OUT_CSV = BASE / "oregon_srs_relational_context_v01.csv"
OUT_QA = BASE / "oregon_srs_relational_context_qa_v01.txt"

KEY = "cell_id"
VALID = "srs_valid_engine_row_v01"

RADII_M = [3000, 10000, 25000]

CONTEXT_VARIABLES = [
    "phi_physical_v01",
    "physical_exceptionality_raw_v03",
    "terrain_drama_v03",
    "scenic_coast_v03",
    "relief_norm_v03",
    "slope_norm_v03",
    "coast_proximity_v03",
    "coast_complexity_v03",
]


def log(msg):
    print(f"[{SCRIPT}] {msg}")


def main():
    log("Reading pointwise spatial OOF layer")
    gdf = gpd.read_file(INPUT_FP)

    required = [KEY, VALID, "centroid_x", "centroid_y"]
    for c in required:
        if c not in gdf.columns:
            raise ValueError(f"Missing required column: {c}")

    variables = [c for c in CONTEXT_VARIABLES if c in gdf.columns]
    missing = [c for c in CONTEXT_VARIABLES if c not in gdf.columns]

    if missing:
        log(f"Missing optional context variables: {missing}")

    model_mask = gdf[VALID].astype(bool)
    context_df = gdf.loc[model_mask].copy()
    context_df = context_df.dropna(subset=["centroid_x", "centroid_y"] + variables)

    log(f"Rows available for relational context: {len(context_df):,}")
    log(f"Context variables: {variables}")

    coords = context_df[["centroid_x", "centroid_y"]].astype(float).values
    tree = BallTree(coords, metric="euclidean")

    for radius in RADII_M:
        label = f"{int(radius / 1000)}km"
        log(f"Computing relational context at {label}")

        neighbors = tree.query_radius(coords, r=radius)

        neighbor_counts = []
        for idxs in neighbors:
            # Exclude self if present
            neighbor_counts.append(max(len(idxs) - 1, 0))

        context_df[f"rel_neighbor_count_{label}_v01"] = neighbor_counts

        for var in variables:
            vals = context_df[var].astype(float).values
            means = []
            maxs = []
            mins = []
            contrasts = []

            for i, idxs in enumerate(neighbors):
                idxs = idxs[idxs != i]

                if len(idxs) == 0:
                    means.append(np.nan)
                    maxs.append(np.nan)
                    mins.append(np.nan)
                    contrasts.append(np.nan)
                    continue

                neigh_vals = vals[idxs]
                mean_val = float(np.nanmean(neigh_vals))
                max_val = float(np.nanmax(neigh_vals))
                min_val = float(np.nanmin(neigh_vals))

                means.append(mean_val)
                maxs.append(max_val)
                mins.append(min_val)
                contrasts.append(float(vals[i] - mean_val))

            context_df[f"{var}_mean_{label}_v01"] = means
            context_df[f"{var}_max_{label}_v01"] = maxs
            context_df[f"{var}_min_{label}_v01"] = mins
            context_df[f"{var}_contrast_{label}_v01"] = contrasts

    log("Joining relational context back to full grid")
    new_cols = [
        c for c in context_df.columns
        if c not in gdf.columns or c == KEY
    ]

    context_join = pd.DataFrame(context_df[new_cols]).drop(columns="geometry", errors="ignore")
    out = gdf.merge(context_join, on=KEY, how="left", validate="one_to_one")

    qa = []
    qa.append(f"{SCRIPT} QA")
    qa.append("=" * 80)
    qa.append(f"Input: {INPUT_FP}")
    qa.append(f"Output: {OUT_GPKG}")
    qa.append(f"Rows total: {len(out):,}")
    qa.append(f"Rows context modeled: {len(context_df):,}")
    qa.append(f"Radii meters: {RADII_M}")
    qa.append("")
    qa.append("Context variables:")
    qa.extend([f"- {v}" for v in variables])
    qa.append("")
    qa.append("Neighbor count summaries:")

    for radius in RADII_M:
        label = f"{int(radius / 1000)}km"
        col = f"rel_neighbor_count_{label}_v01"
        qa.append("")
        qa.append(col)
        qa.append(out[col].describe().to_string())

    qa.append("")
    qa.append("Top null rates among new relational columns:")
    rel_cols = [c for c in out.columns if c.endswith("_v01") and ("_3km_" in c or "_10km_" in c or "_25km_" in c or c.startswith("rel_neighbor_count"))]
    qa.append(out[rel_cols].isna().mean().sort_values(ascending=False).head(30).to_string())

    log("Writing outputs")
    out.to_file(OUT_GPKG, driver="GPKG")
    out.drop(columns="geometry").to_csv(OUT_CSV, index=False)
    OUT_QA.write_text("\n".join(qa), encoding="utf-8")

    log(f"Created {OUT_GPKG}")
    log(f"Created {OUT_CSV}")
    log(f"Created {OUT_QA}")
    log("Done")


if __name__ == "__main__":
    main()