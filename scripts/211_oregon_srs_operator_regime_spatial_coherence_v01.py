from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from scipy.stats import permutation_test
from sklearn.cluster import DBSCAN


SCRIPT = "211_oregon_srs_operator_regime_spatial_coherence_v01"

BASE = Path("data/processed")

INPUT_FP = BASE / "oregon_srs_relational_gain_regimes_v01.gpkg"

OUT_GPKG = BASE / "oregon_srs_operator_regime_spatial_coherence_v01.gpkg"
OUT_CLUSTER_CSV = BASE / "oregon_srs_operator_regime_clusters_v01.csv"
OUT_PERM_CSV = BASE / "oregon_srs_operator_regime_permutation_v01.csv"
OUT_QA = BASE / "oregon_srs_operator_regime_spatial_coherence_qa_v01.txt"


VALID = "srs_valid_engine_row_v01"
GAIN = "relational_gain_v01"

NEIGHBOR_RADIUS_M = 1500.0
CLUSTER_EPS_M = 1600.0
CLUSTER_MIN_SAMPLES = 5

N_PERMUTATIONS = 999
RANDOM_SEED = 42


def log(msg):
    print(f"[{SCRIPT}] {msg}")


def local_same_sign_share(values, neighbor_lists):
    result = np.full(len(values), np.nan)

    signs = np.sign(values)

    for i, neighbors in enumerate(neighbor_lists):
        neighbors = [j for j in neighbors if j != i]

        if len(neighbors) == 0:
            continue

        own_sign = signs[i]
        neighbor_signs = signs[neighbors]

        result[i] = np.mean(neighbor_signs == own_sign)

    return result


def mean_neighbor_gain(values, neighbor_lists):
    result = np.full(len(values), np.nan)

    for i, neighbors in enumerate(neighbor_lists):
        neighbors = [j for j in neighbors if j != i]

        if len(neighbors) == 0:
            continue

        result[i] = np.mean(values[neighbors])

    return result


def global_same_sign_agreement(values, neighbor_lists):
    signs = np.sign(values)

    agreements = []

    for i, neighbors in enumerate(neighbor_lists):
        for j in neighbors:
            if j <= i:
                continue

            agreements.append(signs[i] == signs[j])

    if len(agreements) == 0:
        return np.nan

    return float(np.mean(agreements))


def permutation_same_sign_test(
    values,
    neighbor_lists,
    n_permutations=999,
    seed=42,
):
    rng = np.random.default_rng(seed)

    observed = global_same_sign_agreement(
        values,
        neighbor_lists,
    )

    permuted_stats = []

    for _ in range(n_permutations):
        permuted = rng.permutation(values)

        stat = global_same_sign_agreement(
            permuted,
            neighbor_lists,
        )

        permuted_stats.append(stat)

    permuted_stats = np.asarray(permuted_stats)

    p_value = (
        1
        + np.sum(permuted_stats >= observed)
    ) / (
        n_permutations + 1
    )

    return observed, permuted_stats, p_value


def build_clusters(coords, gains):
    labels = np.full(len(coords), -1, dtype=int)

    relational_idx = np.where(gains > 0)[0]
    pointwise_idx = np.where(gains < 0)[0]

    cluster_counter = 0

    for idx in [relational_idx, pointwise_idx]:

        if len(idx) == 0:
            continue

        model = DBSCAN(
            eps=CLUSTER_EPS_M,
            min_samples=CLUSTER_MIN_SAMPLES,
        )

        local_labels = model.fit_predict(coords[idx])

        unique_clusters = sorted(
            c for c in np.unique(local_labels)
            if c >= 0
        )

        for local_cluster in unique_clusters:
            members = idx[
                local_labels == local_cluster
            ]

            labels[members] = cluster_counter
            cluster_counter += 1

    return labels


def main():

    log("Reading relational gain regime layer")

    gdf = gpd.read_file(INPUT_FP)

    required = [
        "cell_id",
        VALID,
        GAIN,
        "gain_regime_v01",
    ]

    for col in required:
        if col not in gdf.columns:
            raise ValueError(
                f"Missing required column: {col}"
            )

    mask = (
        gdf[VALID].astype(bool)
        & gdf[GAIN].notna()
    )

    work = gdf.loc[mask].copy()

    log(
        f"Rows available for spatial coherence analysis: "
        f"{len(work):,}"
    )

    centroids = work.geometry.centroid

    coords = np.column_stack(
        [
            centroids.x.to_numpy(),
            centroids.y.to_numpy(),
        ]
    )

    gains = work[GAIN].to_numpy(dtype=float)

    log(
        f"Building {NEIGHBOR_RADIUS_M / 1000:.1f} km "
        "local neighborhood graph"
    )

    tree = cKDTree(coords)

    neighbor_lists = tree.query_ball_point(
        coords,
        r=NEIGHBOR_RADIUS_M,
    )

    neighbor_counts = np.array(
        [
            len(n) - 1
            for n in neighbor_lists
        ]
    )

    work["operator_neighbor_count_v01"] = neighbor_counts

    work["operator_same_sign_share_v01"] = (
        local_same_sign_share(
            gains,
            neighbor_lists,
        )
    )

    work["operator_neighbor_mean_gain_v01"] = (
        mean_neighbor_gain(
            gains,
            neighbor_lists,
        )
    )

    work["operator_local_coherence_v01"] = (
        work["operator_same_sign_share_v01"]
    )

    work["operator_spatial_regime_v01"] = None

    work.loc[
        (
            (work[GAIN] > 0)
            & (
                work["operator_same_sign_share_v01"]
                >= 0.75
            )
        ),
        "operator_spatial_regime_v01",
    ] = "coherent_relational"

    work.loc[
        (
            (work[GAIN] < 0)
            & (
                work["operator_same_sign_share_v01"]
                >= 0.75
            )
        ),
        "operator_spatial_regime_v01",
    ] = "coherent_pointwise"

    work.loc[
        work["operator_spatial_regime_v01"].isna(),
        "operator_spatial_regime_v01",
    ] = "mixed_operator_zone"


    log("Running global sign-coherence permutation test")

    observed_agreement, permuted_stats, p_value = (
        permutation_same_sign_test(
            gains,
            neighbor_lists,
            n_permutations=N_PERMUTATIONS,
            seed=RANDOM_SEED,
        )
    )


    log("Building contiguous operator-preference clusters")

    cluster_labels = build_clusters(
        coords,
        gains,
    )

    work["operator_cluster_id_v01"] = cluster_labels


    cluster_rows = []

    valid_clusters = sorted(
        c
        for c in np.unique(cluster_labels)
        if c >= 0
    )

    for cluster_id in valid_clusters:

        sub = work[
            work["operator_cluster_id_v01"]
            == cluster_id
        ]

        mean_gain = sub[GAIN].mean()

        if mean_gain > 0:
            preference = "relational"
        else:
            preference = "pointwise"

        cluster_rows.append(
            {
                "operator_cluster_id_v01": cluster_id,
                "operator_preference": preference,
                "n_cells": len(sub),
                "mean_gain": mean_gain,
                "median_gain": sub[GAIN].median(),
                "mean_abs_gain": sub[GAIN].abs().mean(),
                "mean_phi": sub[
                    "phi_physical_v01"
                ].mean(),
                "mean_y": sub[
                    "y_observed_v01"
                ].mean(),
                "mean_coast_proximity": sub[
                    "coast_proximity_v03"
                ].mean(),
                "mean_coast_complexity": sub[
                    "coast_complexity_v03"
                ].mean(),
                "mean_same_sign_share": sub[
                    "operator_same_sign_share_v01"
                ].mean(),
                "dominant_fold": sub[
                    "spatial_fold_v01"
                ].mode().iloc[0],
            }
        )

    cluster_summary = pd.DataFrame(cluster_rows)

    if len(cluster_summary):
        cluster_summary = cluster_summary.sort_values(
            [
                "n_cells",
                "mean_abs_gain",
            ],
            ascending=[
                False,
                False,
            ],
        )


    regime_summary = (
        work.groupby("operator_spatial_regime_v01")
        .agg(
            n=(GAIN, "size"),
            mean_gain=(GAIN, "mean"),
            median_gain=(GAIN, "median"),
            mean_abs_gain=(GAIN, lambda s: s.abs().mean()),
            mean_same_sign_share=(
                "operator_same_sign_share_v01",
                "mean",
            ),
            mean_phi=(
                "phi_physical_v01",
                "mean",
            ),
            mean_y=(
                "y_observed_v01",
                "mean",
            ),
            mean_coast_proximity=(
                "coast_proximity_v03",
                "mean",
            ),
            mean_coast_complexity=(
                "coast_complexity_v03",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "mean_gain",
            ascending=False,
        )
    )


    permutation_df = pd.DataFrame(
        {
            "permutation_id": np.arange(
                1,
                len(permuted_stats) + 1,
            ),
            "same_sign_agreement": permuted_stats,
        }
    )


    log("Joining spatial coherence results to full grid")

    result_cols = [
        "cell_id",
        "operator_neighbor_count_v01",
        "operator_same_sign_share_v01",
        "operator_neighbor_mean_gain_v01",
        "operator_local_coherence_v01",
        "operator_spatial_regime_v01",
        "operator_cluster_id_v01",
    ]

    result = gdf.merge(
        work[result_cols],
        on="cell_id",
        how="left",
    )


    qa = []

    qa.append(f"{SCRIPT} QA")
    qa.append("=" * 80)

    qa.append(f"Input: {INPUT_FP}")
    qa.append(f"Output: {OUT_GPKG}")

    qa.append("")
    qa.append(f"Rows total: {len(result):,}")
    qa.append(f"Rows analyzed: {len(work):,}")

    qa.append("")
    qa.append("Spatial parameters:")
    qa.append(
        f"Neighborhood radius m: {NEIGHBOR_RADIUS_M}"
    )
    qa.append(
        f"Cluster eps m: {CLUSTER_EPS_M}"
    )
    qa.append(
        f"Cluster minimum samples: {CLUSTER_MIN_SAMPLES}"
    )

    qa.append("")
    qa.append("Neighbor count summary:")
    qa.append(
        work[
            "operator_neighbor_count_v01"
        ].describe().to_string()
    )

    qa.append("")
    qa.append("Global sign-coherence test:")
    qa.append(
        f"Observed adjacent same-sign agreement: "
        f"{observed_agreement}"
    )
    qa.append(
        f"Permutation mean: "
        f"{np.mean(permuted_stats)}"
    )
    qa.append(
        f"Permutation std: "
        f"{np.std(permuted_stats)}"
    )
    qa.append(
        f"Permutation p-value: {p_value}"
    )

    qa.append("")
    qa.append("Spatial regime counts:")
    qa.append(
        work[
            "operator_spatial_regime_v01"
        ].value_counts().to_string()
    )

    qa.append("")
    qa.append("Spatial regime summary:")
    qa.append(
        regime_summary.to_string(index=False)
    )

    qa.append("")
    qa.append(
        f"Detected operator clusters: "
        f"{len(valid_clusters)}"
    )

    qa.append("")
    qa.append("Top operator clusters:")
    if len(cluster_summary):
        qa.append(
            cluster_summary.head(40).to_string(
                index=False
            )
        )
    else:
        qa.append("No clusters detected.")


    log("Writing outputs")

    result.to_file(
        OUT_GPKG,
        driver="GPKG",
    )

    cluster_summary.to_csv(
        OUT_CLUSTER_CSV,
        index=False,
    )

    permutation_df.to_csv(
        OUT_PERM_CSV,
        index=False,
    )

    OUT_QA.write_text(
        "\n".join(qa),
        encoding="utf-8",
    )

    log(f"Created {OUT_GPKG}")
    log(f"Created {OUT_CLUSTER_CSV}")
    log(f"Created {OUT_PERM_CSV}")
    log(f"Created {OUT_QA}")

    log("Done")


if __name__ == "__main__":
    main()