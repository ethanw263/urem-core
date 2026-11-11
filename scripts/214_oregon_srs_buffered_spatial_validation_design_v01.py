from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans


SCRIPT = "214_oregon_srs_buffered_spatial_validation_design_v01"

BASE = Path("data/processed")

INPUT_FP = (
    BASE
    / "oregon_srs_relational_context_v01.gpkg"
)

OUT_GPKG = (
    BASE
    / "oregon_srs_buffered_spatial_validation_design_v01.gpkg"
)

OUT_FOLD_CSV = (
    BASE
    / "oregon_srs_buffered_spatial_validation_folds_v01.csv"
)

OUT_BUFFER_CSV = (
    BASE
    / "oregon_srs_buffered_spatial_validation_buffers_v01.csv"
)

OUT_QA = (
    BASE
    / "oregon_srs_buffered_spatial_validation_design_qa_v01.txt"
)


VALID = "srs_valid_engine_row_v01"

N_FOLDS = 8

BUFFER_DISTANCES_M = [
    0,
    3000,
    10000,
    25000,
]

RANDOM_SEED = 42


def log(msg):
    print(f"[{SCRIPT}] {msg}")


def main():

    log("Reading relational context layer")

    gdf = gpd.read_file(INPUT_FP)

    required = [
        "cell_id",
        VALID,
        "centroid_x",
        "centroid_y",
    ]

    for col in required:

        if col not in gdf.columns:

            raise ValueError(
                f"Missing required column: {col}"
            )


    # -----------------------------------------------------
    # Valid modeling population
    # -----------------------------------------------------

    valid_mask = (
        gdf[VALID].astype(bool)
    )

    work = gdf.loc[valid_mask].copy()

    log(
        f"Valid rows for validation design: "
        f"{len(work):,}"
    )


    # -----------------------------------------------------
    # Build coordinate matrix
    # -----------------------------------------------------

    coords = work[
        [
            "centroid_x",
            "centroid_y",
        ]
    ].to_numpy(dtype=float)


    # -----------------------------------------------------
    # Normalize coordinate axes before clustering
    #
    # This prevents the longer north-south Oregon coast
    # dimension from completely dominating the fold design.
    # -----------------------------------------------------

    coords_scaled = (
        coords
        - coords.mean(axis=0)
    ) / coords.std(axis=0)


    # -----------------------------------------------------
    # Spatially compact KMeans regions
    #
    # These are geographically compact test blocks.
    # KMeans is used only to construct validation geometry,
    # not as a scientific predictive model.
    # -----------------------------------------------------

    log(
        f"Constructing {N_FOLDS} compact spatial folds"
    )

    km = KMeans(
        n_clusters=N_FOLDS,
        random_state=RANDOM_SEED,
        n_init=50,
    )

    fold_labels = km.fit_predict(
        coords_scaled
    )

    work[
        "buffered_spatial_fold_v01"
    ] = fold_labels


    # -----------------------------------------------------
    # Fold geometry summary
    # -----------------------------------------------------

    fold_rows = []

    for fold in sorted(
        work[
            "buffered_spatial_fold_v01"
        ].unique()
    ):

        sub = work[
            work[
                "buffered_spatial_fold_v01"
            ] == fold
        ]

        fold_rows.append(
            {
                "fold": int(fold),

                "n_test": len(sub),

                "share_of_valid_rows":
                    len(sub) / len(work),

                "x_min":
                    sub["centroid_x"].min(),

                "x_max":
                    sub["centroid_x"].max(),

                "y_min":
                    sub["centroid_y"].min(),

                "y_max":
                    sub["centroid_y"].max(),

                "x_span_m":
                    sub["centroid_x"].max()
                    - sub["centroid_x"].min(),

                "y_span_m":
                    sub["centroid_y"].max()
                    - sub["centroid_y"].min(),
            }
        )


    fold_df = pd.DataFrame(
        fold_rows
    ).sort_values(
        "fold"
    )


    # -----------------------------------------------------
    # Buffered train/test audit
    #
    # For each test fold and buffer distance:
    #
    # test = fold cells
    #
    # excluded = non-test cells within buffer distance
    #            of any test centroid
    #
    # train = remaining cells
    # -----------------------------------------------------

    buffer_rows = []


    all_xy = coords


    for fold in sorted(
        work[
            "buffered_spatial_fold_v01"
        ].unique()
    ):

        test_mask = (
            work[
                "buffered_spatial_fold_v01"
            ].to_numpy()
            == fold
        )

        test_xy = all_xy[test_mask]

        non_test_mask = ~test_mask
        non_test_xy = all_xy[non_test_mask]


        # ---------------------------------------------
        # Chunked nearest-test-point distance
        #
        # Avoids creating a huge full distance matrix.
        # ---------------------------------------------

        from scipy.spatial import cKDTree

        tree = cKDTree(test_xy)

        nearest_distance, _ = tree.query(
            non_test_xy,
            k=1,
        )


        for buffer_m in BUFFER_DISTANCES_M:

            if buffer_m == 0:

                excluded_non_test = np.zeros(
                    len(non_test_xy),
                    dtype=bool,
                )

            else:

                excluded_non_test = (
                    nearest_distance
                    <= buffer_m
                )


            n_test = int(
                test_mask.sum()
            )

            n_excluded = int(
                excluded_non_test.sum()
            )

            n_train = int(
                len(non_test_xy)
                - n_excluded
            )


            buffer_rows.append(
                {
                    "fold": int(fold),

                    "buffer_m":
                        int(buffer_m),

                    "n_total_valid":
                        len(work),

                    "n_test":
                        n_test,

                    "n_buffer_excluded":
                        n_excluded,

                    "n_train":
                        n_train,

                    "test_share":
                        n_test / len(work),

                    "excluded_share":
                        n_excluded / len(work),

                    "train_share":
                        n_train / len(work),
                }
            )


    buffer_df = pd.DataFrame(
        buffer_rows
    ).sort_values(
        [
            "buffer_m",
            "fold",
        ]
    )


    # -----------------------------------------------------
    # Join fold labels back to full grid
    # -----------------------------------------------------

    result = gdf.merge(
        work[
            [
                "cell_id",
                "buffered_spatial_fold_v01",
            ]
        ],
        on="cell_id",
        how="left",
    )


    # -----------------------------------------------------
    # QA report
    # -----------------------------------------------------

    qa = []

    qa.append(f"{SCRIPT} QA")
    qa.append("=" * 80)

    qa.append(
        f"Input: {INPUT_FP}"
    )

    qa.append(
        f"Output: {OUT_GPKG}"
    )

    qa.append("")

    qa.append(
        f"Rows total: {len(result):,}"
    )

    qa.append(
        f"Valid modeling rows: {len(work):,}"
    )

    qa.append(
        f"Spatial folds: {N_FOLDS}"
    )


    qa.append("")
    qa.append("Fold size summary:")

    qa.append(
        fold_df[
            [
                "fold",
                "n_test",
                "share_of_valid_rows",
                "x_span_m",
                "y_span_m",
            ]
        ].to_string(
            index=False
        )
    )


    qa.append("")
    qa.append(
        "Fold test-size distribution:"
    )

    qa.append(
        fold_df[
            "n_test"
        ].describe().to_string()
    )


    qa.append("")
    qa.append(
        "Buffered train/test summary:"
    )

    qa.append(
        buffer_df.to_string(
            index=False
        )
    )


    qa.append("")
    qa.append(
        "Mean train/test/exclusion shares "
        "by buffer distance:"
    )

    qa.append(
        buffer_df.groupby(
            "buffer_m"
        )[
            [
                "test_share",
                "excluded_share",
                "train_share",
            ]
        ]
        .mean()
        .to_string()
    )


    qa.append("")
    qa.append(
        "Minimum training rows by buffer:"
    )

    qa.append(
        buffer_df.groupby(
            "buffer_m"
        )[
            "n_train"
        ]
        .min()
        .to_string()
    )


    qa.append("")
    qa.append("Interpretation guardrail:")

    qa.append(
        "This script constructs and audits validation "
        "geometry only. It does not evaluate model "
        "performance. Buffered model comparison should "
        "proceed only after confirming that fold sizes "
        "and retained training populations are adequate."
    )


    # -----------------------------------------------------
    # Write outputs
    # -----------------------------------------------------

    log("Writing outputs")

    result.to_file(
        OUT_GPKG,
        driver="GPKG",
    )

    fold_df.to_csv(
        OUT_FOLD_CSV,
        index=False,
    )

    buffer_df.to_csv(
        OUT_BUFFER_CSV,
        index=False,
    )

    OUT_QA.write_text(
        "\n".join(qa),
        encoding="utf-8",
    )


    log(f"Created {OUT_GPKG}")
    log(f"Created {OUT_FOLD_CSV}")
    log(f"Created {OUT_BUFFER_CSV}")
    log(f"Created {OUT_QA}")

    log("Done")


if __name__ == "__main__":
    main()