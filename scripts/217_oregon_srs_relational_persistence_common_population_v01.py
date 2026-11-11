from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


SCRIPT = "217_oregon_srs_relational_persistence_common_population_v01"

BASE = Path("data/processed")

INPUT_DECAY_GPKG = BASE / "oregon_srs_relational_gain_decay_profile_v01.gpkg"
INPUT_DECAY_CELLS = BASE / "oregon_srs_relational_gain_decay_cells_v01.csv"
INPUT_DECAY_FOLDS = BASE / "oregon_srs_relational_gain_decay_folds_v01.csv"

OUT_GPKG = BASE / "oregon_srs_relational_persistence_common_population_v01.gpkg"
OUT_CELL_CSV = BASE / "oregon_srs_relational_persistence_common_population_cells_v01.csv"
OUT_FOLD_CSV = BASE / "oregon_srs_relational_persistence_common_population_folds_v01.csv"
OUT_QA = BASE / "oregon_srs_relational_persistence_common_population_qa_v01.txt"

BUFFERS_M = [0, 1000, 3000, 5000, 10000, 15000, 20000, 25000, 35000, 50000]


def log(msg):
    print(f"[{SCRIPT}] {msg}")


def classify_persistence(row):
    share = row["share_buffers_relational_better_v01"]
    sign_changes = row["gain_sign_changes_v01"]
    mean_gain = row["mean_gain_across_buffers_v01"]

    if share >= 0.8 and mean_gain > 0:
        return "persistent_relational"
    if share <= 0.2 and mean_gain < 0:
        return "persistent_pointwise"
    if sign_changes >= 3:
        return "unstable_multiscale"
    if mean_gain > 0:
        return "weak_relational_persistence"
    if mean_gain < 0:
        return "weak_pointwise_persistence"
    return "neutral"


def main():
    log("Reading decay spatial output")
    gdf = gpd.read_file(INPUT_DECAY_GPKG)

    log("Reading decay cell summary")
    cells = pd.read_csv(INPUT_DECAY_CELLS)

    log("Reading decay fold summary")
    folds = pd.read_csv(INPUT_DECAY_FOLDS)

    required_cell_cols = [
        "cell_id",
        "mean_gain_across_buffers_v01",
        "median_gain_across_buffers_v01",
        "share_buffers_relational_better_v01",
        "max_tested_positive_buffer_m_v01",
        "gain_sign_changes_v01",
    ]

    for c in required_cell_cols:
        if c not in cells.columns:
            raise ValueError(f"Missing required cell column: {c}")

    common = cells.dropna(subset=["mean_gain_across_buffers_v01"]).copy()

    log(f"Common modeled cell population: {len(common):,}")

    common["persistence_regime_v01"] = common.apply(classify_persistence, axis=1)

    # Fold-level persistence summaries
    fold_profile = (
        folds.groupby("fold")
        .agg(
            n_buffers=("buffer_m", "size"),
            mean_gain=("mae_gain_pt_minus_rel", "mean"),
            median_gain=("mae_gain_pt_minus_rel", "median"),
            min_gain=("mae_gain_pt_minus_rel", "min"),
            max_gain=("mae_gain_pt_minus_rel", "max"),
            share_buffers_relational_better=("mae_gain_pt_minus_rel", lambda s: float((s > 0).mean())),
            gain_sign_changes=("mae_gain_pt_minus_rel", lambda s: int(np.sum(np.sign(s.to_numpy())[1:] != np.sign(s.to_numpy())[:-1]))),
        )
        .reset_index()
    )

    def classify_fold(row):
        if row["share_buffers_relational_better"] >= 0.8 and row["mean_gain"] > 0:
            return "persistent_relational_fold"
        if row["share_buffers_relational_better"] <= 0.2 and row["mean_gain"] < 0:
            return "persistent_pointwise_fold"
        if row["gain_sign_changes"] >= 3:
            return "unstable_multiscale_fold"
        if row["mean_gain"] > 0:
            return "weak_relational_fold"
        if row["mean_gain"] < 0:
            return "weak_pointwise_fold"
        return "neutral_fold"

    fold_profile["fold_persistence_regime_v01"] = fold_profile.apply(classify_fold, axis=1)

    out = gdf.merge(
        common[
            [
                "cell_id",
                "mean_gain_across_buffers_v01",
                "median_gain_across_buffers_v01",
                "share_buffers_relational_better_v01",
                "max_tested_positive_buffer_m_v01",
                "gain_sign_changes_v01",
                "persistence_regime_v01",
            ]
        ],
        on="cell_id",
        how="left",
        suffixes=("", "_common"),
    )

    qa = []
    qa.append(f"{SCRIPT} QA")
    qa.append("=" * 80)
    qa.append(f"Input decay GPKG: {INPUT_DECAY_GPKG}")
    qa.append(f"Input decay cells: {INPUT_DECAY_CELLS}")
    qa.append(f"Input decay folds: {INPUT_DECAY_FOLDS}")
    qa.append(f"Output: {OUT_GPKG}")
    qa.append("")
    qa.append(f"Rows total spatial: {len(out):,}")
    qa.append(f"Common modeled cells: {len(common):,}")
    qa.append("")
    qa.append("Cell persistence regime counts:")
    qa.append(common["persistence_regime_v01"].value_counts().to_string())
    qa.append("")
    qa.append("Cell persistence summary:")
    qa.append(
        common[
            [
                "mean_gain_across_buffers_v01",
                "median_gain_across_buffers_v01",
                "share_buffers_relational_better_v01",
                "max_tested_positive_buffer_m_v01",
                "gain_sign_changes_v01",
            ]
        ]
        .describe()
        .to_string()
    )
    qa.append("")
    qa.append("Fold persistence profile:")
    qa.append(fold_profile.to_string(index=False))
    qa.append("")
    qa.append("Interpretation guardrail:")
    qa.append(
        "These persistence regimes are descriptive summaries of tested buffer behavior. "
        "They are not yet a validated operator-selection rule, interaction horizon, or final SRS theorem."
    )

    log("Writing outputs")
    out.to_file(OUT_GPKG, driver="GPKG")
    common.to_csv(OUT_CELL_CSV, index=False)
    fold_profile.to_csv(OUT_FOLD_CSV, index=False)
    OUT_QA.write_text("\n".join(qa), encoding="utf-8")

    log(f"Created {OUT_GPKG}")
    log(f"Created {OUT_CELL_CSV}")
    log(f"Created {OUT_FOLD_CSV}")
    log(f"Created {OUT_QA}")
    log("Done")


if __name__ == "__main__":
    main()