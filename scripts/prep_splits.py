"""
Train / validation / test split for the VCPI prediction contest.

The 1,064 compounds in test_compounds.csv are the final leaderboard set —
they are never used here.

The 14,021 non-control VCPI compounds are split three ways at compound level:
  train  70 %  — model fitting
  val    15 %  — hyperparameter tuning
  test   15 %  — held-out evaluation before final submission

Outputs
-------
splits.csv   — columns: user_compound_id, split  (values: "train" | "val" | "test")
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT         = Path(__file__).parent.parent
CONTEST_REPO = ROOT.parent / "vcpi-prediction-contest-2026"
MASTER_META  = CONTEST_REPO / "data/combined_files/master_df_all.csv"
TEST_COMPOUNDS_CSV = CONTEST_REPO / "src/vcpi_prediction_contest/data_files/test_compounds.csv"
OUT_SPLITS   = ROOT / "data/splits.csv"

TRAIN_FRACTION = 0.70
VAL_FRACTION   = 0.15
# test gets the remainder (0.15)
RANDOM_SEED    = 42


def main() -> None:
    master = pd.read_csv(MASTER_META)
    leaderboard_cpds = pd.read_csv(TEST_COMPOUNDS_CSV, dtype={"compound": str})

    active = master[~master["is_control"]].copy()
    active["user_compound_id"] = active["user_compound_id"].astype(str)

    unique_ids = sorted(active["user_compound_id"].unique())
    print(f"Unique VCPI compounds (non-control): {len(unique_ids)}")
    print(f"Leaderboard test compounds:          {len(leaderboard_cpds)}")

    overlap = set(unique_ids) & set(leaderboard_cpds["compound"])
    if overlap:
        print(f"WARNING: {len(overlap)} compounds appear in both VCPI and leaderboard set!", file=sys.stderr)
    else:
        print("No overlap with leaderboard set. Clean.")

    rng = np.random.default_rng(RANDOM_SEED)
    ids = np.array(unique_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * TRAIN_FRACTION)
    n_val   = int(n * VAL_FRACTION)
    # remainder goes to local test
    train_ids = set(ids[:n_train])
    val_ids   = set(ids[n_train : n_train + n_val])

    def assign(cid: str) -> str:
        if cid in train_ids:
            return "train"
        if cid in val_ids:
            return "val"
        return "test"

    splits = pd.DataFrame({
        "user_compound_id": unique_ids,
        "split": [assign(cid) for cid in unique_ids],
    })

    counts = splits["split"].value_counts()
    print(f"\nSplit sizes → train: {counts['train']}  |  val: {counts['val']}  |  test: {counts['test']}")

    splits.to_csv(OUT_SPLITS, index=False)
    print(f"Saved splits to {OUT_SPLITS.resolve()}")


if __name__ == "__main__":
    main()
