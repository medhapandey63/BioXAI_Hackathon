"""
kNN expression predictor based on Tanimoto similarity of Morgan (ECFP4) fingerprints.

For each query compound, predicts expression as a similarity-weighted average
of the k most similar training compounds' expression profiles.

Usage
-----
python knn_predict.py            # evaluate on val + local test, generate submission
python knn_predict.py --k 10     # tune k
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

ROOT         = Path(__file__).parent.parent
CONTEST_REPO = ROOT.parent / "vcpi-prediction-contest-2026"
sys.path.insert(0, str(CONTEST_REPO / "src"))

from vcpi_prediction_contest.data import load_gene_filter, load_test_compounds  # noqa: E402
from vcpi_prediction_contest.metrics import score_compounds, aggregate_leaderboards  # noqa: E402

EXPR_WIDE       = ROOT / "data/train_expression_wide.parquet"
SPLITS_CSV      = ROOT / "data/splits.csv"
COMPOUNDS_CSV   = ROOT / "data/compounds_all.csv"
WEIGHTS_PARQUET = CONTEST_REPO / "weights.parquet"
OUT_SUBMISSION  = ROOT / "predictions.parquet"

FP_RADIUS = 2
FP_BITS   = 1024


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def smiles_to_fp(smiles: str, gen) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return np.zeros(FP_BITS, dtype=np.uint8)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(FP_BITS, dtype=np.uint8)
    fp.SetBitsInList(list(fp.GetOnBits()))  # not needed — use direct conversion
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr


def compute_fingerprints(smiles_series: pd.Series) -> np.ndarray:
    """Return (n, FP_BITS) uint8 array of Morgan ECFP4 fingerprints."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    fps = []
    for smi in tqdm(smiles_series, desc="Computing fingerprints", unit="mol"):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            fps.append(np.zeros(FP_BITS, dtype=np.uint8))
        else:
            arr = np.zeros(FP_BITS, dtype=np.uint8)
            from rdkit.DataStructs import ConvertToNumpyArray
            ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
            fps.append(arr)
    return np.stack(fps)


# ---------------------------------------------------------------------------
# Tanimoto similarity
# ---------------------------------------------------------------------------

def tanimoto_matrix(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Compute Tanimoto similarity between every (query, ref) pair.

    Parameters
    ----------
    query : (n_query, bits)
    ref   : (n_ref,   bits)

    Returns
    -------
    (n_query, n_ref) float32 similarity matrix
    """
    query = query.astype(np.float32)
    ref   = ref.astype(np.float32)
    intersection = query @ ref.T                          # (n_query, n_ref)
    sum_q = query.sum(axis=1, keepdims=True)              # (n_query, 1)
    sum_r = ref.sum(axis=1, keepdims=True).T              # (1, n_ref)
    union = sum_q + sum_r - intersection
    union = np.where(union == 0, 1.0, union)              # avoid /0 for all-zero fps
    return (intersection / union).astype(np.float32)


# ---------------------------------------------------------------------------
# kNN prediction
# ---------------------------------------------------------------------------

def knn_predict(
    query_fps: np.ndarray,
    train_fps: np.ndarray,
    train_expr: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Predict expression for each query compound.

    Parameters
    ----------
    query_fps  : (n_query, bits)
    train_fps  : (n_train, bits)
    train_expr : (n_train, n_genes)
    k          : number of neighbours

    Returns
    -------
    (n_query, n_genes) float32 predictions
    """
    print(f"Computing Tanimoto similarity ({len(query_fps)} queries x {len(train_fps)} ref)...",
          flush=True)
    sim = tanimoto_matrix(query_fps, train_fps)           # (n_query, n_train)

    n_query, n_genes = len(query_fps), train_expr.shape[1]
    predictions = np.zeros((n_query, n_genes), dtype=np.float32)

    print(f"Predicting with k={k}...", flush=True)
    for i in tqdm(range(n_query), desc="kNN predict", unit="compound"):
        top_k_idx = np.argpartition(sim[i], -k)[-k:]
        w = sim[i, top_k_idx].astype(np.float64)
        w_sum = w.sum()
        if w_sum == 0:
            w = np.ones(k) / k
        else:
            w = w / w_sum
        predictions[i] = w @ train_expr[top_k_idx]

    return predictions


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def wide_to_long(compounds: list[str], gene_ids: list[str],
                 values: np.ndarray, value_col: str) -> pd.DataFrame:
    """Convert (n_compounds x n_genes) array to long-format DataFrame."""
    df = pd.DataFrame(values, index=compounds, columns=gene_ids)
    df.index.name = "compound"
    return df.reset_index().melt(id_vars="compound", var_name="gene_id",
                                 value_name=value_col)


def evaluate(
    pred_wide: pd.DataFrame,
    truth_wide: pd.DataFrame,
    weights: pd.DataFrame,
    gene_filter: list[str],
    label: str,
) -> None:
    """Print wMSE for a set of compounds."""
    compounds = pred_wide["compound"].tolist()

    truth_long = truth_wide[truth_wide["compound"].isin(compounds)].copy()
    truth_long = truth_long.melt(id_vars="compound", var_name="gene_id",
                                 value_name="expression") \
                           if "gene_id" not in truth_wide.columns \
                           else truth_wide[truth_wide["compound"].isin(compounds)]

    # truth_wide is already wide — melt
    truth_long = (
        truth_wide[truth_wide["compound"].isin(compounds)]
        .melt(id_vars="compound", var_name="gene_id", value_name="expression")
    )

    pred_long = pred_wide.melt(id_vars="compound", var_name="gene_id",
                               value_name="predicted_expression")

    # Use weights for these specific compounds (subset of weight matrix columns)
    available = [c for c in compounds if c in weights.columns]
    if available:
        w = weights[available]
    else:
        w = None  # fall back to variance weights

    scores = score_compounds(truth_long, pred_long, gene_filter=gene_filter, weights=w)
    board  = aggregate_leaderboards(scores)
    print(f"[{label}]  n={board['n_compounds']}  wMSE={board['wmse_mean']:.4f}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Number of neighbours")
    args = parser.parse_args()
    k = args.k

    print("Loading data...", flush=True)
    splits   = pd.read_csv(SPLITS_CSV, dtype={"user_compound_id": str})
    expr     = pd.read_parquet(EXPR_WIDE)
    expr["compound"] = expr["compound"].astype(str)
    compounds_df = pd.read_csv(COMPOUNDS_CSV, dtype={"user_compound_id": str})
    gene_filter  = load_gene_filter()
    test_cpds    = load_test_compounds()   # leaderboard compounds
    test_cpds["compound"] = test_cpds["compound"].astype(str)

    # Split expression table
    train_ids = set(splits[splits["split"] == "train"]["user_compound_id"])
    val_ids   = set(splits[splits["split"] == "val"]["user_compound_id"])
    test_ids  = set(splits[splits["split"] == "test"]["user_compound_id"])

    expr_train = expr[expr["compound"].isin(train_ids)].copy()
    expr_val   = expr[expr["compound"].isin(val_ids)].copy()
    expr_test  = expr[expr["compound"].isin(test_ids)].copy()

    print(f"Train: {len(expr_train)}  Val: {len(expr_val)}  Test: {len(expr_test)}", flush=True)

    # Merge SMILES onto expression rows
    smiles_map = compounds_df.set_index("user_compound_id")["smiles"]

    def add_smiles(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["smiles"] = df["compound"].map(smiles_map)
        return df

    expr_train = add_smiles(expr_train)
    expr_val   = add_smiles(expr_val)
    expr_test  = add_smiles(expr_test)
    test_cpds  = test_cpds.copy()  # already has smiles column

    # Compute fingerprints
    print("\nComputing fingerprints...", flush=True)
    train_fps  = compute_fingerprints(expr_train["smiles"])
    val_fps    = compute_fingerprints(expr_val["smiles"])
    test_fps   = compute_fingerprints(expr_test["smiles"])
    lb_fps     = compute_fingerprints(test_cpds["smiles"])

    # Expression matrices (n_compounds x n_genes)
    gene_cols   = gene_filter
    train_expr  = expr_train[gene_cols].values.astype(np.float32)
    val_expr    = expr_val[gene_cols].values.astype(np.float32)
    test_expr   = expr_test[gene_cols].values.astype(np.float32)

    # Load official weights for local scoring
    print("\nLoading scoring weights...", flush=True)
    W = pd.read_parquet(WEIGHTS_PARQUET)

    # ---- Per-gene mean baseline on val ----
    print("\n--- Baseline: per-gene mean ---", flush=True)
    per_gene_mean = train_expr.mean(axis=0)  # (n_genes,)
    baseline_preds = np.tile(per_gene_mean, (len(expr_val), 1))
    baseline_wide = pd.DataFrame(baseline_preds,
                                 index=expr_val["compound"].tolist(),
                                 columns=gene_cols)
    baseline_wide.index.name = "compound"
    evaluate(baseline_wide.reset_index(),
             expr_val[["compound"] + gene_cols].copy(),
             W, gene_filter, label="val per-gene-mean")

    # Print average nearest-neighbour similarity to understand library diversity
    sim_sample = tanimoto_matrix(val_fps[:200], train_fps)
    top1_sim = sim_sample.max(axis=1)
    print(f"Avg top-1 Tanimoto similarity (val vs train): "
          f"{top1_sim.mean():.3f} ± {top1_sim.std():.3f}", flush=True)

    # ---- Evaluate on val ----
    print(f"\n--- Validation (k={k}) ---", flush=True)
    val_preds = knn_predict(val_fps, train_fps, train_expr, k)
    val_pred_wide = pd.DataFrame(val_preds,
                                 index=expr_val["compound"].tolist(),
                                 columns=gene_cols)
    val_pred_wide.index.name = "compound"
    val_pred_wide = val_pred_wide.reset_index()
    evaluate(val_pred_wide, expr_val[["compound"] + gene_cols].copy(),
             W, gene_filter, label=f"val k={k}")

    # ---- Evaluate on local test ----
    print(f"\n--- Local test (k={k}) ---", flush=True)
    # For local test, use all train+val compounds as reference
    all_train_val_ids = train_ids | val_ids
    expr_tv   = expr[expr["compound"].isin(all_train_val_ids)].copy()
    expr_tv   = add_smiles(expr_tv)
    tv_fps    = compute_fingerprints(expr_tv["smiles"])
    tv_expr   = expr_tv[gene_cols].values.astype(np.float32)

    test_preds = knn_predict(test_fps, tv_fps, tv_expr, k)
    test_pred_wide = pd.DataFrame(test_preds,
                                  index=expr_test["compound"].tolist(),
                                  columns=gene_cols)
    test_pred_wide.index.name = "compound"
    test_pred_wide = test_pred_wide.reset_index()
    evaluate(test_pred_wide, expr_test[["compound"] + gene_cols].copy(),
             W, gene_filter, label=f"test k={k}")

    # ---- Generate leaderboard submission ----
    print(f"\n--- Generating submission (k={k}, trained on all {len(expr)} compounds) ---",
          flush=True)
    all_fps  = compute_fingerprints(expr["compound"].map(smiles_map))

    # Use all 14k training compounds for final submission
    all_expr = expr[gene_cols].values.astype(np.float32)

    lb_preds = knn_predict(lb_fps, all_fps, all_expr, k)
    lb_pred_wide = pd.DataFrame(lb_preds,
                                index=test_cpds["compound"].tolist(),
                                columns=gene_cols)
    lb_pred_wide.index.name = "compound"

    # Convert to long format for submission
    submission = (
        lb_pred_wide.reset_index()
        .melt(id_vars="compound", var_name="gene_id", value_name="predicted_log2(CPM+1)")
    )
    submission.to_parquet(OUT_SUBMISSION, index=False)
    print(f"Submission saved: {OUT_SUBMISSION}  "
          f"({len(submission):,} rows = {test_cpds['compound'].nunique()} compounds "
          f"x {len(gene_filter)} genes)", flush=True)


if __name__ == "__main__":
    main()
