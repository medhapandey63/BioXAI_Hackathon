"""Quick k sweep to find optimal number of neighbours."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT         = Path(__file__).parent.parent
CONTEST_REPO = ROOT.parent / "vcpi-prediction-contest-2026"
sys.path.insert(0, str(CONTEST_REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from vcpi_prediction_contest.data import load_gene_filter
from knn_predict import compute_fingerprints, tanimoto_matrix, evaluate

splits       = pd.read_csv(ROOT / "data/splits.csv", dtype={'user_compound_id': str})
expr         = pd.read_parquet(ROOT / "data/train_expression_wide.parquet")
expr['compound'] = expr['compound'].astype(str)
compounds_df = pd.read_csv(ROOT / "data/compounds_all.csv", dtype={'user_compound_id': str})
gene_filter  = load_gene_filter()
W            = pd.read_parquet(CONTEST_REPO / "weights.parquet")
smiles_map   = compounds_df.set_index('user_compound_id')['smiles']

train_ids = set(splits[splits['split'] == 'train']['user_compound_id'])
val_ids   = set(splits[splits['split'] == 'val']['user_compound_id'])
expr_train = expr[expr['compound'].isin(train_ids)].copy()
expr_val   = expr[expr['compound'].isin(val_ids)].copy()
gene_cols  = gene_filter

print("Computing fingerprints...", flush=True)
train_fps  = compute_fingerprints(expr_train['compound'].map(smiles_map))
val_fps    = compute_fingerprints(expr_val['compound'].map(smiles_map))
train_expr = expr_train[gene_cols].values.astype(np.float32)

# Baseline: per-gene mean
mean_pred = np.tile(train_expr.mean(axis=0), (len(expr_val), 1))
bw = pd.DataFrame(mean_pred, index=expr_val['compound'].tolist(), columns=gene_cols)
bw.index.name = 'compound'
evaluate(bw.reset_index(), expr_val[['compound'] + gene_cols], W, gene_filter, 'baseline per-gene-mean')

# Compute similarity matrix once, reuse for all k
print("Computing similarity matrix...", flush=True)
sim = tanimoto_matrix(val_fps, train_fps)   # (n_val, n_train)
print(f"Avg top-1 Tanimoto: {sim.max(axis=1).mean():.3f} ± {sim.max(axis=1).std():.3f}",
      flush=True)

n_query, n_genes = len(val_fps), train_expr.shape[1]

for k in [1, 3, 5, 10, 20, 50]:
    preds = np.zeros((n_query, n_genes), dtype=np.float32)
    for i in range(n_query):
        top_k_idx = np.argpartition(sim[i], -k)[-k:]
        w = sim[i, top_k_idx].astype(np.float64)
        s = w.sum()
        w = w / s if s > 0 else np.ones(k) / k
        preds[i] = w @ train_expr[top_k_idx]
    pw = pd.DataFrame(preds, index=expr_val['compound'].tolist(), columns=gene_cols)
    pw.index.name = 'compound'
    evaluate(pw.reset_index(), expr_val[['compound'] + gene_cols], W, gene_filter, f'kNN k={k}')
