"""
Build the per-compound gene expression table from raw VCPI counts.

Output format (wide)
--------------------
train_expression_wide.parquet
  - index / first column: compound  (user_compound_id string)
  - remaining columns:    gene_id   (12,995 scored genes)
  - values:               mean log2(CPM+1) across replicates

Wide format is natural for modelling (each compound is one row).
Convert to long for scoring:
    long = wide.reset_index().melt(id_vars='compound',
                                   var_name='gene_id',
                                   value_name='expression')

Checkpoints
-----------
Partial results are written to checkpoints/ after every CHECKPOINT_EVERY
compounds. Re-running the script skips compounds already in checkpoints.
"""

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT         = Path(__file__).parent.parent
CONTEST_REPO = ROOT.parent / "vcpi-prediction-contest-2026"
sys.path.insert(0, str(CONTEST_REPO / "src"))

from vcpi_prediction_contest.data import load_gene_filter  # noqa: E402

MASTER_META = CONTEST_REPO / "data/combined_files/master_df_all.csv"
COUNTS_FILES = {
    "tvc-bhr-009": CONTEST_REPO / "data/vcpi_tvc-bhr-009_counts.parquet",
    "tvc-kdl-010": CONTEST_REPO / "data/vcpi_tvc-kdl-010_counts.parquet",
    "tvc-qnu-012": CONTEST_REPO / "data/vcpi_tvc-qnu-012_counts.parquet",
}
OUT_WIDE        = ROOT / "data/train_expression_wide.parquet"
CHECKPOINT_DIR  = ROOT / "data/checkpoints"
CHECKPOINT_EVERY = 2000   # save after this many compounds
SAMPLE_BATCH_SIZE = 3000  # samples read per parquet pass


def already_done() -> set[str]:
    """Return compound IDs already saved in checkpoint files."""
    if not CHECKPOINT_DIR.exists():
        return set()
    done = set()
    for f in sorted(CHECKPOINT_DIR.glob("expr_*.parquet")):
        df = pd.read_parquet(f, columns=["compound"])
        done.update(df["compound"].tolist())
    return done


def save_checkpoint(wide: pd.DataFrame, tag: str) -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    path = CHECKPOINT_DIR / f"expr_{tag}.parquet"
    wide.to_parquet(path, index=False)


def process_experiment(
    job_id: str,
    counts_path: Path,
    meta: pd.DataFrame,
    gene_filter: list[str],
    skip_compounds: set[str],
    pbar: tqdm,
) -> pd.DataFrame:
    """
    Return wide DataFrame (n_compounds x n_genes) for one experiment.
    Compounds in skip_compounds are excluded (already checkpointed).
    """
    gene_set = set(gene_filter)

    # Drop already-processed compounds
    meta = meta[~meta["user_compound_id"].isin(skip_compounds)].copy()
    if meta.empty:
        pbar.write(f"  {job_id}: all compounds already done, skipping.")
        return pd.DataFrame()

    # Group samples by compound, build batches that keep replicates together
    groups = meta.groupby("user_compound_id")["sequenced_id"].apply(list)
    batches: list[list[str]] = []
    batch_compounds: list[list[str]] = []
    cur_samples: list[str] = []
    cur_compounds: list[str] = []

    for compound_id, samples in groups.items():
        if cur_samples and len(cur_samples) + len(samples) > SAMPLE_BATCH_SIZE:
            batches.append(cur_samples)
            batch_compounds.append(cur_compounds)
            cur_samples, cur_compounds = [], []
        cur_samples.extend(samples)
        cur_compounds.append(compound_id)
    if cur_samples:
        batches.append(cur_samples)
        batch_compounds.append(cur_compounds)

    wide_pieces: list[pd.DataFrame] = []
    checkpoint_buf: list[pd.DataFrame] = []
    checkpoint_count = 0
    checkpoint_idx = 0

    for batch_samples, batch_cids in zip(batches, batch_compounds):
        # Read all genes for correct library sizes, then filter
        counts = pd.read_parquet(
            counts_path, columns=["gene_id"] + batch_samples
        ).set_index("gene_id")

        library_sizes = counts.sum(axis=0).replace(0, np.nan)
        counts = counts.loc[counts.index.isin(gene_set)]
        cpm = counts.div(library_sizes, axis=1) * 1e6
        del counts
        log_cpm = np.log2(cpm + 1.0)
        del cpm

        # Map sample -> compound, mean within compound
        batch_meta = meta[meta["sequenced_id"].isin(batch_samples)].set_index("sequenced_id")
        log_cpm.columns = pd.Index(
            log_cpm.columns.map(batch_meta["user_compound_id"]), name="user_compound_id"
        )
        per_compound = log_cpm.T.groupby(level="user_compound_id").mean()
        del log_cpm

        # Reindex to canonical gene order
        per_compound = per_compound.reindex(columns=gene_filter)

        # per_compound is (n_compounds x n_genes), reset index to get compound col
        per_compound.index.name = "compound"
        per_compound = per_compound.reset_index()

        checkpoint_buf.append(per_compound)
        checkpoint_count += len(batch_cids)
        pbar.update(len(batch_cids))
        gc.collect()

        # Write checkpoint every CHECKPOINT_EVERY compounds
        if checkpoint_count >= CHECKPOINT_EVERY:
            chunk = pd.concat(checkpoint_buf, ignore_index=True)
            save_checkpoint(chunk, f"{job_id}_{checkpoint_idx:03d}")
            pbar.write(f"  checkpoint saved: {job_id}_{checkpoint_idx:03d} "
                       f"({len(chunk)} compounds)")
            wide_pieces.append(chunk)
            checkpoint_buf = []
            checkpoint_count = 0
            checkpoint_idx += 1

    # Flush remaining buffer
    if checkpoint_buf:
        chunk = pd.concat(checkpoint_buf, ignore_index=True)
        save_checkpoint(chunk, f"{job_id}_{checkpoint_idx:03d}")
        pbar.write(f"  checkpoint saved: {job_id}_{checkpoint_idx:03d} "
                   f"({len(chunk)} compounds)")
        wide_pieces.append(chunk)

    return pd.concat(wide_pieces, ignore_index=True) if wide_pieces else pd.DataFrame()


def main() -> None:
    print("Loading metadata and gene filter...", flush=True)
    master = pd.read_csv(MASTER_META)
    master["sequenced_id"] = master["sequenced_id"].astype(str)
    master["user_compound_id"] = master["user_compound_id"].astype(str)

    gene_filter = load_gene_filter()
    print(f"Gene filter: {len(gene_filter)} genes", flush=True)

    skip = already_done()
    if skip:
        print(f"Resuming: {len(skip)} compounds already in checkpoints.", flush=True)

    active = master[~master["is_control"]].copy()
    total_compounds = active["user_compound_id"].nunique() - len(skip)

    all_pieces: list[pd.DataFrame] = []

    with tqdm(total=total_compounds, unit="compound", desc="Building expression") as pbar:
        for job_id, counts_path in COUNTS_FILES.items():
            meta_job = active[active["job_id"] == job_id].copy()
            pbar.write(f"\n{job_id}: {meta_job['user_compound_id'].nunique()} compounds")
            result = process_experiment(
                job_id, counts_path, meta_job, gene_filter, skip, pbar
            )
            if not result.empty:
                all_pieces.append(result)

    if not all_pieces:
        print("Nothing to do — all compounds already checkpointed.", flush=True)
        # Load from checkpoints directly
        all_pieces = [
            pd.read_parquet(f)
            for f in sorted(CHECKPOINT_DIR.glob("expr_*.parquet"))
        ]

    print("\nMerging all experiments...", flush=True)
    expression_wide = pd.concat(all_pieces, ignore_index=True)

    # Handle any compound appearing in multiple experiments
    dup = expression_wide["compound"].duplicated().sum()
    if dup:
        print(f"WARNING: {dup} duplicate compound rows — averaging.", flush=True)
        expression_wide = expression_wide.groupby("compound", as_index=False).mean()

    print(f"Final table: {len(expression_wide):,} compounds x "
          f"{len(expression_wide.columns) - 1:,} genes", flush=True)

    expression_wide.to_parquet(OUT_WIDE, index=False)
    print(f"Saved to {OUT_WIDE.resolve()}", flush=True)


if __name__ == "__main__":
    main()
