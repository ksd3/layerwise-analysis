"""Parquet I/O utilities."""

import os
import glob
import numpy as np
import pandas as pd


def make_parquet_safe(df):
    """Normalize DataFrame for Parquet: lists stay lists, scalars stay scalars."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            sample = non_null.iloc[:100]
            has_lists = sample.apply(lambda x: isinstance(x, (list, np.ndarray))).any()
            if has_lists:
                def _norm(x):
                    if isinstance(x, np.ndarray):
                        return x.tolist()
                    if isinstance(x, list):
                        return x
                    return None
                df[col] = df[col].apply(_norm)
    return df


def save_shard(df, shard_dir, shard_idx):
    """Write a Parquet shard to disk. Returns the file path."""
    os.makedirs(shard_dir, exist_ok=True)
    path = os.path.join(shard_dir, f"{shard_idx:05d}.parquet")
    df = make_parquet_safe(df)
    df.to_parquet(path, index=False)
    print(f"    [SHARD] {path} ({len(df)} rows, {len(df.columns)} cols)")
    return path


def load_shards(shard_dir):
    """Load and concatenate all Parquet shards from a directory.

    Returns DataFrame or None if no shards found.
    """
    if not os.path.isdir(shard_dir):
        return None
    shard_files = sorted(glob.glob(os.path.join(shard_dir, "*.parquet")))
    if not shard_files:
        return None
    dfs = []
    for sf in shard_files:
        try:
            dfs.append(pd.read_parquet(sf))
        except Exception as e:
            print(f"    WARN: failed to read {sf}: {e}")
    if not dfs:
        return None
    combined = pd.concat(dfs, ignore_index=True)
    # NOTE: do NOT deduplicate here — finalize handles dedup with
    # match-separation sorting for correct closest-match selection.
    return combined
