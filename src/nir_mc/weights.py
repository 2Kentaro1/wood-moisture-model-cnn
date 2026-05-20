from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float32)
    return w / (w.mean() + 1e-8)


def species_balanced(df: pd.DataFrame, group_col: str = "species number") -> np.ndarray:
    counts = df[group_col].map(df[group_col].value_counts()).to_numpy(dtype=np.float32)
    return _normalize(1.0 / counts)


def species_index_balanced(
    df: pd.DataFrame,
    group_col: str = "species number",
    id_col: str = "sample number",
    n_bins: int = 10,
) -> np.ndarray:
    work = df[[group_col, id_col]].copy()
    work["_orig"] = np.arange(len(work))
    work = work.sort_values([group_col, id_col])
    work["_local_index"] = work.groupby(group_col).cumcount()
    sizes = work.groupby(group_col)[id_col].transform("size").to_numpy(dtype=np.float32)
    denom = np.maximum(sizes - 1.0, 1.0)
    norm = work["_local_index"].to_numpy(dtype=np.float32) / denom
    bins = np.minimum((norm * n_bins).astype(int), n_bins - 1)
    work["_bin"] = bins
    counts = work.groupby([group_col, "_bin"])[id_col].transform("size").to_numpy(dtype=np.float32)
    work["_weight"] = 1.0 / counts
    restored = work.sort_values("_orig")["_weight"].to_numpy(dtype=np.float32)
    return _normalize(restored)
