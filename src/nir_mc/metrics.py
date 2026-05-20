from __future__ import annotations

import numpy as np
import pandas as pd


MC_BANDS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 120), (120, 200), (200, np.inf)]


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def overall_rmse(y_true, y_pred) -> float:
    return rmse(y_true, y_pred)


def species_rmse(df: pd.DataFrame, y_true_col: str, y_pred_col: str, group_col: str = "species number") -> pd.DataFrame:
    rows = []
    for group, g in df.groupby(group_col):
        rows.append({group_col: group, "n": len(g), "rmse": rmse(g[y_true_col], g[y_pred_col])})
    return pd.DataFrame(rows).sort_values(group_col)


def species_mean_rmse(df: pd.DataFrame, y_true_col: str, y_pred_col: str, group_col: str = "species number") -> float:
    return float(species_rmse(df, y_true_col, y_pred_col, group_col)["rmse"].mean())


def mc_band_rmse(df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> pd.DataFrame:
    rows = []
    y = df[y_true_col]
    for lo, hi in MC_BANDS:
        mask = (y >= lo) & (y < hi)
        label = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f"{lo:g}+"
        if mask.any():
            rows.append({"mc_band": label, "n": int(mask.sum()), "rmse": rmse(df.loc[mask, y_true_col], df.loc[mask, y_pred_col])})
    return pd.DataFrame(rows)
