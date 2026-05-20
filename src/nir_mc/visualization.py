from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


def _savefig(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_actual_vs_pred_by_species(df: pd.DataFrame, target_col: str, pred_col: str, group_col: str, id_col: str, path: str | Path):
    species = sorted(df[group_col].unique())
    ncols = 3
    nrows = int(np.ceil(len(species) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)
    for ax, sp in zip(axes.ravel(), species):
        g = df[df[group_col] == sp].sort_values(id_col).copy()
        g["_idx"] = np.arange(len(g))
        ax.plot(g["_idx"], g[target_col], label="actual", marker="o", linewidth=1)
        ax.plot(g["_idx"], g[pred_col], label="pred", marker="o", linewidth=1)
        ax.set_title(f"species {sp}")
        ax.set_xlabel("species-local sample index")
        ax.set_ylabel("MC")
    for ax in axes.ravel()[len(species) :]:
        ax.axis("off")
    axes[0, 0].legend()
    _savefig(path)


def plot_test_prediction_distribution(pred_dict: dict[str, np.ndarray], path: str | Path):
    rows = []
    for name, pred in pred_dict.items():
        rows.extend({"model": name, "pred": float(v)} for v in pred)
    plt.figure(figsize=(9, 5))
    sns.kdeplot(data=pd.DataFrame(rows), x="pred", hue="model", common_norm=False)
    _savefig(path)


def plot_embedding_pca(embedding: np.ndarray, y, species, path: str | Path):
    emb2 = PCA(n_components=2, random_state=42).fit_transform(embedding)
    df = pd.DataFrame({"pc1": emb2[:, 0], "pc2": emb2[:, 1], "MC": y, "species": species.astype(str)})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="pc1", y="pc2", hue="MC", style="species", palette="viridis", s=45)
    _savefig(path)


def plot_embedding_feature_correlation(embedding: np.ndarray, feature_df: pd.DataFrame, path: str | Path):
    emb_df = pd.DataFrame(embedding, columns=[f"emb_{i:02d}" for i in range(embedding.shape[1])])
    corr = pd.concat([emb_df, feature_df.reset_index(drop=True)], axis=1).corr().loc[emb_df.columns, feature_df.columns]
    plt.figure(figsize=(max(8, 0.6 * corr.shape[1]), max(4, 0.35 * corr.shape[0])))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    _savefig(path)
