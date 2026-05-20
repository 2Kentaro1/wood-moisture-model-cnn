from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_extra_trees(random_state: int = 42, n_estimators: int = 700, min_samples_leaf: int = 2):
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )


def make_rf(random_state: int = 42, n_estimators: int = 700, min_samples_leaf: int = 2):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )


def make_ridge(alpha: float = 1.0):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def fit_with_optional_weight(model, X, y, sample_weight=None):
    if sample_weight is None:
        return model.fit(X, y)
    if hasattr(model, "steps"):
        last_name = model.steps[-1][0]
        return model.fit(X, y, **{f"{last_name}__sample_weight": sample_weight})
    return model.fit(X, y, sample_weight=sample_weight)


def as_2d_embedding(embedding: np.ndarray, prefix: str = "cnn_emb") -> tuple[np.ndarray, list[str]]:
    cols = [f"{prefix}_{i:02d}" for i in range(embedding.shape[1])]
    return embedding, cols
