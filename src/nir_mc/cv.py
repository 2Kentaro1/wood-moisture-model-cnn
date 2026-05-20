from __future__ import annotations

from sklearn.model_selection import GroupKFold


def make_group_kfold(n_splits: int = 5) -> GroupKFold:
    return GroupKFold(n_splits=n_splits)
