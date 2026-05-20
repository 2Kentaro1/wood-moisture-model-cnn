from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence


def _auto_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass
class Config:
    TRAIN_PATH: str = "data/train.csv"
    TEST_PATH: str = "data/test.csv"
    OUTPUT_DIR: str = "outputs/nir_cnn_embedding"
    TARGET_COL: str = "含水率"
    SPECIES_COL: str = "樹種"
    GROUP_COL: str = "species number"
    ID_COL: str = "sample number"
    RANDOM_STATE: int = 42
    N_SPLITS: int = 5
    DEVICE: str = field(default_factory=_auto_device)

    band: str = "full"
    channels: Sequence[str] = field(default_factory=lambda: ("raw", "snv", "snv_sg1", "snv_sg2"))
    embedding_dim: int = 16
    target_transform: str = "none"
    epochs: int = 100
    early_stopping_patience: int = 15
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0

    extra_trees_estimators: int = 700
    rf_estimators: int = 700
    min_samples_leaf: int = 2

    def output_path(self) -> Path:
        return Path(self.OUTPUT_DIR)

    def setting_name(self) -> str:
        ch = "-".join(self.channels)
        return f"band-{self.band}_ch-{ch}_emb-{self.embedding_dim}_tf-{self.target_transform}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channels"] = list(self.channels)
        return d

    def update(self, **kwargs) -> "Config":
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
        return self
