from __future__ import annotations

import torch
from torch import nn


class CNN1DRegressor(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, x, return_embedding: bool = False):
        emb = self.encoder(x)
        pred = self.head(emb).squeeze(-1)
        if return_embedding:
            return pred, emb
        return pred


def weighted_mse_loss(pred, target, weight):
    return torch.mean(weight * (pred - target) ** 2)
