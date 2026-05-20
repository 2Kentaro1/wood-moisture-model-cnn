from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


BANDS = {
    "1000_1300": (1000, 1300),
    "1300_1600": (1300, 1600),
    "1600_1800": (1600, 1800),
    "1800_2000": (1800, 2000),
    "2000_2300": (2000, 2300),
    "2300_2500": (2300, 2500),
    "full": None,
}


def snv(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mean) / (std + 1e-8)


def _valid_window(window_length: int, n_features: int) -> int:
    w = min(window_length, n_features if n_features % 2 == 1 else n_features - 1)
    return max(w, 3 if n_features >= 3 else n_features)


def savgol_derivative(X: np.ndarray, window_length: int = 21, polyorder: int = 3, deriv: int = 1) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    w = _valid_window(window_length, X.shape[1])
    p = min(polyorder, w - 1)
    return savgol_filter(X, window_length=w, polyorder=p, deriv=deriv, axis=1).astype(np.float32)


def build_spectra_dict(X_raw: np.ndarray) -> dict[str, np.ndarray]:
    X_snv = snv(X_raw)
    return {
        "raw": X_raw.astype(np.float32),
        "snv": X_snv.astype(np.float32),
        "raw_sg1": savgol_derivative(X_raw, deriv=1),
        "raw_sg2": savgol_derivative(X_raw, deriv=2),
        "snv_sg1": savgol_derivative(X_snv, deriv=1),
        "snv_sg2": savgol_derivative(X_snv, deriv=2),
    }


def band_mask(wavelengths: np.ndarray, band: str | tuple[float, float] | None) -> np.ndarray:
    if band is None or band == "full":
        return np.ones_like(wavelengths, dtype=bool)
    if isinstance(band, str):
        if band not in BANDS:
            raise ValueError(f"Unknown band: {band}")
        bounds = BANDS[band]
        if bounds is None:
            return np.ones_like(wavelengths, dtype=bool)
    else:
        bounds = band
    lo, hi = bounds
    return (wavelengths >= lo) & (wavelengths <= hi)


def build_cnn_tensor(
    spectra_dict: dict[str, np.ndarray],
    channels: list[str] | tuple[str, ...],
    band: str | tuple[float, float] | None = None,
    wavelengths: np.ndarray | None = None,
) -> np.ndarray:
    arrays = []
    mask = None
    if wavelengths is not None:
        mask = band_mask(np.asarray(wavelengths), band)
        if not np.any(mask):
            raise ValueError(f"Band {band} selects no wavelengths.")
    for channel in channels:
        if channel not in spectra_dict:
            raise KeyError(f"Unknown spectra channel: {channel}")
        X = spectra_dict[channel]
        arrays.append(X[:, mask] if mask is not None else X)
    return np.stack(arrays, axis=-1).astype(np.float32)
