from __future__ import annotations

import numpy as np
import pandas as pd


def _floatable_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        try:
            float(col)
        except (TypeError, ValueError):
            continue
        cols.append(str(col))
    return cols


def get_wavelength_axis(train_df: pd.DataFrame, test_df: pd.DataFrame | None = None):
    spectral_cols = _floatable_columns(train_df)
    if not spectral_cols:
        raise ValueError("No spectral columns convertible to float were found.")

    wavenumbers = np.array([float(c) for c in spectral_cols], dtype=float)
    order = np.argsort(1e7 / wavenumbers)
    wn_cols_sorted = [spectral_cols[i] for i in order]
    wavenumbers_sorted = wavenumbers[order]
    wavelengths_sorted = 1e7 / wavenumbers_sorted
    wavelength_labels = [f"{w:.2f}" for w in wavelengths_sorted]

    if test_df is not None:
        test_cols = set(_floatable_columns(test_df))
        missing = [c for c in spectral_cols if c not in test_cols]
        extra = [c for c in test_cols if c not in set(spectral_cols)]
        if missing or extra:
            raise ValueError(f"Train/test spectral columns differ. missing={missing[:5]}, extra={extra[:5]}")

    metadata = {
        "original_wavenumber_min": float(wavenumbers.min()),
        "original_wavenumber_max": float(wavenumbers.max()),
        "wavelength_min": float(wavelengths_sorted.min()),
        "wavelength_max": float(wavelengths_sorted.max()),
        "n_spectral_cols": int(len(wn_cols_sorted)),
        "sorted_column_names": wn_cols_sorted,
        "wavelength_labels": wavelength_labels,
    }
    return wn_cols_sorted, wavenumbers_sorted, wavelengths_sorted, wavelength_labels, metadata


def extract_spectra(df: pd.DataFrame, sorted_cols: list[str]) -> np.ndarray:
    return df.loc[:, sorted_cols].to_numpy(dtype=np.float32)
