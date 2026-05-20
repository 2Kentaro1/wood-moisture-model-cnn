from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


BASELINE_3 = ["snv_1450", "sg1_ratio_pos_neg", "snv_sg2_slope_1000_1300"]
PHASE_7 = [
    "sg2_1720_1740_contrast",
    "sg2_1370_peak",
    "snv_slope_1650_1750",
    "sg1_1900_left_right_ratio",
    "snv_slope_1600_1800",
    "sg1_ratio_pos_neg",
    "sg2_1685_1720_contrast",
]
STABLE_4 = ["sg2_1720_1740_contrast", "sg1_slope_2000_2050", "sg2_1370_peak", "sg1_1900_left_right_ratio"]


def nearest_wavelength_index(wavelengths: np.ndarray, target_nm: float) -> int:
    return int(np.argmin(np.abs(np.asarray(wavelengths) - target_nm)))


def band_index(wavelengths: np.ndarray, wl_min: float, wl_max: float) -> np.ndarray:
    wavelengths = np.asarray(wavelengths)
    mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    if not np.any(mask):
        warnings.warn(f"No wavelengths found in band [{wl_min}, {wl_max}]. Using nearest endpoints.", stacklevel=2)
        lo = nearest_wavelength_index(wavelengths, wl_min)
        hi = nearest_wavelength_index(wavelengths, wl_max)
        a, b = sorted((lo, hi))
        mask = np.zeros_like(wavelengths, dtype=bool)
        mask[a : b + 1] = True
    return mask


def band_mean(X: np.ndarray, wavelengths: np.ndarray, wl_min: float, wl_max: float) -> np.ndarray:
    return X[:, band_index(wavelengths, wl_min, wl_max)].mean(axis=1)


def build_handcrafted_features(spectra: dict[str, np.ndarray], wavelengths: np.ndarray) -> pd.DataFrame:
    snv = spectra["snv"]
    sg1 = spectra["snv_sg1"]
    sg2 = spectra["snv_sg2"]
    feats = {
        "snv_1450": snv[:, nearest_wavelength_index(wavelengths, 1450)],
        "sg1_ratio_pos_neg": np.clip(sg1[:, band_index(wavelengths, 2230, 2280)], 0, None).mean(axis=1)
        / (np.abs(np.clip(sg1[:, band_index(wavelengths, 2000, 2150)], None, 0)).mean(axis=1) + 1e-8),
        "snv_sg2_slope_1000_1300": (sg2[:, nearest_wavelength_index(wavelengths, 1300)] - sg2[:, nearest_wavelength_index(wavelengths, 1000)]) / 300.0,
        "sg2_1720_1740_contrast": band_mean(sg2, wavelengths, 1720, 1740) - band_mean(sg2, wavelengths, 1685, 1720),
        "sg2_1370_peak": sg2[:, nearest_wavelength_index(wavelengths, 1370)],
        "snv_slope_1650_1750": (snv[:, nearest_wavelength_index(wavelengths, 1750)] - snv[:, nearest_wavelength_index(wavelengths, 1650)]) / 100.0,
        "sg1_1900_left_right_ratio": np.abs(sg1[:, band_index(wavelengths, 1850, 1900)]).mean(axis=1)
        / (np.abs(sg1[:, band_index(wavelengths, 1900, 1950)]).mean(axis=1) + 1e-8),
        "snv_slope_1600_1800": (snv[:, nearest_wavelength_index(wavelengths, 1800)] - snv[:, nearest_wavelength_index(wavelengths, 1600)]) / 200.0,
        "sg2_1685_1720_contrast": band_mean(sg2, wavelengths, 1685, 1720) - band_mean(sg2, wavelengths, 1720, 1740),
        "sg1_slope_2000_2050": band_mean(sg1, wavelengths, 2000, 2050),
    }
    return pd.DataFrame(feats)


def select_feature_set(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    present = [c for c in names if c in df.columns]
    missing = [c for c in names if c not in df.columns]
    if missing:
        warnings.warn(f"Missing handcrafted features skipped: {missing}", stacklevel=2)
    return df.loc[:, present].copy()
