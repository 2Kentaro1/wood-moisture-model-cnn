from __future__ import annotations

import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from nir_mc.config import Config
from nir_mc.features import PHASE_7, STABLE_4, build_handcrafted_features, select_feature_set
from nir_mc.io import ensure_output_dirs, read_csv, save_json
from nir_mc.metrics import mc_band_rmse, overall_rmse, species_mean_rmse, species_rmse
from nir_mc.models.cnn1d import CNN1DRegressor, weighted_mse_loss
from nir_mc.models.tabular import fit_with_optional_weight, make_extra_trees, make_rf, make_ridge
from nir_mc.preprocessing import build_cnn_tensor, build_spectra_dict
from nir_mc.submission import save_submission
from nir_mc.utils import seed_everything
from nir_mc.visualization import (
    plot_actual_vs_pred_by_species,
    plot_embedding_feature_correlation,
    plot_embedding_pca,
    plot_test_prediction_distribution,
)
from nir_mc.wavelengths import extract_spectra, get_wavelength_axis
from nir_mc.weights import species_index_balanced


class SpectraDataset(Dataset):
    def __init__(self, X, y=None, sample_weight=None):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32)
        self.w = None if sample_weight is None else torch.tensor(sample_weight, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx], self.w[idx]


def transform_target(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return y.astype(np.float32)
    if mode == "log1p":
        return np.log1p(y).astype(np.float32)
    raise ValueError(f"Unknown target_transform: {mode}")


def inverse_target(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return y
    if mode == "log1p":
        return np.expm1(y)
    raise ValueError(f"Unknown target_transform: {mode}")


def train_one_fold(X_train, y_train, w_train, X_valid, y_valid, config: Config):
    device = torch.device(config.DEVICE)
    model = CNN1DRegressor(in_channels=X_train.shape[-1], embedding_dim=config.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    y_train_t = transform_target(y_train, config.target_transform)
    y_valid_t = transform_target(y_valid, config.target_transform)
    train_loader = DataLoader(
        SpectraDataset(X_train, y_train_t, w_train),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    valid_loader = DataLoader(SpectraDataset(X_valid, y_valid_t, np.ones_like(y_valid_t)), batch_size=config.batch_size, shuffle=False)

    best_state = None
    best_loss = np.inf
    wait = 0
    history = []
    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = weighted_mse_loss(pred, yb, wb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, wb in valid_loader:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                val_losses.append(float(weighted_mse_loss(model(xb), yb, wb).cpu()))
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)), "valid_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= config.early_stopping_patience:
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def predict_cnn(model, X, config: Config) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(SpectraDataset(X), batch_size=config.batch_size, shuffle=False)
    device = torch.device(config.DEVICE)
    with torch.no_grad():
        for xb in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    return inverse_target(np.concatenate(preds), config.target_transform)


def extract_embedding(model, X, config: Config) -> np.ndarray:
    model.eval()
    embs = []
    loader = DataLoader(SpectraDataset(X), batch_size=config.batch_size, shuffle=False)
    device = torch.device(config.DEVICE)
    with torch.no_grad():
        for xb in loader:
            _, emb = model(xb.to(device), return_embedding=True)
            embs.append(emb.cpu().numpy())
    return np.concatenate(embs, axis=0)


def run_cnn_cv(X_cnn, X_test_cnn, y, train_df: pd.DataFrame, weights: np.ndarray, config: Config, dirs: dict[str, Path]):
    setting = config.setting_name()
    groups = train_df[config.GROUP_COL].to_numpy()
    gkf = GroupKFold(n_splits=config.N_SPLITS)
    oof_pred = np.zeros(len(y), dtype=np.float32)
    oof_emb = np.zeros((len(y), config.embedding_dim), dtype=np.float32)
    test_emb_folds = []
    test_pred_folds = []
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(tqdm(gkf.split(X_cnn, y, groups), total=config.N_SPLITS, desc="CNN folds")):
        model, hist = train_one_fold(X_cnn[tr_idx], y[tr_idx], weights[tr_idx], X_cnn[va_idx], y[va_idx], config)
        pred_va = predict_cnn(model, X_cnn[va_idx], config)
        oof_pred[va_idx] = pred_va
        oof_emb[va_idx] = extract_embedding(model, X_cnn[va_idx], config)
        test_emb_folds.append(extract_embedding(model, X_test_cnn, config))
        test_pred_folds.append(predict_cnn(model, X_test_cnn, config))
        torch.save(
            {"model_state_dict": model.state_dict(), "config": config.to_dict(), "fold": fold},
            dirs["models"] / f"cnn_fold{fold}_{setting}.pt",
        )
        hist.to_csv(dirs["results"] / f"cnn_history_fold{fold}_{setting}.csv", index=False)
        fold_rows.append({"fold": fold, "rmse": overall_rmse(y[va_idx], pred_va), "n_valid": len(va_idx)})

    test_emb = np.mean(test_emb_folds, axis=0)
    test_pred = np.mean(test_pred_folds, axis=0)
    return oof_pred, oof_emb, test_pred, test_emb, pd.DataFrame(fold_rows)


def _feature_matrix(parts: list[pd.DataFrame | np.ndarray]) -> np.ndarray:
    arrays = [p.to_numpy(dtype=np.float32) if isinstance(p, pd.DataFrame) else np.asarray(p, dtype=np.float32) for p in parts]
    return np.concatenate(arrays, axis=1)


def run_downstream_models(
    feature_train: pd.DataFrame,
    feature_test: pd.DataFrame,
    emb_train: np.ndarray,
    emb_test: np.ndarray,
    y: np.ndarray,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    weights: np.ndarray,
    config: Config,
    dirs: dict[str, Path],
):
    setting = config.setting_name()
    groups = train_df[config.GROUP_COL].to_numpy()
    specs = {
        "C1_phase7_extra_species_index": (_feature_matrix([select_feature_set(feature_train, PHASE_7)]), _feature_matrix([select_feature_set(feature_test, PHASE_7)]), "extra"),
        "C2_cnn_embedding_ridge": (emb_train, emb_test, "ridge"),
        "C2_cnn_embedding_extra": (emb_train, emb_test, "extra"),
        "C3_phase7_cnn_embedding_extra": (_feature_matrix([select_feature_set(feature_train, PHASE_7), emb_train]), _feature_matrix([select_feature_set(feature_test, PHASE_7), emb_test]), "extra"),
        "C4_phase7_stable4_cnn_embedding_extra": (_feature_matrix([select_feature_set(feature_train, PHASE_7 + STABLE_4), emb_train]), _feature_matrix([select_feature_set(feature_test, PHASE_7 + STABLE_4), emb_test]), "extra"),
        "C5_phase7_cnn_embedding_rf": (_feature_matrix([select_feature_set(feature_train, PHASE_7), emb_train]), _feature_matrix([select_feature_set(feature_test, PHASE_7), emb_test]), "rf"),
        "C6_phase7_cnn_embedding_ridge": (_feature_matrix([select_feature_set(feature_train, PHASE_7), emb_train]), _feature_matrix([select_feature_set(feature_test, PHASE_7), emb_test]), "ridge"),
    }
    rows = []
    pred_test = {}
    pred_oof = {}
    for name, (X, Xt, kind) in specs.items():
        oof = np.zeros(len(y), dtype=np.float32)
        for tr_idx, va_idx in GroupKFold(n_splits=config.N_SPLITS).split(X, y, groups):
            model = _make_downstream(kind, config)
            fit_with_optional_weight(model, X[tr_idx], y[tr_idx], weights[tr_idx])
            oof[va_idx] = model.predict(X[va_idx])
        final_model = _make_downstream(kind, config)
        fit_with_optional_weight(final_model, X, y, weights)
        pred_test[name] = final_model.predict(Xt)
        pred_oof[name] = oof
        joblib.dump(final_model, dirs["models"] / f"downstream_{name}_{setting}.pkl")
        eval_df = train_df[[config.ID_COL, config.GROUP_COL, config.SPECIES_COL]].copy()
        eval_df[config.TARGET_COL] = y
        eval_df["pred"] = oof
        rows.append(
            {
                "model": name,
                "overall_rmse": overall_rmse(y, oof),
                "species_mean_rmse": species_mean_rmse(eval_df, config.TARGET_COL, "pred", config.GROUP_COL),
            }
        )
    compare = pd.DataFrame(rows).sort_values("overall_rmse")
    compare.to_csv(dirs["results"] / f"downstream_compare_{setting}.csv", index=False)
    return compare, pred_oof, pred_test


def _make_downstream(kind: str, config: Config):
    if kind == "extra":
        return make_extra_trees(config.RANDOM_STATE, config.extra_trees_estimators, config.min_samples_leaf)
    if kind == "rf":
        return make_rf(config.RANDOM_STATE, config.rf_estimators, config.min_samples_leaf)
    if kind == "ridge":
        return make_ridge()
    raise ValueError(kind)


def run_full_experiment(config: Config) -> dict:
    seed_everything(config.RANDOM_STATE)
    dirs = ensure_output_dirs(config.OUTPUT_DIR)
    setting = config.setting_name()

    train_df = read_csv(config.TRAIN_PATH)
    test_df = read_csv(config.TEST_PATH)
    wn_cols, _, wavelengths, _, wl_meta = get_wavelength_axis(train_df, test_df)
    X_raw = extract_spectra(train_df, wn_cols)
    X_test_raw = extract_spectra(test_df, wn_cols)
    y = train_df[config.TARGET_COL].to_numpy(dtype=np.float32)
    weights = species_index_balanced(train_df, config.GROUP_COL, config.ID_COL)

    spectra = build_spectra_dict(X_raw)
    spectra_test = build_spectra_dict(X_test_raw)
    handcrafted = build_handcrafted_features(spectra, wavelengths)
    handcrafted_test = build_handcrafted_features(spectra_test, wavelengths)
    handcrafted.to_csv(dirs["features"] / "handcrafted_features_train.csv", index=False)
    handcrafted_test.to_csv(dirs["features"] / "handcrafted_features_test.csv", index=False)

    X_cnn = build_cnn_tensor(spectra, list(config.channels), band=config.band, wavelengths=wavelengths)
    X_test_cnn = build_cnn_tensor(spectra_test, list(config.channels), band=config.band, wavelengths=wavelengths)
    oof_pred, oof_emb, cnn_test_pred, test_emb, fold_summary = run_cnn_cv(X_cnn, X_test_cnn, y, train_df, weights, config, dirs)

    emb_cols = [f"cnn_emb_{i:02d}" for i in range(config.embedding_dim)]
    pd.DataFrame(oof_emb, columns=emb_cols).to_csv(dirs["features"] / f"cnn_embedding_train_{setting}.csv", index=False)
    pd.DataFrame(test_emb, columns=emb_cols).to_csv(dirs["features"] / f"cnn_embedding_test_{setting}.csv", index=False)

    oof_df = train_df[[config.ID_COL, config.GROUP_COL, config.SPECIES_COL, config.TARGET_COL]].copy()
    oof_df["pred"] = oof_pred
    oof_df.to_csv(dirs["results"] / f"cnn_oof_predictions_{setting}.csv", index=False)
    fold_summary.loc[len(fold_summary)] = {"fold": "overall", "rmse": overall_rmse(y, oof_pred), "n_valid": len(y)}
    fold_summary.to_csv(dirs["results"] / f"cnn_cv_summary_{setting}.csv", index=False)
    species_rmse(oof_df, config.TARGET_COL, "pred", config.GROUP_COL).to_csv(dirs["results"] / f"cnn_species_rmse_{setting}.csv", index=False)
    mc_band_rmse(oof_df, config.TARGET_COL, "pred").to_csv(dirs["results"] / f"cnn_mc_band_rmse_{setting}.csv", index=False)

    compare, downstream_oof, downstream_test = run_downstream_models(
        handcrafted, handcrafted_test, oof_emb, test_emb, y, train_df, test_df, weights, config, dirs
    )
    cnn_row = pd.DataFrame([{"model": "CNN_only", "overall_rmse": overall_rmse(y, oof_pred), "species_mean_rmse": species_mean_rmse(oof_df, config.TARGET_COL, "pred", config.GROUP_COL)}])
    final_compare = pd.concat([cnn_row, compare], ignore_index=True).sort_values("overall_rmse")
    final_compare.to_csv(dirs["results"] / "final_compare_df.csv", index=False)

    save_json({"config": config.to_dict(), "wavelength_metadata": wl_meta}, dirs["results"] / f"config_{setting}.json")
    save_submission(test_df[config.ID_COL], cnn_test_pred, dirs["submissions"] / "submission_CNN_only.csv")
    sub_map = {
        "submission_phase7_extra_species_index.csv": "C1_phase7_extra_species_index",
        "submission_phase7_cnn_embedding_extra.csv": "C3_phase7_cnn_embedding_extra",
        "submission_phase7_stable4_cnn_embedding_extra.csv": "C4_phase7_stable4_cnn_embedding_extra",
    }
    for filename, model_name in sub_map.items():
        save_submission(test_df[config.ID_COL], downstream_test[model_name], dirs["submissions"] / filename)
    ensemble = np.mean([cnn_test_pred, downstream_test["C1_phase7_extra_species_index"], downstream_test["C3_phase7_cnn_embedding_extra"]], axis=0)
    save_submission(test_df[config.ID_COL], ensemble, dirs["submissions"] / "submission_ensemble_cnn_phase7.csv")

    try:
        plot_actual_vs_pred_by_species(oof_df, config.TARGET_COL, "pred", config.GROUP_COL, config.ID_COL, dirs["figures"] / f"actual_vs_pred_by_species_{setting}.png")
        plot_test_prediction_distribution(
            {
                "CNN_only": cnn_test_pred,
                "phase7_extra": downstream_test["C1_phase7_extra_species_index"],
                "phase7_cnn_extra": downstream_test["C3_phase7_cnn_embedding_extra"],
            },
            dirs["figures"] / f"test_prediction_distribution_{setting}.png",
        )
        plot_embedding_pca(oof_emb, y, train_df[config.GROUP_COL].astype(str), dirs["figures"] / f"embedding_pca_{setting}.png")
        plot_embedding_feature_correlation(oof_emb, handcrafted, dirs["figures"] / f"embedding_feature_correlation_{setting}.png")
    except Exception as e:
        print(f"[WARN] Figure generation failed: {e}")

    return {"final_compare": final_compare, "setting": setting, "output_dir": str(dirs["root"])}
