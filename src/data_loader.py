"""Chargement et fenetrage des series temporelles pour la prevision long terme.

Conventions LTSF (Informer/Autoformer/PatchTST) :
    - Split chronologique fixe pour ETT : 12 mois train / 4 mois val / 4 mois test.
    - Normalisation : StandardScaler ajuste uniquement sur le TRAIN.
    - Chaque echantillon est une paire (x, y) :
        x : fenetre d'entree de longueur `input_len` (historique)
        y : fenetre cible de longueur `pred_len`   (futur a predire)

Modes supportes :
    'M'  : multivarie -> multivarie (toutes les variables en entree et en sortie)
    'S'  : univarie  -> univarie   (uniquement la colonne cible)
    'MS' : multivarie -> univarie  (toutes en entree, cible unique en sortie)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


# -----------------------------------------------------------------------------
# Configuration par dataset
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetConfig:
    """Specifie le fichier, la colonne temporelle, la cible et les bornes de split."""

    filename: str
    date_col: str
    target_col: str
    # Bornes [debut, fin[ de chaque split, exprimees en nombre de points (pas en dates).
    train_end: int
    val_end: int
    test_end: int


# Bornes ETT issues des papiers Informer/Autoformer :
#   ETTh* : horaire   -> 12*30*24, 16*30*24, 20*30*24
#   ETTm* : 15-minutes -> 12*30*24*4, 16*30*24*4, 20*30*24*4
_H = 30 * 24
_M = 30 * 24 * 4

DATASETS: dict[str, DatasetConfig] = {
    "ETTh1": DatasetConfig("ETTh1.csv", "date", "OT", 12 * _H, 16 * _H, 20 * _H),
    "ETTh2": DatasetConfig("ETTh2.csv", "date", "OT", 12 * _H, 16 * _H, 20 * _H),
    "ETTm1": DatasetConfig("ETTm1.csv", "date", "OT", 12 * _M, 16 * _M, 20 * _M),
    "ETTm2": DatasetConfig("ETTm2.csv", "date", "OT", 12 * _M, 16 * _M, 20 * _M),
}


# -----------------------------------------------------------------------------
# Dataset PyTorch
# -----------------------------------------------------------------------------
class TimeSeriesWindowDataset(Dataset):
    """Fenetrage glissant (x, y) sur une serie deja normalisee.

    Pour i dans [0, N - input_len - pred_len] :
        x = data[i : i + input_len]              shape = (input_len, n_features)
        y = data[i + input_len : i + input_len + pred_len]  shape = (pred_len, n_targets)
    """

    def __init__(
        self,
        data_x: np.ndarray,
        data_y: np.ndarray,
        input_len: int,
        pred_len: int,
    ) -> None:
        assert len(data_x) == len(data_y), "data_x et data_y doivent avoir la meme longueur"
        self.data_x = torch.from_numpy(data_x).float()
        self.data_y = torch.from_numpy(data_y).float()
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.data_x) - self.input_len - self.pred_len + 1

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data_x[i : i + self.input_len]
        y = self.data_y[i + self.input_len : i + self.input_len + self.pred_len]
        return x, y


# -----------------------------------------------------------------------------
# Factory principale
# -----------------------------------------------------------------------------
def _select_columns(df: pd.DataFrame, cfg: DatasetConfig, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (features_x, features_y) selon le mode."""
    feature_cols = [c for c in df.columns if c != cfg.date_col]
    target_col = cfg.target_col

    if mode == "M":
        x_cols = y_cols = feature_cols
    elif mode == "S":
        x_cols = y_cols = [target_col]
    elif mode == "MS":
        x_cols = feature_cols
        y_cols = [target_col]
    else:
        raise ValueError(f"Mode inconnu : {mode!r} (attendu : 'M', 'S' ou 'MS')")

    return df[x_cols].to_numpy(dtype=np.float32), df[y_cols].to_numpy(dtype=np.float32)


def load_dataset(
    name: str,
    input_len: int = 96,
    pred_len: int = 96,
    batch_size: int = 32,
    mode: str = "M",
    num_workers: int = 0,
    data_dir: Path | str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """Retourne (train_loader, val_loader, test_loader, scaler).

    Le `scaler` est ajuste uniquement sur la partie train (features x).
    La meme transformation est appliquee aux cibles y pour obtenir des predictions
    comparables, puis l'inversion se fait au moment de l'evaluation.
    """
    if name not in DATASETS:
        raise KeyError(f"Dataset inconnu : {name!r}. Disponibles : {list(DATASETS)}")
    cfg = DATASETS[name]

    path = Path(data_dir) if data_dir else DATA_DIR
    df = pd.read_csv(path / cfg.filename)
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])

    data_x, data_y = _select_columns(df, cfg, mode)

    # Bornes des splits (on prefixe val/test par input_len pour pouvoir fenetrer
    # jusqu'au tout premier point du split sans sauter d'exemples).
    borders_start = [0, cfg.train_end - input_len, cfg.val_end - input_len]
    borders_end = [cfg.train_end, cfg.val_end, cfg.test_end]

    # StandardScaler : fit uniquement sur le train
    scaler_x = StandardScaler().fit(data_x[: cfg.train_end])
    scaler_y = StandardScaler().fit(data_y[: cfg.train_end])
    data_x = scaler_x.transform(data_x).astype(np.float32)
    data_y = scaler_y.transform(data_y).astype(np.float32)

    def _make(split_idx: int, shuffle: bool) -> DataLoader:
        x_slice = data_x[borders_start[split_idx] : borders_end[split_idx]]
        y_slice = data_y[borders_start[split_idx] : borders_end[split_idx]]
        ds = TimeSeriesWindowDataset(x_slice, y_slice, input_len, pred_len)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
        )

    train_loader = _make(0, shuffle=True)
    val_loader = _make(1, shuffle=False)
    test_loader = _make(2, shuffle=False)

    # On attache scaler_y en priorite : c'est lui qu'on utilise pour inverser les predictions.
    scaler_y.scaler_x_ = scaler_x
    return train_loader, val_loader, test_loader, scaler_y


# -----------------------------------------------------------------------------
# Verification rapide quand lance en script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    train, val, test, scaler = load_dataset("ETTh1", input_len=96, pred_len=96, batch_size=32)
    print(f"Train batches : {len(train):4d}  | n_samples = {len(train.dataset)}")
    print(f"Val   batches : {len(val):4d}  | n_samples = {len(val.dataset)}")
    print(f"Test  batches : {len(test):4d}  | n_samples = {len(test.dataset)}")

    x, y = next(iter(train))
    print("\nExemple de batch :")
    print(f"  x shape = {tuple(x.shape)}  (batch, input_len, n_features)")
    print(f"  y shape = {tuple(y.shape)}  (batch, pred_len, n_targets)")
    print(f"  x stats : mean={x.mean():+.3f}  std={x.std():.3f}")
