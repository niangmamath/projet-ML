"""Metriques et evaluation des modeles de prevision long terme.

Metriques standard LTSF :
    MSE  : mean squared error (principale)
    MAE  : mean absolute error (robuste aux outliers)
    RMSE : racine de MSE (meme unite que la cible)
    MAPE : erreur relative en %
    sMAPE : variante symetrique de MAPE
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


# -----------------------------------------------------------------------------
# Metriques
# -----------------------------------------------------------------------------
def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, true)))


def mape(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((true - pred) / (np.abs(true) + eps))) * 100)


def smape(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(2 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + eps)) * 100)


def all_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    return {
        "MSE": mse(pred, true),
        "MAE": mae(pred, true),
        "RMSE": rmse(pred, true),
        "MAPE": mape(pred, true),
        "sMAPE": smape(pred, true),
    }


# -----------------------------------------------------------------------------
# Boucle d'evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    inverse_fn=None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fait tourner le modele sur un dataloader et retourne (preds, trues, metriques).

    `inverse_fn` : fonction optionnelle qui prend un array et le denormalise
                   (ex. scaler.inverse_transform). Si fournie, les metriques
                   sont calculees sur les valeurs denormalisees.
    """
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        preds.append(out.cpu().numpy())
        trues.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    if inverse_fn is not None:
        shape = preds.shape
        preds = inverse_fn(preds.reshape(-1, shape[-1])).reshape(shape)
        trues = inverse_fn(trues.reshape(-1, shape[-1])).reshape(shape)

    return preds, trues, all_metrics(preds, trues)


# -----------------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------------
def plot_prediction(
    pred: np.ndarray,
    true: np.ndarray,
    sample_idx: int = 0,
    feature_idx: int = -1,
    title: str = "Prediction vs verite",
    save_path: str | Path | None = None,
) -> None:
    """Trace une prediction individuelle contre la cible reelle.

    Args:
        pred, true : arrays (N, pred_len, n_features)
        sample_idx : quel echantillon du batch tracer
        feature_idx : quelle variable tracer (-1 par defaut = derniere, souvent OT)
    """
    y_true = true[sample_idx, :, feature_idx]
    y_pred = pred[sample_idx, :, feature_idx]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_true, label="reel", linewidth=2)
    ax.plot(y_pred, label="prediction", linewidth=2, linestyle="--")
    ax.set_xlabel("horizon (pas)")
    ax.set_ylabel("valeur")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_error_per_horizon(
    pred: np.ndarray,
    true: np.ndarray,
    metric: str = "MAE",
    save_path: str | Path | None = None,
):
    """Trace l'erreur en fonction du pas d'horizon (0 a pred_len-1).

    Utile pour voir comment chaque architecture degrade sur un long horizon.
    """
    err = np.abs(pred - true) if metric == "MAE" else (pred - true) ** 2
    err_per_step = err.mean(axis=(0, 2))  # moyenne sur batch et features

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(err_per_step, linewidth=2)
    ax.set_xlabel("pas d'horizon")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} par pas d'horizon")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
