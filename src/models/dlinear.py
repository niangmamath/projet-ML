"""DLinear — Decomposition-Linear baseline (Zeng et al., AAAI 2023).

Papier : 'Are Transformers Effective for Time Series Forecasting?'

Idee : decomposer la serie en TENDANCE (moyenne mobile) + RESIDU, puis
appliquer un simple layer lineaire sur chaque composante. Souvent competitif
voire superieur aux Transformers complexes sur les benchmarks LTSF.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesDecomposition(nn.Module):
    """Decomposition additive : serie = tendance (moyenne mobile) + residu."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size doit etre impair pour conserver la longueur"
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x : (B, L, C) -> (B, C, L) pour AvgPool1d
        x_perm = x.transpose(1, 2)
        pad = self.kernel_size // 2
        # Padding par repetition des bords pour eviter l'effet de bord en debut/fin
        x_padded = F.pad(x_perm, (pad, pad), mode="replicate")
        trend = self.avg_pool(x_padded).transpose(1, 2)
        residual = x - trend
        return residual, trend


class DLinear(nn.Module):
    """Une couche lineaire par composante (tendance + residu), par canal independant.

    Args:
        input_len  : longueur de la fenetre d'entree
        pred_len   : longueur de l'horizon de prediction
        n_channels : nombre de variables (features)
        kernel_size : fenetre de la moyenne mobile pour decomposer
        individual : si True, une tete lineaire differente par canal (plus de parametres,
                     souvent meilleur) ; sinon, poids partages sur tous les canaux
    """

    def __init__(
        self,
        input_len: int,
        pred_len: int,
        n_channels: int,
        kernel_size: int = 25,
        individual: bool = True,
    ):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.individual = individual
        self.decomp = SeriesDecomposition(kernel_size)

        if individual:
            self.linear_residual = nn.ModuleList(
                [nn.Linear(input_len, pred_len) for _ in range(n_channels)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(input_len, pred_len) for _ in range(n_channels)]
            )
        else:
            self.linear_residual = nn.Linear(input_len, pred_len)
            self.linear_trend = nn.Linear(input_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, C)
        residual, trend = self.decomp(x)              # (B, L, C) chacun
        residual = residual.transpose(1, 2)           # (B, C, L)
        trend = trend.transpose(1, 2)                 # (B, C, L)

        if self.individual:
            out_r = torch.stack(
                [self.linear_residual[c](residual[:, c, :]) for c in range(self.n_channels)],
                dim=1,
            )
            out_t = torch.stack(
                [self.linear_trend[c](trend[:, c, :]) for c in range(self.n_channels)],
                dim=1,
            )
        else:
            out_r = self.linear_residual(residual)
            out_t = self.linear_trend(trend)

        out = (out_r + out_t).transpose(1, 2)          # (B, pred_len, C)
        return out
