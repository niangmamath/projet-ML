"""Autoformer (Wu et al., NeurIPS 2021).

Papier : 'Autoformer: Decomposition Transformers with Auto-Correlation
          for Long-Term Series Forecasting'.

Deux idees centrales :
    1. DECOMPOSITION dans l'architecture : chaque couche extrait et met a jour
       separement la composante TENDANCE (moyenne mobile) et la composante
       SAISONNIERE (residu). La tendance est propagee et accumulee dans un
       chemin dedie ; la saisonnalite passe par les blocs 'attention'.
    2. AUTO-CORRELATION : remplace la self-attention par un mecanisme qui,
       au lieu de ponderer des points, PONDERE DES DECALAGES TEMPORELS.
       On trouve via FFT les k lags tau_1..tau_k les plus auto-correles
       avec la serie, puis on fait une 'roll-aggregation' : chaque token
       devient la combinaison ponderee de la serie decalee de ces lags.
       -> complexite O(L log L) grace a la FFT.

Implementation encoder-only simplifiee pour comparaison equitable.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Decomposition (moyenne mobile)
# -----------------------------------------------------------------------------
class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x : (B, L, D)
        pad = self.kernel_size // 2
        xp = F.pad(x.transpose(1, 2), (pad, pad), mode="replicate")
        trend = self.avg(xp).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


# -----------------------------------------------------------------------------
# Auto-Correlation
# -----------------------------------------------------------------------------
class AutoCorrelation(nn.Module):
    """Auto-correlation efficace par FFT, puis roll-aggregation sur les top-k lags."""

    def __init__(self, d_model: int, n_heads: int, factor: int = 3, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.factor = factor

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _time_delay_agg(self, V: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """Agrege les valeurs V decalees des top-k lags les plus auto-correles."""
        # V, corr : (B, H, D_head, L)  (V contient d_head channels par tete)
        B, H, D, L = V.shape
        k = min(self.factor * int(math.log(L)), L)

        # Moyenne de la correlation sur les batchs et tetes pour choisir les lags globaux
        mean_corr = corr.mean(dim=(0, 1))                       # (D, L)
        mean_corr = mean_corr.mean(dim=0)                       # (L,)
        top_lags = mean_corr.topk(k).indices                    # (k,)
        top_weights = F.softmax(mean_corr[top_lags], dim=-1)    # (k,)

        # Rotation cyclique de V pour chaque lag, ponderee par la correlation
        agg = torch.zeros_like(V)
        for i in range(k):
            lag = int(top_lags[i].item())
            agg = agg + torch.roll(V, shifts=-lag, dims=-1) * top_weights[i]
        return agg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).permute(0, 2, 3, 1)  # (B,H,D,L)
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_head).permute(0, 2, 3, 1)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_head).permute(0, 2, 3, 1)

        # Auto-correlation via FFT : corr(tau) = IFFT( FFT(q) * conj(FFT(k)) )
        q_fft = torch.fft.rfft(Q, dim=-1)
        k_fft = torch.fft.rfft(K, dim=-1)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=L, dim=-1)    # (B,H,D,L)

        out = self._time_delay_agg(V, corr)                               # (B,H,D,L)
        out = out.permute(0, 3, 1, 2).contiguous().view(B, L, self.d_model)
        return self.dropout(self.out_proj(out))


# -----------------------------------------------------------------------------
# Bloc encoder Autoformer : auto-corr + decomp + FFN + decomp
# -----------------------------------------------------------------------------
class AutoformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, kernel_size: int = 25,
                 factor: int = 3, dropout: float = 0.1):
        super().__init__()
        self.autocorr = AutoCorrelation(d_model, n_heads, factor, dropout)
        self.decomp1 = SeriesDecomp(kernel_size)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.decomp2 = SeriesDecomp(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Branche saisonniere : passe par auto-corr et FFN ; les tendances extraites
        # sont accumulees par le modele via leur retour en sortie.
        x = x + self.autocorr(x)
        x, trend1 = self.decomp1(x)
        x = x + self.ffn(x)
        x, trend2 = self.decomp2(x)
        return x, trend1 + trend2


# -----------------------------------------------------------------------------
# Modele complet
# -----------------------------------------------------------------------------
class Autoformer(nn.Module):
    def __init__(
        self,
        input_len: int,
        pred_len: int,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 256,
        kernel_size: int = 25,
        factor: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.n_features = n_features

        self.decomp_init = SeriesDecomp(kernel_size)
        self.value_embed = nn.Linear(n_features, d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [AutoformerBlock(d_model, n_heads, d_ff, kernel_size, factor, dropout)
             for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

        # Branche saisonniere : flatten -> pred
        self.seasonal_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_len * d_model, pred_len * n_features),
        )
        # Branche tendance : projection temporelle (moyenne appliquee comme init)
        self.trend_head = nn.Linear(input_len, pred_len)
        # Projection de la tendance (d_model -> n_features)
        self.trend_proj = nn.Linear(d_model, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Decomposition initiale sur l'entree brute
        seasonal_init, trend_init = self.decomp_init(x)       # (B, L, n_features) chacun

        # Embedding des deux branches
        s = self.value_embed(seasonal_init)
        s = self.input_dropout(s)
        trend_accum = self.value_embed(trend_init)            # tendance sur l'espace d_model

        for block in self.blocks:
            s, trend_delta = block(s)
            trend_accum = trend_accum + trend_delta

        s = self.norm(s)

        # Predictions
        seasonal_out = self.seasonal_head(s).view(-1, self.pred_len, self.n_features)
        trend_proj = self.trend_proj(trend_accum)              # (B, L, n_features)
        trend_out = self.trend_head(trend_proj.transpose(1, 2)).transpose(1, 2)  # (B, pred_len, n_features)
        return seasonal_out + trend_out
