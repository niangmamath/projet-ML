"""FEDformer (Zhou et al., ICML 2022).

Papier : 'FEDformer: Frequency Enhanced Decomposed Transformer for Long-term
          Series Forecasting'.

Idee centrale : au lieu d'operer sur les L tokens temporels, on passe dans le
DOMAINE FREQUENTIEL via FFT et on n'attend qu'un petit sous-ensemble de M modes
(M << L, choisis au hasard ou par magnitude). Les operations sont faites par
des projections LINEAIRES COMPLEXES sur ces M modes, puis on revient au
domaine temporel par IFFT.

Avantages :
    - complexite O(L log L) grace a la FFT ;
    - filtrage implicite des hautes frequences (souvent du bruit) ;
    - signal compact : les series temporelles ont souvent un spectre
      concentre sur quelques modes.

Deux variantes dans le papier : Fourier (celle qu'on implemente ici) et Wavelet.
Decomposition tendance/saison heritee d'Autoformer.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.autoformer import SeriesDecomp


# -----------------------------------------------------------------------------
# Bloc Fourier
# -----------------------------------------------------------------------------
class FourierBlock(nn.Module):
    """Remplace l'attention par une projection lineaire complexe sur M modes
    frequentiels selectionnes au hasard (echantillonnage, cf. papier)."""

    def __init__(self, d_model: int, n_heads: int, seq_len: int, modes: int = 32):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Nombre de modes a garder : borne par le nombre reel de modes dispo apres rfft
        max_modes = seq_len // 2 + 1
        self.modes = min(modes, max_modes)

        # Indices des modes (ici : les plus basses frequences = les plus informatives
        # pour une serie temporelle usuelle)
        self.register_buffer("mode_idx", torch.arange(self.modes))

        # Poids complexes par tete : (H, d_head, d_head, modes), parties R et I separees
        scale = 1.0 / (self.d_head * self.d_head)
        self.weight_real = nn.Parameter(scale * torch.randn(n_heads, self.d_head, self.d_head, self.modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(n_heads, self.d_head, self.d_head, self.modes))

    def _complex_matmul(self, x: torch.Tensor) -> torch.Tensor:
        """Multiplication (B, H, d_head, modes) x (H, d_head, d_head, modes) -> (B, H, d_head, modes)."""
        w = torch.complex(self.weight_real, self.weight_imag)                 # (H, d_head, d_head, modes)
        # einsum : pour chaque (b, h, modes), multiplier le vecteur d_head par la matrice (d_head, d_head)
        out = torch.einsum("bhdm,hodm->bhom", x, w)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, d_model)
        B, L, _ = x.shape
        # (B, L, H, d_head) -> (B, H, d_head, L)
        xr = x.view(B, L, self.n_heads, self.d_head).permute(0, 2, 3, 1)

        # FFT reelle sur la dimension temporelle
        x_fft = torch.fft.rfft(xr, dim=-1)                                     # (B, H, d_head, L//2 + 1)

        # Selectionne les M modes (basses frequences)
        x_fft_sel = x_fft[..., self.mode_idx]                                  # (B, H, d_head, modes)

        # Projection complexe
        out_fft_sel = self._complex_matmul(x_fft_sel)                          # (B, H, d_head, modes)

        # Reconstruit un spectre pleine taille avec zeros ailleurs
        out_fft = torch.zeros_like(x_fft)
        out_fft[..., self.mode_idx] = out_fft_sel

        # IFFT pour revenir au domaine temporel
        out = torch.fft.irfft(out_fft, n=L, dim=-1)                            # (B, H, d_head, L)
        out = out.permute(0, 3, 1, 2).contiguous().view(B, L, self.d_model)
        return out


# -----------------------------------------------------------------------------
# Bloc encoder FEDformer : Fourier + decomp + FFN + decomp (comme Autoformer)
# -----------------------------------------------------------------------------
class FEDformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, seq_len: int,
                 modes: int = 32, kernel_size: int = 25, dropout: float = 0.1):
        super().__init__()
        self.fourier = FourierBlock(d_model, n_heads, seq_len, modes)
        self.decomp1 = SeriesDecomp(kernel_size)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.decomp2 = SeriesDecomp(kernel_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.dropout(self.fourier(x))
        x, trend1 = self.decomp1(x)
        x = x + self.ffn(x)
        x, trend2 = self.decomp2(x)
        return x, trend1 + trend2


# -----------------------------------------------------------------------------
# Modele complet
# -----------------------------------------------------------------------------
class FEDformer(nn.Module):
    def __init__(
        self,
        input_len: int,
        pred_len: int,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 256,
        modes: int = 32,
        kernel_size: int = 25,
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
            [FEDformerBlock(d_model, n_heads, d_ff, input_len, modes, kernel_size, dropout)
             for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

        self.seasonal_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_len * d_model, pred_len * n_features),
        )
        self.trend_head = nn.Linear(input_len, pred_len)
        self.trend_proj = nn.Linear(d_model, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal_init, trend_init = self.decomp_init(x)
        s = self.value_embed(seasonal_init)
        s = self.input_dropout(s)
        trend_accum = self.value_embed(trend_init)

        for block in self.blocks:
            s, trend_delta = block(s)
            trend_accum = trend_accum + trend_delta

        s = self.norm(s)
        seasonal_out = self.seasonal_head(s).view(-1, self.pred_len, self.n_features)
        trend_proj = self.trend_proj(trend_accum)
        trend_out = self.trend_head(trend_proj.transpose(1, 2)).transpose(1, 2)
        return seasonal_out + trend_out
