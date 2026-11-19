"""PatchTST (Nie et al., ICLR 2023).

Papier : 'A Time Series is Worth 64 Words: Long-term Forecasting with Transformers'.

Deux idees centrales :
    1. PATCHING : la serie est decoupee en patches de longueur `patch_len`
       avec un pas `stride`. Chaque patch devient un 'token' du Transformer.
       -> reduit la longueur de la sequence (donc la complexite de l'attention)
       -> chaque token agrege un vrai motif local de la serie.
    2. CHANNEL-INDEPENDENT : chaque variable est traitee par le MEME modele
       mais independamment des autres. Le Transformer ne voit jamais les
       variables ensemble. Contre-intuitif mais souvent plus robuste
       que l'approche multivariee en LTSF.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PatchTST(nn.Module):
    """Args:
        input_len  : longueur de la fenetre d'entree
        pred_len   : longueur de l'horizon de prediction
        n_features : nombre de variables (canaux)
        patch_len  : taille d'un patch
        stride     : pas entre deux patches (si = patch_len, patches disjoints)
        d_model    : dimension des tokens Transformer
        n_heads    : tetes d'attention
        n_layers   : blocs d'encodeur
        d_ff       : dimension FFN
        dropout    : taux de dropout
        revin      : si True, on normalise chaque fenetre et chaque canal
                     avant le modele puis on denormalise la sortie (astuce
                     RevIN du papier PatchTST pour gerer la non-stationnarite).
    """

    def __init__(
        self,
        input_len: int,
        pred_len: int,
        n_features: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        revin: bool = True,
    ):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.patch_len = patch_len
        self.stride = stride
        self.revin = revin

        # Nombre de patches apres unfold, en incluant un padding au debut
        # pour couvrir le tout premier pas de la serie.
        self.padding = stride
        self.n_patches = (input_len + self.padding - patch_len) // stride + 1

        # Projection d'un patch (R^patch_len) -> R^d_model
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = PositionalEncoding(d_model, max_len=self.n_patches + 1)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Tete flatten -> pred_len
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.n_patches * d_model, pred_len),
        )

    def _revin_normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x : (B, L, C) -> normalisation par fenetre et par canal
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-5
        return (x - mean) / std, mean, std

    def _revin_denormalize(self, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return y * std + mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, C)
        if self.revin:
            x, mean, std = self._revin_normalize(x)

        B, L, C = x.shape
        # Permute -> (B, C, L) pour patcher sur l'axe temporel
        xc = x.permute(0, 2, 1)
        # Padding par repetition du dernier pas (important : PAD a DROITE pour preserver
        # l'alignement temporel avec la sortie)
        xc = nn.functional.pad(xc, (0, self.padding), mode="replicate")

        # Unfold : (B, C, L + padding) -> (B, C, n_patches, patch_len)
        patches = xc.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        n_patches = patches.size(2)

        # Fusionner (B, C) pour traiter chaque canal independamment par le MEME modele
        patches = patches.reshape(B * C, n_patches, self.patch_len)

        tokens = self.patch_embed(patches)              # (B*C, n_patches, d_model)
        tokens = self.pos_embed(tokens)
        tokens = self.input_dropout(tokens)
        tokens = self.encoder(tokens)                   # (B*C, n_patches, d_model)

        y = self.head(tokens)                           # (B*C, pred_len)
        y = y.view(B, C, self.pred_len).permute(0, 2, 1)   # (B, pred_len, C)

        if self.revin:
            y = self._revin_denormalize(y, mean, std)
        return y
