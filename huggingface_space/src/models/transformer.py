"""Vanilla Transformer pour la prevision de series temporelles long terme.

Architecture encoder-only, suivant la pratique LTSF :
    1. Embedding : projection lineaire (n_features -> d_model)
    2. Encodage positionnel sinusoidal (Vaswani et al. 2017)
    3. N blocs d'encodeur Transformer (self-attention + FFN)
    4. Tete de prediction : (B, L, d_model) -> (B, pred_len, n_features)

Sert de BASELINE attention pour comparer aux variantes
(Informer, Autoformer, FEDformer, PatchTST).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Encodage positionnel sinusoidal, fige (Vaswani 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))     # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, d_model)
        return x + self.pe[:, : x.size(1)]


class VanillaTransformer(nn.Module):
    """Encoder-only Transformer pour la prevision multivariee.

    Args:
        input_len   : longueur de la fenetre d'entree
        pred_len    : longueur de l'horizon de prediction
        n_features  : nombre de variables en entree (= en sortie en mode M)
        d_model     : dimension d'embedding
        n_heads     : tetes d'attention
        n_layers    : blocs d'encodeur empiles
        d_ff        : dimension du feed-forward intra-bloc
        dropout     : taux de dropout
    """

    def __init__(
        self,
        input_len: int,
        pred_len: int,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.d_model = d_model

        self.value_embed = nn.Linear(n_features, d_model)
        self.pos_embed = PositionalEncoding(d_model, max_len=max(5000, input_len + pred_len))
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

        # Tete de prediction : on applique un linear sur chaque pas d'embedding,
        # puis on flatten -> projection vers pred_len * n_features.
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_len * d_model, pred_len * n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, n_features)
        z = self.value_embed(x)                 # (B, L, d_model)
        z = self.pos_embed(z)
        z = self.input_dropout(z)
        z = self.encoder(z)                     # (B, L, d_model)
        y = self.head(z)                        # (B, pred_len * n_features)
        return y.view(-1, self.pred_len, self.n_features)
