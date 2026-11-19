"""Informer (Zhou et al., AAAI 2021 — Best Paper).

Papier : 'Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting'.

Trois contributions cles :
    1. PROBSPARSE ATTENTION : au lieu de calculer l'attention pour les L queries,
       on identifie les Top-u (u = c*log L) queries les plus 'informatives'
       (celles dont la distribution d'attention s'eloigne le plus de l'uniforme,
       via une mesure KL approchee). Les autres queries recoivent la moyenne
       des valeurs. -> complexite O(L log L) au lieu de O(L^2).
    2. SELF-ATTENTION DISTILLING : entre deux couches d'encodeur, on applique
       une convolution 1D + MaxPool stride 2 qui divise par 2 la longueur
       de la sequence. Reduit encore la memoire sur les tres longues entrees.
    3. GENERATIVE DECODER : predit toute la sequence future en UN pas (pas
       d'autoregression) via un token placeholder + cross-attention.

Implementation simplifiee (encoder-only + flatten head) pour une comparaison
directe et equitable avec les autres modeles du projet.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# ProbSparse Self-Attention
# -----------------------------------------------------------------------------
class ProbSparseAttention(nn.Module):
    """Attention multi-tete avec selection Top-u des queries les plus informatives."""

    def __init__(self, d_model: int, n_heads: int, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.factor = factor                      # c : facteur pour u = c * ln(L)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _prob_qk(self, Q: torch.Tensor, K: torch.Tensor, n_top: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Selectionne les n_top queries dont la distribution est la plus 'eloignee'
        de l'uniforme, via un sampling aleatoire des cles pour approximer la
        mesure de sparsite (KL approchee)."""
        # Q, K : (B, H, L, d_head)
        B, H, L_q, _ = Q.shape
        L_k = K.size(-2)

        # Sample au hasard ln(L_k)*factor cles
        n_sample = min(self.factor * int(np.ceil(np.log(L_k))), L_k)
        idx_sample = torch.randint(L_k, (n_sample,), device=Q.device)
        K_sample = K[:, :, idx_sample, :]                                   # (B, H, n_sample, d_head)

        # Produit scalaire Q x K_sample
        Q_K_sample = torch.einsum("bhld,bhnd->bhln", Q, K_sample)           # (B, H, L_q, n_sample)

        # Mesure de 'sparsite' : max - mean (plus elle est grande, plus la distribution
        # concentre son attention sur quelques cles = 'interessante').
        M = Q_K_sample.max(dim=-1).values - Q_K_sample.mean(dim=-1)         # (B, H, L_q)

        # Garde les n_top queries les plus informatives
        M_top = M.topk(n_top, dim=-1, sorted=False).indices                 # (B, H, n_top)
        Q_reduced = torch.gather(
            Q, 2, M_top.unsqueeze(-1).expand(-1, -1, -1, self.d_head)
        )                                                                   # (B, H, n_top, d_head)

        # Attention complete sur ces queries
        scores = torch.einsum("bhud,bhkd->bhuk", Q_reduced, K) / math.sqrt(self.d_head)
        return scores, M_top

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        n_top = min(self.factor * int(np.ceil(np.log(L))), L)
        scores, idx_top = self._prob_qk(Q, K, n_top)                        # (B, H, n_top, L)
        attn = F.softmax(scores, dim=-1)
        context_top = torch.einsum("bhuk,bhkd->bhud", attn, V)              # (B, H, n_top, d_head)

        # Pour les autres queries : on leur attribue la moyenne des valeurs
        # (approximation sous-tendue par la distribution uniforme).
        context = V.mean(dim=-2, keepdim=True).expand(-1, -1, L, -1).clone()
        context.scatter_(
            2,
            idx_top.unsqueeze(-1).expand(-1, -1, -1, self.d_head),
            context_top,
        )

        out = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.dropout(self.out_proj(out))


# -----------------------------------------------------------------------------
# Bloc encodeur + distilling
# -----------------------------------------------------------------------------
class InformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, n_heads, factor, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DistillingLayer(nn.Module):
    """Conv1D + MaxPool stride 2 pour diviser la sequence par 2."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(d_model)
        self.act = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, d_model)
        x = x.transpose(1, 2)
        x = self.pool(self.act(self.norm(self.conv(x))))
        return x.transpose(1, 2)


# -----------------------------------------------------------------------------
# Modele complet
# -----------------------------------------------------------------------------
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


class Informer(nn.Module):
    """Informer simplifie : encoder ProbSparse + distilling + tete flatten.

    Args:
        input_len   : longueur de la fenetre d'entree
        pred_len    : longueur de l'horizon de prediction
        n_features  : nombre de variables
        d_model     : dimension d'embedding
        n_heads     : tetes d'attention
        n_layers    : nombre de blocs encoder
        d_ff        : dimension FFN
        factor      : facteur c pour le top-u = c * ln(L)
        dropout     : taux de dropout
        distil      : si True, applique un distilling entre chaque bloc
    """

    def __init__(
        self,
        input_len: int,
        pred_len: int,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        factor: int = 5,
        dropout: float = 0.1,
        distil: bool = True,
    ):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.distil = distil

        self.value_embed = nn.Linear(n_features, d_model)
        self.pos_embed = PositionalEncoding(d_model, max_len=max(5000, input_len))
        self.input_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [InformerEncoderBlock(d_model, n_heads, d_ff, factor, dropout) for _ in range(n_layers)]
        )
        self.distils = nn.ModuleList(
            [DistillingLayer(d_model) for _ in range(n_layers - 1)] if distil else []
        )

        # Longueur finale apres distillations
        self.final_len = input_len
        if distil:
            for _ in range(n_layers - 1):
                self.final_len = (self.final_len + 1) // 2

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.final_len * d_model, pred_len * n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.value_embed(x)
        z = self.pos_embed(z)
        z = self.input_dropout(z)

        for i, block in enumerate(self.blocks):
            z = block(z)
            if self.distil and i < len(self.distils):
                z = self.distils[i](z)

        z = self.norm(z)
        y = self.head(z)
        return y.view(-1, self.pred_len, self.n_features)
