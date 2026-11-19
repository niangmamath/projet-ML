"""Interpretabilite des modeles de prevision long terme.

Trois analyses :
    1. Erreur par pas d'horizon (tous modeles) : montre comment chaque
       architecture degrade quand l'horizon s'allonge.
    2. Importance des variables par permutation : mesure la hausse de MSE
       quand on permute aleatoirement une feature a l'entree -> importance.
    3. Cartes d'attention du Vanilla Transformer : hooks sur les couches
       pour recuperer les matrices d'attention et moyenner sur les tetes.

Sortie : figures PNG dans results/figures/ et CSV d'importance dans
         results/metrics/feature_importance.csv.

Usage :
    python src/interpret.py --model PatchTST --data ETTh1 --pred_len 96
    python src/interpret.py --model Transformer --data ETTh1 --pred_len 96 --attention
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.data_loader import load_dataset
from src.evaluate import evaluate_model, mse
from src.train import build_model
from src.utils import get_device, get_logger, set_seed

ROOT = Path(__file__).resolve().parent.parent
FEATURES_ETT = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


# -----------------------------------------------------------------------------
# 1) Erreur par horizon
# -----------------------------------------------------------------------------
def error_per_horizon(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Retourne MAE par pas d'horizon, moyennee sur echantillons et features."""
    return np.abs(pred - true).mean(axis=(0, 2))


# -----------------------------------------------------------------------------
# 2) Importance des variables par permutation
# -----------------------------------------------------------------------------
@torch.no_grad()
def permutation_importance(
    model: nn.Module,
    loader,
    device: torch.device,
    n_features: int,
    n_repeats: int = 3,
    seed: int = 2026,
) -> np.ndarray:
    """Pour chaque feature, permute aleatoirement ses valeurs dans la fenetre
    d'entree (le long de l'axe temps, par batch) et mesure la hausse de MSE.

    Retourne un vecteur (n_features,) : importance = MSE_perm - MSE_base.
    """
    rng = np.random.default_rng(seed)
    model.eval()

    # MSE de reference
    all_preds, all_trues = [], []
    all_inputs = []
    for x, y in loader:
        all_inputs.append(x)
        all_preds.append(model(x.to(device)).cpu())
        all_trues.append(y)
    base_pred = torch.cat(all_preds).numpy()
    true = torch.cat(all_trues).numpy()
    base_mse = mse(base_pred, true)

    importances = np.zeros(n_features)
    for f in range(n_features):
        scores = []
        for _ in range(n_repeats):
            perm_preds = []
            for x in all_inputs:
                xp = x.clone()
                # Permute la feature f sur l'axe temporel, independemment par batch
                perm = torch.from_numpy(rng.permutation(xp.shape[1]))
                xp[:, :, f] = xp[:, perm, f]
                perm_preds.append(model(xp.to(device)).cpu())
            pm = torch.cat(perm_preds).numpy()
            scores.append(mse(pm, true) - base_mse)
        importances[f] = float(np.mean(scores))

    return importances


# -----------------------------------------------------------------------------
# 3) Cartes d'attention (Vanilla Transformer uniquement)
# -----------------------------------------------------------------------------
def collect_attention(model: nn.Module, loader, device: torch.device, n_batches: int = 1) -> np.ndarray | None:
    """Extrait les cartes d'attention moyennes d'un Vanilla Transformer.

    `nn.TransformerEncoderLayer` emprunte en interne un fast path qui ignore
    `need_weights=True` sur `self_attn`. On remplace donc directement sa
    methode `_sa_block` pour appeler `self_attn(..., need_weights=True,
    average_attn_weights=True)` et stocker les poids d'attention.
    Retourne None si le modele ne contient aucune couche standard.
    """
    layers = [m for m in model.modules() if isinstance(m, nn.TransformerEncoderLayer)]
    if not layers:
        return None

    attns = []
    orig_forwards = []

    def make_forward(layer: nn.TransformerEncoderLayer):
        """Reimplementation 'slow path' qui appelle self_attn avec need_weights=True.

        PyTorch utilise en inference un fast path C++ qui court-circuite _sa_block
        et ne retourne jamais les poids d'attention. On remplace donc le forward
        du layer par une version explicite equivalente (norm_first=True ici).
        """
        def _forward(src, src_mask=None, src_key_padding_mask=None, is_causal=False):
            x = src
            # Branche attention (pre-norm car norm_first=True dans notre modele)
            y = layer.norm1(x)
            attn_out, w = layer.self_attn(
                y, y, y,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            attns.append(w.detach().cpu())              # (B, L, L)
            x = x + layer.dropout1(attn_out)
            # Branche FFN
            x = x + layer._ff_block(layer.norm2(x))
            return x
        return _forward

    for layer in layers:
        orig_forwards.append((layer, layer.forward))
        layer.forward = make_forward(layer)

    try:
        model.eval()
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                model(x.to(device))
                if i + 1 >= n_batches:
                    break
    finally:
        for layer, orig in orig_forwards:
            layer.forward = orig

    if not attns:
        return None
    # Concatene toutes les cartes puis moyenne sur (couches * appels * batch)
    stacked = torch.cat(attns, dim=0).numpy()  # (n_layers * n_batches * B, L, L)
    return stacked.mean(axis=0)                # (L, L)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(args):
    log = get_logger(f"interpret.{args.model}")
    set_seed(args.seed)
    device = get_device()
    tag = f"{args.model}_{args.data}_h{args.pred_len}"
    log.info(f"interpretation de {tag}")

    # Donnees + modele
    _, _, test_loader, _ = load_dataset(args.data, args.input_len, args.pred_len, batch_size=args.batch_size)
    features = FEATURES_ETT
    n_features = len(features)
    model = build_model(args.model, args.input_len, args.pred_len, n_features).to(device)

    ckpt = ROOT / "results" / "checkpoints" / f"{tag}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint introuvable : {ckpt}")
    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
    log.info(f"checkpoint charge : {ckpt.name}")

    fig_dir = ROOT / "results" / "figures" / tag
    fig_dir.mkdir(parents=True, exist_ok=True)

    # (a) Predictions test
    pred, true, metrics = evaluate_model(model, test_loader, device)
    log.info(f"test MSE={metrics['MSE']:.4f}  MAE={metrics['MAE']:.4f}")

    # (b) Erreur par pas d'horizon
    err_h = error_per_horizon(pred, true)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(err_h, linewidth=2, color="#c0392b")
    ax.set_xlabel("pas d'horizon")
    ax.set_ylabel("MAE")
    ax.set_title(f"{args.model} - MAE par pas d'horizon (h={args.pred_len})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "error_per_horizon.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("sauve error_per_horizon.png")

    # (c) Exemples de predictions (3 fenetres)
    for i, idx in enumerate([0, len(pred) // 2, len(pred) - 1]):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(true[idx, :, -1], label="reel OT", linewidth=2)
        ax.plot(pred[idx, :, -1], label="prediction", linewidth=2, linestyle="--")
        ax.set_xlabel("pas d'horizon")
        ax.set_ylabel("OT (normalise)")
        ax.set_title(f"{args.model} h={args.pred_len} - exemple {i+1}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"pred_sample_{i+1}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    log.info("sauve 3 exemples de predictions")

    # (d) Importance des variables (permutation)
    log.info("calcul de l'importance des variables (permutation)...")
    imp = permutation_importance(model, test_loader, device, n_features, n_repeats=args.n_repeats)

    fig, ax = plt.subplots(figsize=(8, 4))
    order = np.argsort(imp)[::-1]
    ax.bar([features[i] for i in order], imp[order], color="#2980b9")
    ax.set_ylabel("DeltaMSE (permutation)")
    ax.set_title(f"{args.model} - importance des variables")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # CSV d'importance (append)
    out_csv = ROOT / "results" / "metrics" / "feature_importance.csv"
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["model", "data", "pred_len", "feature", "delta_mse"])
        for feat, val in zip(features, imp):
            w.writerow([args.model, args.data, args.pred_len, feat, round(float(val), 6)])
    log.info("sauve feature_importance.png + .csv")

    # (e) Attention (Transformer uniquement)
    if args.attention:
        attn = collect_attention(model, test_loader, device, n_batches=2)
        if attn is None:
            log.info("pas d'attention standard dans ce modele (Informer, Autoformer, etc. ont une attention custom)")
        else:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(attn, aspect="auto", cmap="viridis")
            ax.set_xlabel("cle (pas d'entree)")
            ax.set_ylabel("requete (pas d'entree)")
            ax.set_title(f"{args.model} - attention moyenne")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(fig_dir / "attention_map.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            log.info("sauve attention_map.png")

    log.info(f"interpretation terminee -> {fig_dir}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   choices=["DLinear", "Transformer", "PatchTST", "Informer", "Autoformer", "FEDformer"])
    p.add_argument("--data", default="ETTh1", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"])
    p.add_argument("--input_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_repeats", type=int, default=3)
    p.add_argument("--attention", action="store_true",
                   help="tenter d'extraire les cartes d'attention (Vanilla Transformer)")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
