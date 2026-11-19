"""Boucle d'entrainement unifiee pour tous les modeles du projet.

Usage :
    python src/train.py --model PatchTST --data ETTh1 --pred_len 96
    python src/train.py --model Informer --data ETTh1 --pred_len 192 --epochs 20

Pour un benchmark equitable, tous les modeles utilisent :
    - Adam, lr=1e-3 (lr=1e-4 pour les gros Transformer)
    - MSELoss
    - Early stopping (patience=5) sur val loss
    - Memes splits et memes normalisations (via data_loader)

Les resultats sont AJOUTES a results/metrics/comparison.csv.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.data_loader import load_dataset
from src.evaluate import evaluate_model
from src.utils import EarlyStopping, count_parameters, get_device, get_logger, set_seed

ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# Factory des modeles : importe uniquement celui demande (evite les cycles)
# -----------------------------------------------------------------------------
def build_model(name: str, input_len: int, pred_len: int, n_features: int) -> nn.Module:
    name = name.lower()
    if name == "dlinear":
        from src.models.dlinear import DLinear
        return DLinear(input_len, pred_len, n_features)
    if name == "transformer":
        from src.models.transformer import VanillaTransformer
        return VanillaTransformer(input_len, pred_len, n_features)
    if name == "patchtst":
        from src.models.patchtst import PatchTST
        return PatchTST(input_len, pred_len, n_features)
    if name == "informer":
        from src.models.informer import Informer
        return Informer(input_len, pred_len, n_features)
    if name == "autoformer":
        from src.models.autoformer import Autoformer
        return Autoformer(input_len, pred_len, n_features)
    if name == "fedformer":
        from src.models.fedformer import FEDformer
        return FEDformer(input_len, pred_len, n_features)
    raise ValueError(f"Modele inconnu : {name!r}")


# -----------------------------------------------------------------------------
# Boucle d'entrainement
# -----------------------------------------------------------------------------
def run_one_epoch(model: nn.Module, loader, optimizer, loss_fn, device, train: bool) -> float:
    model.train() if train else model.eval()
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
    return total / max(n, 1)


def train_and_eval(args) -> dict:
    log = get_logger(f"train.{args.model}")
    set_seed(args.seed)
    device = get_device()
    log.info(f"device={device}  model={args.model}  data={args.data}  pred_len={args.pred_len}")

    # Donnees
    train_loader, val_loader, test_loader, _ = load_dataset(
        args.data,
        input_len=args.input_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
    )
    n_features = next(iter(train_loader))[0].shape[-1]

    # Modele
    model = build_model(args.model, args.input_len, args.pred_len, n_features).to(device)
    n_params = count_parameters(model)
    log.info(f"n_params={n_params:,}")

    # Optim & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    loss_fn = nn.MSELoss()

    # Checkpoint + TensorBoard
    tag = f"{args.model}_{args.data}_h{args.pred_len}"
    ckpt_path = ROOT / "results" / "checkpoints" / f"{tag}.pt"
    es = EarlyStopping(patience=args.patience, path=ckpt_path)
    writer = SummaryWriter(log_dir=ROOT / "results" / "runs" / tag) if args.tensorboard else None

    # Boucle
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr = run_one_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        vl = run_one_epoch(model, val_loader, optimizer, loss_fn, device, train=False)
        scheduler.step(vl)
        es(vl, model)
        marker = "  *best*" if es.counter == 0 else ""
        log.info(f"ep {ep:2d}  train={tr:.4f}  val={vl:.4f}  lr={optimizer.param_groups[0]['lr']:.1e}{marker}")
        if writer:
            writer.add_scalar("loss/train", tr, ep)
            writer.add_scalar("loss/val", vl, ep)
        if es.should_stop:
            log.info(f"early stopping a l'epoch {ep}")
            break
    train_time = time.time() - t0

    # Eval test
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    _, _, metrics = evaluate_model(model, test_loader, device)
    log.info(f"TEST  MSE={metrics['MSE']:.4f}  MAE={metrics['MAE']:.4f}  (train_time={train_time:.1f}s)")

    if writer:
        for k, v in metrics.items():
            writer.add_scalar(f"test/{k}", v, 0)
        writer.close()

    result = {
        "model": args.model,
        "data": args.data,
        "input_len": args.input_len,
        "pred_len": args.pred_len,
        "n_params": n_params,
        "train_time_s": round(train_time, 1),
        **{k: round(v, 4) for k, v in metrics.items()},
    }

    # Append dans comparison.csv
    out_csv = ROOT / "results" / "metrics" / "comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result.keys()))
        if write_header:
            w.writeheader()
        w.writerow(result)

    return result


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   choices=["DLinear", "Transformer", "PatchTST", "Informer", "Autoformer", "FEDformer"])
    p.add_argument("--data", default="ETTh1", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"])
    p.add_argument("--input_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--tensorboard", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train_and_eval(parse_args())
