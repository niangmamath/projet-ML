"""Pipeline de retraining.

Usage :
    # Retrain force (une combinaison) :
    python -m pipelines.retrain --model PatchTST --pred_len 96 --reason manual

    # Decide en fonction du drift observe sur 7j, pour toutes les combinaisons :
    python -m pipelines.retrain --check-drift

    # Combinaison particuliere avec check drift :
    python -m pipelines.retrain --model PatchTST --pred_len 96 --check-drift

Produit :
    results/checkpoints/<Model>_ETTh1_h<H>.pt   (staging, ecrit par train.py)
    + une row dans retrain_runs (Neon) avec MSE avant/apres.
Ne pousse PAS sur HF Hub : c'est le job de pipelines/promote.py.
"""
from __future__ import annotations

import argparse
import logging
import sys
from argparse import Namespace
from pathlib import Path

# .env pour DATA_POSTGRESQL (optionnel : la DB est no-op si absent)
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# Ordre important : ROOT doit l'emporter sur huggingface_space (qui a un
# stub src/ pour ses imports de models/ a l'exec du Space).
sys.path.insert(0, str(ROOT / "huggingface_space"))
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from db import drift_rate_last_days, last_production_mse, log_retrain  # noqa: E402

from pipelines.policy import decide_retrain  # noqa: E402
from src.train import train_and_eval  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("retrain")

MODELS = ["DLinear", "Transformer", "PatchTST", "Informer", "Autoformer", "FEDformer"]
HORIZONS = [96, 192, 336, 720]


def _make_train_args(model: str, pred_len: int, epochs: int) -> Namespace:
    """Construit un Namespace compatible avec src.train.train_and_eval."""
    return Namespace(
        model=model, data="ETTh1", input_len=96, pred_len=pred_len,
        epochs=epochs, batch_size=32, lr=1e-3,
        patience=5, seed=2026, tensorboard=False,
    )


def _retrain_one(model: str, pred_len: int, reason: str, epochs: int) -> dict:
    """Entrainement + log DB. Retourne le resultat enrichi (mse_before/after)."""
    log.info(f"=== RETRAIN {model} h={pred_len} (reason={reason}) ===")
    mse_before = last_production_mse(model, pred_len)
    log.info(f"baseline MSE en prod : {mse_before}")

    args = _make_train_args(model, pred_len, epochs=epochs)
    result = train_and_eval(args)
    mse_after = float(result["MSE"])
    mae_after = float(result["MAE"])

    run_id = log_retrain(
        model=model, pred_len=pred_len, reason=reason,
        mse_before=mse_before, mse_after=mse_after, mae_after=mae_after,
        train_time_s=float(result["train_time_s"]),
        promoted=False, notes=None,
    )
    log.info(
        f"OK MSE_before={mse_before} MSE_after={mse_after:.4f} "
        f"MAE={mae_after:.4f} run_id={run_id}"
    )
    return {**result, "mse_before": mse_before, "run_id": run_id}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=MODELS, help="Retrain cette combinaison uniquement")
    ap.add_argument("--pred_len", type=int, choices=HORIZONS, help="Horizon cible")
    ap.add_argument("--reason", default="manual",
                    choices=["manual", "drift", "scheduled"],
                    help="Motif loggue en base")
    ap.add_argument("--check-drift", action="store_true",
                    help="Appliquer la policy (drift >= 20%% sur 7j) avant retrain")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true",
                    help="Affiche la decision mais ne lance pas l'entrainement")
    args = ap.parse_args()

    # Decide quelles combinaisons retrain
    if args.check_drift:
        drift_stats = drift_rate_last_days(days=7)
        log.info(f"drift stats (7j) : {drift_stats or '(aucun log)'}")
        targets = (
            [(args.model, args.pred_len)]
            if args.model and args.pred_len
            else [(m, h) for m in MODELS for h in HORIZONS]
        )
        selected = []
        for m, h in targets:
            d = decide_retrain(m, h, drift_stats, manual=False)
            tag = "RETRAIN" if d.should_retrain else "skip   "
            log.info(f"  [{tag}] {m}/h{h} -> {d.reason}")
            if d.should_retrain:
                selected.append((m, h, "drift"))
    else:
        if not args.model or not args.pred_len:
            ap.error("--model et --pred_len requis (ou utiliser --check-drift)")
        selected = [(args.model, args.pred_len, args.reason)]

    if not selected:
        log.info("Rien a retrain. Fin.")
        return 0

    if args.dry_run:
        log.info(f"[DRY-RUN] {len(selected)} combinaison(s) a retrain : {selected}")
        return 0

    for model, pred_len, reason in selected:
        try:
            _retrain_one(model, pred_len, reason, epochs=args.epochs)
        except Exception as e:
            log.exception(f"retrain failed for {model}/h{pred_len} : {e}")

    log.info(f"Fin : {len(selected)} combinaison(s) traitee(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
