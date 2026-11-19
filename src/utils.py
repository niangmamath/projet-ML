"""Utilitaires transverses : reproductibilite, early stopping, logging."""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 2026) -> None:
    """Fixe les seeds pour reproductibilite (CPU et CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Retourne cuda si dispo, sinon cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    """Arret anticipe sur val_loss avec sauvegarde du meilleur modele.

    Usage :
        es = EarlyStopping(patience=5, path='checkpoint.pt')
        for epoch in ...:
            val_loss = validate(...)
            es(val_loss, model)
            if es.should_stop:
                break
        model.load_state_dict(torch.load(es.path))
    """

    def __init__(self, patience: int = 5, delta: float = 0.0, path: str | Path = "checkpoint.pt"):
        self.patience = patience
        self.delta = delta
        self.path = Path(path)
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def get_logger(name: str = "projet7", level: int = logging.INFO) -> logging.Logger:
    """Logger simple avec format homogene, sans doublons de handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-5s %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
