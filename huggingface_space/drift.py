"""Detection de drift par Kolmogorov-Smirnov (feature-wise).

Pour chaque appel a /api/predict, on compare la fenetre d'entree (96 x 7) a
un echantillon de reference tire du split train d'ETTh1 (les 12 premiers mois).
On rejette H0 (= meme distribution) quand p-value < alpha (defaut : 0.01).

Le drift_flag est True des qu'AU MOINS UNE feature derive. Les p-values sont
loguees en DB pour l'analyse historique.

Interpretation metier : un drift signale que les valeurs capteurs recues sont
statistiquement differentes de celles vues a l'entrainement -> les predictions
restent calculables mais leur fiabilite diminue (on affiche un warning user).
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd
from predictor import DATA_CSV, FEATURES, TRAIN_END
from scipy.stats import ks_2samp

log = logging.getLogger("drift")

ALPHA = 0.01         # seuil de significativite
REF_SIZE = 1000      # taille de l'echantillon de reference (train)


@lru_cache(maxsize=1)
def _reference_sample() -> np.ndarray:
    """Echantillon de reference (REF_SIZE, 7) tire du split train."""
    df = pd.read_csv(DATA_CSV)
    train = df[FEATURES].values[:TRAIN_END]            # (8640, 7)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(train), size=min(REF_SIZE, len(train)), replace=False)
    return train[idx].astype(np.float32)


def compute_input_stats(window: np.ndarray) -> dict:
    """Stats par feature pour une fenetre (96, 7) ou (pred_len, 7)."""
    arr = np.asarray(window, dtype=np.float32)
    return {
        feat: {
            "mean": float(arr[:, i].mean()),
            "std": float(arr[:, i].std()),
            "min": float(arr[:, i].min()),
            "max": float(arr[:, i].max()),
        }
        for i, feat in enumerate(FEATURES)
    }


def ks_drift(window: np.ndarray, alpha: float = ALPHA) -> tuple[dict[str, float], bool]:
    """Compare chaque feature de `window` au split train via KS-test.

    Returns:
        pvalues : {feature_name: p_value}
        drift_flag : True si p-value < alpha pour au moins 1 feature
    """
    ref = _reference_sample()
    pvals: dict[str, float] = {}
    for i, feat in enumerate(FEATURES):
        try:
            _, p = ks_2samp(window[:, i], ref[:, i], method="auto")
            pvals[feat] = float(p)
        except Exception as e:
            log.warning(f"drift: ks failed for {feat}: {e}")
            pvals[feat] = float("nan")
    drift = any(not np.isnan(p) and p < alpha for p in pvals.values())
    return pvals, drift
