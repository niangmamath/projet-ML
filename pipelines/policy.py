"""Regles de decision pour le pipeline MLOps.

Deux decisions distinctes :
    1) RETRAIN - faut-il relancer un entrainement ?
       Oui si :
         - trigger explicite 'manual' (CLI), OU
         - sur une (model, pred_len) donnee, drift_rate(7j) >= RETRAIN_DRIFT_RATE
           ET nombre d'echantillons >= RETRAIN_MIN_SAMPLES

    2) PROMOTE - faut-il publier le nouveau checkpoint sur HF Hub ?
       Oui si :
         - mse_after < mse_before * (1 - PROMOTE_MIN_IMPROVEMENT), OU
         - il n'y a aucun baseline en prod (bootstrap : on promeut d'office)

Les seuils sont volontairement conservateurs : mieux vaut ne pas promouvoir
qu'ecraser un bon modele avec du bruit statistique.
"""
from __future__ import annotations

from dataclasses import dataclass

# Seuils par defaut (modifiables par CLI dans retrain/promote)
RETRAIN_DRIFT_RATE = 0.20      # 20 % des predictions en drift sur 7j
RETRAIN_MIN_SAMPLES = 50       # on exige un minimum d'historique
PROMOTE_MIN_IMPROVEMENT = 0.02 # 2 % d'amelioration MSE pour promouvoir


@dataclass
class RetrainDecision:
    should_retrain: bool
    reason: str
    drift_rate: float | None = None
    n_samples: int | None = None


def decide_retrain(
    model: str,
    pred_len: int,
    drift_stats: dict,
    manual: bool = False,
    drift_threshold: float = RETRAIN_DRIFT_RATE,
    min_samples: int = RETRAIN_MIN_SAMPLES,
) -> RetrainDecision:
    """Decide si on doit retrain (model, pred_len) donne.

    Args:
        drift_stats : output de db.drift_rate_last_days(), ex.
                      {("PatchTST", 96): {"n": 120, "drift_rate": 0.35}}
    """
    if manual:
        return RetrainDecision(True, "manual")

    stats = drift_stats.get((model, pred_len))
    if stats is None:
        return RetrainDecision(False, "no_data", drift_rate=None, n_samples=0)

    n = stats["n"]
    rate = stats["drift_rate"]
    if n < min_samples:
        return RetrainDecision(
            False, f"insufficient_samples (n={n} < {min_samples})",
            drift_rate=rate, n_samples=n,
        )
    if rate < drift_threshold:
        return RetrainDecision(
            False, f"drift_ok (rate={rate:.2f} < {drift_threshold:.2f})",
            drift_rate=rate, n_samples=n,
        )
    return RetrainDecision(
        True, f"drift_exceeded (rate={rate:.2f} >= {drift_threshold:.2f})",
        drift_rate=rate, n_samples=n,
    )


@dataclass
class PromoteDecision:
    should_promote: bool
    reason: str
    improvement: float | None = None


def decide_promote(
    mse_before: float | None,
    mse_after: float,
    min_improvement: float = PROMOTE_MIN_IMPROVEMENT,
) -> PromoteDecision:
    """Decide si le nouveau modele merite d'etre publie."""
    if mse_before is None:
        return PromoteDecision(True, "bootstrap (aucun baseline en prod)")
    improvement = (mse_before - mse_after) / mse_before
    if improvement >= min_improvement:
        return PromoteDecision(
            True, f"improved ({improvement*100:.2f}% >= {min_improvement*100:.0f}%)",
            improvement=improvement,
        )
    return PromoteDecision(
        False, f"not_better ({improvement*100:+.2f}% < {min_improvement*100:.0f}%)",
        improvement=improvement,
    )
