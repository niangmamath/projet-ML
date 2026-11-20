"""Service d'inference pour l'interface Gradio.

Charge n'importe quel checkpoint (model, pred_len) present dans ./checkpoints/,
prepare une fenetre d'entree (96 points, 7 variables) depuis le CSV de test,
l'applique au modele, denormalise et retourne (fenetre, prediction, verite).

La normalisation est refittee sur la partie train du CSV embarque pour
reproduire exactement le pipeline d'entrainement.
"""
from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.models.autoformer import Autoformer
from src.models.dlinear import DLinear
from src.models.fedformer import FEDformer
from src.models.informer import Informer
from src.models.patchtst import PatchTST
from src.models.transformer import VanillaTransformer

log = logging.getLogger("predictor")

ROOT = Path(__file__).resolve().parent
# Dossiers consultes pour les checkpoints, dans l'ordre :
#   1) huggingface_space/checkpoints  (image Docker legacy)
#   2) <project_root>/checkpoints     (dev local depuis la racine)
#   3) fallback HF Hub (hf_hub_download, cf. _resolve_checkpoint)
CKPT_DIRS = [ROOT / "checkpoints", ROOT.parent / "checkpoints"]
# Alias historique (utilise par ailleurs) : pointe vers le premier dossier existant.
CKPT_DIR = next((d for d in CKPT_DIRS if d.is_dir()), CKPT_DIRS[0])
DATA_CSV = ROOT / "data" / "ETTh1.csv"
FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
INPUT_LEN = 96
N_FEATURES = len(FEATURES)

# Registry HF Hub : si CKPT_DIR local est vide, on telecharge a la demande depuis
# MODEL_REPO_ID @ MODEL_VERSION. Les fichiers sont caches dans
# ~/.cache/huggingface/hub (HF_HOME overridable).
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "Momoniang/ltsf-transformers-etth1")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")

# Manifest statique : combinaisons connues presentes dans le registry.
_MANIFEST_MODELS = ["DLinear", "Transformer", "PatchTST", "Informer", "Autoformer", "FEDformer"]
_MANIFEST_HORIZONS = [96, 192, 336, 720]

# Bornes LTSF standard pour ETTh1 (12/4/4 mois en pas horaires)
_H = 30 * 24
TRAIN_END = 12 * _H
VAL_END = 16 * _H
TEST_END = 20 * _H

MODEL_BUILDERS = {
    "DLinear": lambda h: DLinear(INPUT_LEN, h, N_FEATURES),
    "Transformer": lambda h: VanillaTransformer(INPUT_LEN, h, N_FEATURES),
    "PatchTST": lambda h: PatchTST(INPUT_LEN, h, N_FEATURES),
    "Informer": lambda h: Informer(INPUT_LEN, h, N_FEATURES),
    "Autoformer": lambda h: Autoformer(INPUT_LEN, h, N_FEATURES),
    "FEDformer": lambda h: FEDformer(INPUT_LEN, h, N_FEATURES),
}


# -----------------------------------------------------------------------------
# Donnees et scaler (caches)
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_data() -> tuple[np.ndarray, StandardScaler]:
    """Charge ETTh1.csv et fit le StandardScaler sur la partie train uniquement."""
    df = pd.read_csv(DATA_CSV)
    arr = df[FEATURES].values.astype(np.float32)
    scaler = StandardScaler().fit(arr[:TRAIN_END])
    arr_norm = scaler.transform(arr).astype(np.float32)
    return arr_norm, scaler


@lru_cache(maxsize=1)
def test_set_size() -> int:
    arr, _ = _load_data()
    return TEST_END - VAL_END       # 2880 points pour ETTh1


def max_window_start(pred_len: int) -> int:
    """Index maximal (dans le test set) permettant une fenetre + horizon valide."""
    arr, _ = _load_data()
    # Position absolue du debut du test (avec overlap pour construire les fenetres)
    # La derniere fenetre doit satisfaire : start + INPUT_LEN + pred_len <= TEST_END
    return TEST_END - INPUT_LEN - pred_len


def last_hours_window(n_hours: int = INPUT_LEN) -> tuple[np.ndarray, pd.Timestamp]:
    """Retourne les n_hours dernieres lignes du CSV ETTh1 (non normalisees) et le
    timestamp de la derniere ligne. Sert de fenetre par defaut dans l'onglet
    'Utilisation reelle'."""
    df = pd.read_csv(DATA_CSV)
    last = df.iloc[-n_hours:][FEATURES].values.astype(np.float32)
    last_ts = pd.to_datetime(df["date"].iloc[-1])
    return last, last_ts


def _window_start_abs(sample_idx: int, pred_len: int) -> int:
    """Convertit un sample_idx [0, N] en index absolu dans le CSV, borne dans le test."""
    first = VAL_END - INPUT_LEN     # premiere fenetre du test (cf. data_loader standard)
    last = max_window_start(pred_len)
    start = first + sample_idx
    return int(min(max(start, first), last))


# -----------------------------------------------------------------------------
# Catalogue des checkpoints
# -----------------------------------------------------------------------------
_CKPT_RE = re.compile(r"^(?P<model>[A-Za-z]+)_(?P<data>[A-Za-z0-9]+)_h(?P<h>\d+)\.pt$")


@lru_cache(maxsize=1)
def available_checkpoints() -> dict[str, list[int]]:
    """Retourne {model_name: sorted([pred_len, ...])}.

    Priorite 1 : scan de CKPT_DIR local si des .pt y sont presents (dev local,
                 ou image Docker qui les embarque).
    Priorite 2 : manifest statique des 6 modeles x 4 horizons (= registry HF Hub).
    """
    out: dict[str, list[int]] = {}
    for d in CKPT_DIRS:
        if not d.is_dir():
            continue
        for f in d.iterdir():
            m = _CKPT_RE.match(f.name)
            if not m or m.group("data") != "ETTh1":
                continue
            out.setdefault(m.group("model"), []).append(int(m.group("h")))
        if out:
            break
    if out:
        return {k: sorted(set(v)) for k, v in sorted(out.items())}
    # Fallback registry : on annonce ce que MODEL_REPO_ID @ MODEL_VERSION contient.
    return {m: list(_MANIFEST_HORIZONS) for m in _MANIFEST_MODELS}


@lru_cache(maxsize=1)
def model_params(reference_horizon: int = 96) -> dict[str, int]:
    """Retourne {model_name: nombre de parametres entrainables} a un horizon de reference.

    L'horizon par defaut (96) sert de "taille nominale" pour la comparaison UI.
    Note : la derniere couche de projection depend de pred_len, donc le count varie
    entre horizons (ex. DLinear h=96 = 130k, h=720 = 977k). Voir rapport section 6.2.
    """
    out: dict[str, int] = {}
    for name, builder in MODEL_BUILDERS.items():
        try:
            m = builder(reference_horizon)
            out[name] = sum(p.numel() for p in m.parameters() if p.requires_grad)
        except Exception as e:
            log.warning(f"model_params: {name} failed: {e}")
    return out


def _resolve_checkpoint(model_name: str, pred_len: int) -> Path:
    """Localise le .pt a charger : local d'abord, sinon pull depuis HF Hub.

    Le cache HF est reutilise entre requetes; au premier appel pour un couple
    (model, h), le fichier descend sur disque puis reste la pour toute la duree
    de vie du conteneur.
    """
    fname = f"{model_name}_ETTh1_h{pred_len}.pt"
    for d in CKPT_DIRS:
        local = d / fname
        if local.is_file():
            return local
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise FileNotFoundError(
            f"Checkpoint local absent ({local}) et huggingface_hub indisponible."
        ) from e
    log.info(f"downloading {fname} from {MODEL_REPO_ID}@{MODEL_VERSION}")
    path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=fname,
        revision=MODEL_VERSION,
        repo_type="model",
        token=os.getenv("HF_TOKEN"),  # None accepte si repo public
    )
    return Path(path)


# -----------------------------------------------------------------------------
# Modele (cache par (model, pred_len))
# -----------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _load_model(model_name: str, pred_len: int) -> torch.nn.Module:
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Modele inconnu : {model_name}")
    try:
        ckpt = _resolve_checkpoint(model_name, pred_len)
    except Exception as e:
        raise FileNotFoundError(
            f"Checkpoint introuvable pour {model_name}/h{pred_len} : {e}"
        ) from e
    model = MODEL_BUILDERS[model_name](pred_len)
    state = torch.load(ckpt, weights_only=True, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# API principale
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict_window(
    model_name: str,
    pred_len: int,
    window: np.ndarray,
) -> dict:
    """Predit a partir d'une fenetre fournie par l'utilisateur (donnees brutes, non normalisees).

    Args:
        window : (INPUT_LEN, N_FEATURES) en unites reelles, dans l'ordre FEATURES.

    Returns:
        dict avec :
            context     (INPUT_LEN, 7)   fenetre entree telle quelle (denormalisee)
            prediction  (pred_len, 7)    prediction denormalisee
    """
    window = np.asarray(window, dtype=np.float32)
    if window.shape != (INPUT_LEN, N_FEATURES):
        raise ValueError(
            f"window doit avoir la forme ({INPUT_LEN}, {N_FEATURES}), recu {window.shape}"
        )
    _, scaler = _load_data()
    window_norm = scaler.transform(window).astype(np.float32)

    model = _load_model(model_name, pred_len)
    x_t = torch.from_numpy(window_norm).float().unsqueeze(0)
    yhat_norm = model(x_t).squeeze(0).cpu().numpy()

    pred = scaler.inverse_transform(yhat_norm)
    return {
        "context": window.copy(),
        "prediction": pred,
    }


def predict_and_log(
    model_name: str,
    pred_len: int,
    window: np.ndarray,
    alert: str | None = None,
) -> dict:
    """Chemin unifie predict + drift KS + log DB (best-effort).

    Appele par /api/predict, /api/compare ET par l'UI Gradio, pour que toute
    prediction (HTTP ou interface) soit tracee dans la table `predictions`.

    Returns:
        dict avec cles :
            prediction      (pred_len, 7)  denormalisee
            latency_ms      float          temps d'inference mur
            drift_pvalues   dict | None    p-values KS par feature
            drift_flag      bool           true si une feature derive (seuil ALPHA)
    """
    import time

    t0 = time.perf_counter()
    res = predict_window(model_name, pred_len, window)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    pred = res["prediction"]

    drift_pvals: dict | None = None
    drift_flag = False
    try:
        from drift import ks_drift
        drift_pvals, drift_flag = ks_drift(window)
    except Exception as e:
        log.warning(f"predict_and_log: drift check failed: {e}")

    try:
        from db import log_prediction
        from drift import compute_input_stats
        log_prediction(
            model=model_name,
            pred_len=pred_len,
            latency_ms=round(latency_ms, 2),
            input_stats=compute_input_stats(window),
            output_stats=compute_input_stats(pred),
            drift_pvalues=drift_pvals,
            drift_flag=drift_flag,
            alert=alert,
        )
    except Exception as e:
        log.warning(f"predict_and_log: db log failed: {e}")

    return {
        "prediction": pred,
        "latency_ms": latency_ms,
        "drift_pvalues": drift_pvals,
        "drift_flag": drift_flag,
    }


@torch.no_grad()
def predict_sample(model_name: str, pred_len: int, sample_idx: int) -> dict:
    """Retourne la prediction pour un echantillon donne du test set.

    Returns:
        dict avec cles :
            context     (INPUT_LEN, N_FEATURES)  denormalise
            prediction  (pred_len, N_FEATURES)   denormalise
            ground_truth(pred_len, N_FEATURES)   denormalise
            mse, mae : float
    """
    arr, scaler = _load_data()
    start = _window_start_abs(sample_idx, pred_len)
    x_norm = arr[start : start + INPUT_LEN]                       # (L, 7)
    y_norm = arr[start + INPUT_LEN : start + INPUT_LEN + pred_len]

    model = _load_model(model_name, pred_len)
    x_t = torch.from_numpy(x_norm).float().unsqueeze(0)           # (1, L, 7)
    yhat_norm = model(x_t).squeeze(0).cpu().numpy()               # (pred_len, 7)

    # Denormalisation
    context = scaler.inverse_transform(x_norm)
    pred = scaler.inverse_transform(yhat_norm)
    truth = scaler.inverse_transform(y_norm)

    mse = float(np.mean((pred - truth) ** 2))
    mae = float(np.mean(np.abs(pred - truth)))

    return {
        "context": context,
        "prediction": pred,
        "ground_truth": truth,
        "mse": mse,
        "mae": mae,
        "start_idx": start,
    }
