"""API FastAPI servant a la fois l'interface Gradio (route /) et l'API REST
(routes /api/*, /health, /metrics).

Points d'entree :
    GET  /                 -> UI Gradio (3 onglets)
    GET  /docs             -> Swagger UI auto
    GET  /health           -> statut du service
    GET  /api/models       -> liste des modeles+horizons dispo
    POST /api/predict      -> prediction a partir d'une fenetre fournie
    GET  /metrics          -> metriques format Prometheus (phase C)

Lancement local :
    uvicorn api:app --host 0.0.0.0 --port 7860

Dans HF Spaces : le Dockerfile execute cette commande au CMD.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
import numpy as np
from db import init_schema, recent_predictions, summary_stats
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from predictor import available_checkpoints, predict_and_log

# Metriques Prometheus (phase C)
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from schemas import (
    CompareModelResult,
    CompareRequest,
    CompareResponse,
    HealthResponse,
    ModelCard,
    ModelsResponse,
    MonitoringResponse,
    PredictionRow,
    PredictRequest,
    PredictResponse,
)
from ui import build_demo

PRED_COUNTER = Counter(
    "ltsf_predictions_total", "Total predictions served", ["model", "pred_len"]
)
DRIFT_COUNTER = Counter(
    "ltsf_drift_detected_total", "Predictions flagged as drifted", ["model"]
)
LATENCY_HIST = Histogram(
    "ltsf_predict_latency_seconds", "Inference latency (s)", ["model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# Chargement .env : d'abord dans le CWD (deploiement HF ou test dans ce dossier),
# puis a la racine du projet (dev local avec .env a la racine).
_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")
load_dotenv(_HERE.parent / ".env")

# Logger JSON-friendly
logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}',
)
log = logging.getLogger("api")

VERSION = "0.2.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks.

    init_schema() tourne en background : si Neon est froid (cold start ~30s),
    on ne bloque pas le bind du port. Le logging DB s'activera des que
    le schema est pret, sinon les predictions tournent sans log (best-effort).
    """
    import asyncio
    log.info("API starting")
    ckpts = available_checkpoints()
    n_loaded = sum(len(v) for v in ckpts.values())
    log.info(f"available checkpoints: {n_loaded} across {len(ckpts)} models")

    async def _init_db_bg():
        # to_thread evite de bloquer l'event loop sur le cold start Neon.
        ok = await asyncio.to_thread(init_schema)
        if ok:
            log.info("db: schema ready, prediction logging enabled")
        else:
            log.info("db: logging disabled (no URL or init failed)")

    asyncio.create_task(_init_db_bg())
    yield
    log.info("API shutting down")


app = FastAPI(
    title="LTSF Transformers",
    description="Serving + comparison UI + MLOps monitoring",
    version=VERSION,
    lifespan=lifespan,
)


# -----------------------------------------------------------------------------
# API REST
# -----------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    ckpts = available_checkpoints()
    n_loaded = sum(len(v) for v in ckpts.values())
    return HealthResponse(status="ok", version=VERSION, loaded_models=n_loaded)


@app.get("/api/models", response_model=ModelsResponse, tags=["inference"])
def list_models():
    ckpts = available_checkpoints()
    cards = [ModelCard(name=name, pred_lens=hs) for name, hs in ckpts.items()]
    return ModelsResponse(
        models=cards,
        default_model="PatchTST" if "PatchTST" in ckpts else next(iter(ckpts), ""),
        default_pred_len=96,
    )


@app.post("/api/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest):
    try:
        window = np.asarray(req.window, dtype=np.float32)
        out = predict_and_log(req.model, req.pred_len, window)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        PRED_COUNTER.labels(model=req.model, pred_len=str(req.pred_len)).inc()
        LATENCY_HIST.labels(model=req.model).observe(out["latency_ms"] / 1000.0)
        if out["drift_flag"]:
            DRIFT_COUNTER.labels(model=req.model).inc()
    except Exception as e:
        log.warning(f"metrics bump failed: {e}")

    log.info(
        f'predict ok model={req.model} pred_len={req.pred_len} '
        f'latency_ms={out["latency_ms"]:.1f} drift={out["drift_flag"]}'
    )
    return PredictResponse(
        model=req.model,
        pred_len=req.pred_len,
        features=["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
        prediction=out["prediction"].tolist(),
        latency_ms=round(out["latency_ms"], 2),
        drift_flag=out["drift_flag"],
        drift_pvalues=out["drift_pvalues"],
    )


def _check_admin(token: str | None) -> None:
    """Guard pour endpoints admin. 401 si token manquant/invalide.

    Si ADMIN_TOKEN n'est pas defini, l'endpoint reste ferme (fail-closed).
    """
    expected = os.getenv("ADMIN_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="Admin endpoints disabled (ADMIN_TOKEN not set)")
    if not token or token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Token")


@app.post("/api/compare", response_model=CompareResponse, tags=["inference"])
def compare(req: CompareRequest):
    """Compare plusieurs modeles sur une meme fenetre.

    Retourne la prediction et la latence par modele, plus le drift KS sur la fenetre
    d'entree (commun a tous puisque les donnees d'entree sont identiques).
    """
    try:
        window = np.asarray(req.window, dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Drift : calcule par predict_and_log a chaque appel, identique puisque meme fenetre.
    # On recupere la valeur du premier modele pour l'exposer au niveau reponse.
    drift_pvals: dict | None = None
    drift_flag = False

    results: list[CompareModelResult] = []
    for name in req.models:
        try:
            out = predict_and_log(name, req.pred_len, window)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=f"{name}: {e}")
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"{name}: {e}")

        try:
            PRED_COUNTER.labels(model=name, pred_len=str(req.pred_len)).inc()
            LATENCY_HIST.labels(model=name).observe(out["latency_ms"] / 1000.0)
            if out["drift_flag"]:
                DRIFT_COUNTER.labels(model=name).inc()
        except Exception as e:
            log.warning(f"metrics bump failed in /api/compare: {e}")

        if drift_pvals is None:
            drift_pvals = out["drift_pvalues"]
            drift_flag = out["drift_flag"]

        results.append(CompareModelResult(
            model=name,
            prediction=out["prediction"].tolist(),
            latency_ms=round(out["latency_ms"], 2),
        ))

    log.info(
        f"compare ok models={req.models} pred_len={req.pred_len} drift={drift_flag}"
    )
    return CompareResponse(
        pred_len=req.pred_len,
        features=["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
        drift_flag=drift_flag,
        drift_pvalues=drift_pvals,
        results=results,
    )


@app.get("/api/monitoring", response_model=MonitoringResponse, tags=["ops"])
def monitoring(limit: int = 50, x_admin_token: str | None = Header(default=None)):
    """Dashboard data admin : totaux + N dernieres predictions.

    Protege par header X-Admin-Token (doit matcher la variable d'env ADMIN_TOKEN).
    """
    _check_admin(x_admin_token)
    stats = summary_stats()
    rows = recent_predictions(limit=limit)
    return MonitoringResponse(
        total=stats["total"],
        drifted=stats["drifted"],
        avg_latency_ms=stats["avg_latency_ms"],
        by_model=stats["by_model"],
        recent=[
            PredictionRow(
                ts=str(r["ts"]),
                model=r["model"],
                pred_len=r["pred_len"],
                latency_ms=r["latency_ms"],
                drift_flag=r["drift_flag"],
                alert=r["alert"],
            )
            for r in rows
        ],
    )


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
def metrics():
    """Metriques au format Prometheus (compteurs, histogrammes latence)."""
    body = generate_latest().decode("utf-8")
    body += (
        "# HELP ltsf_build_info Build info gauge\n"
        "# TYPE ltsf_build_info gauge\n"
        f'ltsf_build_info{{version="{VERSION}"}} 1\n'
    )
    return PlainTextResponse(body, media_type=CONTENT_TYPE_LATEST)


_STATIC_DIR = _HERE / "static"


@app.get("/admin", include_in_schema=False)
def admin_page():
    """Dashboard admin autonome (HTML+JS statique). L'auth se fait cote client
    via l'endpoint /api/monitoring + header X-Admin-Token : rien de secret dans
    la page HTML elle-meme, elle est sans valeur pour un visiteur non admin."""
    return FileResponse(_STATIC_DIR / "admin.html", media_type="text/html")


# -----------------------------------------------------------------------------
# Montage de l'interface Gradio sur /  (DOIT etre dernier : Gradio capture /*)
# -----------------------------------------------------------------------------
demo = build_demo()
app = gr.mount_gradio_app(app, demo, path="/")
