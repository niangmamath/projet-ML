"""Smoke tests de l'API FastAPI (sans DB : init_schema est no-op)."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import api
    with TestClient(api.app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["loaded_models"] == 24


def test_models_endpoint(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    body = r.json()
    assert body["default_model"] == "PatchTST"
    assert len(body["models"]) == 6
    assert body["default_pred_len"] == 96


def test_predict_endpoint_valid(client, sample_window):
    r = client.post("/api/predict", json={
        "model": "DLinear",
        "pred_len": 96,
        "window": sample_window.tolist(),
    })
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "DLinear"
    assert body["pred_len"] == 96
    assert len(body["prediction"]) == 96
    assert len(body["prediction"][0]) == 7
    assert "drift_flag" in body
    assert "drift_pvalues" in body


def test_predict_endpoint_bad_shape(client):
    r = client.post("/api/predict", json={
        "model": "DLinear",
        "pred_len": 96,
        "window": [[0.0] * 7] * 10,       # 10 lignes au lieu de 96
    })
    assert r.status_code == 422               # validation Pydantic


def test_monitoring_endpoint_without_db(client):
    """Sans DB configuree, l'endpoint doit repondre un dashboard vide (pas 500)."""
    r = client.get("/api/monitoring")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["recent"] == []


def test_metrics_endpoint_format(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "ltsf_build_info" in r.text
    assert r.headers["content-type"].startswith("text/plain")
