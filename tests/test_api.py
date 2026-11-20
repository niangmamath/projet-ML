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


def test_compare_endpoint_valid(client, sample_window):
    """Compare 2 modeles sur la meme fenetre, verifie structure + drift unique."""
    r = client.post("/api/compare", json={
        "models": ["DLinear", "PatchTST"],
        "pred_len": 96,
        "window": sample_window.tolist(),
    })
    assert r.status_code == 200
    body = r.json()
    assert body["pred_len"] == 96
    assert "drift_flag" in body
    assert "drift_pvalues" in body
    assert len(body["results"]) == 2
    names = {r["model"] for r in body["results"]}
    assert names == {"DLinear", "PatchTST"}
    for r_ in body["results"]:
        assert len(r_["prediction"]) == 96
        assert len(r_["prediction"][0]) == 7
        assert r_["latency_ms"] >= 0


def test_compare_endpoint_bad_model(client, sample_window):
    """Un nom de modele hors enum doit etre rejete en 422 par Pydantic."""
    r = client.post("/api/compare", json={
        "models": ["DLinear", "NotAModel"],
        "pred_len": 96,
        "window": sample_window.tolist(),
    })
    assert r.status_code == 422


def test_predict_endpoint_bad_shape(client):
    r = client.post("/api/predict", json={
        "model": "DLinear",
        "pred_len": 96,
        "window": [[0.0] * 7] * 10,       # 10 lignes au lieu de 96
    })
    assert r.status_code == 422               # validation Pydantic


def test_monitoring_requires_admin_token(client, monkeypatch):
    """Sans ADMIN_TOKEN configure cote env -> 503."""
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    r = client.get("/api/monitoring")
    assert r.status_code == 503


def test_monitoring_rejects_bad_token(client, monkeypatch):
    """ADMIN_TOKEN configure mais header absent/incorrect -> 401."""
    monkeypatch.setenv("ADMIN_TOKEN", "s3cr3t")
    r = client.get("/api/monitoring")
    assert r.status_code == 401
    r = client.get("/api/monitoring", headers={"X-Admin-Token": "wrong"})
    assert r.status_code == 401


def test_monitoring_accepts_good_token(client, monkeypatch):
    """ADMIN_TOKEN configure + header correct -> 200 (dashboard vide, pas de DB)."""
    monkeypatch.setenv("ADMIN_TOKEN", "s3cr3t")
    r = client.get("/api/monitoring", headers={"X-Admin-Token": "s3cr3t"})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["recent"] == []


def test_metrics_endpoint_format(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "ltsf_build_info" in r.text
    assert r.headers["content-type"].startswith("text/plain")
