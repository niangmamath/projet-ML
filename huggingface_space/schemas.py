"""Schemas Pydantic pour l'API REST."""
from __future__ import annotations

from typing import Literal

from predictor import FEATURES, INPUT_LEN
from pydantic import BaseModel, Field, field_validator

ModelName = Literal["DLinear", "Transformer", "PatchTST", "Informer", "Autoformer", "FEDformer"]
Horizon = Literal[96, 192, 336, 720]


class PredictRequest(BaseModel):
    """Requete de prediction a partir d'une fenetre fournie par l'utilisateur.

    `window` doit etre une matrice (96, 7) avec les 7 features ETT dans l'ordre :
    HUFL, HULL, MUFL, MULL, LUFL, LULL, OT.
    """
    model: ModelName = "PatchTST"
    pred_len: Horizon = 96
    window: list[list[float]] = Field(
        ..., description=f"Matrice ({INPUT_LEN}, {len(FEATURES)}) des 96 dernieres heures"
    )

    @field_validator("window")
    @classmethod
    def _check_shape(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != INPUT_LEN:
            raise ValueError(f"window doit avoir exactement {INPUT_LEN} lignes, recu {len(v)}")
        for i, row in enumerate(v):
            if len(row) != len(FEATURES):
                raise ValueError(
                    f"ligne {i} : {len(row)} colonnes, attendu {len(FEATURES)} ({FEATURES})"
                )
        return v


class PredictResponse(BaseModel):
    model: str
    pred_len: int
    features: list[str]
    prediction: list[list[float]]           # (pred_len, 7) denormalise
    latency_ms: float
    drift_flag: bool = False                # true si une feature derive (KS-test)
    drift_pvalues: dict[str, float] | None = None   # p-value par feature
    alert: dict | None = None               # rempli si seuil fourni par user


class CompareRequest(BaseModel):
    """Compare plusieurs modeles sur une meme fenetre (1 a 6 modeles)."""
    models: list[ModelName] = Field(..., min_length=1, max_length=6)
    pred_len: Horizon = 96
    window: list[list[float]] = Field(..., description=f"Matrice ({INPUT_LEN}, {len(FEATURES)})")

    @field_validator("window")
    @classmethod
    def _check_shape(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != INPUT_LEN:
            raise ValueError(f"window doit avoir exactement {INPUT_LEN} lignes, recu {len(v)}")
        for i, row in enumerate(v):
            if len(row) != len(FEATURES):
                raise ValueError(
                    f"ligne {i} : {len(row)} colonnes, attendu {len(FEATURES)} ({FEATURES})"
                )
        return v

    @field_validator("models")
    @classmethod
    def _dedup(cls, v: list[str]) -> list[str]:
        seen, out = set(), []
        for m in v:
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out


class CompareModelResult(BaseModel):
    model: str
    prediction: list[list[float]]
    latency_ms: float


class CompareResponse(BaseModel):
    pred_len: int
    features: list[str]
    drift_flag: bool = False
    drift_pvalues: dict[str, float] | None = None
    results: list[CompareModelResult]


class PredictionRow(BaseModel):
    ts: str
    model: str
    pred_len: int
    latency_ms: float | None = None
    drift_flag: bool | None = None
    alert: str | None = None


class MonitoringResponse(BaseModel):
    total: int
    drifted: int
    avg_latency_ms: float | None
    by_model: dict[str, int]
    recent: list[PredictionRow]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    loaded_models: int


class ModelCard(BaseModel):
    name: str
    pred_lens: list[int]


class ModelsResponse(BaseModel):
    models: list[ModelCard]
    default_model: str
    default_pred_len: int
