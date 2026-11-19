"""Couche persistance pour les logs de prediction (Postgres Neon, phase C).

Design :
    - Connection pool lazy (singleton) via psycopg_pool.
    - Si aucune variable d'env (DATABASE_URL ou DATA_POSTGRESQL) n'est fournie,
      toutes les fonctions deviennent des no-op : l'app reste fonctionnelle
      sans DB (utile en dev, en CI, et si Neon est indisponible).
    - Les erreurs DB ne doivent JAMAIS casser l'inference : try/except partout
      avec log en WARNING. L'inference est la fonctionnalite principale.

Schema :
    predictions(
        id             bigserial primary key,
        ts             timestamptz not null default now(),
        model          text        not null,
        pred_len       int         not null,
        latency_ms     double precision,
        input_stats    jsonb,           -- {feature: {mean, std, min, max}}
        output_stats   jsonb,
        drift_pvalues  jsonb,           -- {feature: p_value}
        drift_flag     boolean,
        alert          text             -- message user, nullable
    )
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from typing import Any

log = logging.getLogger("db")

_POOL = None            # singleton
_POOL_INIT_TRIED = False
_SCHEMA_INIT_DONE = False

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id             bigserial PRIMARY KEY,
    ts             timestamptz NOT NULL DEFAULT now(),
    model          text        NOT NULL,
    pred_len       int         NOT NULL,
    latency_ms     double precision,
    input_stats    jsonb,
    output_stats   jsonb,
    drift_pvalues  jsonb,
    drift_flag     boolean,
    alert          text
);
CREATE INDEX IF NOT EXISTS predictions_ts_idx ON predictions (ts DESC);
CREATE INDEX IF NOT EXISTS predictions_model_idx ON predictions (model, pred_len);

CREATE TABLE IF NOT EXISTS retrain_runs (
    id             bigserial PRIMARY KEY,
    ts             timestamptz NOT NULL DEFAULT now(),
    model          text        NOT NULL,
    pred_len       int         NOT NULL,
    reason         text,                     -- 'manual' | 'drift' | 'scheduled'
    mse_before     double precision,
    mse_after      double precision,
    mae_after      double precision,
    train_time_s   double precision,
    promoted       boolean     DEFAULT false,
    hub_version    text,                     -- tag HF Hub si promu
    notes          text
);
CREATE INDEX IF NOT EXISTS retrain_runs_ts_idx ON retrain_runs (ts DESC);
"""


def _db_url() -> str | None:
    """Accepte DATABASE_URL (standard) ou DATA_POSTGRESQL (alias Neon)."""
    return os.getenv("DATABASE_URL") or os.getenv("DATA_POSTGRESQL")


def get_pool():
    """Pool singleton. Retourne None si pas de DB configuree ou init echoue."""
    global _POOL, _POOL_INIT_TRIED
    if _POOL is not None:
        return _POOL
    if _POOL_INIT_TRIED:
        return None
    _POOL_INIT_TRIED = True

    url = _db_url()
    if not url:
        log.info("db: DATABASE_URL/DATA_POSTGRESQL absent -> monitoring desactive")
        return None

    try:
        from psycopg_pool import ConnectionPool
    except ImportError:
        log.warning("db: psycopg_pool non installe -> monitoring desactive")
        return None

    try:
        # Lazy : le pool s'ouvrira au 1er check-out. Evite de bloquer le startup
        # si Neon est en cold-start (free tier dort apres ~5min d'inactivite).
        _POOL = ConnectionPool(
            conninfo=url,
            min_size=0,
            max_size=4,
            timeout=30,            # 30s au checkout pour couvrir cold-start Neon
            kwargs={"autocommit": True},
            open=False,
        )
        _POOL.open(wait=False)
        log.info("db: pool created (lazy, opens on first use)")
        return _POOL
    except Exception as e:
        log.warning(f"db: pool init failed -> monitoring desactive ({e})")
        _POOL = None
        return None


@contextmanager
def _conn():
    pool = get_pool()
    if pool is None:
        yield None
        return
    try:
        with pool.connection() as c:
            yield c
    except Exception as e:
        log.warning(f"db: connection error -> {e}")
        yield None


def init_schema() -> bool:
    """A appeler au startup. Idempotent. Retourne True si OK."""
    global _SCHEMA_INIT_DONE
    if _SCHEMA_INIT_DONE:
        return True
    with _conn() as c:
        if c is None:
            return False
        try:
            with c.cursor() as cur:
                cur.execute(SCHEMA_SQL)
            _SCHEMA_INIT_DONE = True
            log.info("db: schema OK")
            return True
        except Exception as e:
            log.warning(f"db: schema init failed -> {e}")
            return False


def log_prediction(
    *,
    model: str,
    pred_len: int,
    latency_ms: float,
    input_stats: dict[str, Any],
    output_stats: dict[str, Any],
    drift_pvalues: dict[str, float] | None,
    drift_flag: bool | None,
    alert: str | None = None,
) -> None:
    """Insert best-effort ; swallows erreurs (inference ne doit pas en patir)."""
    with _conn() as c:
        if c is None:
            return
        try:
            with c.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions
                        (model, pred_len, latency_ms,
                         input_stats, output_stats,
                         drift_pvalues, drift_flag, alert)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        model, pred_len, latency_ms,
                        json.dumps(input_stats),
                        json.dumps(output_stats),
                        json.dumps(drift_pvalues) if drift_pvalues else None,
                        drift_flag,
                        alert,
                    ),
                )
        except Exception as e:
            log.warning(f"db: insert failed -> {e}")


def recent_predictions(limit: int = 50) -> list[dict]:
    """Retourne les N dernieres predictions pour l'onglet Monitoring."""
    with _conn() as c:
        if c is None:
            return []
        try:
            with c.cursor() as cur:
                cur.execute(
                    """
                    SELECT ts, model, pred_len, latency_ms,
                           drift_flag, alert
                    FROM predictions
                    ORDER BY ts DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception as e:
            log.warning(f"db: select recent failed -> {e}")
            return []


def drift_rate_last_days(days: int = 7) -> dict:
    """Taux de drift sur les N derniers jours, par (model, pred_len).

    Utilise par la policy de retraining (pipelines/policy.py).
    Returns: {(model, pred_len): {"n": int, "drift_rate": float}}
    """
    out: dict = {}
    with _conn() as c:
        if c is None:
            return out
        try:
            with c.cursor() as cur:
                cur.execute(
                    """
                    SELECT model, pred_len,
                           COUNT(*) AS n,
                           SUM(CASE WHEN drift_flag THEN 1 ELSE 0 END)::float / COUNT(*) AS rate
                    FROM predictions
                    WHERE ts >= now() - (%s || ' days')::interval
                    GROUP BY model, pred_len
                    """,
                    (str(days),),
                )
                for model, pred_len, n, rate in cur.fetchall():
                    out[(model, int(pred_len))] = {"n": int(n), "drift_rate": float(rate or 0)}
        except Exception as e:
            log.warning(f"db: drift_rate_last_days failed -> {e}")
    return out


def log_retrain(
    *,
    model: str,
    pred_len: int,
    reason: str,
    mse_before: float | None,
    mse_after: float,
    mae_after: float,
    train_time_s: float,
    promoted: bool = False,
    hub_version: str | None = None,
    notes: str | None = None,
) -> int | None:
    """Insert dans retrain_runs, retourne l'id ou None si DB indispo."""
    with _conn() as c:
        if c is None:
            return None
        try:
            with c.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO retrain_runs
                        (model, pred_len, reason, mse_before, mse_after, mae_after,
                         train_time_s, promoted, hub_version, notes)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    RETURNING id
                    """,
                    (model, pred_len, reason, mse_before, mse_after, mae_after,
                     train_time_s, promoted, hub_version, notes),
                )
                row = cur.fetchone()
                return int(row[0]) if row else None
        except Exception as e:
            log.warning(f"db: log_retrain failed -> {e}")
            return None


def mark_promoted(run_id: int, hub_version: str) -> None:
    """Met a jour une row retrain_runs apres promotion effective."""
    with _conn() as c:
        if c is None:
            return
        try:
            with c.cursor() as cur:
                cur.execute(
                    "UPDATE retrain_runs SET promoted = true, hub_version = %s WHERE id = %s",
                    (hub_version, run_id),
                )
        except Exception as e:
            log.warning(f"db: mark_promoted failed -> {e}")


def last_production_mse(model: str, pred_len: int) -> float | None:
    """Dernier MSE marque comme 'prod' (promu). Sert de baseline pour la policy."""
    with _conn() as c:
        if c is None:
            return None
        try:
            with c.cursor() as cur:
                cur.execute(
                    """
                    SELECT mse_after FROM retrain_runs
                    WHERE model = %s AND pred_len = %s AND promoted = true
                    ORDER BY ts DESC LIMIT 1
                    """,
                    (model, pred_len),
                )
                row = cur.fetchone()
                return float(row[0]) if row else None
        except Exception as e:
            log.warning(f"db: last_production_mse failed -> {e}")
            return None


def summary_stats() -> dict:
    """Compteurs globaux pour la carte 'Monitoring'."""
    out = {"total": 0, "drifted": 0, "by_model": {}, "avg_latency_ms": None}
    with _conn() as c:
        if c is None:
            return out
        try:
            with c.cursor() as cur:
                cur.execute("SELECT COUNT(*), AVG(latency_ms), SUM(CASE WHEN drift_flag THEN 1 ELSE 0 END) FROM predictions")
                total, avg_lat, drifted = cur.fetchone()
                out["total"] = int(total or 0)
                out["drifted"] = int(drifted or 0)
                out["avg_latency_ms"] = float(avg_lat) if avg_lat is not None else None
                cur.execute("SELECT model, COUNT(*) FROM predictions GROUP BY model ORDER BY 2 DESC")
                out["by_model"] = {r[0]: int(r[1]) for r in cur.fetchall()}
        except Exception as e:
            log.warning(f"db: summary failed -> {e}")
    return out
