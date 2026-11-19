# Architecture

Vue d'ensemble du système de bout en bout, des choix de conception et de la façon dont les composants interagissent.

## 1. Vue d'ensemble

Le projet se décompose en **4 sous-systèmes**, cloisonnés mais connectés :

```
┌─────────────────────────────────────────────────────────────────────┐
│                         1. ENTRAÎNEMENT                             │
│   src/train.py --model X --pred_len H   →   results/checkpoints/    │
│   scripts/run_grid.ps1  (24 combinaisons)                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │  upload_models_to_hub.py
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     2. MODEL REGISTRY (HF Hub)                      │
│   Momoniang/ltsf-transformers-etth1  @ v1.0.0                       │
│   24 fichiers .pt, versionnés par tag git                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │  hf_hub_download (lazy, au boot)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      3. SERVING (HF Space)                          │
│   FastAPI (api.py)                                                  │
│    ├─ GET  /health, /api/models, /docs, /metrics                    │
│    ├─ POST /api/predict        ← logs + drift + metrics             │
│    ├─ GET  /api/monitoring                                          │
│    └─ Gradio UI monté sur /    (3 onglets)                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │  prédictions loggées
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     4. OBSERVABILITÉ + MLOps                        │
│   Neon Postgres                                                     │
│    ├─ predictions   (1 row par /api/predict)                        │
│    └─ retrain_runs  (1 row par cycle retrain)                       │
│                                                                     │
│   pipelines/retrain.py  ← drift >20% sur 7j OU --reason manual      │
│   pipelines/promote.py  → recopie + push HF Hub (v1.0.1, v1.1.0…)   │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. Choix de conception structurants

### 2.1 Séparation stricte `src/` ↔ `huggingface_space/`

- `src/` : code d'entraînement et de recherche. Peut dépendre de `torch`, `tensorboard`, `shap`, tout ce qui est lourd.
- `huggingface_space/` : code **de serving** uniquement. Même les implémentations de modèles sont dupliquées (via `huggingface_space/src/models/`) pour que l'image Docker n'embarque pas `tensorboard` ni `shap`.

Rationale : une image Docker de serving ne doit pas traîner les dépendances de recherche. Cela coupe aussi l'image en deux surfaces d'attaque distinctes.

### 2.2 Les checkpoints ne vivent PAS dans l'image Docker

Avant Phase B : `COPY checkpoints ./checkpoints` → image ~1.7 GB.
Après Phase B : `hf_hub_download` au premier appel d'inférence → image ~200 MB.

Avantages :
- **Décorrelation code/modèle** : on peut déployer un nouveau modèle sans rebuilder l'image (pas de re-CI).
- **Versioning explicite** via `MODEL_VERSION=v1.0.0` → rollback trivial.
- **Cache persistant** via `HF_HOME=/home/user/.cache/huggingface` → les checkpoints restent en RAM-disk pendant la vie du conteneur.

Fallback : si `./checkpoints/` est présent localement (dev), le resolver préfère cette source — pas besoin de réseau.

### 2.3 FastAPI monte Gradio, pas l'inverse

```python
# api.py (dernière ligne)
app = gr.mount_gradio_app(app, demo, path="/")
```

Un seul port (7860), une seule image, un seul process. Pas de reverse proxy à configurer, pas de double-serveur. HF Space lance uvicorn, qui sert l'API REST sur `/api/*` et l'UI sur `/`.

### 2.4 DB en best-effort

```python
# db.py
def log_prediction(...) -> None:
    with _conn() as c:
        if c is None: return      # no-op si DB indispo
        try:
            cur.execute(INSERT ...)
        except Exception as e:
            log.warning(...)      # jamais une HTTPException
```

L'inférence ne doit **jamais** échouer parce que la DB est down. Chaque opération DB est entourée de try/except et dégrade silencieusement. Les tests `test_api.py` tournent sans DB et valident ce chemin.

### 2.5 Drift calculé au sens KS, pas par modèle supervisé

```python
# drift.py
from scipy.stats import ks_2samp
pvalue = ks_2samp(window[:, i], reference_train[:, i]).pvalue
```

Pas besoin d'un "modèle de drift" séparé : Kolmogorov-Smirnov sur la distribution des features vs train suffit pour signaler qu'on sort de la distribution d'entraînement. Simple, interprétable, et calculable en < 1 ms par feature.

## 3. Flux de données complet pour un `POST /api/predict`

```
 Client POST (model=PatchTST, pred_len=96, window=[96,7])
    │
    ├─▶ Pydantic valide shape (96, 7)
    │
    ├─▶ _load_model(PatchTST, 96)   [cache lru, 32 slots]
    │     │
    │     ├─ CKPT_DIR / PatchTST_ETTh1_h96.pt présent ? → yes → load
    │     └─                                            → no  → hf_hub_download
    │
    ├─▶ scaler.transform(window)    [fit sur train seulement]
    │
    ├─▶ model(window) → yhat_norm (96, 7)
    │
    ├─▶ scaler.inverse_transform(yhat_norm) → prediction (96, 7)
    │
    ├─▶ ks_drift(window)            → {HUFL: p=0.3, …}, drift_flag
    │
    ├─▶ PRED_COUNTER.inc(), LATENCY_HIST.observe()
    │
    ├─▶ log_prediction(...)         [best-effort, async-safe]
    │
    └─▶ return PredictResponse(prediction, drift_flag, pvalues, latency_ms)
```

Latences typiques (CPU, 4 vCPU) :
- DLinear : ~3 ms
- PatchTST : ~160 ms
- Informer : ~200 ms
- Autoformer / FEDformer / Transformer : ~400-800 ms

## 4. Versioning des modèles

Le tag `MODEL_VERSION` est la source de vérité unique pour "quel modèle tourne en prod" :

```
MODEL_REPO_ID  = Momoniang/ltsf-transformers-etth1
MODEL_VERSION  = v1.0.0                          # modifiable via HF Space Settings
```

- **v1.0.0** : release initiale, 24 checkpoints
- **v1.0.1** : patch — un modèle réentraîné via `pipelines/promote.py`
- **v1.1.0** : minor — nouvelle architecture ou tous les modèles réentraînés
- **v2.0.0** : major — breaking (changement input_len, nouveau dataset, etc.)

Rollback : `MODEL_VERSION=v1.0.0` + restart du Space → 1 min.

## 5. Dépendances externes et dégradations

| Service | Rôle | Si down |
|---|---|---|
| **HF Hub** | Pull checkpoints au boot | Fallback local si CKPT_DIR présent, sinon 503 au 1er appel |
| **Neon Postgres** | Logging + monitoring | Inférence OK, onglet 3 affiche "aucune prédiction" |
| **HF Space** | Hébergement démo | Rien côté code ; seul le site de démo tombe |

Aucun SPOF applicatif : tant que le code tourne et que les checkpoints sont joignables (HF Hub ou cache local), l'inférence est servie. Les composants périphériques (DB, HF Space) dégradent le service sans le casser.

## 6. Ce qui N'EST PAS dans le périmètre

- **Authentification** : l'API est publique. Pas d'OAuth, pas d'API-key. Si besoin un jour → middleware FastAPI standard.
- **Rate limiting** : HF Space apporte un quota d'infra. Un middleware applicatif (slowapi, redis-leaky-bucket) peut être ajouté sur l'app FastAPI si besoin.
- **Multi-tenant** : un seul utilisateur logique (tout le monde partage la même DB `predictions`).
- **GPU** : image CPU-only. Les modèles sont assez petits pour que ça suffise (< 1 s par prédiction au pire).
