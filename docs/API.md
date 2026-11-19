# Spécification API REST

Base URL locale : `http://127.0.0.1:7860`
Base URL production : `https://Momoniang-projet7-ltsf-transformers.hf.space`

Documentation OpenAPI interactive : [`/docs`](http://127.0.0.1:7860/docs) (Swagger UI auto-généré par FastAPI).

---

## Endpoints

### `GET /health`

Statut du service.

**Réponse 200** :
```json
{
  "status": "ok",
  "version": "0.2.0",
  "loaded_models": 24
}
```

**Usage** :
```bash
curl http://127.0.0.1:7860/health
```

---

### `GET /api/models`

Catalogue des modèles disponibles.

**Réponse 200** :
```json
{
  "models": [
    {"name": "Autoformer",  "pred_lens": [96, 192, 336, 720]},
    {"name": "DLinear",     "pred_lens": [96, 192, 336, 720]},
    {"name": "FEDformer",   "pred_lens": [96, 192, 336, 720]},
    {"name": "Informer",    "pred_lens": [96, 192, 336, 720]},
    {"name": "PatchTST",    "pred_lens": [96, 192, 336, 720]},
    {"name": "Transformer", "pred_lens": [96, 192, 336, 720]}
  ],
  "default_model": "PatchTST",
  "default_pred_len": 96
}
```

**Usage** :
```bash
curl http://127.0.0.1:7860/api/models
```

---

### `POST /api/predict`

Prédit à partir d'une fenêtre d'entrée fournie.

**Corps requête** :
```json
{
  "model": "PatchTST",
  "pred_len": 96,
  "window": [[hufl, hull, mufl, mull, lufl, lull, ot], ... 96 lignes ...]
}
```

**Paramètres** :
| Champ | Type | Valeurs | Défaut |
|---|---|---|---|
| `model` | string | DLinear, Transformer, PatchTST, Informer, Autoformer, FEDformer | PatchTST |
| `pred_len` | int | 96, 192, 336, 720 | 96 |
| `window` | float[96][7] | matrice 96 lignes × 7 colonnes (ordre : HUFL, HULL, MUFL, MULL, LUFL, LULL, OT) | requis |

**Réponse 200** :
```json
{
  "model": "PatchTST",
  "pred_len": 96,
  "features": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
  "prediction": [[p0_hufl, …, p0_ot], … 96 lignes …],
  "latency_ms": 161.0,
  "drift_flag": false,
  "drift_pvalues": {
    "HUFL": 0.42, "HULL": 0.31, "MUFL": 0.18,
    "MULL": 0.09, "LUFL": 0.27, "LULL": 0.33, "OT": 0.11
  },
  "alert": null
}
```

**Erreurs** :
- `422` : shape invalide (window pas en (96, 7))
- `422` : model/pred_len hors enum
- `404` : checkpoint introuvable (ne devrait pas arriver en prod normale)

**Exemple complet** (Python) :
```python
import json, urllib.request
import pandas as pd

df = pd.read_csv("huggingface_space/data/ETTh1.csv")
window = df[["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]].iloc[-96:].values.tolist()

body = json.dumps({"model":"PatchTST","pred_len":96,"window":window}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:7860/api/predict",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=30) as r:
    resp = json.loads(r.read())
print(resp["latency_ms"], "ms, drift:", resp["drift_flag"])
```

**Exemple curl** (fenêtre bidon) :
```bash
python -c "
import json
window = [[10.0, 2.0, 8.0, 1.5, 3.0, 0.5, 20.0]] * 96
print(json.dumps({'model': 'DLinear', 'pred_len': 96, 'window': window}))
" > /tmp/body.json

curl -X POST http://127.0.0.1:7860/api/predict \
  -H "Content-Type: application/json" \
  -d @/tmp/body.json
```

---

### `GET /api/monitoring?limit=50` 🔒 admin

Dashboard data : agrégats + N dernières prédictions.

**Authentification** : header `X-Admin-Token` requis, doit matcher la variable d'env `ADMIN_TOKEN` côté serveur.

- `401 Unauthorized` si header absent ou valeur incorrecte
- `503 Service Unavailable` si `ADMIN_TOKEN` n'est pas configuré côté serveur (fail-closed)

**Paramètres query** :
| Champ | Type | Défaut |
|---|---|---|
| `limit` | int | 50 |

**Exemple** :
```bash
curl -H "X-Admin-Token: $ADMIN_TOKEN" "$SPACE/api/monitoring?limit=20"
```

**Réponse 200** :
```json
{
  "total": 1247,
  "drifted": 183,
  "avg_latency_ms": 178.4,
  "by_model": {"PatchTST": 823, "DLinear": 312, "Informer": 112},
  "recent": [
    {
      "ts": "2026-11-19 17:08:23+00:00",
      "model": "PatchTST",
      "pred_len": 96,
      "latency_ms": 158.2,
      "drift_flag": true,
      "alert": null
    },
    ...
  ]
}
```

Si la DB est indisponible : `{"total": 0, "drifted": 0, "avg_latency_ms": null, "by_model": {}, "recent": []}` — réponse valide, pas d'erreur 500.

---

### `GET /metrics`

Métriques Prometheus, format texte.

**Réponse 200** (extrait) :
```
# HELP ltsf_predictions_total Total predictions served
# TYPE ltsf_predictions_total counter
ltsf_predictions_total{model="PatchTST",pred_len="96"} 823.0

# HELP ltsf_drift_detected_total Predictions flagged as drifted
# TYPE ltsf_drift_detected_total counter
ltsf_drift_detected_total{model="PatchTST"} 142.0

# HELP ltsf_predict_latency_seconds Inference latency (s)
# TYPE ltsf_predict_latency_seconds histogram
ltsf_predict_latency_seconds_bucket{le="0.05",model="PatchTST"} 0
ltsf_predict_latency_seconds_bucket{le="0.1",model="PatchTST"} 15
ltsf_predict_latency_seconds_bucket{le="0.25",model="PatchTST"} 801
...

# HELP ltsf_build_info Build info gauge
# TYPE ltsf_build_info gauge
ltsf_build_info{version="0.2.0"} 1
```

Scrapable par un Prometheus distant :
```yaml
# prometheus.yml
scrape_configs:
  - job_name: ltsf
    scrape_interval: 15s
    static_configs:
      - targets: ["Momoniang-projet7-ltsf-transformers.hf.space:443"]
    scheme: https
```

---

### `GET /docs`

Swagger UI généré par FastAPI à partir des schémas Pydantic. Interactif, permet de tester les endpoints depuis le navigateur.

### `GET /openapi.json`

Spec OpenAPI 3.1 brute, consommable par n'importe quel client (Postman, Insomnia, generateurs SDK).

---

## Codes d'erreur standards

| Code | Cause | Payload |
|---|---|---|
| **200** | OK | voir schémas ci-dessus |
| **404** | Modèle ou checkpoint introuvable | `{"detail": "Checkpoint absent : X_ETTh1_h96.pt"}` |
| **422** | Validation Pydantic (shape, enum) | `{"detail": [{"loc": [...], "msg": "..."}]}` |
| **500** | Erreur interne (bug) | `{"detail": "Internal Server Error"}` |

---

## Limites et considérations

- **Pas d'authentification** : API publique dans cette version. Un déploiement en environnement contrôlé devra ajouter un middleware d'auth (JWT, API-key, OAuth) — voir [docs/MLOPS.md](MLOPS.md#runbook) pour les crochets à ajouter.
- **Pas de rate limit applicatif** : HF Space impose un quota de base, mais rien de granulaire.
- **Pas de batch** : un `/api/predict` = une fenêtre. Pour du batch, enchaîner côté client.
- **CPU-only** : les 24 checkpoints tournent sur CPU. Latence worst-case ~800 ms pour les plus gros modèles.
- **Timeout recommandé côté client** : 30 s (pour couvrir le cold-start initial + pull HF Hub).
