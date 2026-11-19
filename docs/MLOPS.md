# MLOps : les 5 phases + boucle complète

Ce projet implémente une chaîne MLOps **de bout en bout**, pas un simple notebook qui pousse un modèle en démo. Cinq phases ont été livrées (A → E), chacune avec un fichier dédié et des tests.

## Table des matières
1. [Phase A — Serving unifié](#phase-a--serving-unifié-fastapi--gradio)
2. [Phase B — Model registry](#phase-b--model-registry-hf-hub)
3. [Phase C — Monitoring + drift](#phase-c--monitoring--drift-detection)
4. [Phase D — Retraining pipeline](#phase-d--retraining--promotion)
5. [Phase E — Tests + CI/CD](#phase-e--tests--cicd)
6. [Boucle complète](#boucle-complète-mlops)
7. [Runbook](#runbook)

---

## Phase A — Serving unifié FastAPI + Gradio

**Objectif** : exposer les 24 checkpoints derrière une seule URL, avec **UI pour humains** et **API pour machines**.

**Livré** :
- [`huggingface_space/api.py`](../huggingface_space/api.py) — FastAPI avec 6 endpoints (`/health`, `/api/models`, `/api/predict`, `/api/monitoring`, `/metrics`, `/docs`)
- [`huggingface_space/ui.py`](../huggingface_space/ui.py) — 3 onglets Gradio : comparaison, utilisation opérationnelle, monitoring
- [`huggingface_space/predictor.py`](../huggingface_space/predictor.py) — catalogue des checkpoints + inférence
- [`huggingface_space/schemas.py`](../huggingface_space/schemas.py) — Pydantic (validation shape 96×7, horizons autorisés 96/192/336/720)

**Point remarquable** : `app = gr.mount_gradio_app(app, demo, path="/")` fusionne les deux surfaces sur le port 7860. Un seul conteneur, un seul process.

---

## Phase B — Model registry HF Hub

**Problème** : avant, l'image Docker embarquait les 1.5 GB de checkpoints. Chaque mise à jour de modèle imposait un rebuild complet de l'image et un re-déploiement du Space.

**Solution** : les checkpoints vivent sur **HuggingFace Hub** (repo modèle séparé), versionnés par tag git.

**Livré** :
- [`scripts/upload_models_to_hub.py`](../scripts/upload_models_to_hub.py) — push les 24 `.pt` sur `Momoniang/ltsf-transformers-etth1` + tag `v1.0.0`
- Patch de [`huggingface_space/predictor.py`](../huggingface_space/predictor.py) — fonction `_resolve_checkpoint()` qui tombe sur `hf_hub_download()` si le fichier local est absent
- [`huggingface_space/Dockerfile`](../huggingface_space/Dockerfile) — `COPY checkpoints` supprimé, variables `MODEL_REPO_ID` et `MODEL_VERSION` ajoutées
- Cache HF via `HF_HOME=/home/user/.cache/huggingface` → persistant pendant la vie du conteneur

**Bénéfices** :

| | Avant B | Après B |
|---|---|---|
| Taille image | 1.7 GB | ~200 MB |
| Temps rebuild sur nouveau modèle | ~15 min (CI + push) | 0 (juste repoint `MODEL_VERSION`) |
| Rollback | Rebuild + redeploy | Change var env + restart (1 min) |

---

## Phase C — Monitoring + drift detection

**Objectif** : observer ce qui se passe en production, détecter quand la distribution des inputs s'éloigne de la distribution d'entraînement.

**Livré** :

### DB Neon Postgres

Table [`predictions`](../huggingface_space/db.py) :
```sql
CREATE TABLE predictions (
    id             bigserial PRIMARY KEY,
    ts             timestamptz DEFAULT now(),
    model          text,
    pred_len       int,
    latency_ms     double precision,
    input_stats    jsonb,        -- {HUFL: {mean, std, min, max}, ...}
    output_stats   jsonb,
    drift_pvalues  jsonb,        -- {HUFL: p-value, ...}
    drift_flag     boolean,
    alert          text
);
```

### KS-test par feature

[`huggingface_space/drift.py`](../huggingface_space/drift.py) :
```python
def ks_drift(window):
    for feat in FEATURES:
        _, p = ks_2samp(window[:, i], ref_sample[:, i])
        pvals[feat] = p
    return pvals, any(p < 0.01 for p in pvals.values())
```

Reference sample : 1000 points tirés du split train (`[:TRAIN_END]`). Seuil α = 0.01. Chaque appel `/api/predict` calcule p-values + flag en < 5 ms.

### Métriques Prometheus

Exposées sur `/metrics` :
- `ltsf_predictions_total{model, pred_len}` — compteur
- `ltsf_drift_detected_total{model}` — compteur
- `ltsf_predict_latency_seconds{model}` — histogramme

Scrapables par un Prometheus/Grafana externe. Suffisant pour alerter dans un vrai setup prod.

### Dashboard Gradio onglet 3

Lit `summary_stats()` + `recent_predictions(50)` → affiche :
- KPIs (total, % drift, latence moyenne, répartition par modèle)
- Camembert nominal vs drift
- Tableau des 50 dernières prédictions

---

## Phase D — Retraining + promotion

**Objectif** : fermer la boucle MLOps — quand le drift dépasse un seuil, un pipeline relance un entraînement, évalue, et promeut (ou pas) le nouveau modèle.

### Politique ([pipelines/policy.py](../pipelines/policy.py))

```python
RETRAIN_DRIFT_RATE = 0.20       # 20% des pred en drift sur 7j
RETRAIN_MIN_SAMPLES = 50        # minimum d'historique
PROMOTE_MIN_IMPROVEMENT = 0.02  # +2% MSE pour promouvoir
```

Deux décisions séparées :
- **`decide_retrain(model, h, drift_stats)`** → `RetrainDecision(should_retrain, reason)`
- **`decide_promote(mse_before, mse_after)`** → `PromoteDecision(should_promote, improvement)`

### Retraining ([pipelines/retrain.py](../pipelines/retrain.py))

```bash
# Ad-hoc
python -m pipelines.retrain --model PatchTST --pred_len 96 --reason manual

# Automatique selon la DB
python -m pipelines.retrain --check-drift

# Preview (dry-run)
python -m pipelines.retrain --check-drift --dry-run
```

Process :
1. Lit `drift_rate_last_days(7)` depuis Neon
2. Applique `decide_retrain()` pour chaque (model, pred_len)
3. Pour chaque cible retenue : appelle `src.train.train_and_eval()`
4. Logue dans `retrain_runs` avec `mse_before` (dernière prod) et `mse_after`

### Promotion ([pipelines/promote.py](../pipelines/promote.py))

```bash
# Promeut le dernier retrain non-promu
python -m pipelines.promote --model PatchTST --pred_len 96

# Force malgré PROMOTE_MIN_IMPROVEMENT
python -m pipelines.promote --model PatchTST --pred_len 96 --force
```

Process :
1. Récupère la dernière row `retrain_runs` non promue
2. `decide_promote()` → OK ou skip
3. Copie `results/checkpoints/<tag>.pt` → `checkpoints/<tag>.pt`
4. Upload sur HF Hub avec un **nouveau tag** (auto-increment patch : v1.0.0 → v1.0.1)
5. `mark_promoted(run_id, new_version)` en DB

---

## Phase E — Tests + CI/CD

### Tests (23 au total, ~37 s)

```
tests/
├── conftest.py            # fixtures : no_real_db, sample_window
├── test_predictor.py      # 5 tests : catalogue, resolver, shapes
├── test_drift.py          # 3 tests : stats, p-values, drift détecté
├── test_api.py            # 6 tests : tous les endpoints
└── test_policy.py         # 9 tests : décisions retrain/promote
```

Markers :
- `@pytest.mark.slow` — charge un vrai checkpoint (skip en CI)
- `@pytest.mark.integration` — nécessite DB (skip en CI)

CI command : `pytest tests/ -q -m "not slow and not integration"` → 22 tests passants en ~15 s.

### CI (`.github/workflows/ci.yml`)

3 jobs :
1. **lint** — `ruff check .`
2. **test** — pytest (dépend de lint)
3. **docker-build** — Docker build dry-run du Dockerfile HF Space (dépend de lint)

Trigger : push main, PR main, workflow_dispatch.

### CD (`.github/workflows/deploy-hf.yml`)

Push automatique de `huggingface_space/` → HF Space sur merge main.

Secrets requis côté GitHub :
- `HF_TOKEN` — token write HuggingFace

Variables optionnelles :
- `HF_SPACE_ID` — defaut `Momoniang/projet7-ltsf-transformers`

---

## Boucle complète MLOps

```
1. ENTRAÎNEMENT           src/train.py        → results/checkpoints/X.pt
2. UPLOAD REGISTRY        scripts/upload…     → HF Hub v1.0.0
3. BUILD + DEPLOY         git push main       → GH Actions → HF Space (v1.0.0)
4. INFÉRENCE EN PROD      POST /api/predict   → Neon predictions
                                              → drift détecté
5. ALERTE                 drift_rate > 20%    → trigger retrain
6. RETRAINING             pipelines/retrain   → retrain_runs row
7. ÉVALUATION             MSE_new < MSE_prod  → decide_promote = True
8. PROMOTION              pipelines/promote   → push HF Hub v1.0.1 +
                                                mark DB promoted=true
9. DEPLOY                 bump MODEL_VERSION  → restart Space
                          (ou nouveau push)
                          → retour à l'étape 4
```

Cette boucle peut tourner **automatiquement** via un cron (GitHub Actions scheduled ou cron sur un worker) qui lance `retrain.py --check-drift && promote.py …` toutes les N heures.

---

## Runbook

### Déployer une nouvelle version de modèle

```bash
# 1. Retrain (local ou worker)
python -m pipelines.retrain --model PatchTST --pred_len 96 --reason manual

# 2. Vérifier le résultat dans Neon
# SELECT * FROM retrain_runs ORDER BY ts DESC LIMIT 1;

# 3. Promouvoir
$env:HF_TOKEN = "hf_xxx"
python -m pipelines.promote --model PatchTST --pred_len 96

# 4. Repoint le Space sur la nouvelle version
# HF Space → Settings → Variables → MODEL_VERSION = v1.0.1
# Restart Space (bouton).
```

### Rollback

```
HF Space → Settings → Variables → MODEL_VERSION = v1.0.0 → Restart.
```

Temps : ~1 min. Les checkpoints v1.0.0 sont téléchargés via `hf_hub_download` au 1er appel après restart.

### Désactiver le monitoring temporairement

```
HF Space → Settings → Secrets → retirer DATA_POSTGRESQL → Restart.
```

L'API reste fonctionnelle. L'onglet Monitoring affichera "aucune prédiction enregistrée".

### Investigation : "les prédictions sont bizarres aujourd'hui"

```sql
-- Dans Neon SQL Editor :
SELECT ts, model, drift_flag, drift_pvalues
FROM predictions
WHERE ts > now() - interval '2 hours'
ORDER BY ts DESC;

-- Taux de drift par feature sur 24h
SELECT
  key AS feature,
  COUNT(*) FILTER (WHERE (value::float) < 0.01) * 1.0 / COUNT(*) AS drift_rate
FROM predictions, jsonb_each(drift_pvalues)
WHERE ts > now() - interval '24 hours'
GROUP BY key ORDER BY drift_rate DESC;
```

### Tests cassent en CI mais passent en local

Vérifier :
1. Tu as bien `DATABASE_URL=""` et `DATA_POSTGRESQL=""` dans l'env CI (le workflow les set déjà)
2. Les tests marqués `@slow` ne doivent PAS être lancés en CI (filtre `-m "not slow"`)
3. Le dossier `checkpoints/` n'est pas committé — c'est normal, les tests non-slow n'en ont pas besoin
