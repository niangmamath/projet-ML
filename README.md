# LTSF Transformers — Benchmark + serving MLOps

Système de prévision long-horizon de séries temporelles sur **ETTh1**, avec 6 architectures (5 Transformers + 1 baseline linéaire) comparées rigoureusement et déployées via une chaîne **MLOps complète** : API REST, monitoring, détection de drift, pipeline de retraining, CI/CD.

[![CI](https://github.com/niangmamath/projet-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/niangmamath/projet-ML/actions/workflows/ci.yml)
[![HF Space](https://img.shields.io/badge/HF%20Space-live-yellow)](https://huggingface.co/spaces/Momoniang/projet7-ltsf-transformers)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](#licence)

---

## En une phrase

Un benchmark reproductible des 6 modèles LTSF (DLinear, Transformer, Informer, Autoformer, FEDformer, PatchTST) sur ETTh1 × {96, 192, 336, 720} — **24 checkpoints** — exposé via une **démo publique** (Gradio + FastAPI) avec **registry de modèles HF Hub**, **logging Postgres**, **détection de drift KS-test**, **pipeline de retraining/promotion** et **CI/CD GitHub Actions**.

## Résultats clés

| Modèle | Params | MSE @h=96 | MSE @h=192 | MSE @h=336 | MSE @h=720 |
|---|---:|---:|---:|---:|---:|
| **DLinear**    | ~0.5 M  | voir `results/metrics/comparison.csv` ||||
| **PatchTST**   | ~2.2 M  ||||  |
| Transformer    | ~33 M   ||||  |
| Informer       | ~13 M   ||||  |
| Autoformer     | ~33 M   ||||  |
| FEDformer      | ~33 M   ||||  |

> Détails chiffrés + figures dans [`results/`](results/) et [`docs/MODELS.md`](docs/MODELS.md).

## Architecture du système (MLOps complet)

```
┌────────────┐   POST /api/predict   ┌─────────────┐   logs    ┌──────────────┐
│  Client    │ ────────────────────▶ │  FastAPI    │ ────────▶ │ Neon Postgres│
│ (UI, REST) │                       │  + Gradio   │           │ predictions  │
└────────────┘ ◀──── drift + pred ── │  + metrics  │           │ retrain_runs │
                                     └─────┬───────┘           └──────┬───────┘
                                           │ lazy pull                │
                                           ▼                          │
                                     ┌─────────────┐                  │
                                     │   HF Hub    │  ◀──── push ─────┤
                                     │ (checkpoints│       (promote)  │
                                     │   v1.X.Y)   │                  │
                                     └─────────────┘                  │
                                                                      │
                                     ┌─────────────┐                  │
                                     │ pipelines/  │ ◀── drift >20% ──┘
                                     │  retrain +  │
                                     │  promote    │
                                     └─────────────┘
```

Détail : [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/MLOPS.md`](docs/MLOPS.md)

## Démarrage rapide

```bash
# Cloner + environnement
git clone https://github.com/niangmamath/projet-ML.git
cd projet-ML
python -m venv venv
venv\Scripts\activate                 # Windows
pip install -r huggingface_space/requirements.txt

# Récupérer les données ETTh1
python src/download_data.py

# Entraîner un modèle
python src/train.py --model PatchTST --data ETTh1 --pred_len 96

# Lancer la démo (FastAPI + Gradio sur http://127.0.0.1:7860)
cd huggingface_space
python app.py
```

Démo live : **https://huggingface.co/spaces/Momoniang/projet7-ltsf-transformers**

## Structure du dépôt

```
projet-ML/
├── src/                        # Code source entraînement + modèles
│   ├── models/                 #   6 architectures (DLinear, Transformer, …)
│   ├── data_loader.py          #   splits 12/4/4 mois, StandardScaler train-only
│   ├── train.py                #   boucle d'entraînement unifiée
│   ├── evaluate.py             #   MSE, MAE
│   └── interpret.py            #   erreur/horizon, feat. importance, attention
├── huggingface_space/          # App déployable sur HF Spaces (Docker)
│   ├── api.py                  #   FastAPI + Gradio mount
│   ├── ui.py                   #   3 onglets Gradio
│   ├── predictor.py            #   catalogue + resolver (local / HF Hub)
│   ├── schemas.py              #   Pydantic (PredictRequest/Response, …)
│   ├── db.py                   #   Neon Postgres (logging)
│   ├── drift.py                #   KS-test par feature
│   └── Dockerfile              #   image CPU, checkpoints pulled lazily
├── pipelines/                  # MLOps Phase D
│   ├── policy.py               #   règles retrain + promote
│   ├── retrain.py              #   orchestrateur entraînement
│   └── promote.py              #   copy prod + push HF Hub + mark DB
├── scripts/
│   ├── run_grid.ps1            #   grille 6×4 = 24 entraînements
│   ├── run_interpret.ps1       #   figures interprétation
│   └── upload_models_to_hub.py #   push checkpoints vers HF Hub
├── tests/                      # pytest (23 tests, ~40s)
├── .github/workflows/          # CI (lint + tests + docker) + CD (HF Space)
├── docs/                       # Documentation technique détaillée
├── results/
│   ├── checkpoints/            #   .pt staging (sortie train)
│   ├── metrics/                #   comparison.csv, feature_importance.csv
│   └── figures/                #   PNG attention + error-per-horizon
└── checkpoints/                # .pt production (ce qui alimente la démo)
```

## Modèles comparés

| Modèle | Année | Idée clé | Référence |
|---|---|---|---|
| **DLinear** | 2023 | Baseline linéaire décomposée (trend + residual) | [Zeng et al.](https://arxiv.org/abs/2205.13504) |
| **Transformer** | 2017 | Attention multi-tête (encoder-only pour LTSF) | [Vaswani et al.](https://arxiv.org/abs/1706.03762) |
| **Informer** | 2021 | ProbSparse attention + self-attention distilling | [Zhou et al.](https://arxiv.org/abs/2012.07436) |
| **Autoformer** | 2021 | Décomposition série + Auto-Correlation | [Wu et al.](https://arxiv.org/abs/2106.13008) |
| **FEDformer** | 2022 | Attention dans le domaine fréquentiel | [Zhou et al.](https://arxiv.org/abs/2201.12740) |
| **PatchTST** | 2023 | Patching + indépendance canaux | [Nie et al.](https://arxiv.org/abs/2211.14730) |

Implémentations from-scratch inspirées de [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library), détails dans [`docs/MODELS.md`](docs/MODELS.md).

## Documentation

| Doc | Contenu |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Vue d'ensemble système, composants, flux de données |
| [`docs/MLOPS.md`](docs/MLOPS.md) | Les 5 phases MLOps (A→E) + boucle complète + runbook |
| [`docs/API.md`](docs/API.md) | Spécification REST (endpoints, exemples curl, erreurs) |
| [`docs/MODELS.md`](docs/MODELS.md) | Détails d'implémentation des 6 architectures, hyperparams |
| [`docs/DEPLOY.md`](docs/DEPLOY.md) | Procédure déploiement local + HF Space + Neon |

## CI/CD

- **CI** : ruff (lint) + pytest (23 tests, 37 s) + Docker build dry-run sur chaque PR/push main
- **CD** : push automatique de `huggingface_space/` vers HF Space sur merge main (via `HF_TOKEN` en secret)
- **Registry** : checkpoints versionnés sur HF Hub (`Momoniang/ltsf-transformers-etth1@v1.X.Y`)

## Auteur

**Mamath Yacine Niang**

## Licence

MIT. Les implémentations s'inspirent du dépôt [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) (Apache 2.0) ; chaque modèle est réimplémenté from scratch.
