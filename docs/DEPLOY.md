# Déploiement

Guide complet pour reproduire le projet en local, pousser les modèles sur HuggingFace Hub, et déployer la démo sur HuggingFace Spaces.

## Table des matières
1. [Déploiement local (dev)](#1-déploiement-local-dev)
2. [Setup HuggingFace Hub (registry)](#2-setup-huggingface-hub-registry)
3. [Setup Neon Postgres (monitoring)](#3-setup-neon-postgres-monitoring)
4. [Déploiement HuggingFace Space](#4-déploiement-huggingface-space)
5. [CI/CD GitHub Actions](#5-cicd-github-actions)
6. [Mise à jour d'un modèle en prod](#6-mise-à-jour-dun-modèle-en-prod)
7. [Rollback](#7-rollback)

---

## 1. Déploiement local (dev)

### Prérequis
- Python **3.11** (testé)
- 4 Go de RAM libre, 2 Go de disque
- Git, Git LFS *(uniquement si tu pousses sur HF Space en direct — cf. §4)*

### Installation

```powershell
git clone https://github.com/niangmamath/projet-ML.git
cd projet-ML

python -m venv venv
venv\Scripts\activate

# Deps de base (serving + inférence)
pip install -r huggingface_space/requirements.txt

# Deps dev (tests + lint)
pip install pytest ruff
```

### Entraînement (optionnel — les checkpoints sont déjà sur HF Hub)

```powershell
# Télécharger le dataset ETTh1 (1 MB)
python src/download_data.py

# Un modèle, un horizon
python src/train.py --model PatchTST --data ETTh1 --pred_len 96

# Toute la grille (6 modèles × 4 horizons = 24 runs, ~6-10 h CPU)
./scripts/run_grid.ps1
```

Sortie dans `results/checkpoints/<Model>_ETTh1_h<H>.pt` + stats dans `results/metrics/comparison.csv`.

### Lancer la démo

```powershell
cd huggingface_space
python app.py
# Ouvrir http://127.0.0.1:7860
```

Comportement :
- Si `checkpoints/` est présent à la racine du projet → utilise ces fichiers locaux.
- Sinon → télécharge depuis HF Hub (`MODEL_REPO_ID@MODEL_VERSION`) et les met en cache dans `~/.cache/huggingface/hub/`.

### Tests

```powershell
# Tous les tests (incluant ceux marqués @slow qui chargent de vrais checkpoints)
pytest tests/

# Seulement les tests CI (rapides, pas de .pt réel)
pytest tests/ -m "not slow and not integration"

# Lint
ruff check .
```

---

## 2. Setup HuggingFace Hub (registry)

### Créer le repo modèle

```powershell
# Récupérer ton token write : https://huggingface.co/settings/tokens
$env:HF_TOKEN = "hf_xxxxxxxx"

# Push les 24 checkpoints + tag v1.0.0
python scripts/upload_models_to_hub.py `
    --repo-id Momoniang/ltsf-transformers-etth1 `
    --version v1.0.0 `
    --public
```

Remplacer `Momoniang` par ton username HF. Le script :
1. Crée le repo si absent (idempotent)
2. Upload les 24 `.pt` (1.5 GB, 5-30 min selon bande passante)
3. Pose le tag git `v1.0.0`

### Pour accélérer l'upload (lent par défaut)

```powershell
pip install hf_transfer
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
python scripts/upload_models_to_hub.py --repo-id ... --version v1.0.0 --public
```

L'uploader Rust est 5-10× plus rapide. Les blobs déjà uploadés sont reconnus par hash → les relances sont rapides.

### Vérifier

Browser → `https://huggingface.co/Momoniang/ltsf-transformers-etth1/tree/v1.0.0` → tu dois voir les 24 fichiers.

---

## 3. Setup Neon Postgres (monitoring)

Optionnel mais recommandé — sans DB, l'onglet 3 (Monitoring) restera vide.

### Créer la base

1. Compte gratuit : https://console.neon.tech/signup
2. Create project → nom `projet7` → region proche
3. Copier la **Pooled connection string** (format `postgresql://user:pass@ep-xxx-pooler...`)

### Configurer en local

```bash
# .env à la racine du projet (créé par toi, gitignored)
DATA_POSTGRESQL=postgresql://user:pass@ep-xxx-pooler.c-9.us-east-2.aws.neon.tech/projet7?sslmode=require&channel_binding=require
```

Le schema (`predictions`, `retrain_runs`) sera créé automatiquement au premier démarrage de l'app.

### Configurer sur HF Space

Settings → **Secrets** → add :
- Name : `DATA_POSTGRESQL`
- Value : la connection string

Ne JAMAIS mettre dans le code, le Dockerfile, ou un fichier commit.

---

## 4. Déploiement HuggingFace Space

### 4a. Première création (une seule fois)

1. https://huggingface.co/new-space
2. Owner : ton compte, Name : ex. `projet7-ltsf-transformers`
3. License : MIT
4. **SDK : Docker** (important, pas "Gradio")
5. Visibilité : Public
6. Hardware : CPU basic (gratuit)
7. Create Space

### 4b. Configurer les secrets et variables

Settings → Variables and secrets :

**Secrets** (private, chiffrés) :
- `DATA_POSTGRESQL` = ta connection string Neon
- `HF_TOKEN` *(optionnel, si tu veux que l'app push des artefacts)*

**Variables** (publiques, visibles) :
- `MODEL_REPO_ID` = `Momoniang/ltsf-transformers-etth1`
- `MODEL_VERSION` = `v1.0.0`

### 4c. Push manuel (premier déploiement)

Après Phase B, le dossier `huggingface_space/` ne contient plus les 1.5 GB de checkpoints. Pas besoin de Git LFS.

```powershell
cd huggingface_space

git init -b main
git add .
git commit -m "Initial deploy"

git remote add origin https://huggingface.co/spaces/Momoniang/projet7-ltsf-transformers
git push -u origin main
```

Build HF (~5-10 min) : Docker image + download des 24 checkpoints au boot.

### 4d. Déploiement automatique (recommandé)

Voir [§5 CI/CD](#5-cicd-github-actions).

---

## 5. CI/CD GitHub Actions

### Setup unique

1. Créer le repo GitHub : `github.com/niangmamath/projet-ML`
2. Push le code :
   ```powershell
   cd projet-ML    # racine du projet
   git remote add origin https://github.com/niangmamath/projet-ML.git
   git branch -M main
   git push -u origin main
   ```
3. GitHub → Settings → Secrets and variables → Actions :
   - **Secret** `HF_TOKEN` = ton token write HF
   - **Variable** `HF_SPACE_ID` = `Momoniang/projet7-ltsf-transformers` *(optionnel, défaut dans le workflow)*

### Ce qui tourne

**`.github/workflows/ci.yml`** (sur chaque PR + push main) :
- `lint` : ruff
- `test` : pytest (23 tests, ~15 s en CI sans les slow)
- `docker-build` : build dry-run du Dockerfile HF Space

**`.github/workflows/deploy-hf.yml`** (sur merge main avec changements dans `huggingface_space/`) :
- Push `huggingface_space/` → HF Space
- Rebuild automatique du Space (~5-10 min)

### Voir les runs

- https://github.com/niangmamath/projet-ML/actions
- Le badge CI dans le README se met à jour automatiquement

---

## 6. Mise à jour d'un modèle en prod

Workflow complet quand tu veux promouvoir un nouveau modèle :

```powershell
# 1. Retrain
python -m pipelines.retrain --model PatchTST --pred_len 96 --reason manual

# 2. Vérifier le résultat (lire retrain_runs dans Neon)
#    MSE_after doit être < MSE_before pour que decide_promote passe

# 3. Promouvoir (uploade sur HF Hub avec nouveau tag v1.0.1)
$env:HF_TOKEN = "hf_xxx"
python -m pipelines.promote --model PatchTST --pred_len 96

# 4. Repoint le Space sur la nouvelle version
#    HF Space → Settings → Variables → MODEL_VERSION = v1.0.1
#    Restart Space (bouton)

# 5. Vérifier
curl https://Momoniang-projet7-ltsf-transformers.hf.space/health
# {"status":"ok","version":"0.2.0","loaded_models":24}
```

---

## 7. Rollback

Si un nouveau modèle fait moins bien que prévu :

```
HF Space → Settings → Variables → MODEL_VERSION = v1.0.0 → Restart
```

Temps : **~1 minute**. Les checkpoints v1.0.0 sont téléchargés via `hf_hub_download` au 1er appel après restart (le cache HF est reset avec le conteneur).

Les tags HF Hub ne se suppriment pas automatiquement → toutes les versions historiques restent disponibles pour rollback.

---

## 8. Bonnes pratiques de sécurité

**À faire** :
- `.env` toujours dans `.gitignore` et `.dockerignore`
- Token HF stocké dans un **Secret** GitHub ou HF Space, jamais en clair
- Token write HF révoqué dès qu'il n'est plus utilisé
- Connection string Neon avec `sslmode=require` et `channel_binding=require`

**À ne jamais faire** :
- Commiter `.env`, un token ou une connection string
- Mettre un secret dans le Dockerfile (visible publiquement)
- Pousser `checkpoints/` directement dans le repo HF Space (utiliser HF Hub)
- Partager ton token write dans un chat/mail/issue
