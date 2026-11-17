# Projet 7 — Architectures Transformer pour la prévision de séries temporelles à long terme

Étude comparative des architectures Informer, Autoformer, FEDformer et PatchTST pour la prévision de séries temporelles à long horizon (LTSF).

## Objectifs

- Étudier théoriquement les architectures Transformer dédiées aux séries temporelles.
- Comparer leurs performances sur des benchmarks standards (ETT, Electricity, Weather, Traffic).
- Analyser l'interprétabilité des modèles.
- Déployer une solution accessible via une interface web.

## Structure du dépôt

```
projet7-transformers-ts/
├── data/
│   ├── raw/            # Jeux de données bruts
│   └── processed/      # Données prétraitées
├── notebooks/
│   ├── 01_eda.ipynb            # Analyse exploratoire
│   ├── 02_baseline.ipynb       # Modèles de référence
│   └── 03_models.ipynb         # Modèles Transformer
├── src/
│   ├── data_loader.py          # Chargement et fenêtrage
│   ├── models/                 # Implémentations des architectures
│   ├── train.py                # Boucle d'entraînement
│   ├── evaluate.py             # Métriques et évaluation
│   └── interpret.py            # Interprétabilité (attention, SHAP)
├── results/
│   ├── figures/                # Graphiques
│   ├── metrics/                # Tableaux de résultats
│   └── checkpoints/            # Modèles entraînés
├── app/
│   └── streamlit_app.py        # Interface de démonstration
├── requirements.txt
└── README.md
```

## Installation

```bash
# Cloner le dépôt
git clone <url-du-depot>
cd projet7-transformers-ts

# Créer un environnement virtuel
python -m venv venv
# Windows :
venv\Scripts\activate
# Linux/Mac :
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### 1. Télécharger les données
Les jeux ETT, Electricity, Weather et Traffic sont disponibles sur le dépôt officiel
[thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library).
Les placer dans `data/raw/`.

### 2. Lancer l'EDA
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 3. Entraîner un modèle
```bash
python src/train.py --model PatchTST --data ETTh1 --pred_len 96
```

### 4. Lancer la démo
```bash
streamlit run app/streamlit_app.py
```

## Modèles comparés

| Modèle      | Année | Idée clé |
|-------------|-------|----------|
| Transformer | 2017  | Attention multi-tête (baseline) |
| Informer    | 2021  | ProbSparse attention + distillation |
| Autoformer  | 2021  | Décomposition + Auto-Correlation |
| FEDformer   | 2022  | Attention dans le domaine fréquentiel |
| PatchTST    | 2023  | Patching + indépendance des canaux |
| DLinear     | 2023  | Baseline linéaire forte |

## Auteur

Projet ML_ECC — n°7

## Licence

Usage académique.
