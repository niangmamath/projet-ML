# Modèles : détails d'implémentation

Six architectures implémentées **from scratch** (inspirées de [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)), toutes dans le même format `(input_len=96, pred_len=H, n_features=7)` sur ETTh1.

## Vue d'ensemble

| Modèle | Année | Paramètres | Idée clé | Fichier |
|---|---|---:|---|---|
| **DLinear** | 2023 | ~0.5 M | Baseline linéaire décomposée | [dlinear.py](../src/models/dlinear.py) |
| **Transformer** | 2017 | ~33 M | Attention multi-tête (encoder-only) | [transformer.py](../src/models/transformer.py) |
| **Informer** | 2021 | ~13 M | ProbSparse attention | [informer.py](../src/models/informer.py) |
| **Autoformer** | 2021 | ~33 M | Décomposition + Auto-Correlation | [autoformer.py](../src/models/autoformer.py) |
| **FEDformer** | 2022 | ~33 M | Attention fréquentielle (FFT) | [fedformer.py](../src/models/fedformer.py) |
| **PatchTST** | 2023 | ~2.2 M | Patching + canaux indépendants | [patchtst.py](../src/models/patchtst.py) |

> Nombre de paramètres indicatif pour `pred_len=96`. Varie avec l'horizon.

---

## 1. DLinear (baseline)

**Idée** : décomposer la série en `trend` (moving average) + `residual`, prédire les deux via deux `nn.Linear(input_len → pred_len)` indépendants par canal.

```python
class DLinear(nn.Module):
    def __init__(self, input_len, pred_len, n_features):
        self.trend = nn.Linear(input_len, pred_len)
        self.resid = nn.Linear(input_len, pred_len)
```

**Pourquoi c'est un baseline fort** : [Zeng et al. 2023](https://arxiv.org/abs/2205.13504) ont montré qu'un modèle linéaire décomposé bat souvent les Transformers complexes sur LTSF, parce que les benchmarks LTSF sont dominés par des tendances déterministes et de la périodicité.

**Rôle dans ce projet** : borne basse à battre. Si un Transformer à 33M params ne bat pas DLinear, il n'apporte rien.

---

## 2. Transformer vanilla (encoder-only)

**Idée** : un bloc encoder standard (attention + FFN + LayerNorm) sur la séquence `(batch, 96, 7)`, puis projection linéaire `96×d_model → pred_len×7`.

```python
class VanillaTransformer(nn.Module):
    def __init__(self, input_len, pred_len, n_features):
        self.embed = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8,
                                       dim_feedforward=2048, norm_first=True,
                                       batch_first=True),
            num_layers=3,
        )
        self.proj = nn.Linear(input_len * d_model, pred_len * n_features)
```

**Référence** : [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762).

**Note** : `norm_first=True` → architecture pre-norm, plus stable pour l'entraînement. L'attention extraction (pour interprétation) a nécessité de remplacer la méthode `forward` entière, car la fast-path PyTorch bypasse les hooks sur `_sa_block`.

---

## 3. Informer

**Idée** : attention `ProbSparse` qui ne garde que les top-k queries actives (log-k, pas n) → complexité `O(L log L)` au lieu de `O(L²)`, et `self-attention distilling` qui réduit la longueur de séquence par moitié à chaque couche.

```python
class ProbAttention(nn.Module):
    def forward(self, Q, K, V):
        # 1) sample log(L) indices de K, calcule la "sparsity" de chaque Q
        # 2) garde les top-u queries (u = c * log(L))
        # 3) attention standard uniquement sur ces top-u
```

**Référence** : [Zhou et al. 2021](https://arxiv.org/abs/2012.07436).

**Intérêt pour ce projet** : montre que la sparsification de l'attention ne perd pas significativement en qualité sur ETTh1, et reduit drastiquement la mémoire.

---

## 4. Autoformer

**Idée** : remplace la self-attention par un bloc `AutoCorrelation` basé sur une FFT — détecte les décalages périodiques les plus corrélés dans la série, et agrège les représentations sur ces décalages.

```python
def auto_correlation(q, k, v):
    # 1) FFT(q), FFT(k)
    # 2) multiplication élément-par-élément → auto-corrélation
    # 3) IFFT → scores par délai temporel
    # 4) top-k délais, agrège V aux lags correspondants
```

Décomposition `trend + seasonal` intégrée à chaque bloc.

**Référence** : [Wu et al. 2021](https://arxiv.org/abs/2106.13008).

**Intérêt** : exploite explicitement la périodicité (24h pour ETTh1 → températures + charges électriques quotidiennes).

---

## 5. FEDformer

**Idée** : attention dans le **domaine fréquentiel** via DFT ou wavelets. Au lieu d'attendre sur les séquences temporelles, on sélectionne aléatoirement M fréquences, on fait l'attention sur ces modes, puis on reconstruit.

Implémentation : `FourierBlock` qui garde M modes Fourier aléatoires (defaut : M=64 modes sur une séquence de 96 → gain mémoire massif).

**Référence** : [Zhou et al. 2022](https://arxiv.org/abs/2201.12740).

**Note** : combiné avec la décomposition d'Autoformer (trend + seasonal).

---

## 6. PatchTST

**Idée** : deux ruptures par rapport aux Transformers classiques LTSF :
1. **Patching** : au lieu de traiter chaque pas de temps comme un token, on regroupe des patchs de 16 pas → séquence 10× plus courte → attention plus efficace, moins de bruit.
2. **Channel-independence** : on traite chaque feature séparément (`batch_size × n_features` → batch effectif plus gros), puis on concatène les sorties.

```python
class PatchTST(nn.Module):
    def __init__(self, input_len=96, pred_len=96, n_features=7):
        self.patch_len, self.stride = 16, 8
        n_patches = (input_len - self.patch_len) // self.stride + 1  # = 11
        self.encoder = nn.TransformerEncoder(...)
        self.head = nn.Linear(n_patches * d_model, pred_len)
    def forward(self, x):                           # x: (B, 96, 7)
        x = x.permute(0, 2, 1)                       # (B, 7, 96)
        # patching + encoder par canal
        ...
```

**Référence** : [Nie et al. 2023](https://arxiv.org/abs/2211.14730).

**Résultat attendu** : parmi les meilleurs sur ETTh1, avec 15× moins de paramètres que Transformer vanilla.

---

## Pipeline d'entraînement

Un seul fichier [`src/train.py`](../src/train.py) pour les 6 modèles :

```python
# Hyperparamètres communs (fairness du benchmark)
optimizer = Adam(lr=1e-3)                            # 1e-4 pour les gros Transformers
loss_fn   = MSELoss()
scheduler = ReduceLROnPlateau(factor=0.5, patience=2)
early_stop= EarlyStopping(patience=5)                 # sur val loss
batch_size= 32
epochs    = 15 (max, avec early stopping)
seed      = 2026
```

**Splits** (standard LTSF ETTh1) :
- Train : mois 1-12
- Val : mois 13-16
- Test : mois 17-20

**Normalisation** : `StandardScaler` **fit uniquement sur train**, appliqué à tous les splits. Garantit pas de fuite de données.

**Lancement grille complète** :
```powershell
./scripts/run_grid.ps1    # 6 modèles × 4 horizons = 24 runs
```

---

## Interprétabilité

[`src/interpret.py`](../src/interpret.py) produit 3 analyses par modèle :

### 1. Erreur par horizon
MSE/MAE par pas `t+1, t+2, ..., t+H` → identifie à quel moment la prédiction décroche.

### 2. Permutation feature importance
Pour chaque feature, on la permute aléatoirement dans le test set et on mesure la dégradation MSE → importance relative.

Résultats dans [`results/metrics/feature_importance.csv`](../results/metrics/feature_importance.csv).

### 3. Cartes d'attention (Transformers uniquement)
Extraction des poids d'attention au forward pass. Technique : **remplacer la méthode `forward` entière** du `TransformerEncoderLayer` (pas juste `_sa_block`) parce que le fast-path C++ de PyTorch bypasse sinon les hooks :

```python
def make_forward(layer):
    def _forward(src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        y = layer.norm1(x)
        attn_out, w = layer.self_attn(y, y, y,
            need_weights=True, average_attn_weights=True)
        attns.append(w.detach().cpu())
        x = x + layer.dropout1(attn_out)
        x = x + layer._ff_block(layer.norm2(x))
        return x
    return _forward
```

Figures dans [`results/figures/<model>_ETTh1_h96/`](../results/figures/).

---

## Ressources

- Repo d'inspiration : [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- Benchmark LTSF original : [Informer repo](https://github.com/zhouhaoyi/Informer2020)
- Dataset ETT : [https://github.com/zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)
