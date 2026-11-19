# Deployer la demo sur Hugging Face Spaces

Ce dossier (`huggingface_space/`) est **auto-contenu** : il peut etre pousse tel
quel comme un repo Hugging Face Space de type Docker. Il contient le code
applicatif, les modeles source, les checkpoints embarques, le CSV ETTh1 de
demonstration, le Dockerfile et le README avec le frontmatter attendu par HF.

## 1. Prerequis

- Un compte Hugging Face : <https://huggingface.co/join>
- Un **token d'acces** avec portee *write* : <https://huggingface.co/settings/tokens>
- Git installe en local (deja le cas).
- Git LFS : HF Spaces s'attend a ce que les fichiers > 10 Mo soient en LFS.
  Installation Windows : <https://git-lfs.com/>

Verifier :
```powershell
git lfs version
```

## 2. Creer le Space (interface web, une seule fois)

1. Aller sur <https://huggingface.co/new-space>
2. Owner : votre compte. Name : ex. `ltsf-transformers-ett`
3. License : MIT
4. **Space SDK : Docker** (important : pas "Gradio", meme si on utilise Gradio
   a l'interieur — c'est le Dockerfile qui orchestre tout)
5. Visibilite : **Public** (permet l'acces direct a l'API et a l'UI)
6. Hardware : CPU basic (gratuit, suffisant)
7. Valider

HF fournit alors l'URL `https://huggingface.co/spaces/<votre_compte>/ltsf-transformers-ett`
et l'URL git `https://huggingface.co/spaces/<votre_compte>/ltsf-transformers-ett.git`.

## 3. Pousser le contenu du dossier

Depuis la racine du projet local, dans PowerShell :

```powershell
cd huggingface_space

# Initialise un repo git dedie a ce sous-dossier
git init -b main
git lfs install

# Suivre les gros fichiers en LFS (checkpoints et donnees)
git lfs track "checkpoints/*.pt"
git lfs track "data/*.csv"
git add .gitattributes

# Stage tout le reste
git add .

# Commit initial
git commit -m "Initial deploy: LTSF Transformers comparison demo"

# Branche a utiliser par HF
git branch -M main

# Remote HF
git remote add origin https://huggingface.co/spaces/<votre_compte>/ltsf-transformers-ett
```

Pousser (demande le token HF comme mot de passe) :

```powershell
git push -u origin main
```

HF demarrera automatiquement la construction Docker. Le build prend 15-25 minutes
en gratuit (installation de torch + wheels + copie des 1.5 Go de checkpoints).
Suivre le log sur l'onglet "Logs" du Space. A la fin, un onglet "App" affiche
l'interface Gradio.

**Push initial long** : git doit transferer 1.5 Go de checkpoints via LFS la
premiere fois. Compter 10-30 minutes selon la connexion. Les pushs suivants ne
renvoient que les fichiers modifies.

## 4. Mises a jour

Pour redeployer apres un changement (nouveau checkpoint, UI, etc.) :

```powershell
cd huggingface_space
git add <fichiers modifies>
git commit -m "<message>"
git push
```

## 5. Debug courants

- **Build echoue sur torch** : le build HF peut parfois rate le telechargement
  (195 MB CPU-only). Relancer le build via l'onglet "Settings -> Restart this Space".
- **App se lance mais plante au premier clic** : verifier que les checkpoints
  sont bien pousses en LFS (`git lfs ls-files` doit les lister). Un .pt pousse
  en non-LFS apparait comme un pointeur texte et PyTorch rate le load.
- **Port 7860** : Dockerfile et `app.py` utilisent tous les deux 7860 — c'est le
  port par defaut lu par HF. Ne pas changer.
- **README frontmatter** : le bloc YAML en haut de `README.md` est obligatoire.
  Si HF ne detecte pas le SDK docker, verifier que le bloc est intact et qu'il
  n'y a aucun BOM / caractere invisible en debut de fichier.

## 6. Reproduire l'app en local (sans Docker)

```powershell
cd huggingface_space
..\venv\Scripts\python.exe app.py
# Puis ouvrir http://127.0.0.1:7860 dans un navigateur
```

## 7. Reproduire le build Docker en local (optionnel)

```powershell
cd huggingface_space
docker build -t ltsf-demo .
docker run --rm -p 7860:7860 ltsf-demo
# http://127.0.0.1:7860
```
