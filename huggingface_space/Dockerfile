# Image pour Hugging Face Spaces (Docker SDK).
# - Python 3.11 slim
# - PyTorch CPU-only (via l'index wheel de PyTorch declare dans requirements.txt)
# - l'utilisateur non-root 'user' (uid 1000) est exige par HF

FROM python:3.11-slim

# Dependences systeme minimales pour matplotlib (freetype, libpng) et gcc le temps
# d'eventuelles compilations de roues scientifiques.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libfreetype6-dev \
        libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Utilisateur non-root impose par HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /home/user/app

# On installe les deps d'abord (couche cacheable) avant de copier le code.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Code applicatif + modules modeles + donnees d'inference.
# Les checkpoints ne sont PAS embarques : ils sont telecharges au boot depuis
# le registry HF Hub (MODEL_REPO_ID @ MODEL_VERSION, cf. predictor.py).
COPY --chown=user api.py app.py predictor.py schemas.py ui.py ./
COPY --chown=user src ./src
COPY --chown=user data ./data

# Dossier cache HF (persiste au sein du conteneur pour reutiliser les .pt).
ENV HF_HOME=/home/user/.cache/huggingface \
    MODEL_REPO_ID=Momoniang/ltsf-transformers-etth1 \
    MODEL_VERSION=v1.0.0

EXPOSE 7860

# Uvicorn gere Gradio monte + API REST sur le meme port.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
