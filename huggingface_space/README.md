---
title: LTSF Transformers Comparison
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Comparative study of Transformers for long-term time series forecasting on ETTh1
---

# LTSF Transformers — comparison + MLOps serving

Interactive demo for a **long-term time series forecasting (LTSF)** benchmark on **ETTh1**, with six models (one linear baseline + five Transformer variants) implemented from scratch and served behind a full MLOps stack.

Pick a model, a prediction horizon (96 / 192 / 336 / 720 hours), a sample
from the test set and a variable to visualize: the app plots the
input context (96 past hours) together with the model's forecast and the
ground truth over the chosen horizon.

## Models

Six architectures, all implemented from scratch in PyTorch 2.4:

| Model | Idea |
|---|---|
| **DLinear** | Linear baseline with trend/seasonality decomposition (Zeng et al., 2023) |
| **PatchTST** | Patching + channel-independence + RevIN (Nie et al., 2023) |
| **Transformer** | Vanilla encoder-only, attention baseline |
| **Informer** | ProbSparse attention + self-attention distilling (Zhou et al., 2021) |
| **Autoformer** | Auto-Correlation via FFT + series decomposition (Wu et al., 2021) |
| **FEDformer** | Fourier-enhanced attention + series decomposition (Zhou et al., 2022) |

All models were trained with the same recipe (Adam, lr=1e-3, MSE loss,
early stopping, `input_len=96`) on the standard LTSF train/val/test split
(12/4/4 months).

## Test-set results on ETTh1 (MSE)

| Model       | h=96  | h=192 | h=336 | h=720 |
|-------------|------:|------:|------:|------:|
| PatchTST    | **0.387** | 0.457 | 0.479 | 0.519 |
| DLinear     | 0.388 | **0.441** | **0.485** | **0.507** |
| Autoformer  | 0.611 | 0.779 | 1.038 | 1.262 |
| Informer    | 0.611 | 0.953 | 0.963 | 1.038 |
| FEDformer   | 0.620 | 0.816 | 0.993 | 1.274 |
| Transformer | 0.822 | 1.027 | 0.929 | 1.227 |

These results reproduce the well-known finding of Zeng et al. (2023,
"Are Transformers Effective for Time Series Forecasting?"): a simple
linear decomposition baseline matches or beats most heavy Transformer
variants on this benchmark, with PatchTST being the exception.

## What's in this Space

Three Gradio tabs + a REST API, all served from a single FastAPI process:

1. **Model comparison**: pick (model, horizon, test sample, variable) → plot forecast vs ground truth, with MSE / MAE.
2. **Operational use**: upload your own 96-hour CSV (or use the embedded ETTh1 tail), set an alert threshold on OT, export the forecast as CSV.
3. **Monitoring** (live): aggregate stats from the production DB — total predictions, drift ratio (KS-test), latency, breakdown by model.

REST endpoints under `/api/*` (full spec at [`/docs`](./docs)):

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service status, number of loaded checkpoints |
| `GET /api/models` | Catalog of 6 models × 4 horizons |
| `POST /api/predict` | Inference from a `(96, 7)` input window — returns forecast + per-feature KS drift p-values |
| `GET /api/monitoring` | Dashboard data (totals, latency, recent predictions) |
| `GET /metrics` | Prometheus-format metrics (counters, latency histograms) |

## MLOps design

This Space is a thin serving layer. The actual MLOps pipeline lives in the GitHub repo:

- **Model registry**: checkpoints are versioned on HuggingFace Hub (`Momoniang/ltsf-transformers-etth1@v1.X.Y`), downloaded lazily at inference time.
- **Monitoring**: every `/api/predict` call is logged to Neon Postgres with input/output stats and KS drift p-values.
- **Drift detection**: Kolmogorov-Smirnov test per feature against the ETTh1 training distribution.
- **Retraining pipeline**: `pipelines/retrain.py --check-drift` triggers a retrain when drift rate > 20% over the last 7 days. `pipelines/promote.py` compares MSE, promotes the new checkpoint to HF Hub with an auto-incremented tag, and marks the DB row.
- **CI/CD**: GitHub Actions run ruff + pytest + Docker build on every PR; automatic push to this Space on merge to main.

## Repository

Full source, docs and pipelines: **https://github.com/niangmamath/projet-ML**

Inspiration (credited where used): [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library).
