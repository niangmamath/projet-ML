"""Interface Gradio a 3 onglets :
    1. Comparaison des modeles (benchmark sur le test set)
    2. Utilisation reelle (upload CSV de ses propres donnees, alertes, export)
    3. Monitoring (dashboard live sur la base Postgres)

L'objet `demo` est monte sur FastAPI dans api.py.
"""
from __future__ import annotations

import tempfile
from datetime import timedelta
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from predictor import (
    FEATURES,
    INPUT_LEN,
    available_checkpoints,
    last_hours_window,
    max_window_start,
    predict_sample,
    predict_window,
)

CKPTS = available_checkpoints()
MODEL_NAMES = list(CKPTS.keys())


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _sample_max(pred_len: int) -> int:
    return max(0, max_window_start(int(pred_len)) - (16 * 30 * 24 - 96))


def _plot_pred_vs_truth(result: dict, pred_len: int, feature_idx: int) -> plt.Figure:
    ctx = result["context"][:, feature_idx]
    pred = result["prediction"][:, feature_idx]
    truth = result["ground_truth"][:, feature_idx]
    feat_name = FEATURES[feature_idx]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    t_ctx = np.arange(-INPUT_LEN, 0)
    t_fut = np.arange(0, pred_len)
    ax.plot(t_ctx, ctx, color="#555", linewidth=1.5, label=f"{feat_name} - contexte")
    ax.plot(t_fut, truth, color="#2c7fb8", linewidth=2, label=f"{feat_name} - verite")
    ax.plot(t_fut, pred, color="#c0392b", linewidth=2, linestyle="--", label=f"{feat_name} - prediction")
    ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("pas horaires (0 = debut de l'horizon)")
    ax.set_ylabel(feat_name)
    ax.set_title(f"Horizon {pred_len} h - echantillon #{result['start_idx']}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_pred_only(window: np.ndarray, pred: np.ndarray, feature_idx: int,
                    pred_len: int, threshold: float | None, start_ts: pd.Timestamp) -> plt.Figure:
    feat_name = FEATURES[feature_idx]
    ctx = window[:, feature_idx]
    pr = pred[:, feature_idx]

    t_ctx = pd.date_range(end=start_ts, periods=INPUT_LEN, freq="h")
    t_fut = pd.date_range(start=start_ts + timedelta(hours=1), periods=pred_len, freq="h")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(t_ctx, ctx, color="#555", linewidth=1.5, label=f"{feat_name} - historique (96 h)")
    ax.plot(t_fut, pr, color="#c0392b", linewidth=2, linestyle="--", label=f"{feat_name} - prediction")
    ax.axvline(start_ts, color="black", linewidth=0.5, linestyle=":")

    if threshold is not None and feat_name == "OT":
        ax.axhline(threshold, color="#e67e22", linewidth=1.2, linestyle="-.",
                   label=f"seuil d'alerte = {threshold:.1f}")
        over = pr > threshold
        if over.any():
            idx = int(np.argmax(over))
            ax.plot(t_fut[idx], pr[idx], marker="o", color="red", markersize=10,
                    label=f"depassement t+{idx+1}h")

    ax.set_xlabel("date (horaire)")
    ax.set_ylabel(feat_name)
    ax.set_title(f"Prevision sur {pred_len} h")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# =============================================================================
# ONGLET 1 : Comparaison des modeles (benchmark test set)
# =============================================================================
def run_comparison(model_name: str, pred_len: int, sample_idx: int, feature: str):
    result = predict_sample(model_name, int(pred_len), int(sample_idx))
    feat_idx = FEATURES.index(feature)
    fig = _plot_pred_vs_truth(result, int(pred_len), feat_idx)
    md = (
        f"**MSE** (denormalise) : `{result['mse']:.4f}`\n\n"
        f"**MAE** (denormalise) : `{result['mae']:.4f}`\n\n"
        f"Modele : `{model_name}` | Horizon : `{pred_len}` h | "
        f"Variable : `{feature}` | Fenetre d'entree : `{INPUT_LEN}` h"
    )
    return fig, md


def on_model_change(model_name: str):
    horizons = CKPTS.get(model_name, [96])
    default = horizons[0]
    return gr.update(choices=horizons, value=default), gr.update(maximum=_sample_max(default), value=0)


def on_horizon_change(pred_len):
    return gr.update(maximum=_sample_max(int(pred_len)), value=0)


# =============================================================================
# ONGLET 2 : Utilisation reelle (operateur)
# =============================================================================
def _read_user_csv(file) -> tuple[np.ndarray, pd.Timestamp, str]:
    """Lit un CSV uploade et retourne (window[96,7], last_timestamp, message).

    Le CSV doit contenir au moins 96 lignes avec les colonnes FEATURES. Une colonne
    'date' optionnelle sert a afficher la timeline.
    """
    if file is None:
        return None, None, ""
    df = pd.read_csv(file.name) if hasattr(file, "name") else pd.read_csv(file)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise gr.Error(f"Colonnes manquantes dans le CSV : {missing}. Attendu : {FEATURES}")
    if len(df) < INPUT_LEN:
        raise gr.Error(f"Le CSV doit contenir au moins {INPUT_LEN} lignes, recu {len(df)}")
    window = df[FEATURES].iloc[-INPUT_LEN:].values.astype(np.float32)
    if "date" in df.columns:
        last_ts = pd.to_datetime(df["date"].iloc[-1])
    else:
        last_ts = pd.Timestamp("now").floor("h")
    msg = f"OK : {len(df)} lignes lues, 96 dernieres heures utilisees. Fin : {last_ts}."
    return window, last_ts, msg


def run_operator(file, model_name: str, pred_len: int, feature: str, threshold: float, use_threshold: bool):
    # Si pas de fichier, on prend les 96 dernieres heures du CSV embarque (demo)
    if file is None:
        window, last_ts = last_hours_window()
        info_msg = "Aucun CSV uploade - utilisation des 96 dernieres heures d'ETTh1 embarquees."
    else:
        window, last_ts, info_msg = _read_user_csv(file)

    result = predict_window(model_name, int(pred_len), window)
    pred = result["prediction"]
    feat_idx = FEATURES.index(feature)

    th = float(threshold) if use_threshold else None
    fig = _plot_pred_only(window, pred, feat_idx, int(pred_len), th, last_ts)

    # Analyse alerte sur OT
    ot_pred = pred[:, FEATURES.index("OT")]
    alert_md = ""
    if use_threshold:
        over = ot_pred > th
        if over.any():
            first = int(np.argmax(over))
            peak = float(ot_pred.max())
            peak_t = int(np.argmax(ot_pred))
            alert_md = (
                f"### Alerte depassement\n"
                f"- Seuil OT : **{th:.1f}**\n"
                f"- Premier depassement : **t+{first+1} h** (= {last_ts + timedelta(hours=first+1):%Y-%m-%d %H:%M})\n"
                f"- Pic prevu : **{peak:.2f}** a t+{peak_t+1} h\n"
            )
        else:
            alert_md = f"### Aucun depassement prevu\nOT max prevu : **{ot_pred.max():.2f}** (< seuil {th:.1f}).\n"

    # Prepare CSV export (chemin cross-plateforme)
    future_ts = pd.date_range(start=last_ts + timedelta(hours=1), periods=int(pred_len), freq="h")
    export_df = pd.DataFrame(pred, columns=FEATURES)
    export_df.insert(0, "date", future_ts)
    tmp_dir = Path(tempfile.gettempdir()) / "ltsf_exports"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = str(tmp_dir / "prediction_export.csv")
    export_df.to_csv(csv_path, index=False)

    status_md = (
        f"{info_msg}\n\n"
        f"Modele : `{model_name}` | Horizon : `{pred_len}` h | Variable affichee : `{feature}`"
    )
    return fig, status_md, alert_md, csv_path


# =============================================================================
# ONGLET 3 : Monitoring (live - lit la base Postgres via db.py)
# =============================================================================
def _monitor_refresh(password: str = "") -> tuple[str, pd.DataFrame, plt.Figure | None]:
    """Appelle la couche DB directement. Gate par ADMIN_TOKEN (admin uniquement)."""
    import os as _os

    from db import recent_predictions, summary_stats

    expected = _os.getenv("ADMIN_TOKEN", "")
    if not expected:
        return ("### Monitoring\n\n_Monitoring desactive : `ADMIN_TOKEN` non configure._",
                pd.DataFrame(), None)
    if password != expected:
        return ("### Monitoring (admin)\n\n"
                "_Entre le mot de passe admin ci-dessus puis clique Afficher._",
                pd.DataFrame(), None)

    stats = summary_stats()
    rows = recent_predictions(limit=50)

    if stats["total"] == 0:
        md = (
            "### Monitoring\n\n"
            "_Aucune prediction enregistree pour l'instant._\n\n"
            "Lance quelques predictions dans l'onglet 'Utilisation reelle' ou via "
            "`POST /api/predict`, puis clique sur 'Rafraichir'."
        )
        return md, pd.DataFrame(), None

    avg_lat = stats["avg_latency_ms"]
    drift_rate = (stats["drifted"] / stats["total"] * 100) if stats["total"] else 0
    md = (
        "### Monitoring (live)\n\n"
        f"- **Predictions totales** : {stats['total']}\n"
        f"- **Drift detecte** : {stats['drifted']} ({drift_rate:.1f} %)\n"
        f"- **Latence moyenne** : {avg_lat:.1f} ms\n"
        f"- **Repartition par modele** : "
        + ", ".join(f"`{k}`={v}" for k, v in stats["by_model"].items())
    )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Camembert drift / non-drift
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ok = stats["total"] - stats["drifted"]
    ax.pie([ok, stats["drifted"]], labels=["Nominal", "Drift"],
           colors=["#2ecc71", "#e74c3c"], autopct="%1.1f%%", startangle=90)
    ax.set_title("Repartition drift sur l'historique")
    fig.tight_layout()
    return md, df, fig


# =============================================================================
# Blocks
# =============================================================================
def build_demo() -> gr.Blocks:
    CSS = ".gradio-container { max-width: 1200px !important; }"
    default_h = CKPTS.get(MODEL_NAMES[0], [96])[0]

    with gr.Blocks(title="LTSF Transformers - Demo & MLOps", css=CSS) as demo:
        gr.Markdown(
            "# LTSF Transformers - Benchmark & Serving MLOps (ETTh1)\n\n"
            "Demo multi-onglets : benchmark des 6 modeles, utilisation operationnelle "
            "sur donnees reelles, et dashboard de monitoring en production."
        )

        with gr.Tab("1. Comparaison des modeles"):
            gr.Markdown(
                "Evaluation sur le test set ETTh1 : choisir un modele, un horizon, un "
                "echantillon et une variable pour visualiser prediction vs verite."
            )
            with gr.Row():
                model_dd = gr.Dropdown(MODEL_NAMES, value=MODEL_NAMES[0], label="Modele")
                horizon_dd = gr.Dropdown(CKPTS[MODEL_NAMES[0]], value=default_h,
                                         label="Horizon (h)")
                feature_dd = gr.Dropdown(FEATURES, value="OT", label="Variable")
            sample_sl = gr.Slider(0, _sample_max(default_h), value=0, step=1,
                                  label="Echantillon du test set")
            btn = gr.Button("Predire", variant="primary")
            plot1 = gr.Plot(label="Contexte + prediction vs verite")
            md1 = gr.Markdown()

            model_dd.change(on_model_change, inputs=model_dd, outputs=[horizon_dd, sample_sl])
            horizon_dd.change(on_horizon_change, inputs=horizon_dd, outputs=sample_sl)
            btn.click(run_comparison,
                      inputs=[model_dd, horizon_dd, sample_sl, feature_dd],
                      outputs=[plot1, md1])

        with gr.Tab("2. Utilisation reelle"):
            gr.Markdown(
                "Upload un CSV de tes 96 dernieres heures (colonnes : "
                f"`{', '.join(FEATURES)}`, +`date` optionnel) ou laisse vide pour "
                "utiliser les dernieres heures d'ETTh1 embarquees. Le modele predit "
                "les prochaines heures et signale un depassement de seuil sur OT."
            )
            with gr.Row():
                file_in = gr.File(label="CSV historique (optionnel)",
                                  file_types=[".csv"])
            with gr.Row():
                op_model = gr.Dropdown(MODEL_NAMES, value="PatchTST", label="Modele")
                op_horizon = gr.Dropdown([96, 192, 336, 720], value=96, label="Horizon (h)")
                op_feature = gr.Dropdown(FEATURES, value="OT", label="Variable affichee")
            with gr.Row():
                use_th = gr.Checkbox(value=True, label="Activer seuil d'alerte (sur OT)")
                th_sl = gr.Slider(0, 60, value=30, step=0.5, label="Seuil OT")
            op_btn = gr.Button("Predire", variant="primary")

            plot2 = gr.Plot(label="Prevision")
            status2 = gr.Markdown()
            alert2 = gr.Markdown()
            csv_out = gr.File(label="Telecharger la prediction (CSV)")

            op_btn.click(run_operator,
                         inputs=[file_in, op_model, op_horizon, op_feature, th_sl, use_th],
                         outputs=[plot2, status2, alert2, csv_out])

        with gr.Tab("3. Monitoring (admin)"):
            gr.Markdown(
                "Dashboard live des predictions loguees en base (Postgres Neon). "
                "**Acces restreint** : saisir le mot de passe admin (variable "
                "d'env `ADMIN_TOKEN` cote Space)."
            )
            mon_pw = gr.Textbox(label="Mot de passe admin", type="password",
                                placeholder="ADMIN_TOKEN")
            mon_btn = gr.Button("Afficher / Rafraichir", variant="primary")
            mon_md = gr.Markdown()
            mon_fig = gr.Plot(label="Drift sur l'historique")
            mon_df = gr.Dataframe(label="50 dernieres predictions",
                                  headers=["ts", "model", "pred_len",
                                           "latency_ms", "drift_flag", "alert"],
                                  wrap=True)
            mon_btn.click(_monitor_refresh, inputs=mon_pw,
                          outputs=[mon_md, mon_df, mon_fig])

        gr.Markdown(
            "---\n"
            "API REST publique sur `/api/*` (voir [/docs](./docs) pour la "
            "specification OpenAPI). Monitoring Prometheus sur `/metrics`. "
            "Code : [github.com/niangmamath/projet-ML]"
            "(https://github.com/niangmamath/projet-ML)."
        )

    return demo
