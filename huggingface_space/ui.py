"""Interface Gradio a 2 onglets :
    1. Prevision sur mes donnees (multi-modeles OU 1 modele + seuil d'alerte)
    2. Monitoring (dashboard admin, lit la base Postgres)

Le benchmark test set figure dans le rapport et le README (resultats figes
calcules a l'entrainement) ; la plateforme sert les cas operationnels.

L'objet `demo` est monte sur FastAPI dans api.py.
"""
from __future__ import annotations

import tempfile
import time
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
    predict_window,
)

CKPTS = available_checkpoints()
MODEL_NAMES = list(CKPTS.keys())

# Palette coherente modele -> couleur, partagee entre onglets 1 et 2.
MODEL_COLORS = {
    "DLinear":     "#2E86DE",
    "PatchTST":    "#10AC84",
    "Transformer": "#8E44AD",
    "Informer":    "#E67E22",
    "Autoformer":  "#C0392B",
    "FEDformer":   "#F1C40F",
}
# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _plot_multi_pred_only(
    window: np.ndarray, preds_by_model: dict[str, np.ndarray], feature_idx: int,
    pred_len: int, threshold: float | None, start_ts: pd.Timestamp,
) -> plt.Figure:
    """Overlay de 1..N predictions sur une fenetre fournie par l'utilisateur.

    Le seuil d'alerte n'est trace que si un seul modele est selectionne.
    """
    feat_name = FEATURES[feature_idx]
    ctx = window[:, feature_idx]
    t_ctx = pd.date_range(end=start_ts, periods=INPUT_LEN, freq="h")
    t_fut = pd.date_range(start=start_ts + timedelta(hours=1), periods=pred_len, freq="h")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(t_ctx, ctx, color="#333", linewidth=1.5, label=f"{feat_name} - historique (96 h)")

    for model_name, pred in preds_by_model.items():
        pr = pred[:, feature_idx]
        color = MODEL_COLORS.get(model_name, "#888")
        ax.plot(t_fut, pr, color=color, linewidth=1.8, linestyle="--",
                label=f"{model_name}")

    ax.axvline(start_ts, color="black", linewidth=0.5, linestyle=":")

    # Seuil + marqueur de depassement uniquement si un seul modele + variable OT
    if threshold is not None and feat_name == "OT" and len(preds_by_model) == 1:
        ax.axhline(threshold, color="#e67e22", linewidth=1.2, linestyle="-.",
                   label=f"seuil d'alerte = {threshold:.1f}")
        only_pred = next(iter(preds_by_model.values()))[:, feature_idx]
        over = only_pred > threshold
        if over.any():
            idx = int(np.argmax(over))
            ax.plot(t_fut[idx], only_pred[idx], marker="o", color="red", markersize=10,
                    label=f"depassement t+{idx+1}h")

    ax.set_xlabel("date (horaire)")
    ax.set_ylabel(feat_name)
    ax.set_title(f"Prevision sur {pred_len} h")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# =============================================================================
# ONGLET 1 : Utilisation reelle (operateur)
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


def run_operator(file, selected_models: list[str], pred_len: int, feature: str,
                 threshold: float, use_threshold: bool):
    """Prevoit avec 1..N modeles sur une fenetre fournie par l'utilisateur.

    - 1 modele coche  : mode operationnel (seuil d'alerte actif, export CSV).
    - N modeles coches : mode comparatif (overlay, tableau latence + drift + valeurs cles).
    """
    if not selected_models:
        raise gr.Error("Selectionne au moins un modele.")

    if file is None:
        window, last_ts = last_hours_window()
        info_msg = "Aucun CSV uploade - utilisation des 96 dernieres heures d'ETTh1 embarquees."
    else:
        window, last_ts, info_msg = _read_user_csv(file)

    preds_by_model: dict[str, np.ndarray] = {}
    rows = []
    for name in selected_models:
        t0 = time.perf_counter()
        res = predict_window(name, int(pred_len), window)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        pred_arr = res["prediction"]
        preds_by_model[name] = pred_arr
        # KS drift sur la fenetre d'entree (meme calcul que l'API)
        try:
            from drift import ks_drift
            _, drift_flag = ks_drift(window)
        except Exception:
            drift_flag = None
        ot_pred = pred_arr[:, FEATURES.index("OT")]
        rows.append({
            "Modele":          name,
            "Latence (ms)":    round(latency_ms, 1),
            "Drift (KS)":      "oui" if drift_flag else ("non" if drift_flag is False else "n/a"),
            "OT a t+1 h":      round(float(ot_pred[0]), 2),
            "OT a t+H":        round(float(ot_pred[-1]), 2),
            "Amplitude OT":    round(float(ot_pred.max() - ot_pred.min()), 2),
        })

    feat_idx = FEATURES.index(feature)
    single_mode = (len(selected_models) == 1)

    # Seuil : actif uniquement en mode 1 modele + variable OT
    th = float(threshold) if (use_threshold and single_mode) else None
    fig = _plot_multi_pred_only(window, preds_by_model, feat_idx, int(pred_len), th, last_ts)

    # Export CSV : toujours fourni. Format large (wide) : date + 1 colonne par (modele, feature).
    future_ts = pd.date_range(start=last_ts + timedelta(hours=1), periods=int(pred_len), freq="h")
    export_df = pd.DataFrame({"date": future_ts})
    for name, pred_arr in preds_by_model.items():
        for i, feat in enumerate(FEATURES):
            export_df[f"{name}_{feat}"] = pred_arr[:, i]
    tmp_dir = Path(tempfile.gettempdir()) / "ltsf_exports"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    csv_name = ("prediction_export.csv" if single_mode
                else f"predictions_compare_{len(selected_models)}models.csv")
    csv_path = str(tmp_dir / csv_name)
    export_df.to_csv(csv_path, index=False)

    # Alerte textuelle : mode 1 modele uniquement
    alert_md = ""
    if single_mode:
        only_name = selected_models[0]
        only_pred = preds_by_model[only_name]
        ot_pred = only_pred[:, FEATURES.index("OT")]
        if use_threshold:
            over = ot_pred > th
            if over.any():
                first = int(np.argmax(over))
                peak = float(ot_pred.max())
                peak_t = int(np.argmax(ot_pred))
                alert_md = (
                    f"### Alerte depassement\n"
                    f"- Modele : **{only_name}**\n"
                    f"- Seuil OT : **{th:.1f}**\n"
                    f"- Premier depassement : **t+{first+1} h** "
                    f"(= {last_ts + timedelta(hours=first+1):%Y-%m-%d %H:%M})\n"
                    f"- Pic prevu : **{peak:.2f}** a t+{peak_t+1} h\n"
                )
            else:
                alert_md = (f"### Aucun depassement prevu\n"
                            f"Modele {only_name} : OT max prevu : **{ot_pred.max():.2f}** "
                            f"(< seuil {th:.1f}).\n")
    else:
        alert_md = (
            f"### Mode comparatif ({len(selected_models)} modeles)\n"
            f"Seuil d'alerte desactive. L'export CSV contient les predictions des "
            f"{len(selected_models)} modeles (format large : `date` + une colonne par "
            f"`<Modele>_<Feature>`).")

    status_md = (
        f"{info_msg}\n\n"
        f"Modeles : {', '.join(f'`{m}`' for m in selected_models)} | "
        f"Horizon : `{pred_len}` h | Variable affichee : `{feature}`"
    )
    cmp_df = pd.DataFrame(rows, columns=[
        "Modele", "Latence (ms)", "Drift (KS)", "OT a t+1 h", "OT a t+H", "Amplitude OT",
    ])
    return fig, status_md, alert_md, csv_path, cmp_df


# =============================================================================
# ONGLET 2 : Monitoring (live - lit la base Postgres via db.py)
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

    with gr.Blocks(title="LTSF Transformers - Serving MLOps", css=CSS) as demo:
        gr.Markdown(
            "# LTSF Transformers - Serving MLOps (ETTh1)\n\n"
            "Plateforme de prevision : charge tes 96 dernieres heures, choisis "
            "un ou plusieurs modeles, obtiens la prevision + alerte + export CSV. "
            "La comparaison globale des modeles (MSE/MAE sur le test set) figure "
            "dans le rapport et le README."
        )

        with gr.Tab("1. Prevision sur mes donnees"):
            gr.Markdown(
                "Upload un CSV de tes 96 dernieres heures (colonnes : "
                f"`{', '.join(FEATURES)}`, +`date` optionnel) ou laisse vide pour utiliser les "
                "dernieres heures d'ETTh1 embarquees.\n\n"
                "- **1 modele coche** : mode operationnel (seuil d'alerte sur OT + export CSV).\n"
                "- **Plusieurs modeles coches** : mode comparatif (overlay + tableau latence / drift / "
                "valeurs cles)."
            )
            with gr.Row():
                file_in = gr.File(label="CSV historique (optionnel)", file_types=[".csv"])
            with gr.Row():
                op_models = gr.CheckboxGroup(
                    MODEL_NAMES, value=["DLinear"] if "DLinear" in MODEL_NAMES else MODEL_NAMES[:1],
                    label="Modeles a utiliser (1 a 6)",
                )
            with gr.Row():
                op_horizon = gr.Dropdown([96, 192, 336, 720], value=96, label="Horizon (h)")
                op_feature = gr.Dropdown(FEATURES, value="OT", label="Variable affichee")
            with gr.Row():
                use_th = gr.Checkbox(value=True,
                                     label="Activer seuil d'alerte sur OT (actif en mode 1 modele)")
                th_sl = gr.Slider(0, 60, value=30, step=0.5, label="Seuil OT")
            op_btn = gr.Button("Predire", variant="primary")

            plot2 = gr.Plot(label="Prevision")
            status2 = gr.Markdown()
            alert2 = gr.Markdown()
            cmp_table = gr.Dataframe(label="Comparaison par modele",
                                     interactive=False, wrap=True)
            csv_out = gr.File(label="Telecharger la prediction (CSV)")

            op_btn.click(run_operator,
                         inputs=[file_in, op_models, op_horizon, op_feature, th_sl, use_th],
                         outputs=[plot2, status2, alert2, csv_out, cmp_table])

        with gr.Tab("2. Monitoring (admin)"):
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
