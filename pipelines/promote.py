"""Promotion d'un checkpoint staging vers la production.

Usage :
    # Promote le dernier retrain d'une combinaison (prend l'id DB le plus recent)
    python -m pipelines.promote --model PatchTST --pred_len 96

    # Promote en forcant (saute la regle PROMOTE_MIN_IMPROVEMENT)
    python -m pipelines.promote --model PatchTST --pred_len 96 --force

    # Promote une run par son id
    python -m pipelines.promote --run-id 42

Workflow :
    1. Charge la derniere row retrain_runs(model, pred_len) non promue (ou --run-id).
    2. Compare mse_after a la baseline prod (last_production_mse) -> policy.decide_promote.
    3. Si OK ou --force :
        a) copie results/checkpoints/<tag>.pt -> checkpoints/<tag>.pt
        b) upload sur HF Hub avec un nouveau tag (--version, defaut : auto-increment patch)
        c) marque la row promoted=true, hub_version=<tag>
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "huggingface_space"))
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from db import _conn, mark_promoted  # noqa: E402

from pipelines.policy import decide_promote  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("promote")

STAGING_DIR = ROOT / "results" / "checkpoints"
PROD_DIR = ROOT / "checkpoints"
DEFAULT_REPO = os.getenv("MODEL_REPO_ID", "Momoniang/ltsf-transformers-etth1")


def _last_run(model: str | None, pred_len: int | None, run_id: int | None) -> dict | None:
    """Recupere une row retrain_runs par id ou (model, pred_len) non promue recente."""
    with _conn() as c:
        if c is None:
            raise RuntimeError("DB indisponible : impossible de recuperer le run.")
        with c.cursor() as cur:
            if run_id is not None:
                cur.execute(
                    "SELECT id, model, pred_len, mse_before, mse_after "
                    "FROM retrain_runs WHERE id = %s", (run_id,),
                )
            else:
                cur.execute(
                    """SELECT id, model, pred_len, mse_before, mse_after
                       FROM retrain_runs
                       WHERE model=%s AND pred_len=%s AND promoted = false
                       ORDER BY ts DESC LIMIT 1""",
                    (model, pred_len),
                )
            row = cur.fetchone()
            if not row:
                return None
            cols = ["id", "model", "pred_len", "mse_before", "mse_after"]
            return dict(zip(cols, row))


def _next_version(current: str) -> str:
    """vX.Y.Z -> vX.Y.(Z+1)."""
    m = re.match(r"^v(\d+)\.(\d+)\.(\d+)$", current)
    if not m:
        return current + ".1"
    x, y, z = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return f"v{x}.{y}.{z+1}"


def _upload_to_hub(ckpt_path: Path, repo_id: str, tag: str) -> None:
    """Upload le .pt sur HF Hub + cree le tag."""
    from huggingface_hub import HfApi, create_tag

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN absent : impossible de pousser sur HF Hub.")

    api = HfApi(token=token)
    log.info(f"upload {ckpt_path.name} -> {repo_id}@main")
    api.upload_file(
        path_or_fileobj=str(ckpt_path),
        path_in_repo=ckpt_path.name,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"promote {ckpt_path.stem} -> {tag}",
    )
    log.info(f"create tag {tag}")
    try:
        create_tag(
            repo_id=repo_id, repo_type="model", tag=tag,
            tag_message=f"Promotion automatique : {ckpt_path.stem}",
            token=token,
        )
    except Exception as e:
        log.warning(f"tag {tag} deja present ou echec : {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="Modele a promouvoir (sinon --run-id)")
    ap.add_argument("--pred_len", type=int, help="Horizon (avec --model)")
    ap.add_argument("--run-id", type=int, help="Id de la row retrain_runs")
    ap.add_argument("--force", action="store_true", help="Saute la regle d'amelioration MSE")
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--version",
                    help="Tag HF Hub. Defaut : auto-increment du dernier hub_version promu.")
    ap.add_argument("--no-upload", action="store_true",
                    help="Copie en prod mais n'uploade pas sur HF Hub (test local)")
    args = ap.parse_args()

    if not args.run_id and (not args.model or not args.pred_len):
        ap.error("Fournir --run-id OU (--model ET --pred_len).")

    run = _last_run(args.model, args.pred_len, args.run_id)
    if not run:
        log.error("Aucune run retrain_runs correspondante (non promue).")
        return 1

    model, pred_len = run["model"], run["pred_len"]
    tag = f"{model}_ETTh1_h{pred_len}"
    log.info(f"candidat : run_id={run['id']} {tag}  MSE {run['mse_before']} -> {run['mse_after']}")

    # 1) Policy
    decision = decide_promote(run["mse_before"], run["mse_after"])
    log.info(f"policy : should_promote={decision.should_promote} ({decision.reason})")
    if not decision.should_promote and not args.force:
        log.info("Pas de promotion (utiliser --force pour forcer).")
        return 0

    # 2) Localisation staging
    src_ckpt = STAGING_DIR / f"{tag}.pt"
    if not src_ckpt.is_file():
        log.error(f"Checkpoint staging introuvable : {src_ckpt}")
        return 2

    # 3) Copie en prod
    PROD_DIR.mkdir(parents=True, exist_ok=True)
    dst_ckpt = PROD_DIR / f"{tag}.pt"
    shutil.copy2(src_ckpt, dst_ckpt)
    log.info(f"prod OK : {dst_ckpt}  ({dst_ckpt.stat().st_size / 1e6:.1f} MB)")

    # 4) Upload HF Hub + versioning
    new_version = args.version or _next_version(os.getenv("MODEL_VERSION", "v1.0.0"))
    if args.no_upload:
        log.info(f"[--no-upload] saute HF Hub. Version theorique : {new_version}")
    else:
        try:
            _upload_to_hub(dst_ckpt, args.repo_id, new_version)
        except Exception as e:
            log.exception(f"Upload HF Hub echoue : {e}")
            return 3

    # 5) Marque la row en DB
    mark_promoted(run["id"], new_version)
    log.info(f"DB: retrain_runs.id={run['id']} promoted=true hub_version={new_version}")
    log.info("OK promotion terminee.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
