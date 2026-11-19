"""Pousse les 24 checkpoints LTSF sur un repo HuggingFace Hub.

Usage:
    # Depuis la racine du projet, avec HF_TOKEN dans l'env (ou huggingface-cli login)
    python scripts/upload_models_to_hub.py \
        --repo-id niangmamathyacine/ltsf-transformers-etth1 \
        --version v1.0.0 \
        --public

Le script :
  1. Cree le repo HF Hub (model repo) s'il n'existe pas.
  2. Verifie que les 24 fichiers attendus sont bien presents dans ./checkpoints/.
  3. Upload tout le dossier.
  4. Pose le tag de version pour permettre un pinning cote Space.

Les checkpoints ne contiennent que des poids (torch.save(state_dict)) : rien
de sensible n'y est embarque.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, create_repo, create_tag

MODELS = ["DLinear", "Transformer", "PatchTST", "Informer", "Autoformer", "FEDformer"]
HORIZONS = [96, 192, 336, 720]


def expected_files() -> list[str]:
    return [f"{m}_ETTh1_h{h}.pt" for m in MODELS for h in HORIZONS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True,
                    help="e.g. niangmamathyacine/ltsf-transformers-etth1")
    ap.add_argument("--version", default="v1.0.0",
                    help="Git tag applique apres upload (ex: v1.0.0)")
    ap.add_argument("--public", action="store_true",
                    help="Rend le repo public (defaut: prive)")
    ap.add_argument("--checkpoints-dir", default="checkpoints",
                    help="Dossier contenant les .pt")
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoints_dir).resolve()
    if not ckpt_dir.is_dir():
        print(f"[ERR] dossier introuvable : {ckpt_dir}", file=sys.stderr)
        return 2

    missing = [f for f in expected_files() if not (ckpt_dir / f).is_file()]
    if missing:
        print(f"[ERR] {len(missing)} checkpoint(s) manquant(s) :", file=sys.stderr)
        for f in missing:
            print(f"  - {f}", file=sys.stderr)
        return 3

    token = os.getenv("HF_TOKEN")
    if not token:
        print("[WARN] HF_TOKEN non defini -> fallback sur le token cache "
              "(huggingface-cli login).")

    api = HfApi(token=token)

    print(f"[1/3] creation (ou reuse) du repo {args.repo_id} "
          f"(private={not args.public})")
    create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=not args.public,
        exist_ok=True,
        token=token,
    )

    print(f"[2/3] upload du dossier {ckpt_dir} vers main")
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        try:
            api.upload_folder(
                folder_path=str(ckpt_dir),
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=f"release {args.version} - 24 LTSF checkpoints ETTh1",
                allow_patterns=["*.pt"],
            )
            break
        except Exception as e:
            if attempt == max_attempts:
                print(f"[ERR] upload echec apres {max_attempts} tentatives : {e}",
                      file=sys.stderr)
                return 4
            wait = min(60, 5 * attempt)
            print(f"[WARN] tentative {attempt}/{max_attempts} echouee ({type(e).__name__}): {e}")
            print(f"       retry dans {wait}s (les blobs deja pousses sont conserves)")
            time.sleep(wait)

    print(f"[3/3] tag {args.version} sur main")
    try:
        create_tag(
            repo_id=args.repo_id,
            repo_type="model",
            tag=args.version,
            tag_message=f"LTSF ETTh1 - release {args.version}",
            token=token,
        )
    except Exception as e:
        print(f"[WARN] tag deja present ou echec : {e}")

    print(f"\nOK : https://huggingface.co/{args.repo_id}/tree/{args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
