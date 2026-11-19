"""Télécharge les jeux de données standards LTSF dans data/raw/.

Datasets couverts :
    - ETT (ETTh1, ETTh2, ETTm1, ETTm2)
    - Weather
    - Electricity (ECL)
    - Traffic
    - Exchange-Rate
    - ILI

Source : miroir Hugging Face Hub du dataset Autoformer (THUML).
Usage :
    python src/download_data.py                 # tous les petits datasets
    python src/download_data.py --all           # inclut Traffic et ECL (gros)
    python src/download_data.py --only ETTh1
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

ETT_BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"

DATASETS: dict[str, dict] = {
    "ETTh1":    {"url": f"{ETT_BASE}/ETTh1.csv", "size_mb": 2,  "heavy": False},
    "ETTh2":    {"url": f"{ETT_BASE}/ETTh2.csv", "size_mb": 2,  "heavy": False},
    "ETTm1":    {"url": f"{ETT_BASE}/ETTm1.csv", "size_mb": 10, "heavy": False},
    "ETTm2":    {"url": f"{ETT_BASE}/ETTm2.csv", "size_mb": 10, "heavy": False},
}


def download(name: str, url: str) -> None:
    out = RAW_DIR / f"{name}.csv"
    if out.exists():
        print(f"  [skip] {name:9s} déjà présent ({out.stat().st_size / 1e6:.1f} MB)")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [get ] {name:9s} <- {url}")
    try:
        urllib.request.urlretrieve(url, out)
        print(f"         OK ({out.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"         ERREUR : {e}", file=sys.stderr)
        if out.exists():
            out.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="inclut Traffic et ECL (gros fichiers)")
    parser.add_argument("--only", nargs="+", choices=list(DATASETS), help="ne télécharge que ces datasets")
    args = parser.parse_args()

    if args.only:
        targets = args.only
    elif args.all:
        targets = list(DATASETS)
    else:
        targets = [n for n, m in DATASETS.items() if not m["heavy"]]

    print(f"Destination : {RAW_DIR}")
    print(f"Datasets ciblés : {', '.join(targets)}\n")
    for name in targets:
        download(name, DATASETS[name]["url"])
    print("\nTerminé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
