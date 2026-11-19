"""Fixtures partagees pour les tests.

Important :
    - Les tests ne doivent JAMAIS toucher a la vraie DB Postgres : on efface
      DATABASE_URL/DATA_POSTGRESQL en session-level autouse.
    - sys.path est amenage pour importer huggingface_space.* et src.* sans
      modifier le code du projet.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent

# Ordre : ROOT en 1er (gagne sur huggingface_space/src stub)
sys.path.insert(0, str(ROOT / "huggingface_space"))
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True, scope="session")
def _no_real_db(monkeypatch_session):
    """Garantit qu'aucun test ne se connecte a Neon par accident.

    On set a "" (pas delenv) car api.py appelle load_dotenv() au import, qui
    repeuplerait sinon les vars depuis le .env a la racine. Avec override=False
    (defaut), load_dotenv ne remplace pas une var deja presente -> "" gagne.
    """
    monkeypatch_session.setenv("DATABASE_URL", "")
    monkeypatch_session.setenv("DATA_POSTGRESQL", "")


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Monkeypatch de scope session (celui par defaut est function-scoped)."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture
def sample_window() -> np.ndarray:
    """Fenetre d'entree valide (96, 7) remplie de valeurs raisonnables."""
    rng = np.random.default_rng(0)
    return rng.normal(loc=10.0, scale=3.0, size=(96, 7)).astype(np.float32)
