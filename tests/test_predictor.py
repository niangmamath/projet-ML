"""Tests du module predictor (catalogue + resolver + inference)."""
from __future__ import annotations

import numpy as np
import pytest
from predictor import (
    FEATURES,
    INPUT_LEN,
    N_FEATURES,
    _resolve_checkpoint,
    available_checkpoints,
    predict_window,
)


def test_features_order_is_canonical():
    assert FEATURES == ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    assert N_FEATURES == 7
    assert INPUT_LEN == 96


def test_catalogue_covers_all_24_combinations():
    c = available_checkpoints()
    assert set(c.keys()) == {
        "DLinear", "Transformer", "PatchTST", "Informer", "Autoformer", "FEDformer"
    }
    for h_list in c.values():
        assert h_list == [96, 192, 336, 720]


def test_resolver_returns_existing_file():
    """Avec checkpoints locaux presents, le resolver ne doit pas aller sur HF Hub."""
    path = _resolve_checkpoint("PatchTST", 96)
    assert path.is_file()
    assert path.name == "PatchTST_ETTh1_h96.pt"


@pytest.mark.slow
def test_predict_window_shapes(sample_window):
    """Smoke test d'inference (charge un checkpoint)."""
    out = predict_window("DLinear", 96, sample_window)      # DLinear = le plus leger
    assert out["prediction"].shape == (96, N_FEATURES)
    assert out["context"].shape == (INPUT_LEN, N_FEATURES)
    assert np.isfinite(out["prediction"]).all()


def test_predict_window_rejects_bad_shape():
    with pytest.raises(ValueError):
        predict_window("DLinear", 96, np.zeros((50, 7), dtype=np.float32))
    with pytest.raises(ValueError):
        predict_window("DLinear", 96, np.zeros((96, 5), dtype=np.float32))
