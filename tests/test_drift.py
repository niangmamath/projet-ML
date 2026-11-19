"""Tests detection de drift par KS-test."""
from __future__ import annotations

import numpy as np
from drift import compute_input_stats, ks_drift
from predictor import FEATURES


def test_input_stats_structure(sample_window):
    s = compute_input_stats(sample_window)
    assert set(s.keys()) == set(FEATURES)
    for feat, d in s.items():
        assert set(d.keys()) == {"mean", "std", "min", "max"}
        assert d["min"] <= d["mean"] <= d["max"]


def test_ks_drift_returns_pvalues_per_feature(sample_window):
    pvals, flag = ks_drift(sample_window)
    assert set(pvals.keys()) == set(FEATURES)
    assert all(0.0 <= v <= 1.0 or np.isnan(v) for v in pvals.values())
    assert isinstance(flag, bool)


def test_ks_drift_flags_strongly_shifted_distribution():
    """Une fenetre tiree d'une distribution tres eloignee du train DOIT drifter."""
    rng = np.random.default_rng(0)
    # Train ETTh1 a des valeurs dans quelques dizaines ; on force une shift forte.
    bad = rng.normal(loc=1000.0, scale=0.1, size=(96, 7)).astype(np.float32)
    _, flag = ks_drift(bad)
    assert flag is True
