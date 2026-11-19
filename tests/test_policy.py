"""Tests des regles de decision MLOps (retrain / promote)."""
from __future__ import annotations

from pipelines.policy import decide_promote, decide_retrain


# -----------------------------------------------------------------------------
# decide_retrain
# -----------------------------------------------------------------------------
def test_retrain_manual_always_wins():
    d = decide_retrain("PatchTST", 96, drift_stats={}, manual=True)
    assert d.should_retrain is True
    assert d.reason == "manual"


def test_retrain_no_data_in_db():
    d = decide_retrain("PatchTST", 96, drift_stats={})
    assert d.should_retrain is False
    assert d.reason == "no_data"


def test_retrain_insufficient_samples():
    d = decide_retrain(
        "PatchTST", 96,
        drift_stats={("PatchTST", 96): {"n": 10, "drift_rate": 0.9}},
    )
    assert d.should_retrain is False
    assert "insufficient_samples" in d.reason


def test_retrain_drift_below_threshold():
    d = decide_retrain(
        "PatchTST", 96,
        drift_stats={("PatchTST", 96): {"n": 500, "drift_rate": 0.05}},
    )
    assert d.should_retrain is False
    assert "drift_ok" in d.reason


def test_retrain_drift_exceeded():
    d = decide_retrain(
        "PatchTST", 96,
        drift_stats={("PatchTST", 96): {"n": 500, "drift_rate": 0.35}},
    )
    assert d.should_retrain is True
    assert "drift_exceeded" in d.reason


# -----------------------------------------------------------------------------
# decide_promote
# -----------------------------------------------------------------------------
def test_promote_bootstrap_when_no_baseline():
    d = decide_promote(mse_before=None, mse_after=0.5)
    assert d.should_promote is True
    assert "bootstrap" in d.reason


def test_promote_meaningful_improvement():
    d = decide_promote(mse_before=1.00, mse_after=0.90)       # -10%
    assert d.should_promote is True


def test_promote_no_significant_improvement():
    d = decide_promote(mse_before=1.00, mse_after=0.995)      # -0.5%
    assert d.should_promote is False


def test_promote_regression():
    d = decide_promote(mse_before=1.00, mse_after=1.20)       # +20%
    assert d.should_promote is False
