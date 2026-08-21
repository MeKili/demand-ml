"""Tests for feature drift monitoring."""

import numpy as np
import pandas as pd

from demand_ml.monitoring import compute_psi, detect_feature_drift


def test_compute_psi_identical_distributions() -> None:
    """PSI should be low for the same distribution."""
    rng = np.random.RandomState(42)
    ref = pd.Series(rng.normal(10, 2, 500))
    cur = pd.Series(rng.normal(10, 2, 500))

    psi = compute_psi(ref, cur)
    assert psi < 0.15


def test_compute_psi_shifted_distribution() -> None:
    """PSI should be higher when distributions differ."""
    ref = pd.Series(np.random.normal(10, 2, 500))
    cur = pd.Series(np.random.normal(15, 2, 500))

    psi = compute_psi(ref, cur)
    assert psi > 0.5


def test_detect_feature_drift_no_drift() -> None:
    """Should not flag drift when distributions are stable."""
    data = np.random.RandomState(42).normal(20, 5, 1000)
    ref_df = pd.DataFrame({"feature_a": data[:500], "feature_b": data[:500] * 2})
    cur_df = pd.DataFrame({"feature_a": data[500:], "feature_b": data[500:] * 2})

    result = detect_feature_drift(ref_df, cur_df, psi_threshold=0.1)
    assert result["has_drift"] is False
    assert len(result["psi_by_feature"]) == 2


def test_detect_feature_drift_with_drift() -> None:
    """Should flag drift when a feature distribution shifts significantly."""
    ref_df = pd.DataFrame(
        {"temp": np.random.normal(15, 5, 200), "humidity": np.random.uniform(30, 90, 200)}
    )
    cur_df = pd.DataFrame(
        {"temp": np.random.normal(5, 5, 200), "humidity": np.random.uniform(30, 90, 200)}
    )

    result = detect_feature_drift(ref_df, cur_df, psi_threshold=0.1)
    assert result["has_drift"] is True
    assert result["psi_by_feature"]["temp"] > 0.1
