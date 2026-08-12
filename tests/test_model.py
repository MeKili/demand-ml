"""Tests for baseline model training (runs on a small synthetic sample)."""

from demand_ml.data import make_synthetic_demand
from demand_ml.model import train_model


def test_train_model_returns_finite_metrics() -> None:
    df = make_synthetic_demand(n_rows=200, seed=1)
    model, metrics = train_model(df, test_size=0.25, random_state=1)

    assert hasattr(model, "predict")
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["mae"] < 1_000_000
