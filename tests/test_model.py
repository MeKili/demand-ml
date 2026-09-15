"""Tests for baseline model training (runs on a small synthetic sample)."""

from demand_ml.data import make_synthetic_demand
from demand_ml.model import train_model, tune_hyperparameters


def test_train_model_returns_finite_metrics() -> None:
    df = make_synthetic_demand(n_rows=200, seed=1)
    model, metrics = train_model(df, test_size=0.25, random_state=1)

    assert hasattr(model, "predict")
    assert metrics["mae_holdout"] >= 0
    assert metrics["rmse_holdout"] >= 0
    assert metrics["mae_holdout"] < 1_000_000
    assert metrics["mae_cv_mean"] >= 0
    assert metrics["rmse_cv_mean"] >= 0
    assert metrics["mae_cv_std"] >= 0
    assert metrics["rmse_cv_std"] >= 0


def test_tune_hyperparameters_returns_best_params() -> None:
    df = make_synthetic_demand(n_rows=200, seed=1)
    model, metrics, best_params = tune_hyperparameters(df, test_size=0.25, random_state=1, cv=2)

    assert hasattr(model, "predict")
    assert isinstance(best_params, dict)
    assert "learning_rate" in best_params
    assert "max_depth" in best_params
    assert "l2_regularization" in best_params
    assert metrics["mae_holdout"] >= 0
    assert metrics["rmse_holdout"] >= 0
    assert metrics["mae_cv_mean"] >= 0
    assert metrics["rmse_cv_mean"] >= 0
