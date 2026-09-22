"""Tests for baseline model training (runs on a small synthetic sample)."""

from demand_ml.data import make_synthetic_demand
from demand_ml.features import FEATURE_COLUMNS
from demand_ml.model import compute_feature_importance, train_model, tune_hyperparameters


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


def test_compute_feature_importance_returns_dict() -> None:
    df = make_synthetic_demand(n_rows=200, seed=1)
    model, _metrics = train_model(df, test_size=0.25, random_state=1)

    importance = compute_feature_importance(model, FEATURE_COLUMNS)
    assert isinstance(importance, dict)
    for feature, score in importance.items():
        assert feature in FEATURE_COLUMNS
        assert isinstance(score, float)
        assert score >= 0
