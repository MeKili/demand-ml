"""Tests for MLflow experiment tracking."""

from __future__ import annotations

from unittest import mock

from demand_ml.data import make_synthetic_demand
from demand_ml.model import train_model
from demand_ml.tracking import log_training_run


def test_log_training_run_logs_metrics() -> None:
    """Verify that log_training_run logs metrics to MLflow."""
    df = make_synthetic_demand(n_rows=200, seed=1)
    model, metrics = train_model(df, test_size=0.25, random_state=1)

    with mock.patch("demand_ml.tracking.mlflow") as mock_mlflow:
        mock_run = mock.MagicMock()
        mock_run.info.run_id = "test-run-123"
        mock_mlflow.active_run.return_value = mock_run
        mock_mlflow.start_run.return_value.__enter__.return_value = None
        mock_mlflow.start_run.return_value.__exit__.return_value = None

        run_id = log_training_run(model, metrics, experiment_name="test-exp")

        assert run_id == "test-run-123"
        mock_mlflow.set_experiment.assert_called_once_with("test-exp")
        mock_mlflow.start_run.assert_called_once()


def test_log_training_run_logs_sklearn_model() -> None:
    """Verify that log_training_run logs the sklearn model."""
    df = make_synthetic_demand(n_rows=200, seed=1)
    model, metrics = train_model(df, test_size=0.25, random_state=1)

    with mock.patch("demand_ml.tracking.mlflow") as mock_mlflow:
        mock_run = mock.MagicMock()
        mock_run.info.run_id = "test-run-456"
        mock_mlflow.active_run.return_value = mock_run
        mock_mlflow.start_run.return_value.__enter__.return_value = None
        mock_mlflow.start_run.return_value.__exit__.return_value = None

        log_training_run(model, metrics)

        mock_mlflow.sklearn.log_model.assert_called_once_with(model, "model")


def test_log_training_run_with_run_name() -> None:
    """Verify that log_training_run respects run_name parameter."""
    df = make_synthetic_demand(n_rows=200, seed=1)
    model, metrics = train_model(df, test_size=0.25, random_state=1)

    with mock.patch("demand_ml.tracking.mlflow") as mock_mlflow:
        mock_run = mock.MagicMock()
        mock_run.info.run_id = "test-run-789"
        mock_mlflow.active_run.return_value = mock_run
        mock_mlflow.start_run.return_value.__enter__.return_value = None
        mock_mlflow.start_run.return_value.__exit__.return_value = None

        log_training_run(model, metrics, run_name="my-baseline")

        mock_mlflow.start_run.assert_called_once_with(run_name="my-baseline")
