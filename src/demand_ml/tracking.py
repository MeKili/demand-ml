"""MLflow experiment tracking for model training and evaluation."""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn


def log_training_run(
    model: Any,
    metrics: dict[str, float],
    *,
    experiment_name: str = "demand-forecasting",
    run_name: str | None = None,
    artifact_uri: str | None = None,
    params: dict[str, Any] | None = None,
    feature_importance: dict[str, float] | None = None,
) -> str:
    """Log model, metrics, hyperparameters, and feature importance to MLflow."""
    if artifact_uri:
        mlflow.set_tracking_uri(f"file:{artifact_uri}")

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        if params:
            for key, value in params.items():
                mlflow.log_param(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        if feature_importance:
            for feature, importance in feature_importance.items():
                mlflow.log_metric(f"feature_importance_{feature}", importance)
        mlflow.sklearn.log_model(model, "model")
        active = mlflow.active_run()
        assert active is not None
        run_id: str = active.info.run_id

    return run_id
