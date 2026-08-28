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
) -> str:
    """Log model and metrics to MLflow; return the run ID."""
    if artifact_uri:
        mlflow.set_tracking_uri(f"file:{artifact_uri}")

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.sklearn.log_model(model, "model")
        active = mlflow.active_run()
        assert active is not None
        run_id: str = active.info.run_id

    return run_id
