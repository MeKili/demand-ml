"""Hyperparameter tuning for the demand-forecasting model."""

from __future__ import annotations

from demand_ml.config import get_settings
from demand_ml.data import load_uci_bike_sharing
from demand_ml.features import FEATURE_COLUMNS
from demand_ml.model import compute_feature_importance, tune_hyperparameters
from demand_ml.persistence import save_model
from demand_ml.tracking import log_training_run


def main() -> None:
    """Tune hyperparameters on real data, save tuned model, and log to MLflow."""
    settings = get_settings()
    df = load_uci_bike_sharing(settings.data_dir)
    model, metrics, best_params = tune_hyperparameters(
        df, test_size=settings.test_size, random_state=settings.random_state
    )
    print(f"Holdout: MAE={metrics['mae_holdout']:.2f} RMSE={metrics['rmse_holdout']:.2f}")
    print(
        f"CV: MAE={metrics['mae_cv_mean']:.2f}±{metrics['mae_cv_std']:.2f} "
        f"RMSE={metrics['rmse_cv_mean']:.2f}±{metrics['rmse_cv_std']:.2f}"
    )
    print(f"Best hyperparameters: {best_params}")

    model_path = f"{settings.model_dir}/tuned.joblib"
    save_model(model, model_path)
    print(f"Tuned model saved to {model_path}")

    feature_importance = compute_feature_importance(model, FEATURE_COLUMNS)
    run_id = log_training_run(
        model,
        metrics,
        experiment_name="demand-forecasting",
        run_name="tuned",
        params=best_params,
        feature_importance=feature_importance,
    )
    print(f"MLflow run ID: {run_id}")


if __name__ == "__main__":
    main()
