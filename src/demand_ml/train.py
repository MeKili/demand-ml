"""Train the baseline model on real UCI Bike Sharing data and report holdout metrics."""

from __future__ import annotations

from demand_ml.config import get_settings
from demand_ml.data import load_uci_bike_sharing
from demand_ml.model import train_model
from demand_ml.persistence import save_model
from demand_ml.tracking import log_training_run


def main() -> None:
    """Load real data, train the baseline model, and save it."""
    settings = get_settings()
    df = load_uci_bike_sharing(settings.data_dir)
    model, metrics = train_model(
        df, test_size=settings.test_size, random_state=settings.random_state
    )
    print(f"MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}")

    model_path = f"{settings.model_dir}/baseline.joblib"
    save_model(model, model_path)
    print(f"Model saved to {model_path}")

    run_id = log_training_run(
        model, metrics, experiment_name="demand-forecasting", run_name="baseline"
    )
    print(f"MLflow run ID: {run_id}")


if __name__ == "__main__":
    main()
