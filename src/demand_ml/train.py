"""Train the baseline model on synthetic data and report holdout metrics."""

from __future__ import annotations

from demand_ml.config import get_settings
from demand_ml.data import make_synthetic_demand
from demand_ml.model import train_model
from demand_ml.persistence import save_model


def main() -> None:
    """Generate data, train the baseline model, and optionally save it."""
    settings = get_settings()
    df = make_synthetic_demand()
    model, metrics = train_model(
        df, test_size=settings.test_size, random_state=settings.random_state
    )
    print(f"MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}")

    model_path = f"{settings.model_dir}/baseline.joblib"
    save_model(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
