"""Train the baseline model on synthetic data and report holdout metrics."""

from __future__ import annotations

from demand_ml.config import get_settings
from demand_ml.data import make_synthetic_demand
from demand_ml.model import train_model


def main() -> None:
    """Generate data, train the baseline model, and print the metrics."""
    settings = get_settings()
    df = make_synthetic_demand()
    _, metrics = train_model(df, test_size=settings.test_size, random_state=settings.random_state)
    print(f"MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}")


if __name__ == "__main__":
    main()
