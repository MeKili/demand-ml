"""Baseline demand-forecasting model and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from demand_ml.features import FEATURE_COLUMNS, TARGET_COLUMN, add_calendar_features


def train_model(
    df: pd.DataFrame, *, test_size: float = 0.2, random_state: int = 42
) -> tuple[Any, dict[str, float]]:
    """Train a gradient-boosting regressor and return it with holdout metrics.

    Returns the fitted estimator and a dict of ``mae`` and ``rmse`` on the holdout.
    """
    featured = add_calendar_features(df)
    features = featured[FEATURE_COLUMNS]
    target = featured[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    model = HistGradientBoostingRegressor(random_state=random_state)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
    }
    return model, metrics
