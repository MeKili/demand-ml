"""Baseline demand-forecasting model and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split

from demand_ml.features import FEATURE_COLUMNS, TARGET_COLUMN, add_calendar_features


def train_model(
    df: pd.DataFrame, *, test_size: float = 0.2, random_state: int = 42, cv: int = 5
) -> tuple[Any, dict[str, float]]:
    """Return fitted estimator and metrics (holdout + cross-validation)."""
    featured = add_calendar_features(df)
    features = featured[FEATURE_COLUMNS]
    target = featured[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    model = HistGradientBoostingRegressor(random_state=random_state)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    mae_holdout = float(mean_absolute_error(y_test, preds))
    rmse_holdout = float(np.sqrt(mean_squared_error(y_test, preds)))

    cv_mae_scores = -cross_val_score(
        HistGradientBoostingRegressor(random_state=random_state),
        features,
        target,
        cv=cv,
        scoring="neg_mean_absolute_error",
    )
    cv_rmse_scores = np.sqrt(
        -cross_val_score(
            HistGradientBoostingRegressor(random_state=random_state),
            features,
            target,
            cv=cv,
            scoring="neg_mean_squared_error",
        )
    )

    metrics = {
        "mae_holdout": mae_holdout,
        "rmse_holdout": rmse_holdout,
        "mae_cv_mean": float(cv_mae_scores.mean()),
        "mae_cv_std": float(cv_mae_scores.std()),
        "rmse_cv_mean": float(cv_rmse_scores.mean()),
        "rmse_cv_std": float(cv_rmse_scores.std()),
    }
    return model, metrics
