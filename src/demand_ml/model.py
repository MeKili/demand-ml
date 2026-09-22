"""Baseline demand-forecasting model and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split

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

    cv_results = cross_validate(
        HistGradientBoostingRegressor(random_state=random_state),
        features,
        target,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
        },
    )
    cv_mae_scores = -cv_results["test_mae"]
    cv_rmse_scores = np.sqrt(-cv_results["test_mse"])

    metrics = {
        "mae_holdout": mae_holdout,
        "rmse_holdout": rmse_holdout,
        "mae_cv_mean": float(cv_mae_scores.mean()),
        "mae_cv_std": float(cv_mae_scores.std()),
        "rmse_cv_mean": float(cv_rmse_scores.mean()),
        "rmse_cv_std": float(cv_rmse_scores.std()),
    }
    return model, metrics


def compute_feature_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    """Return dict mapping feature names to their importance scores."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return dict(zip(feature_names, map(float, importances), strict=True))
    return {}


def tune_hyperparameters(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    cv: int = 3,
) -> tuple[Any, dict[str, float], dict[str, Any]]:
    """Return tuned estimator, metrics, and best hyperparameters via light grid search."""
    featured = add_calendar_features(df)
    features = featured[FEATURE_COLUMNS]
    target = featured[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )

    param_grid = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5, 7],
        "l2_regularization": [0.0, 0.1],
    }
    grid = GridSearchCV(
        HistGradientBoostingRegressor(random_state=random_state),
        param_grid,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    grid.fit(x_train, y_train)
    model = grid.best_estimator_
    preds = model.predict(x_test)

    mae_holdout = float(mean_absolute_error(y_test, preds))
    rmse_holdout = float(np.sqrt(mean_squared_error(y_test, preds)))

    cv_results = cross_validate(
        model,
        features,
        target,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
        },
    )
    cv_mae_scores = -cv_results["test_mae"]
    cv_rmse_scores = np.sqrt(-cv_results["test_mse"])

    metrics = {
        "mae_holdout": mae_holdout,
        "rmse_holdout": rmse_holdout,
        "mae_cv_mean": float(cv_mae_scores.mean()),
        "mae_cv_std": float(cv_mae_scores.std()),
        "rmse_cv_mean": float(cv_rmse_scores.mean()),
        "rmse_cv_std": float(cv_rmse_scores.std()),
    }
    return model, metrics, grid.best_params_
