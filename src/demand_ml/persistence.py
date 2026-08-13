"""Model serialization: save, load, and predict with sklearn estimators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from demand_ml.features import FEATURE_COLUMNS


def save_model(model: Any, filepath: str | Path) -> None:
    """Persist a fitted sklearn estimator to disk using joblib."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(filepath: str | Path) -> Any:
    """Load a fitted sklearn estimator from disk."""
    return joblib.load(filepath)


def predict(
    model: Any | str | Path, features: pd.DataFrame, *, use_feature_subset: bool = True
) -> pd.Series:
    """Predict demand on features, loading the model if a filepath is given.

    Args:
        model: A fitted estimator, or a filepath to load one from.
        features: DataFrame with columns matching FEATURE_COLUMNS.
        use_feature_subset: If True, extract only FEATURE_COLUMNS from the input.
            Set to False if features are already aligned.

    Returns:
        Predicted demand as a Series.
    """
    if isinstance(model, (str, Path)):
        model = load_model(model)

    if use_feature_subset:
        features = features[FEATURE_COLUMNS]

    return pd.Series(model.predict(features), index=features.index)
