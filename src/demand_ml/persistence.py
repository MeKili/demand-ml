"""Model serialization: save, load, and predict with sklearn estimators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from demand_ml.features import FEATURE_COLUMNS


def save_model(model: Any, filepath: str | Path) -> None:
    """Save model to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(filepath: str | Path) -> Any:
    """Load model from disk."""
    return joblib.load(filepath)


def save_feature_bounds(bounds: dict[str, tuple[float, float]], filepath: str | Path) -> None:
    """Save feature bounds to JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(bounds, f)


def load_feature_bounds(filepath: str | Path) -> dict[str, tuple[float, float]]:
    """Load feature bounds from JSON file."""
    path = Path(filepath)
    with open(path) as f:
        raw = json.load(f)
    return {k: tuple(v) for k, v in raw.items()}


def predict(
    model: Any | str | Path, features: pd.DataFrame, *, use_feature_subset: bool = True
) -> pd.Series:
    """Generate demand predictions."""
    if isinstance(model, (str, Path)):
        model = load_model(model)

    if use_feature_subset:
        features = features[FEATURE_COLUMNS]

    return pd.Series(model.predict(features), index=features.index)
