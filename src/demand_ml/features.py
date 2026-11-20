"""Feature engineering for demand forecasting."""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = ["temp", "humidity", "windspeed", "hour", "dayofweek", "month", "is_weekend"]
"""Features used in model training and prediction."""

TARGET_COLUMN = "count"
"""Target variable: hourly bike demand count."""


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from timestamp."""
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    return out


def compute_feature_bounds(features: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Compute min/max bounds for each feature."""
    bounds = {}
    for col in FEATURE_COLUMNS:
        if col in features.columns:
            bounds[col] = (float(features[col].min()), float(features[col].max()))
    return bounds
