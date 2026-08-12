"""Feature engineering for demand forecasting."""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = ["temp", "humidity", "windspeed", "hour", "dayofweek", "month", "is_weekend"]
TARGET_COLUMN = "count"


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with calendar features derived from ``timestamp``."""
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    return out
