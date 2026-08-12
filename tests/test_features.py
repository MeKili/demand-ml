"""Tests for calendar feature engineering."""

from datetime import datetime

import pandas as pd

from demand_ml.features import add_calendar_features


def test_add_calendar_features() -> None:
    # 2026-01-03 is a Saturday.
    df = pd.DataFrame({"timestamp": [datetime(2026, 1, 3, 14, 0)], "count": [10]})
    out = add_calendar_features(df)
    assert out.loc[0, "hour"] == 14
    assert out.loc[0, "dayofweek"] == 5
    assert out.loc[0, "is_weekend"] == 1


def test_features_do_not_mutate_input() -> None:
    df = pd.DataFrame({"timestamp": [datetime(2026, 6, 1, 9, 0)], "count": [3]})
    add_calendar_features(df)
    assert "hour" not in df.columns
