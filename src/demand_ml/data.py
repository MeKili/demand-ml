"""Deterministic synthetic bike-share demand data for offline dev and tests.

A real loader for the public UCI Bike Sharing dataset is a planned addition; the
synthetic generator keeps development and CI fully offline and reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_demand(n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible hourly demand dataset.

    Columns: ``timestamp``, ``temp``, ``humidity``, ``windspeed``, ``count``.
    Demand rises around commute hours and with milder temperatures.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-01", periods=n_rows, freq="h")
    temp = rng.normal(15.0, 8.0, n_rows)
    humidity = rng.uniform(30.0, 90.0, n_rows)
    windspeed = rng.uniform(0.0, 30.0, n_rows)

    hour = timestamps.hour.to_numpy()
    seasonal = 30.0 * np.sin((hour - 8.0) / 24.0 * 2.0 * np.pi)
    base = 50.0 + seasonal + 2.0 * temp - 0.3 * humidity
    count = np.clip(base + rng.normal(0.0, 10.0, n_rows), 0.0, None).round().astype(int)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temp": temp,
            "humidity": humidity,
            "windspeed": windspeed,
            "count": count,
        }
    )
