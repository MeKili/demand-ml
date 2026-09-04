"""Bike-share demand data loaders: UCI dataset and synthetic generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic_demand(n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible synthetic hourly demand dataset."""
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


def load_uci_bike_sharing(data_dir: str = "data") -> pd.DataFrame:
    """Load and cache the UCI Bike Sharing dataset."""
    cache_path = Path(data_dir) / "bike_sharing.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/hour.csv"
        df = pd.read_csv(url)
        df.to_csv(cache_path, index=False)

    df["dteday"] = pd.to_datetime(df["dteday"])
    df["timestamp"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")

    normalized = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "temp": df["temp"] * 41.0,
            "humidity": df["hum"],
            "windspeed": df["windspeed"] * 67.0,
            "count": df["cnt"],
        }
    )
    return normalized
