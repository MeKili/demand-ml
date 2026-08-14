"""Tests for data loaders (offline synthetic generator and UCI caching logic)."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from demand_ml.data import load_uci_bike_sharing, make_synthetic_demand


def test_make_synthetic_demand_reproducible() -> None:
    df1 = make_synthetic_demand(n_rows=100, seed=123)
    df2 = make_synthetic_demand(n_rows=100, seed=123)
    pd.testing.assert_frame_equal(df1, df2)


def test_make_synthetic_demand_shape() -> None:
    n = 250
    df = make_synthetic_demand(n_rows=n)
    assert len(df) == n
    assert set(df.columns) == {"timestamp", "temp", "humidity", "windspeed", "count"}


def test_load_uci_bike_sharing_caches_locally() -> None:
    mock_uci_df = pd.DataFrame(
        {
            "dteday": ["2011-01-01"] * 3,
            "hr": [0, 1, 2],
            "temp": [0.24, 0.22, 0.20],
            "hum": [0.81, 0.80, 0.82],
            "windspeed": [0.0, 0.0, 0.0],
            "cnt": [16, 40, 32],
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "bike_sharing.csv"
        call_count = [0]

        def mock_read_csv(path: str | Path, *args: object, **kwargs: object) -> pd.DataFrame:
            if str(path).startswith("http"):
                call_count[0] += 1
            return mock_uci_df

        with patch("pandas.read_csv", side_effect=mock_read_csv):
            df = load_uci_bike_sharing(data_dir=tmpdir)
            assert call_count[0] == 1
            assert cache_file.exists()
            assert len(df) == 3
            assert set(df.columns) == {"timestamp", "temp", "humidity", "windspeed", "count"}

            df2 = load_uci_bike_sharing(data_dir=tmpdir)
            assert call_count[0] == 1
            pd.testing.assert_frame_equal(df, df2)
