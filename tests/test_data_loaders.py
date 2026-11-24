"""Tests for data loaders (offline synthetic generator and UCI caching logic)."""

import tempfile
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from demand_ml.data import _download_with_retries, load_uci_bike_sharing, make_synthetic_demand


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
        urlopen_count = [0]

        csv_lines = [",".join(mock_uci_df.columns)]
        for _, row in mock_uci_df.iterrows():
            csv_lines.append(",".join(str(v) for v in row))
        csv_data = "\n".join(csv_lines)

        def mock_urlopen(*args: object, **kwargs: object) -> MagicMock:
            urlopen_count[0] += 1
            mock_response = MagicMock()
            mock_response.read.return_value = csv_data.encode("utf-8")
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = None
            return mock_response

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            df = load_uci_bike_sharing(data_dir=tmpdir)
            assert urlopen_count[0] == 1
            assert cache_file.exists()
            assert len(df) == 3
            assert set(df.columns) == {"timestamp", "temp", "humidity", "windspeed", "count"}

            df2 = load_uci_bike_sharing(data_dir=tmpdir)
            assert urlopen_count[0] == 1
            pd.testing.assert_frame_equal(df, df2)


def test_load_uci_bike_sharing_missing_columns() -> None:
    incomplete_df = pd.DataFrame(
        {
            "dteday": ["2011-01-01"],
            "hr": [0],
            "temp": [0.24],
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_lines = [",".join(incomplete_df.columns)]
        for _, row in incomplete_df.iterrows():
            csv_lines.append(",".join(str(v) for v in row))
        csv_data = "\n".join(csv_lines)

        def mock_urlopen(*args: object, **kwargs: object) -> MagicMock:
            mock_response = MagicMock()
            mock_response.read.return_value = csv_data.encode("utf-8")
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = None
            return mock_response

        with (
            patch("urllib.request.urlopen", side_effect=mock_urlopen),
            pytest.raises(ValueError, match="Missing required columns"),
        ):
            load_uci_bike_sharing(data_dir=tmpdir)


def test_download_with_retries_success() -> None:
    csv_data = "a,b\n1,2\n3,4"
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = csv_data.encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        result = _download_with_retries("http://example.com/data.csv")
        assert result == csv_data
        assert mock_urlopen.call_count == 1


def test_download_with_retries_exponential_backoff() -> None:
    csv_data = "a,b\n1,2"
    with patch("urllib.request.urlopen") as mock_urlopen, patch("time.sleep") as mock_sleep:
        mock_response = MagicMock()
        mock_response.read.return_value = csv_data.encode("utf-8")
        mock_response.__enter__.return_value = mock_response

        mock_urlopen.side_effect = [
            urllib.error.URLError("Connection timeout"),
            urllib.error.URLError("Connection timeout"),
            mock_response,
        ]

        result = _download_with_retries("http://example.com/data.csv", max_retries=3)
        assert result == csv_data
        assert mock_urlopen.call_count == 3
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)


def test_download_with_retries_all_fail() -> None:
    with patch("urllib.request.urlopen") as mock_urlopen, patch("time.sleep"):
        mock_urlopen.side_effect = urllib.error.URLError("Connection failed")

        with pytest.raises(urllib.error.URLError):
            _download_with_retries("http://example.com/data.csv", max_retries=2)
