"""Tests for hyperparameter tuning."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from demand_ml.data import make_synthetic_demand
from demand_ml.tune import main


def test_tune_main_runs_without_error(tmp_path: object) -> None:
    """Test that tune.main() runs successfully on synthetic data."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock settings to use temp directories
        mock_settings = MagicMock()
        mock_settings.data_dir = tmpdir
        mock_settings.model_dir = str(Path(tmpdir) / "models")
        mock_settings.test_size = 0.2
        mock_settings.random_state = 42

        with (
            patch("demand_ml.tune.get_settings", return_value=mock_settings),
            patch("demand_ml.tune.load_uci_bike_sharing") as mock_load,
        ):
            # Use synthetic data instead of downloading
            df = make_synthetic_demand(n_rows=100)
            mock_load.return_value = df

            # Run tune.main()
            main()

            # Verify load_uci_bike_sharing was called
            mock_load.assert_called_once()


def test_tune_hyperparameters_produces_model() -> None:
    """Test that tune_hyperparameters returns a fitted model."""
    from demand_ml.model import tune_hyperparameters

    df = make_synthetic_demand(n_rows=100)
    model, metrics, best_params = tune_hyperparameters(df, test_size=0.2, random_state=42, cv=2)

    # Verify model is fitted
    assert hasattr(model, "predict")
    assert model is not None

    # Verify metrics are returned
    assert "mae_holdout" in metrics
    assert "rmse_holdout" in metrics
    assert "mae_cv_mean" in metrics
    assert "rmse_cv_mean" in metrics

    # Verify hyperparameters are returned
    assert isinstance(best_params, dict)
    assert "learning_rate" in best_params
    assert "max_depth" in best_params
    assert "l2_regularization" in best_params
