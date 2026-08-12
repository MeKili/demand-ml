"""Smoke tests: the package imports and configuration loads with valid defaults."""

from demand_ml.config import Settings, get_settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert 0 < settings.test_size < 1
    assert settings.model_dir
    assert settings.mlflow_tracking_uri


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
