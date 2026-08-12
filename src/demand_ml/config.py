"""Configuration for the demand-ml training and serving pipeline."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings, loaded from the environment or an optional ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: str = "data"
    model_dir: str = "models"
    mlflow_tracking_uri: str = "file:./mlruns"
    test_size: float = 0.2
    random_state: int = 42


@lru_cache
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance (configuration is read once)."""
    return Settings()
