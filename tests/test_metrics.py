"""Tests for Prometheus metrics integration."""

from __future__ import annotations

from demand_ml.metrics import (
    ACTIVE_MODELS,
    MODEL_DRIFT_ALERTS,
    PREDICTION_ERRORS,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)


def test_request_count_metric_exists() -> None:
    """Verify REQUEST_COUNT metric is initialized."""
    assert REQUEST_COUNT is not None
    assert REQUEST_COUNT._name == "demand_ml_requests"


def test_request_latency_metric_exists() -> None:
    """Verify REQUEST_LATENCY metric is initialized."""
    assert REQUEST_LATENCY is not None
    assert REQUEST_LATENCY._name == "demand_ml_request_duration_seconds"


def test_prediction_errors_metric_exists() -> None:
    """Verify PREDICTION_ERRORS metric is initialized."""
    assert PREDICTION_ERRORS is not None
    assert PREDICTION_ERRORS._name == "demand_ml_prediction_errors"


def test_model_drift_alerts_metric_exists() -> None:
    """Verify MODEL_DRIFT_ALERTS metric is initialized."""
    assert MODEL_DRIFT_ALERTS is not None
    assert MODEL_DRIFT_ALERTS._name == "demand_ml_drift_alerts"


def test_active_models_gauge_exists() -> None:
    """Verify ACTIVE_MODELS gauge is initialized."""
    assert ACTIVE_MODELS is not None
    assert ACTIVE_MODELS._name == "demand_ml_active_models"
