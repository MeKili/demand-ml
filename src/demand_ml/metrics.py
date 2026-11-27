"""Prometheus metrics for model serving and monitoring."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "demand_ml_requests_total",
    "Total requests to demand-ml service",
    ["endpoint"],
)
"""Counter: Total requests by endpoint."""

REQUEST_LATENCY = Histogram(
    "demand_ml_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
"""Histogram: Request latency by endpoint."""

PREDICTION_ERRORS = Counter(
    "demand_ml_prediction_errors_total",
    "Total prediction errors (validation, model failures)",
    ["error_type"],
)
"""Counter: Prediction errors by type (bounds_violation, model_error, etc)."""

MODEL_DRIFT_ALERTS = Counter(
    "demand_ml_drift_alerts_total",
    "Total drift alerts triggered",
    ["severity"],
)
"""Counter: Drift alerts by severity (warning, critical)."""

ACTIVE_MODELS = Gauge(
    "demand_ml_active_models",
    "Number of active model versions in service",
)
"""Gauge: Count of active models."""
