"""Tests for FastAPI serving endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingRegressor

from demand_ml.data import make_synthetic_demand
from demand_ml.features import FEATURE_COLUMNS, add_calendar_features
from demand_ml.serving import health, predict_batch, predict_demand


def test_health() -> None:
    """Test /health endpoint returns ok status."""

    @asynccontextmanager
    async def test_lifespan(app: FastAPI) -> Any:  # type: ignore
        app.state.model = None
        yield

    app = FastAPI(lifespan=test_lifespan)
    app.get("/health")(health)

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "message" in data


def test_predict_with_model() -> None:
    """Test /predict endpoint with a trained model."""
    df = make_synthetic_demand(n_rows=100, seed=42)
    featured = add_calendar_features(df)
    features = featured[FEATURE_COLUMNS]
    target = featured["count"]

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(features, target)

    @asynccontextmanager
    async def test_lifespan(app: FastAPI) -> Any:  # type: ignore
        app.state.model = model
        yield

    app = FastAPI(lifespan=test_lifespan)
    app.post("/predict")(predict_demand)

    with TestClient(app) as client:
        request_payload = {"features": [15.5, 60.0, 10.0, 9, 3, 8, 1]}
        response = client.post("/predict", json=request_payload)

        assert response.status_code == 200
        data = response.json()
        assert "predicted_demand" in data
        assert isinstance(data["predicted_demand"], float)
        assert data["predicted_demand"] > 0


def test_predict_missing_model() -> None:
    """Test /predict endpoint with missing model."""

    @asynccontextmanager
    async def test_lifespan(app: FastAPI) -> Any:  # type: ignore
        app.state.model = None
        yield

    app = FastAPI(lifespan=test_lifespan)
    app.post("/predict")(predict_demand)

    with TestClient(app) as client:
        request_payload = {"features": [15.5, 60.0, 10.0, 9, 3, 8, 1]}
        response = client.post("/predict", json=request_payload)
        assert response.status_code == 503


def test_predict_invalid_features() -> None:
    """Test /predict endpoint with wrong number of features."""
    df = make_synthetic_demand(n_rows=100, seed=42)
    featured = add_calendar_features(df)
    features = featured[FEATURE_COLUMNS]
    target = featured["count"]

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(features, target)

    @asynccontextmanager
    async def test_lifespan(app: FastAPI) -> Any:  # type: ignore
        app.state.model = model
        yield

    app = FastAPI(lifespan=test_lifespan)
    app.post("/predict")(predict_demand)

    with TestClient(app) as client:
        request_payload = {"features": [15.5, 60.0]}
        response = client.post("/predict", json=request_payload)
        assert response.status_code == 400


def test_predict_batch_with_model() -> None:
    """Test /predict-batch endpoint with multiple feature sets."""
    df = make_synthetic_demand(n_rows=100, seed=42)
    featured = add_calendar_features(df)
    features = featured[FEATURE_COLUMNS]
    target = featured["count"]

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(features, target)

    @asynccontextmanager
    async def test_lifespan(app: FastAPI) -> Any:  # type: ignore
        app.state.model = model
        yield

    app = FastAPI(lifespan=test_lifespan)
    app.post("/predict-batch")(predict_batch)

    with TestClient(app) as client:
        request_payload = {
            "features_list": [
                [15.5, 60.0, 10.0, 9, 3, 8, 1],
                [16.0, 65.0, 11.0, 10, 4, 9, 0],
            ]
        }
        response = client.post("/predict-batch", json=request_payload)

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        assert all(isinstance(p, float) for p in data["predictions"])
        assert all(p > 0 for p in data["predictions"])


def test_predict_batch_empty_list() -> None:
    """Test /predict-batch endpoint with empty features_list."""
    df = make_synthetic_demand(n_rows=100, seed=42)
    featured = add_calendar_features(df)
    features = featured[FEATURE_COLUMNS]
    target = featured["count"]

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(features, target)

    @asynccontextmanager
    async def test_lifespan(app: FastAPI) -> Any:  # type: ignore
        app.state.model = model
        yield

    app = FastAPI(lifespan=test_lifespan)
    app.post("/predict-batch")(predict_batch)

    with TestClient(app) as client:
        request_payload = {"features_list": []}
        response = client.post("/predict-batch", json=request_payload)
        assert response.status_code == 400


def test_predict_batch_invalid_features() -> None:
    """Test /predict-batch endpoint with wrong feature counts."""
    df = make_synthetic_demand(n_rows=100, seed=42)
    featured = add_calendar_features(df)
    features = featured[FEATURE_COLUMNS]
    target = featured["count"]

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(features, target)

    @asynccontextmanager
    async def test_lifespan(app: FastAPI) -> Any:  # type: ignore
        app.state.model = model
        yield

    app = FastAPI(lifespan=test_lifespan)
    app.post("/predict-batch")(predict_batch)

    with TestClient(app) as client:
        request_payload = {
            "features_list": [
                [15.5, 60.0, 10.0, 9, 3, 8, 1],
                [16.0, 65.0],
            ]
        }
        response = client.post("/predict-batch", json=request_payload)
        assert response.status_code == 400
