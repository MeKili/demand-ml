"""FastAPI application for model serving and health checks."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from demand_ml.config import get_settings
from demand_ml.features import FEATURE_COLUMNS
from demand_ml.monitoring import detect_feature_drift
from demand_ml.persistence import load_model, predict

logger = logging.getLogger(__name__)


def _validate_feature_count(feature_list: Any, single: bool = True) -> None:
    """Validate feature count matches expected dimensions."""
    if single:
        if len(feature_list) != len(FEATURE_COLUMNS):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(FEATURE_COLUMNS)} features, got {len(feature_list)}",
            )
    else:
        if any(len(f) != len(FEATURE_COLUMNS) for f in feature_list):
            raise HTTPException(
                status_code=400,
                detail=f"All feature sets must have {len(FEATURE_COLUMNS)} features",
            )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model at startup, cleanup resources at shutdown."""
    settings = get_settings()
    model_path = Path(settings.model_dir) / settings.model_filename
    if model_path.exists():
        app.state.model = load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
    else:
        app.state.model = None
        logger.warning(f"Model not found at {model_path}")
    yield
    logger.info("Shutting down demand-ml service")


app = FastAPI(title="demand-ml", version="0.1.0", lifespan=lifespan)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    message: str


class PredictRequest(BaseModel):
    """Prediction request with feature values."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"features": [15.5, 60, 10, 9, 3, 8, 1]}}
    )

    features: list[float]


class PredictResponse(BaseModel):
    """Prediction response with forecasted demand."""

    predicted_demand: float


class BatchPredictRequest(BaseModel):
    """Batch prediction request with multiple feature sets."""

    features_list: list[list[float]]


class BatchPredictResponse(BaseModel):
    """Batch prediction response with multiple forecasts."""

    predictions: list[float]


class DriftCheckRequest(BaseModel):
    """Drift check request with reference and current feature data."""

    reference_features: list[list[float]]
    current_features: list[list[float]]
    psi_threshold: float = 0.1


class DriftCheckResponse(BaseModel):
    """Drift detection response indicating if drift was found."""

    has_drift: bool
    psi_by_feature: dict[str, float]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check if the service is running."""
    logger.debug("Health check requested")
    return HealthResponse(status="ok", message="demand-ml service is running")


@app.post("/predict", response_model=PredictResponse)
def predict_demand(request: PredictRequest, req: Request) -> PredictResponse:
    """Return demand forecast for given features."""
    model = getattr(req.app.state, "model", None)
    if model is None:
        logger.error("Prediction requested but model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded at startup")
    try:
        _validate_feature_count(request.features, single=True)
    except HTTPException:
        logger.warning(
            f"Invalid feature count: expected {len(FEATURE_COLUMNS)}, got {len(request.features)}"
        )
        raise
    try:
        features_df = pd.DataFrame([request.features], columns=FEATURE_COLUMNS)
        pred = predict(model, features_df, use_feature_subset=False)
        result = float(pred.iloc[0])
        logger.info(f"Prediction completed: demand={result:.2f}")
        return PredictResponse(predicted_demand=result)
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(status_code=503, detail=f"Model not found: {e}") from e
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}") from e


@app.post("/predict-batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest, req: Request) -> BatchPredictResponse:
    """Return demand forecasts for multiple feature sets."""
    model = getattr(req.app.state, "model", None)
    if model is None:
        logger.error("Batch prediction requested but model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded at startup")
    if not request.features_list:
        logger.warning("Batch prediction requested with empty features_list")
        raise HTTPException(status_code=400, detail="features_list cannot be empty")
    try:
        _validate_feature_count(request.features_list, single=False)
    except HTTPException:
        logger.warning("Batch prediction: invalid feature count in some samples")
        raise
    try:
        features_df = pd.DataFrame(request.features_list, columns=FEATURE_COLUMNS)
        preds = predict(model, features_df, use_feature_subset=False)
        logger.info(f"Batch prediction completed: {len(preds)} samples")
        return BatchPredictResponse(predictions=preds.tolist())
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(status_code=503, detail=f"Model not found: {e}") from e
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {e}") from e


@app.post("/drift-check", response_model=DriftCheckResponse)
def check_drift(request: DriftCheckRequest) -> DriftCheckResponse:
    """Detect feature drift between reference and current data via PSI."""
    if not request.reference_features:
        logger.warning("Drift check requested with empty reference_features")
        raise HTTPException(status_code=400, detail="reference_features cannot be empty")
    if not request.current_features:
        logger.warning("Drift check requested with empty current_features")
        raise HTTPException(status_code=400, detail="current_features cannot be empty")
    try:
        _validate_feature_count(request.reference_features, single=False)
    except HTTPException:
        logger.warning("Drift check: invalid feature count in reference_features")
        raise
    try:
        _validate_feature_count(request.current_features, single=False)
    except HTTPException:
        logger.warning("Drift check: invalid feature count in current_features")
        raise
    try:
        reference_df = pd.DataFrame(request.reference_features, columns=FEATURE_COLUMNS)
        current_df = pd.DataFrame(request.current_features, columns=FEATURE_COLUMNS)
        drift_result = detect_feature_drift(
            reference_df, current_df, psi_threshold=request.psi_threshold
        )
        if drift_result["has_drift"]:
            logger.warning(f"Feature drift detected: {drift_result['psi_by_feature']}")
        else:
            logger.info("Drift check completed: no drift detected")
        return DriftCheckResponse(
            has_drift=drift_result["has_drift"],
            psi_by_feature=drift_result["psi_by_feature"],
        )
    except Exception as e:
        logger.error(f"Drift check failed: {e}")
        raise HTTPException(status_code=400, detail=f"Drift check failed: {e}") from e
