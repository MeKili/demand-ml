"""FastAPI application for model serving and health checks."""

from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from demand_ml.persistence import predict

app = FastAPI(title="demand-ml", version="0.1.0")


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


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check if the service is running."""
    return HealthResponse(status="ok", message="demand-ml service is running")


@app.post("/predict", response_model=PredictResponse)
def predict_demand(
    request: PredictRequest, model_path: str = "models/baseline.joblib"
) -> PredictResponse:
    """Predict demand given features.

    Expects features as [temp, humidity, windspeed, hour, dayofweek, month, is_weekend].
    """
    try:
        features_df = pd.DataFrame(
            [request.features],
            columns=["temp", "humidity", "windspeed", "hour", "dayofweek", "month", "is_weekend"],
        )
        pred = predict(model_path, features_df, use_feature_subset=False)
        return PredictResponse(predicted_demand=float(pred.iloc[0]))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not found: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}") from e
