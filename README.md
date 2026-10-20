# demand-ml

Production ML service for urban bike-share demand forecasting. Stack: Python 3.12, uv, pandas, scikit-learn, MLflow, FastAPI, Uvicorn.

## Overview

`demand-ml` trains gradient-boosted demand forecasts on real UCI Bike Sharing data with MLflow tracking, serves predictions via FastAPI, and monitors feature drift in production.

## Features

- **Real UCI Bike Sharing data**: Automatic download, caching, and validation
- **HistGradientBoostingRegressor**: Fast tree-based baseline with hyperparameter tuning
- **Cross-validation & holdout metrics**: MAE/RMSE via sklearn's `cross_validate`
- **MLflow tracking**: Experiment tracking, metrics, feature importance, model artifacts
- **FastAPI serving**: `/health`, `/predict`, `/predict-batch`, `/drift-check` endpoints
- **Feature drift detection**: PSI (Population Stability Index) monitoring
- **Docker ready**: Production-grade Dockerfile with health checks
- **Type strict**: mypy --strict on all source code

## Quick Start

### Install

```bash
# Clone repo and cd into it
git clone https://github.com/MeKili/demand-ml && cd demand-ml

# Install dependencies
make install  # or: uv sync
```

### Train the Model

```bash
make train  # or: uv run python -m demand_ml.train
```

Loads real UCI data (cached to `data/bike_sharing.csv`), trains a baseline model, logs to MLflow (`mlruns/`), and saves to `models/baseline.joblib`.

Output:
```
Holdout: MAE=40.50 RMSE=52.30
CV: MAE=41.20±2.10 RMSE=53.10±2.80
Model saved to models/baseline.joblib
MLflow run ID: abc123def456
```

### Run the API

```bash
make serve  # or: uv run uvicorn demand_ml.serving:app --reload
```

FastAPI auto-docs: http://localhost:8000/docs

### Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [15.5, 60, 10, 9, 3, 8, 1]}'
# Response: {"predicted_demand": 42.5}
```

**Features** (in order): `temp`, `humidity`, `windspeed`, `hour`, `dayofweek`, `month`, `is_weekend`.

### Batch Predictions

```bash
curl -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{"features_list": [[15.5, 60, 10, 9, 3, 8, 1], [12.0, 45, 15, 14, 5, 6, 0]]}'
# Response: {"predictions": [42.5, 35.2]}
```

### Detect Feature Drift

```bash
curl -X POST http://localhost:8000/drift-check \
  -H "Content-Type: application/json" \
  -d '{
    "reference_features": [[15.5, 60, 10, 9, 3, 8, 1]],
    "current_features": [[10.0, 80, 5, 12, 2, 7, 0]],
    "psi_threshold": 0.1
  }'
# Response: {"has_drift": false, "psi_by_feature": {"temp": 0.05, ...}}
```

## Docker

```bash
# Build
docker build -t demand-ml .

# Run (model must be trained locally first; copy models/ into the image)
docker run -p 8000:8000 demand-ml
```

Health check runs every 30s; container is ready after ~5s.

## Project Structure

```
demand-ml/
├── src/demand_ml/
│   ├── __init__.py
│   ├── config.py           # Settings from env or .env
│   ├── data.py             # UCI loader + synthetic generator
│   ├── features.py         # Calendar features (hour, day, month, etc.)
│   ├── model.py            # HistGradientBoosting + tuning
│   ├── persistence.py      # Save/load models (joblib)
│   ├── tracking.py         # MLflow logging
│   ├── monitoring.py       # PSI drift detection
│   ├── serving.py          # FastAPI app (health, predict, batch, drift)
│   └── train.py            # CLI entrypoint
├── tests/
│   ├── test_*.py           # Full coverage (38 tests)
├── models/                 # Trained joblib models
├── data/                   # UCI data cache
├── mlruns/                 # MLflow artifacts (gitignored)
├── pyproject.toml          # Dependencies, ruff/mypy/pytest config
├── Dockerfile              # Production image
├── Makefile                # Task shortcuts
└── README.md               # This file
```

## Configuration

Copy `.env.example` to `.env` and adjust:

```env
DATA_DIR=data                            # Where to cache UCI data
MODEL_DIR=models                         # Where to save trained models
MODEL_FILENAME=baseline.joblib           # Model filename
MLFLOW_TRACKING_URI=file:./mlruns        # MLflow backend
TEST_SIZE=0.2                            # Train/test split
RANDOM_STATE=42                          # RNG seed
```

## Development

```bash
# Check all gates (lint, format, types, tests)
make check

# Individual checks
make lint      # ruff check .
make fmt       # ruff format .
make typecheck # mypy src
make test      # pytest
```

All code is mypy --strict, no untyped sklearn/pandas/mlflow imports.

## API Endpoints

| Method | Path           | Purpose                              |
|--------|----------------|--------------------------------------|
| GET    | `/health`      | Service health (always 200 if running) |
| POST   | `/predict`     | Single prediction                    |
| POST   | `/predict-batch` | Batch predictions (max 1000 per req) |
| POST   | `/drift-check` | Drift detection (PSI-based)          |

All endpoints log request/response and include error details.

## Monitoring

- **MLflow**: Track experiments, metrics, feature importance, and models at `mlruns/`
- **Drift detection**: Monitor feature distributions via PSI (threshold 0.1 default)
- **Structured logging**: All endpoints and training log to stderr

## Future Enhancements

- Advanced monitoring with Evidently (data quality, model performance drift)
- A/B testing support
- Online learning / periodic retraining
- Request tracing (OpenTelemetry)

## License

MIT

---

**Questions?** See [the roadmap](ROADMAP.md) or check test examples in `tests/`.
