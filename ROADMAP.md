# Roadmap

This document tracks the development progress of `demand-ml` toward production-readiness.

## ✅ Completed

### Core ML Pipeline
- [x] **Real UCI Bike Sharing data loader**: Automatic download, caching, and validation
- [x] **HistGradientBoostingRegressor baseline**: Fast tree-based model with structured hyperparameters
- [x] **Cross-validation & metrics**: MAE/RMSE via sklearn's `cross_validate`, holdout evaluation
- [x] **Hyperparameter tuning**: GridSearchCV-based tuning CLI with MLflow integration

### Model Lifecycle
- [x] **MLflow tracking**: Experiment tracking, metrics logging, feature importance, model artifacts
- [x] **Model persistence**: Save/load via joblib with validation helpers
- [x] **Predict helper**: Type-safe prediction interface with bounds validation

### Serving
- [x] **FastAPI endpoints**:
  - GET `/health` — Service readiness check
  - POST `/predict` — Single prediction with bounds validation
  - POST `/predict-batch` — Batch predictions (max 1000/req) with size limits
  - POST `/drift-check` — Feature drift detection via PSI

### Monitoring & Quality
- [x] **Feature drift detection**: PSI (Population Stability Index) monitoring with configurable thresholds
- [x] **Structured logging**: All endpoints and training log request/response with timing
- [x] **Type safety**: mypy --strict on all source code (no untyped imports)
- [x] **Comprehensive tests**: 45 tests covering data, features, model, persistence, serving, tracking, monitoring, and tuning

### Deployment & Documentation
- [x] **Dockerfile**: Production-grade image with health checks
- [x] **Concise docstrings**: All functions and classes documented
- [x] **README**: Quick start, API reference, configuration guide
- [x] **Makefile**: Task shortcuts (install, train, serve, check, test, lint, fmt, typecheck)

---

## 🚀 Future Enhancements

### Monitoring & Observability
- [ ] **Evidently integration**: Advanced data quality, model performance drift, and statistical tests
- [ ] **OpenTelemetry**: Request tracing across training, serving, and monitoring pipelines
- [ ] **Prometheus metrics**: Expose model latency, inference count, and drift alerts

### Model Improvements
- [ ] **AutoML feature engineering**: Polynomial, interaction, and domain-specific features
- [ ] **Ensemble methods**: Stacking, voting, or blending with other regressors (Ridge, Lasso, SVM)
- [ ] **Online learning / periodic retraining**: Scheduled retraining on recent data
- [ ] **Model versioning & rollback**: Serve multiple model versions, A/B test, and rollback on performance dips

### Infrastructure & DevOps
- [ ] **PostgreSQL backend for MLflow**: Replace file-based tracking for production
- [ ] **Kubernetes manifests**: Deployment files (Deployment, Service, Ingress, HPA)
- [ ] **CI/CD pipeline**: GitHub Actions for lint, typecheck, test, build, and push
- [ ] **Helm chart**: Package for easy Kubernetes deployment

### API & Data Handling
- [ ] **Batch async processing**: Async /predict-batch endpoint with async task queue
- [ ] **Feature importance endpoint**: GET `/feature-importance` returning per-model SHAP values
- [ ] **Data validation schemas**: Full validation of incoming features (type, range, distribution)
- [ ] **Explainability**: LIME or SHAP explanations for individual predictions

### Testing & Documentation
- [ ] **Integration tests**: End-to-end train → serve → predict workflows
- [ ] **Load tests**: Benchmark /predict and /predict-batch under load (k6, locust)
- [ ] **Architecture decision records (ADRs)**: Document key design choices
- [ ] **API OpenAPI/Swagger schema**: Auto-generated interactive API docs

---

## Development Notes

### Code Quality Gates (All Must Pass)
```bash
make check  # Runs all checks below
```
- **Lint**: `ruff check . --fix` (E, F, I, UP, B, SIM rules)
- **Format**: `ruff format .` (line-length=100)
- **Type check**: `mypy src` (strict mode, pydantic plugin)
- **Tests**: `pytest` (45 tests, all offline, deterministic)

### Project Structure
```
src/demand_ml/
├── __init__.py         # Package exports
├── config.py           # Settings (env-based)
├── data.py             # UCI loader + synthetic generator
├── features.py         # Calendar feature engineering
├── model.py            # HistGradientBoosting + tuning logic
├── persistence.py      # joblib save/load helpers
├── tracking.py         # MLflow logging wrapper
├── monitoring.py       # PSI drift detection
├── serving.py          # FastAPI app (all 4 endpoints)
├── train.py            # CLI training entrypoint
└── tune.py             # CLI hyperparameter tuning entrypoint

tests/
├── test_config.py      # Settings validation
├── test_data_loaders.py # Data loading and caching
├── test_features.py    # Feature engineering
├── test_model.py       # Model training, prediction, serialization
├── test_persistence.py # joblib save/load round-trips
├── test_serving.py     # FastAPI endpoint integration tests
├── test_tracking.py    # MLflow logging
├── test_tune.py        # Hyperparameter tuning
└── test_monitoring.py  # Drift detection
```

### Dependencies
- **Core ML**: pandas, scikit-learn, numpy, joblib
- **Serving**: FastAPI, uvicorn
- **Tracking**: MLflow
- **Config**: pydantic, pydantic-settings
- **Dev**: ruff, mypy, pytest, httpx

### Known Limitations & TODOs
- [ ] NumPy 2.5+ deprecation warning in joblib (shape assignment); awaiting joblib update
- [ ] Starlette httpx deprecation in test client; consider migrating to httpx2 when stable
- [ ] UCI data downloader retries only once; consider exponential backoff for production
- [ ] MLflow backend is file-based; switch to PostgreSQL for concurrent access in production
- [ ] Feature bounds validation is hardcoded; consider loading from training data statistics

---

## How to Contribute

1. **Pick a task** from "Future Enhancements" or file an issue
2. **Create a branch** from `main`: `git checkout -b feat/my-feature`
3. **Code**: Update src, add tests (tests are required), ensure `make check` passes
4. **Commit**: Use Conventional Commits (`feat(module): description`)
5. **Push & PR**: `git push -u origin feat/my-feature` and open a pull request
6. **Review**: All gates must pass (lint, type, test, format) before merge

---

**Last updated**: 2026-11-15  
**Status**: MVP complete; production enhancements pending
