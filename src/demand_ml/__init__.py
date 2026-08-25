"""demand-ml — a production ML service for urban bike-share demand forecasting.

Full pipeline: real UCI data loader (with local cache) -> feature engineering ->
train/evaluate with gradient boosting -> MLflow tracking -> FastAPI serving
(/health, /predict) -> feature-drift monitoring via PSI. Offline deterministic
tests use a synthetic demand generator.
"""

__version__ = "0.1.0"
