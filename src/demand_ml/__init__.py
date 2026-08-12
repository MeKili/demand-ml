"""demand-ml — a production ML service for urban bike-share demand forecasting.

Lifecycle: data -> features -> train/evaluate -> track (MLflow) -> serve (FastAPI)
-> monitor. This package currently implements the data, features and training slice
on a deterministic synthetic dataset.
"""

__version__ = "0.1.0"
