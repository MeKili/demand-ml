"""Production ML service for urban bike-share demand forecasting."""

from demand_ml.data import load_uci_bike_sharing, make_synthetic_demand
from demand_ml.model import train_model, tune_hyperparameters
from demand_ml.persistence import load_model, predict, save_model
from demand_ml.serving import app
from demand_ml.tracking import log_training_run

__version__ = "0.1.0"

__all__ = [
    "load_uci_bike_sharing",
    "make_synthetic_demand",
    "train_model",
    "tune_hyperparameters",
    "load_model",
    "save_model",
    "predict",
    "log_training_run",
    "app",
]
