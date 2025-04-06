# market_ml_model/models/__init__.py
"""
Models Module Initialization

This module provides classes and functions related to machine learning model
prediction, management, signal generation, and scheduling.
"""

from .manager import PredictionManager

# Import key classes and functions from submodules
from .predictor import (
    ModelPredictorBase,
    get_confidence_levels,
    load_model,
    predict_proba,
    predict_with_threshold,
    preprocess_features,
)
from .scheduler import PredictionScheduler
from .signals import (
    SignalGenerator,
    calibrate_probabilities,
    validate_model_predictions,
)
from .utils import predict_with_model

# Define what gets imported with 'from market_ml_model.models import *'
# Be selective to avoid polluting the namespace
__all__ = [
    "ModelPredictorBase",
    "PredictionManager",
    "SignalGenerator",
    "PredictionScheduler",
    "load_model",
    "preprocess_features",
    "predict_proba",
    "predict_with_threshold",
    "get_confidence_levels",
    "calibrate_probabilities",
    "validate_model_predictions",
    "predict_with_model",
]
