# market_ml_model/models/prediction.py
"""
DEPRECATED: Original prediction module.

Functionality has been refactored into:
- market_ml_model.models.predictor
- market_ml_model.models.manager
- market_ml_model.models.signals
- market_ml_model.models.scheduler
- market_ml_model.models.utils

Please update imports accordingly. See market_ml_model/models/__init__.py
"""

import warnings

warnings.warn(
    "market_ml_model.models.prediction is deprecated. Use components from market_ml_model.models.* submodules.",
    DeprecationWarning,
    stacklevel=2,
)

# Optionally re-export main components for backward compatibility (use with caution)
# from .predictor import ModelPredictorBase, load_model, predict_proba
# from .manager import PredictionManager
# from .signals import SignalGenerator
# from .scheduler import PredictionScheduler
# from .utils import predict_with_model
#
# __all__ = [
#     "ModelPredictorBase", "PredictionManager", "SignalGenerator", "PredictionScheduler",
#     "load_model", "predict_proba", "predict_with_model"
# ]
