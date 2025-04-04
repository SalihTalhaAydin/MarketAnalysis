"""
Model training, evaluation, and prediction module.
"""

# Evaluation and reporting
from .evaluation.metrics import (
    compute_feature_importance,
    evaluate_classifier,
    generate_model_report,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
)

# Model creation and optimization
from .factory.model_factory import MODEL_FACTORY, create_model
from .feature_selection import select_features
from .optimization.hyperparameters import optimize_hyperparameters
from .prediction import (
    ModelPredictorBase,
    PredictionManager,
    PredictionScheduler,
    SignalGenerator,
    calibrate_probabilities,
    load_model,
    predict_proba,
    predict_with_model,
    validate_model_predictions,
)

# Core training and prediction functions
from .training import train_classification_model

__all__ = [
    # Training
    "train_classification_model",
    # Prediction
    "predict_with_model",
    "load_model",
    "predict_proba",
    "ModelPredictorBase",
    "PredictionManager",
    "SignalGenerator",
    "PredictionScheduler",
    "calibrate_probabilities",
    "validate_model_predictions",
    # Feature Selection
    "select_features",
    # Factory
    "create_model",
    "MODEL_FACTORY",
    # Optimization
    "optimize_hyperparameters",
    # Evaluation
    "evaluate_classifier",
    "compute_feature_importance",
    "generate_model_report",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_roc_curve",
    "plot_precision_recall_curve",
]
