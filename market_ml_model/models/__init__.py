"""
Model training, evaluation, and prediction module.
"""

# Core training and prediction functions
from .training import train_classification_model
from .prediction import (
    predict_with_model,
    load_model,
    predict_proba,
    ModelPredictorBase,
    PredictionManager,
    SignalGenerator,
    PredictionScheduler,
    calibrate_probabilities,
    validate_model_predictions
)
from .feature_selection import select_features

# Model creation and optimization
from .factory.model_factory import create_model, MODEL_FACTORY
from .optimization.hyperparameters import optimize_hyperparameters

# Evaluation and reporting
from .evaluation.metrics import (
    evaluate_classifier,
    compute_feature_importance,
    generate_model_report,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    plot_precision_recall_curve
)


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