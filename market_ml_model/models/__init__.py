"""
Model training and evaluation module.
"""

from .training import train_classification_model
from .factory.model_factory import create_model, MODEL_FACTORY
from .evaluation.metrics import evaluate_classifier