# market_ml_model/models/manager.py
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import necessary components from the predictor module
from .predictor import (
    ModelPredictorBase,
    get_confidence_levels,
    predict_with_threshold,
)

logger = logging.getLogger(__name__)


class PredictionManager:
    """Class to manage multiple predictors and ensemble their predictions."""

    def __init__(self):
        """Initialize the prediction manager."""
        self.predictors: Dict[str, ModelPredictorBase] = {}
        self.ensemble_weights: Dict[str, float] = {}
        self.class_names: Optional[List[str]] = None  # Store common class names

    def add_predictor(self, name: str, model_path: str, weight: float = 1.0) -> bool:
        """
        Add a predictor to the manager by loading its model.

        Args:
            name: A unique name for this predictor
            model_path: Path to the model directory or .pkl file
            weight: Weight for this predictor in the ensemble (default: 1.0)

        Returns:
            True if successful, False otherwise
        """
        if name in self.predictors:
            logger.warning(f"Predictor with name '{name}' already exists. Overwriting.")

        try:
            predictor = ModelPredictorBase(model_path)
            if predictor.model is None:
                return False  # Loading failed

            # Check for class name consistency
            if self.class_names is None:
                self.class_names = predictor.class_names
            elif (
                predictor.class_names is not None
                and self.class_names != predictor.class_names
            ):
                logger.error(
                    f"Class name mismatch for predictor '{name}'. Expected {self.class_names}, got {predictor.class_names}."
                )
                return False

            self.predictors[name] = predictor
            self.ensemble_weights[name] = weight
            logger.info(
                f"Added predictor '{name}' from {model_path} with weight {weight}."
            )
            return True

        except Exception as e:
            logger.error(f"Failed to add predictor '{name}': {e}")
            return False

    def remove_predictor(self, name: str) -> bool:
        """
        Remove a predictor from the manager.

        Args:
            name: Name of the predictor to remove

        Returns:
            True if successful, False otherwise
        """
        if name in self.predictors:
            del self.predictors[name]
            del self.ensemble_weights[name]
            logger.info(f"Removed predictor '{name}'.")
            # Reset class names if last predictor removed
            if not self.predictors:
                self.class_names = None
            return True
        else:
            logger.warning(f"Predictor '{name}' not found.")
            return False

    def set_weight(self, name: str, weight: float) -> bool:
        """
        Set the ensemble weight for a predictor.

        Args:
            name: Name of the predictor
            weight: New weight

        Returns:
            True if successful, False otherwise
        """
        if name in self.predictors:
            self.ensemble_weights[name] = weight
            logger.info(f"Set weight for predictor '{name}' to {weight}.")
            return True
        else:
            logger.warning(f"Predictor '{name}' not found.")
            return False

    def predict_proba(
        self,
        features: pd.DataFrame,
        ensemble_method: str = "average",  # 'average', 'weighted_average'
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Generate ensemble class probabilities.

        Args:
            features: Input Feature DataFrame
            ensemble_method: Method for combining probabilities

        Returns:
            Tuple of (ensembled probabilities array, class_names list) or (None, [])
        """
        if not self.predictors:
            logger.error("No predictors added to the manager.")
            return None, []

        all_probs = []
        final_class_names = self.class_names or []
        valid_predictor_names = []  # Keep track of predictors that succeed

        for name, predictor in self.predictors.items():
            probs, names = predictor.predict_proba(features)
            if probs is None:
                logger.warning(
                    f"Predictor '{name}' failed to generate probabilities. Skipping."
                )
                continue
            # Ensure class name consistency was checked during add_predictor
            if not final_class_names:
                final_class_names = names  # Set if first predictor
            all_probs.append(probs * self.ensemble_weights[name])  # Apply weight here
            valid_predictor_names.append(name)  # Add name if successful

        if not all_probs:
            logger.error("All predictors failed.")
            return None, []

        # Combine probabilities using only successful predictors
        if ensemble_method == "average":
            ensembled_probs = np.mean(all_probs, axis=0)
        elif ensemble_method == "weighted_average":
            # Calculate total weight only for predictors that succeeded
            total_weight = sum(
                self.ensemble_weights[name] for name in valid_predictor_names
            )
            if total_weight == 0:
                logger.warning(
                    "Total weight of successful predictors is zero, using simple average."
                )
                ensembled_probs = np.mean(all_probs, axis=0)
            else:
                ensembled_probs = np.sum(all_probs, axis=0) / total_weight
        else:
            logger.error(f"Unsupported ensemble method: {ensemble_method}")
            return None, []

        return ensembled_probs, final_class_names

    def predict(
        self,
        features: pd.DataFrame,
        ensemble_method: str = "average",
        threshold: float = 0.5,
        positive_class_index: int = 1,
    ) -> Optional[np.ndarray]:
        """
        Generate ensemble class predictions.

        Args:
            features: Input Feature DataFrame
            ensemble_method: Method for combining probabilities
            threshold: Decision threshold for binary classification
            positive_class_index: Index of the positive class for binary thresholding

        Returns:
            Predicted classes array, or None if error
        """
        ensembled_probs, _ = self.predict_proba(features, ensemble_method)
        if ensembled_probs is None:
            return None

        return predict_with_threshold(ensembled_probs, threshold, positive_class_index)

    def predict_with_confidence(
        self, features: pd.DataFrame, ensemble_method: str = "average"
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generate ensemble predictions and associated confidence levels.

        Args:
            features: Input Feature DataFrame
            ensemble_method: Method for combining probabilities

        Returns:
            Tuple of (predictions array, confidence array) or (None, None)
        """
        ensembled_probs, _ = self.predict_proba(features, ensemble_method)
        if ensembled_probs is None:
            return None, None

        predictions = predict_with_threshold(
            ensembled_probs
        )  # Use default threshold for prediction
        confidence = get_confidence_levels(ensembled_probs)
        return predictions, confidence
