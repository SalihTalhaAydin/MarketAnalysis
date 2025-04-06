# market_ml_model/models/utils.py
import logging
from typing import Optional

import pandas as pd

# Import necessary components from other modules
from .predictor import (
    ModelPredictorBase,
    predict_with_threshold,
    get_confidence_levels,
)

logger = logging.getLogger(__name__)


# --- High-Level Prediction Function ---


def predict_with_model(
    model_path: str, features: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """
    Convenience function to load a model and generate predictions/probabilities.

    Args:
        model_path: Path to the model directory or .pkl file
        features: Input features DataFrame

    Returns:
        DataFrame containing probabilities, prediction, and confidence, or None
    """
    try:
        predictor = ModelPredictorBase(model_path)
        if predictor.model is None:
            return None

        probabilities, class_names = predictor.predict_proba(features)
        if probabilities is None:
            return None

        # Create result DataFrame
        result_df = pd.DataFrame(
            probabilities,
            index=features.index,
            columns=[
                f"probability_{c}" for c in class_names
            ],  # Ensure column names match
        )
        # Determine positive class index for thresholding
        positive_class_label = "1"  # Assuming '1' is positive
        try:
            pos_idx = class_names.index(positive_class_label)
        except ValueError:
            pos_idx = 1  # Default if '1' not found

        result_df["prediction"] = predict_with_threshold(
            probabilities, positive_class_index=pos_idx
        )
        result_df["confidence"] = get_confidence_levels(probabilities)

        return result_df

    except Exception as e:
        logger.exception(f"Error in predict_with_model: {e}")
        return None
