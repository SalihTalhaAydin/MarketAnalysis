import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError


def predict_with_model(model, features_live: pd.DataFrame) -> np.ndarray:
    """
    Generates class probabilities on new/live data using the trained model.

    Args:
        model: The trained scikit-learn compatible model object.
        features_live: DataFrame of features for which to make predictions.
                       Must have the same columns as the training features.

    Returns:
        Numpy array of shape (n_samples, n_classes) with probabilities
        for each class, or an empty array if prediction fails.
        Based on the mapping in model_training.py (-1->0, 0->1, 1->2):
        Column 0: Probability of class 0 (Stop Loss hit)
        Column 1: Probability of class 1 (Time Barrier hit)
        Column 2: Probability of class 2 (Take Profit hit)
    """
    print("--- Making Predictions ---")
    if model is None:
        print("Error: Invalid model object provided for prediction.")
        return np.array([])
    if not hasattr(model, 'predict_proba'):
        print("Error: Model object does not have a 'predict_proba' method.")
        return np.array([])
    if features_live.empty:
        print("Warning: No live features provided for prediction.")
        return np.array([])

    print(f"Predicting on live features shape: {features_live.shape}")

    try:
        # Ensure the model is fitted before predicting
        # (predict will raise NotFittedError if not)
        # Use predict_proba to get probabilities for each class
        probabilities = model.predict_proba(features_live)
        print(f"Generated probabilities for {len(probabilities)} samples.")
        return probabilities
    except NotFittedError:
        print("Error: Cannot predict probabilities, the model is not fitted.")
        return np.array([])
    except ValueError as ve:
        print(f"Error during probability prediction (ValueError): {ve}")
        # Common issue: features_live columns mismatch training columns, NaNs
        return np.array([])
    except Exception as e:
        print(f"Unexpected error during probability prediction: {e}")
        return np.array([])