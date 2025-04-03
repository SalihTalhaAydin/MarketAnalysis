# import pytest  # Not used directly
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# from sklearn.exceptions import NotFittedError # Not used directly

# Adjust the import path based on your project structure
from market_ml_model.src.main_strategy import predict_with_model

# Fixtures (sample_trained_model, sample_featured_data) are automatically
# discovered from ../conftest.py


# --- Test Functions for Prediction ---


def test_predict_with_model(sample_trained_model, sample_featured_data):
    """Test prediction generation returns probabilities."""
    # Use the same features the sample model was trained on (from fixture)
    features_live = sample_featured_data[['SMA_20', 'RSI_14', 'return',
                                          'close_lag1']]
    probabilities = predict_with_model(sample_trained_model, features_live)

    assert isinstance(probabilities, np.ndarray)
    assert probabilities.ndim == 2  # Should be 2D array (samples, classes)
    assert probabilities.shape[0] == len(features_live)
    assert probabilities.shape[1] == 2  # Assuming binary classification (0, 1)

    # Check if probabilities sum to 1 (approximately) for each sample
    assert np.allclose(np.sum(probabilities, axis=1), 1.0)

    # Check if all probability values are between 0 and 1
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_predict_with_model_not_fitted():
    """Test prediction with a model that hasn't been fitted."""
    model = RandomForestClassifier()  # Not fitted
    # Example features (ensure names don't clash with real model if possible)
    features_live = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    # Need to match potential internal feature names if model expects them
    # If the model wasn't even trained, it might not have feature_names_in_
    # Let's assume simple features for this test case.
    model.n_features_in_ = 2  # Mock attribute if needed for checks
    # model.feature_names_in_ = ['a', 'b'] # Mock attribute if needed

    probabilities = predict_with_model(model, features_live)

    # Function should catch NotFittedError and return empty array
    assert isinstance(probabilities, np.ndarray)
    assert len(probabilities) == 0


def test_predict_with_model_invalid_model():
    """Test prediction with an invalid model object."""
    model = "not a model"
    features_live = pd.DataFrame({'a': [1, 2]})
    probabilities = predict_with_model(model, features_live)
    assert isinstance(probabilities, np.ndarray)
    assert len(probabilities) == 0


def test_predict_with_model_empty_features(sample_trained_model):
    """Test prediction with empty features."""
    features_live = pd.DataFrame()
    probabilities = predict_with_model(sample_trained_model, features_live)
    assert isinstance(probabilities, np.ndarray)
    assert len(probabilities) == 0


def test_predict_with_model_feature_mismatch(sample_trained_model):
    """Test prediction with features different from training."""
    # Model was trained on ['SMA_20', 'RSI_14', 'return', 'close_lag1']
    # Provide features with different columns or order
    features_wrong_cols = pd.DataFrame({
        'SMA_50': [100, 101],  # Wrong column name
        'RSI_14': [50, 55],
        'return': [0.01, -0.01],
        'close_lag1': [105, 106]
    })
    features_wrong_order = pd.DataFrame({
        'RSI_14': [50, 55],  # Wrong order
        'SMA_20': [100, 101],
        'close_lag1': [105, 106],
        'return': [0.01, -0.01]
    })
    features_missing_col = pd.DataFrame({
        'SMA_20': [100, 101],
        'RSI_14': [50, 55],
        'return': [0.01, -0.01]  # Missing close_lag1
    })

    # Scikit-learn models usually raise ValueError for mismatch
    # The function should catch this and return an empty array
    probs_wrong_cols = predict_with_model(
        sample_trained_model, features_wrong_cols
    )
    assert isinstance(probs_wrong_cols, np.ndarray)
    assert len(probs_wrong_cols) == 0

    # Note: Scikit-learn >= 1.0 often handles column order if names match.
    # Let's test if the function still works (it should if sklearn handles it)
    probs_wrong_order = predict_with_model(
        sample_trained_model, features_wrong_order
    )
    if probs_wrong_order.shape[0] > 0:  # If sklearn handled it
        assert probs_wrong_order.shape == (len(features_wrong_order), 2)
    else:  # If our function or sklearn version failed
        assert isinstance(probs_wrong_order, np.ndarray)
        assert len(probs_wrong_order) == 0

    probs_missing_col = predict_with_model(
        sample_trained_model, features_missing_col
    )
    assert isinstance(probs_missing_col, np.ndarray)
    assert len(probs_missing_col) == 0