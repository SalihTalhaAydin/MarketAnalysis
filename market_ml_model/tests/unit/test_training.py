import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

# Adjust the import path based on your project structure
from market_ml_model.src.main_strategy import train_classification_model

# Fixture (sample_featured_data) is automatically discovered from conftest.py


# --- Test Functions for Model Training ---


def test_train_classification_model(sample_featured_data):
    """Test model training returns a fitted model."""
    # Use a subset of features for simplicity in this test
    features = sample_featured_data[['SMA_20', 'RSI_14']]
    target = sample_featured_data['target_direction']

    model = train_classification_model(features, target)

    assert model is not None
    # Check specific model type if desired, though the function might change
    assert isinstance(model, RandomForestClassifier)
    # Check if model is fitted by trying to predict or checking attributes
    try:
        # Check a fitted attribute (specific to RandomForest)
        assert hasattr(model, 'estimators_')
        # Or try predicting
        model.predict(features.iloc[[0]])
    except NotFittedError:
        pytest.fail("Model should be fitted after train_classification_model")
    except Exception as e:
        pytest.fail(
            f"Model prediction/attribute check failed unexpectedly: {e}"
        )


def test_train_classification_model_empty_input():
    """Test training with empty features or target."""
    empty_features = pd.DataFrame()
    empty_target = pd.Series(dtype=int)
    some_features = pd.DataFrame({'a': [1, 2]})
    some_target = pd.Series([0, 1])

    assert train_classification_model(empty_features, some_target) is None
    assert train_classification_model(some_features, empty_target) is None
    assert train_classification_model(empty_features, empty_target) is None


def test_train_classification_model_mismatch_length():
    """Test training with mismatched feature and target lengths."""
    features = pd.DataFrame({'a': [1, 2, 3]})
    target = pd.Series([0, 1])
    assert train_classification_model(features, target) is None


def test_train_classification_model_nan_input(sample_featured_data):
    """Test training with NaN values in features."""
    features = sample_featured_data[['SMA_20', 'RSI_14']].copy()
    target = sample_featured_data['target_direction']
    features.iloc[1, 0] = pd.NA  # Introduce a NaN

    # The function should ideally handle or raise error for NaNs.
    # Current implementation might raise ValueError during fit.
    # Let's assert it returns None as per the function's error handling.
    assert train_classification_model(features, target) is None

    # Test with NaN in target (less common, but good to check)
    features = sample_featured_data[['SMA_20', 'RSI_14']]
    target_nan = target.copy().astype(float)  # Need float for pd.NA
    target_nan.iloc[1] = pd.NA
    assert train_classification_model(features, target_nan) is None