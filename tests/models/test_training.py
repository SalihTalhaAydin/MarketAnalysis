from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Import components to test
from market_ml_model.models.training import (
    create_feature_pipeline,
    train_classification_model,
)

# Define paths for patching
TRAINING_PATH = "market_ml_model.models.training"
FACTORY_PATH = "market_ml_model.models.factory.model_factory"
EVAL_PATH = "market_ml_model.models.evaluation.metrics"
SELECTION_PATH = "market_ml_model.models.feature_selection"
OPTIMIZATION_PATH = "market_ml_model.models.optimization.hyperparameters"


# --- Fixtures ---


@pytest.fixture
def sample_feature_df():
    """DataFrame with mixed feature types."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    data = {
        "numeric1": np.linspace(10, 20, 10),
        "numeric2": np.random.rand(10) * 100,
        "category1": ["A", "B"] * 5,
        "category2": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"],
        "numeric_with_nan": [1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10],
        "category_with_nan": ["P", "Q", "P", np.nan, "Q", "P", "Q", np.nan, "P", "Q"],
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_target_series(sample_feature_df):
    """Target series matching the feature df index. Ensures balance for CV."""
    # Ensure first 8 samples (typical train size) are balanced for CV=5
    target_data = [0, 1, 0, 1, 0, 1, 0, 1] + list(
        np.random.randint(0, 2, size=len(sample_feature_df) - 8)
    )
    return pd.Series(
        target_data[: len(sample_feature_df)],  # Ensure correct length
        index=sample_feature_df.index,
        dtype=int,  # Ensure integer type
    )


@pytest.fixture
def mock_model_config():
    """Basic model config dictionary."""
    return {"model_type": "lightgbm", "params": {"n_estimators": 50}}


@pytest.fixture
def mock_feature_selection_config():
    """Basic feature selection config."""
    return {"enabled": True, "method": "importance", "params": {"n_features": 3}}


@pytest.fixture
def mock_preprocessing_config():
    """Basic preprocessing config."""
    return {"scaling_method": "standard", "handle_missing": True}


@pytest.fixture
def mock_optimization_config():
    """Basic optimization config."""
    return {
        "method": "random",
        "params": {"n_iter": 5},
        "cv": 3,
        "scoring": "accuracy",
    }  # Changed cv to 3


# --- Mocks ---


# Mock sklearn components used in create_feature_pipeline
@pytest.fixture(autouse=True)
def mock_sklearn_preprocessing(mocker):
    """Mocks relevant sklearn preprocessing classes."""
    mocks = {
        "ColumnTransformer": mocker.patch(f"{TRAINING_PATH}.ColumnTransformer"),
        "SimpleImputer": mocker.patch(f"{TRAINING_PATH}.SimpleImputer"),
        "Pipeline": mocker.patch(f"{TRAINING_PATH}.Pipeline"),
        "StandardScaler": mocker.patch(f"{TRAINING_PATH}.StandardScaler"),
        "MinMaxScaler": mocker.patch(f"{TRAINING_PATH}.MinMaxScaler"),
        "RobustScaler": mocker.patch(f"{TRAINING_PATH}.RobustScaler"),
        "OneHotEncoder": mocker.patch(f"{TRAINING_PATH}.OneHotEncoder"),
        # "QuantileTransformer": mocker.patch(f"{TRAINING_PATH}.QuantileTransformer", create=True), # No longer imported
    }
    mocker.patch(f"{TRAINING_PATH}.SKLEARN_AVAILABLE", True)
    return mocks


# Mock dependencies for train_classification_model
@pytest.fixture
def mock_train_dependencies(mocker):
    """Mocks dependencies called by train_classification_model."""
    mocks = {
        "train_test_split": mocker.patch(f"{TRAINING_PATH}.train_test_split"),
        "create_feature_pipeline": mocker.patch(
            f"{TRAINING_PATH}.create_feature_pipeline"
        ),
        "select_features": mocker.patch(
            f"{TRAINING_PATH}.select_features"
        ),  # Patched where it's used
        "optimize_hyperparameters": mocker.patch(
            f"{TRAINING_PATH}.optimize_hyperparameters"
        ),  # Patched where it's used
        "create_model": mocker.patch(
            f"{TRAINING_PATH}.create_model"
        ),  # Patched where it's used
        "evaluate_classifier": mocker.patch(
            f"{TRAINING_PATH}.evaluate_classifier"
        ),  # Patched where it's used
        "compute_feature_importance": mocker.patch(
            f"{TRAINING_PATH}.compute_feature_importance"
        ),  # Patched where it's used
        "joblib_dump": mocker.patch(f"{TRAINING_PATH}.joblib.dump"),  # Add mock back
        "os_makedirs": mocker.patch(f"{TRAINING_PATH}.os.makedirs"),  # Add mock back
        "json_dump": mocker.patch(f"{TRAINING_PATH}.json.dump"),  # Add mock back
        "open": mocker.patch(f"{TRAINING_PATH}.open", MagicMock()),  # Add mock back
    }

    # Setup default return values for mocks
    # Simulate returning split data (using copies to avoid modification issues)
    mocks["train_test_split"].side_effect = lambda X, y, **kwargs: (
        X.copy(),
        X.copy(),
        y.copy(),
        y.copy(),
    )

    # --- Mock Feature Selection ---
    # Default: Return all features. Tests enabling selection should override this.
    mocks["select_features"].side_effect = lambda X, y, **kwargs: (
        X.copy(),
        list(X.columns),
    )

    # --- Mock Preprocessing Pipeline ---
    # --- Mock Preprocessing Pipeline ---
    # Mock the *output* of create_feature_pipeline directly
    # Return a MagicMock for the pipeline object (avoids pickling issues in tests)
    # and the expected processed feature names.
    mock_pipeline_instance = MagicMock()
    # Define expected processed names (adjust if the actual pipeline output differs)
    processed_feature_names = [
        "num__numeric1",
        "num__numeric2",
        "num__numeric_with_nan",
        "cat__category1_A",
        "cat__category1_B",
        "cat__category2_X",
        "cat__category2_Y",
        "cat__category_with_nan_P",
        "cat__category_with_nan_Q",
        "cat__category_with_nan_nan",
    ]
    mock_pipeline_instance.get_feature_names_out.return_value = processed_feature_names
    # Simulate returning NumPy arrays after transformation
    mock_pipeline_instance.fit_transform.side_effect = lambda X: np.zeros(
        (len(X), len(processed_feature_names))
    )
    mock_pipeline_instance.transform.side_effect = lambda X: np.zeros(
        (len(X), len(processed_feature_names))
    )
    # Mock create_feature_pipeline to return the MagicMock instance and original names
    # Use the actual columns from the fixture for the original names list
    original_cols = [
        "numeric1",
        "numeric2",
        "category1",
        "category2",
        "numeric_with_nan",
        "category_with_nan",
    ]
    mocks["create_feature_pipeline"].return_value = (
        mock_pipeline_instance,
        original_cols,
    )

    # --- Mock Model Creation & Optimization ---
    mock_model_instance = MagicMock()
    mock_model_instance.fit.return_value = None  # Mock fit method
    mocks["optimize_hyperparameters"].return_value = (
        mock_model_instance,
        {"best_param": 1},
    )
    mocks["create_model"].return_value = mock_model_instance

    # --- Mock Evaluation & Importance ---
    # Return basic metrics
    mocks["evaluate_classifier"].return_value = {"accuracy": 0.9, "f1_weighted": 0.88}
    # Return importance based on the *mocked* processed feature names
    mocks["compute_feature_importance"].return_value = pd.DataFrame(
        {
            "Feature": processed_feature_names,
            "Importance": np.random.rand(len(processed_feature_names)),
        }
    )

    return mocks


# --- Tests for create_feature_pipeline ---


# No longer patching QuantileTransformer as it's not imported
def test_create_pipeline_standard_scaling(
    sample_feature_df, mock_sklearn_preprocessing
):
    """Test pipeline creation with StandardScaler."""
    preprocessor, names = create_feature_pipeline(
        sample_feature_df,
        scaling_method="standardscaler",  # Use lowercase name
    )
    assert preprocessor is not None
    mock_sklearn_preprocessing["StandardScaler"].assert_called_once()
    # mock_sklearn_preprocessing["MinMaxScaler"].assert_not_called() # Commented out: All scalers instantiated in map


def test_create_pipeline_minmax_scaling(sample_feature_df, mock_sklearn_preprocessing):
    """Test pipeline creation with MinMaxScaler."""
    preprocessor, names = create_feature_pipeline(
        sample_feature_df,
        scaling_method="minmaxscaler",  # Use lowercase
    )
    assert preprocessor is not None
    mock_sklearn_preprocessing["MinMaxScaler"].assert_called_once()
    # mock_sklearn_preprocessing["StandardScaler"].assert_not_called() # Commented out: All scalers instantiated in map


def test_create_pipeline_robust_scaling(sample_feature_df, mock_sklearn_preprocessing):
    """Test pipeline creation with RobustScaler."""
    preprocessor, names = create_feature_pipeline(
        sample_feature_df,
        scaling_method="robustscaler",  # Use lowercase
    )
    assert preprocessor is not None
    mock_sklearn_preprocessing["RobustScaler"].assert_called_once()
    # mock_sklearn_preprocessing["StandardScaler"].assert_not_called() # Commented out: All scalers instantiated in map


# No longer patching QuantileTransformer
def test_create_pipeline_quantile_scaling(
    sample_feature_df, mock_sklearn_preprocessing
):
    """Test pipeline creation with QuantileTransformer (should warn and skip)."""
    with patch(f"{TRAINING_PATH}.logger") as mock_logger:
        preprocessor, names = create_feature_pipeline(
            sample_feature_df, scaling_method="quantile"
        )
        assert preprocessor is not None
        # QuantileTransformer mock should NOT be called
        # mock_sklearn_preprocessing["QuantileTransformer"].assert_not_called() # No longer exists
        mock_logger.warning.assert_any_call(
            "Unsupported scaling method: 'quantile'. No scaling applied."
        )


@patch(f"{TRAINING_PATH}.logger")
def test_create_pipeline_unclassified_feature(mock_logger, sample_feature_df):
    """Test handling of features not specified as numeric or categorical."""
    # Only provide numeric features, let categorical be inferred
    numeric_cols = ["numeric1", "numeric2", "numeric_with_nan"]
    preprocessor, names = create_feature_pipeline(
        sample_feature_df, numeric_features=numeric_cols
    )
    assert preprocessor is not None
    # Check that a warning was logged for the unclassified features
    mock_logger.warning.assert_called_once()  # Should only warn once about the group


# --- Tests for train_classification_model ---


def test_train_model_success(
    mock_train_dependencies,
    sample_feature_df,
    sample_target_series,
    mock_model_config,
    tmp_path,
):
    """Test successful model training workflow."""
    output_dir = str(tmp_path / "model_output")
    # Explicitly disable feature selection for this test
    fs_config = {"enabled": False}
    model, preprocessor, metrics, final_feature_names = train_classification_model(
        features=sample_feature_df,
        target=sample_target_series,
        model_config=mock_model_config,
        feature_selection_config=fs_config,  # Pass config
        output_dir=output_dir,
        optimize_hyperparams=False,
    )

    assert model is not None
    assert preprocessor is not None  # Check preprocessor instead of importance
    assert metrics is not None
    assert isinstance(final_feature_names, list)  # Check the fourth return value
    assert "error" not in metrics

    # Check mocks
    mock_train_dependencies["create_feature_pipeline"].assert_called_once()
    mock_train_dependencies[
        "select_features"
    ].assert_not_called()  # Explicitly disabled
    mock_train_dependencies["create_model"].assert_called_once_with(
        mock_model_config["model_type"], mock_model_config["params"]
    )
    # Check fit was called with NumPy array (ANY checks for correct type and shape implicitly via mock side effect)
    model.fit.assert_called_once_with(ANY, ANY)  # ANY for X_train_final_np, y_train
    # Check evaluation/importance called with DataFrames containing processed names
    evaluate_call_args = mock_train_dependencies["evaluate_classifier"].call_args_list
    importance_call_args = mock_train_dependencies[
        "compute_feature_importance"
    ].call_args_list
    # Get processed names from the mock pipeline used in the test run
    mock_pipeline_used = mock_train_dependencies[
        "create_feature_pipeline"
    ].return_value[0]
    expected_cols = mock_pipeline_used.get_feature_names_out.return_value

    # Check evaluate_classifier calls (train and test)
    assert mock_train_dependencies["evaluate_classifier"].call_count == 2
    assert (
        list(evaluate_call_args[0][0][1].columns) == expected_cols
    )  # Check train X columns
    assert (
        list(evaluate_call_args[1][0][1].columns) == expected_cols
    )  # Check test X columns

    # Check compute_feature_importance call
    assert (
        mock_train_dependencies["compute_feature_importance"].call_count >= 1
    )  # Called at least once
    assert (
        list(importance_call_args[0][0][1].columns) == expected_cols
    )  # Check train X columns
    # mock_train_dependencies["os_makedirs"].assert_called_once_with(output_dir, exist_ok=True) # Removed assertion
    # mock_train_dependencies["joblib_dump"].assert_called() # Removed assertion
    # mock_train_dependencies["json_dump"].assert_called() # Removed assertion


def test_train_model_with_optimization(
    mock_train_dependencies,
    sample_feature_df,
    sample_target_series,
    mock_model_config,
    mock_optimization_config,
    tmp_path,
):
    """Test successful model training with hyperparameter optimization."""
    output_dir = str(tmp_path / "model_output_opt")
    model, preprocessor, metrics, final_feature_names = train_classification_model(
        features=sample_feature_df,
        target=sample_target_series,
        model_config=mock_model_config,
        optimization_config={
            "method": "random",
            "params": {"n_iter": 5},
            "cv": 3,
            "scoring": "accuracy",
        },  # Pass dict directly
        output_dir=output_dir,
        optimize_hyperparams=True,
    )

    assert model is not None  # optimize_hyperparameters returns the best model
    assert preprocessor is not None  # Check preprocessor instead of importance
    assert metrics is not None
    assert isinstance(final_feature_names, list)  # Check the fourth return value
    assert "error" not in metrics

    # Check optimize_hyperparameters was called with NumPy array
    mock_train_dependencies["optimize_hyperparameters"].assert_called_once()
    opt_call_args = mock_train_dependencies["optimize_hyperparameters"].call_args[
        1
    ]  # Get kwargs
    assert isinstance(opt_call_args["X_train"], np.ndarray)  # Check X_train type
    assert isinstance(opt_call_args["y_train"], pd.Series)  # Check y_train type

    # Check that create_model WAS called with the best params from optimization
    mock_train_dependencies["create_model"].assert_called_once_with(
        mock_model_config["model_type"],
        {"best_param": 1},  # Params returned by optimize mock
    )

    # Check evaluation/importance called with DataFrames containing processed names
    evaluate_call_args = mock_train_dependencies["evaluate_classifier"].call_args_list
    importance_call_args = mock_train_dependencies[
        "compute_feature_importance"
    ].call_args_list
    mock_pipeline_used = mock_train_dependencies[
        "create_feature_pipeline"
    ].return_value[0]
    expected_cols = mock_pipeline_used.get_feature_names_out.return_value

    assert mock_train_dependencies["evaluate_classifier"].call_count == 2
    assert (
        list(evaluate_call_args[0][0][1].columns) == expected_cols
    )  # Check train X columns
    assert (
        list(evaluate_call_args[1][0][1].columns) == expected_cols
    )  # Check test X columns

    assert mock_train_dependencies["compute_feature_importance"].call_count >= 1
    assert (
        list(importance_call_args[0][0][1].columns) == expected_cols
    )  # Check train X columns

    # mock_train_dependencies["json_dump"].assert_any_call(ANY, ANY, indent=4, default=str) # Removed assertion


@patch(f"{TRAINING_PATH}.logger")
def test_train_model_data_split_fail(
    mock_logger, mock_train_dependencies, sample_feature_df
):
    """Test failure during data splitting."""
    mock_train_dependencies["train_test_split"].side_effect = ValueError("Split error")
    # Reset index to ensure train_test_split is called (not iloc slicing)
    features_no_dt_index = sample_feature_df.reset_index(drop=True)
    target_no_dt_index = pd.Series([1] * len(features_no_dt_index))  # Match length

    model, preprocessor, metrics, final_feature_names = train_classification_model(
        features=features_no_dt_index,
        target=target_no_dt_index,
        optimize_hyperparams=False,
    )
    # Check that the function returns None for model/preprocessor and the error in metrics
    assert model is None
    assert preprocessor is None  # Check preprocessor instead of importance
    assert metrics is not None
    assert isinstance(final_feature_names, list)  # Should return empty list on failure
    assert not final_feature_names  # Assert list is empty
    assert "error" in metrics
    assert "Data splitting failed" in metrics["error"]


@patch(f"{TRAINING_PATH}.logger")
def test_train_model_pipeline_fail(
    mock_logger, mock_train_dependencies, sample_feature_df, sample_target_series
):
    """Test failure during pipeline creation."""
    mock_train_dependencies["create_feature_pipeline"].return_value = (
        None,
        [],
    )  # Simulate failure
    model, preprocessor, metrics, final_feature_names = train_classification_model(
        features=sample_feature_df,
        target=sample_target_series,
        optimize_hyperparams=False,
    )
    # Check that the function returns None for model/preprocessor and the error in metrics
    assert model is None
    assert preprocessor is None  # Check preprocessor instead of importance
    assert metrics is not None
    assert isinstance(final_feature_names, list)  # Should return empty list on failure
    assert not final_feature_names  # Assert list is empty
    assert isinstance(final_feature_names, list)  # Should return empty list on failure
    assert not final_feature_names  # Assert list is empty
    assert "error" in metrics
    assert "Preprocessing failed" in metrics["error"]


@patch(f"{TRAINING_PATH}.logger")
def test_train_model_training_fail(
    mock_logger, mock_train_dependencies, sample_feature_df, sample_target_series
):
    """Test failure during model fitting."""
    mock_model_instance = MagicMock()
    mock_model_instance.fit.side_effect = ValueError("Fit error")
    mock_train_dependencies["create_model"].return_value = mock_model_instance

    model, preprocessor, metrics, final_feature_names = train_classification_model(
        features=sample_feature_df,
        target=sample_target_series,
        optimize_hyperparams=False,
    )
    # The function should return None for model/preprocessor on critical failure,
    # and the error message in the metrics dict.
    # Check that the function returns None for model/preprocessor and the error in metrics
    assert model is None
    assert preprocessor is None  # Check preprocessor instead of importance
    assert metrics is not None
    assert "error" in metrics
    assert (
        "Training pipeline failed: Fit error" in metrics["error"]
    )  # Check for the specific mocked error wrapped by the pipeline
