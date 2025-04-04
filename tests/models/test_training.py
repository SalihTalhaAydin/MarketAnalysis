from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

# Import sklearn components used internally to check types, etc.

# Import functions to test
from market_ml_model.models.training import (
    create_feature_pipeline,
    train_classification_model,
)

# Define path for patching
TRAINING_PATH = "market_ml_model.models.training"

# --- Fixtures ---


@pytest.fixture
def sample_feature_df():
    """Provides a sample DataFrame with mixed feature types."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    data = {
        "numeric1": np.linspace(10, 20, 10),
        "numeric2": np.random.rand(10) * 100,
        "category1": ["A", "B", "A", "C", "B", "A", "A", "C", "B", "A"],
        "category2": ["X", "X", "Y", "Y", "Z", "X", "Y", "Z", "X", "Y"],
        "numeric_with_nan": [1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10],
        "category_with_nan": ["P", "Q", "R", np.nan, "P", "Q", np.nan, "R", "P", "Q"],
    }
    return pd.DataFrame(data, index=dates)


# --- Tests for create_feature_pipeline ---


# Mock sklearn classes to verify instantiation and parameters
@patch(f"{TRAINING_PATH}.StandardScaler")
@patch(f"{TRAINING_PATH}.MinMaxScaler")
@patch(f"{TRAINING_PATH}.RobustScaler")
@patch(f"{TRAINING_PATH}.QuantileTransformer")
@patch(f"{TRAINING_PATH}.SimpleImputer")
@patch(f"{TRAINING_PATH}.OneHotEncoder")
@patch(f"{TRAINING_PATH}.Pipeline")
@patch(f"{TRAINING_PATH}.ColumnTransformer")
def test_create_pipeline_standard_scaling(
    MockColumnTransformer,
    MockPipeline,
    MockOneHotEncoder,
    MockSimpleImputer,
    MockQuantileTransformer,
    MockRobustScaler,
    MockMinMaxScaler,
    MockStandardScaler,
    sample_feature_df,
):
    """Test pipeline creation with standard scaling and auto-detected features."""
    X = sample_feature_df
    preprocessor, names = create_feature_pipeline(
        X, scaling_method="standard", handle_missing=True
    )

    assert isinstance(preprocessor, MagicMock)  # It's the mocked ColumnTransformer
    assert isinstance(names, list)

    # Check ColumnTransformer was called
    MockColumnTransformer.assert_called_once()
    call_args, call_kwargs = MockColumnTransformer.call_args
    transformers_arg = call_kwargs.get("transformers", [])

    # Check numeric transformer pipeline steps
    numeric_pipeline = None
    numeric_features_passed = []
    for name, pipe, features in transformers_arg:
        if name == "num":
            numeric_pipeline = pipe
            numeric_features_passed = features
            break
    assert numeric_pipeline is not None
    assert set(numeric_features_passed) == {"numeric1", "numeric2", "numeric_with_nan"}
    MockSimpleImputer.assert_any_call(strategy="median")
    MockStandardScaler.assert_called_once()
    assert (
        MockPipeline.call_args_list[0][1]["steps"][0][0] == "imputer"
    )  # First step is imputer
    assert (
        MockPipeline.call_args_list[0][1]["steps"][1][0] == "scaler"
    )  # Second step is scaler

    # Check categorical transformer pipeline steps
    categorical_pipeline = None
    categorical_features_passed = []
    for name, pipe, features in transformers_arg:
        if name == "cat":
            categorical_pipeline = pipe
            categorical_features_passed = features
            break
    assert categorical_pipeline is not None
    assert set(categorical_features_passed) == {
        "category1",
        "category2",
        "category_with_nan",
    }
    MockSimpleImputer.assert_any_call(strategy="most_frequent")
    MockOneHotEncoder.assert_called_once_with(
        handle_unknown="ignore", sparse_output=False
    )
    assert MockPipeline.call_args_list[1][1]["steps"][0][0] == "imputer"
    assert MockPipeline.call_args_list[1][1]["steps"][1][0] == "onehot"

    # Check other scalers were not called
    MockMinMaxScaler.assert_not_called()
    MockRobustScaler.assert_not_called()
    MockQuantileTransformer.assert_not_called()


@patch(f"{TRAINING_PATH}.MinMaxScaler")
def test_create_pipeline_minmax_scaling(MockMinMaxScaler, sample_feature_df):
    """Test pipeline creation with minmax scaling."""
    preprocessor, _ = create_feature_pipeline(
        sample_feature_df[["numeric1", "numeric2"]],  # Only numeric
        scaling_method="minmax",
        handle_missing=False,  # Disable missing handling
    )
    MockMinMaxScaler.assert_called_once()


@patch(f"{TRAINING_PATH}.RobustScaler")
def test_create_pipeline_robust_scaling(MockRobustScaler, sample_feature_df):
    """Test pipeline creation with robust scaling."""
    preprocessor, _ = create_feature_pipeline(
        sample_feature_df[["numeric1", "numeric2"]],
        scaling_method="robust",
        handle_missing=False,
    )
    MockRobustScaler.assert_called_once()


@patch(f"{TRAINING_PATH}.QuantileTransformer")
def test_create_pipeline_quantile_scaling(MockQuantileTransformer, sample_feature_df):
    """Test pipeline creation with quantile scaling."""
    preprocessor, _ = create_feature_pipeline(
        sample_feature_df[["numeric1", "numeric2"]],
        scaling_method="quantile",
        handle_missing=False,
    )
    MockQuantileTransformer.assert_called_once_with(output_distribution="normal")


@patch(f"{TRAINING_PATH}.SimpleImputer")
def test_create_pipeline_no_missing_handling(MockSimpleImputer, sample_feature_df):
    """Test pipeline creation without missing value handling."""
    preprocessor, _ = create_feature_pipeline(sample_feature_df, handle_missing=False)
    MockSimpleImputer.assert_not_called()


def test_create_pipeline_explicit_features(sample_feature_df):
    """Test pipeline creation with explicitly provided feature lists."""
    num_feat = ["numeric1", "numeric_with_nan"]
    cat_feat = ["category1"]
    preprocessor, names = create_feature_pipeline(
        sample_feature_df,
        numeric_features=num_feat,
        categorical_features=cat_feat,
        handle_missing=True,
        scaling_method="standard",
    )
    # Check that only the specified features are included in the transformers
    transformers = preprocessor.transformers  # Correct attribute name
    assert transformers[0][0] == "num"  # Assuming numeric comes first
    assert transformers[0][2] == num_feat
    assert transformers[1][0] == "cat"
    assert transformers[1][2] == cat_feat
    # Check that other columns are passed through
    assert preprocessor.remainder == "passthrough"


@patch(f"{TRAINING_PATH}.logger")
def test_create_pipeline_missing_feature_in_df(mock_logger, sample_feature_df):
    """Test error handling when a specified feature is missing from the DataFrame."""
    num_feat = ["numeric1", "MISSING_FEATURE"]
    cat_feat = ["category1"]
    preprocessor, names = create_feature_pipeline(
        sample_feature_df, numeric_features=num_feat, categorical_features=cat_feat
    )
    assert preprocessor is None
    assert names == []
    mock_logger.error.assert_called_once()
    assert "Features specified but not found" in mock_logger.error.call_args[0][0]


@patch(f"{TRAINING_PATH}.logger")
def test_create_pipeline_unclassified_feature(mock_logger, sample_feature_df):
    """Test warning and auto-classification for unclassified features."""
    num_feat = ["numeric1"]  # Leave out numeric2
    cat_feat = ["category1"]  # Leave out category2
    preprocessor, names = create_feature_pipeline(
        sample_feature_df, numeric_features=num_feat, categorical_features=cat_feat
    )
    assert preprocessor is not None
    mock_logger.warning.assert_called_once()
    assert (
        "Features in DataFrame but not classified"
        in mock_logger.warning.call_args[0][0]
    )
    # Check that numeric2 and category2 were added to the correct lists internally
    transformers = preprocessor.transformers  # Correct attribute name
    assert "numeric2" in transformers[0][2]  # Should be added to numeric
    assert "category2" in transformers[1][2]  # Should be added to categorical
    assert "numeric_with_nan" in transformers[0][2]
    assert "category_with_nan" in transformers[1][2]


def test_create_pipeline_no_numeric(sample_feature_df):
    """Test pipeline creation with only categorical features."""
    cat_feat = ["category1", "category2", "category_with_nan"]
    preprocessor, names = create_feature_pipeline(
        sample_feature_df[cat_feat],
        categorical_features=cat_feat,
        numeric_features=[],  # Explicitly empty
    )
    assert preprocessor is not None
    assert len(preprocessor.transformers) == 1  # Correct attribute name
    assert preprocessor.transformers[0][0] == "cat"  # Correct attribute name
    assert preprocessor.transformers[0][2] == cat_feat  # Correct attribute name


def test_create_pipeline_no_categorical(sample_feature_df):
    """Test pipeline creation with only numeric features."""
    num_feat = ["numeric1", "numeric2", "numeric_with_nan"]
    preprocessor, names = create_feature_pipeline(
        sample_feature_df[num_feat],
        numeric_features=num_feat,
        categorical_features=[],  # Explicitly empty
    )
    assert preprocessor is not None
    assert len(preprocessor.transformers) == 1  # Correct attribute name
    assert preprocessor.transformers[0][0] == "num"  # Correct attribute name
    assert preprocessor.transformers[0][2] == num_feat  # Correct attribute name


@patch(f"{TRAINING_PATH}.SKLEARN_AVAILABLE", False)
@patch(f"{TRAINING_PATH}.logger")
def test_create_pipeline_sklearn_unavailable(mock_logger, sample_feature_df):
    """Test behavior when scikit-learn is not available."""
    preprocessor, names = create_feature_pipeline(sample_feature_df)
    assert preprocessor is None
    assert names == []
    mock_logger.error.assert_called_with(
        "scikit-learn not available for feature pipeline"
    )


# --- Tests for train_classification_model (Basic Setup) ---
# More detailed tests would require more complex mocking setup


# Mock all major dependencies of train_classification_model
@patch(f"{TRAINING_PATH}.create_model")
@patch(f"{TRAINING_PATH}.optimize_hyperparameters")
@patch(f"{TRAINING_PATH}.evaluate_classifier")
@patch(f"{TRAINING_PATH}.compute_feature_importance")
@patch(f"{TRAINING_PATH}.select_features")
@patch(f"{TRAINING_PATH}.create_feature_pipeline")
@patch(f"{TRAINING_PATH}.train_test_split")
@patch(f"{TRAINING_PATH}.joblib.dump")
@patch(f"{TRAINING_PATH}.os.makedirs")
@patch(f"{TRAINING_PATH}.open", new_callable=MagicMock)  # Mock open for saving files
def test_train_classification_model_basic_flow(
    mock_open,
    mock_makedirs,
    mock_joblib_dump,
    mock_train_test_split,
    mock_create_feature_pipeline,
    mock_select_features,
    mock_compute_importance,
    mock_evaluate_classifier,
    mock_optimize_hyperparams,
    mock_create_model,
    sample_feature_df,
):
    """Test the basic successful workflow of train_classification_model."""
    # --- Mock Setup ---
    # Prepare dummy data
    X = sample_feature_df.drop(
        columns=["category_with_nan"]
    )  # Drop col with NaNs for simplicity here
    y = pd.Series(np.random.randint(0, 2, size=len(X)), index=X.index)
    X_train_raw, X_test_raw, y_train, y_test = (
        X.iloc[:8],
        X.iloc[8:],
        y.iloc[:8],
        y.iloc[8:],
    )
    mock_train_test_split.return_value = (X_train_raw, X_test_raw, y_train, y_test)

    # Feature Selection mock
    selected_features_list = ["numeric1", "category1"]
    mock_select_features.return_value = (
        X_train_raw[selected_features_list],
        selected_features_list,
    )

    # Preprocessing mock
    mock_preprocessor = MagicMock(spec=ColumnTransformer)
    mock_preprocessor.get_feature_names_out.return_value = [
        "num__numeric1",
        "cat__category1_A",
        "cat__category1_B",
        "cat__category1_C",
    ]
    X_train_processed = pd.DataFrame(
        np.random.rand(8, 4),
        index=X_train_raw.index,
        columns=mock_preprocessor.get_feature_names_out(),
    )
    X_test_processed = pd.DataFrame(
        np.random.rand(2, 4),
        index=X_test_raw.index,
        columns=mock_preprocessor.get_feature_names_out(),
    )
    mock_preprocessor.fit_transform.return_value = (
        X_train_processed.values
    )  # Return numpy array
    mock_preprocessor.transform.return_value = (
        X_test_processed.values
    )  # Return numpy array
    mock_create_feature_pipeline.return_value = (
        mock_preprocessor,
        selected_features_list,
    )  # Return mock preprocessor

    # Model creation / optimization mock
    mock_model = MagicMock()
    mock_model.fit.return_value = None
    mock_optimize_hyperparams.return_value = (
        mock_model,
        {"param": "best"},
    )  # Return fitted model and best params
    mock_create_model.return_value = (
        mock_model  # For the case where optimization is off
    )

    # Evaluation mock
    mock_evaluate_classifier.side_effect = [
        {"accuracy": 0.9, "f1_weighted": 0.89},  # Train metrics
        {"accuracy": 0.8, "f1_weighted": 0.79},  # Test metrics
    ]

    # Importance mock
    mock_importance_df = pd.DataFrame(
        {
            "Feature": mock_preprocessor.get_feature_names_out(),
            "Importance": np.random.rand(4),
        }
    )
    mock_compute_importance.return_value = mock_importance_df

    # --- Run Function ---
    trained_model, importance_df, metrics = train_classification_model(
        features=X,
        target=y,
        model_config={"model_type": "lightgbm"},
        feature_selection_config={"enabled": True, "method": "importance"},
        preprocessing_config={"scaling_method": "standard"},
        optimize_hyperparams=True,
        optimization_config={"method": "random"},
        output_dir="test_output",
        class_names=["class0", "class1"],
    )

    # --- Assertions ---
    assert trained_model is mock_model
    pd.testing.assert_frame_equal(importance_df, mock_importance_df)
    assert "train" in metrics
    assert "test" in metrics
    assert metrics["test"]["accuracy"] == 0.8

    # Check major steps were called
    # mock_train_test_split.assert_called_once() # Removed: Time series split is used due to DatetimeIndex
    mock_select_features.assert_called_once()
    mock_create_feature_pipeline.assert_called_once()
    mock_preprocessor.fit_transform.assert_called_once()
    mock_preprocessor.transform.assert_called_once()
    mock_optimize_hyperparams.assert_called_once()  # Optimization was enabled
    mock_create_model.assert_not_called()  # Should not be called if optimization returns model
    assert mock_evaluate_classifier.call_count == 2
    mock_compute_importance.assert_called_once()

    # Check saving mocks
    mock_makedirs.assert_called()
    mock_joblib_dump.assert_any_call(mock_preprocessor, ANY)  # Check preprocessor saved
    mock_joblib_dump.assert_any_call(mock_model, ANY)  # Check model saved
    assert (
        mock_open.call_count >= 2
    )  # For best_params.json and training_summary.json (and maybe features)
