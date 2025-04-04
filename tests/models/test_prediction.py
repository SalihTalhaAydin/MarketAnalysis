import os
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

# Import sklearn components for mocking specs
from sklearn.compose import ColumnTransformer

# Import functions and classes to test
from market_ml_model.models.prediction import (
    ModelPredictorBase,
    get_confidence_levels,
    load_model,
    predict_proba,
    predict_with_model,
    predict_with_threshold,
    preprocess_features,
)

# Define path for patching
PREDICTION_PATH = "market_ml_model.models.prediction"

# --- Fixtures ---


@pytest.fixture
def sample_features():
    """Sample features DataFrame."""
    return pd.DataFrame(
        {
            "feat1": np.random.rand(10),
            "feat2_selected": np.random.rand(10),
            "feat3_selected": np.random.rand(10),
            "category_feat": ["A", "B"] * 5,
        },
        index=pd.date_range("2023-01-01", periods=10),
    )


# Copied from test_metrics.py as it's needed here too
@pytest.fixture
def sample_eval_data():
    """Provides sample features, true labels, predictions, and probabilities."""
    X = pd.DataFrame({"feat1": range(10), "feat2": range(10, 20)})
    y_true = pd.Series([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = pd.Series([0, 1, 0, 0, 0, 1, 0, 0, 1, 1])  # Some misclassifications
    # Probabilities for class 1
    y_prob = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.4, 0.6],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.4, 0.6],
        ]
    )
    return X, y_true, y_pred, y_prob


@pytest.fixture
def mock_model_object():
    """Mock trained model object."""
    model = MagicMock()
    model.predict.return_value = np.array([0, 1] * 5)
    model.predict_proba.return_value = np.random.rand(10, 2)
    model.classes_ = np.array([0, 1])
    return model


@pytest.fixture
def mock_preprocessor_object():
    """Mock fitted preprocessor object."""
    preprocessor = MagicMock(spec=ColumnTransformer)
    # Define expected output feature names
    output_feature_names = [
        "proc_feat_0",
        "proc_feat_1",
        "proc_feat_2",
        "proc_feat_3",
        "proc_feat_4",
    ]
    preprocessor.transform.return_value = np.random.rand(
        10, len(output_feature_names)
    )  # Example transformed shape
    preprocessor.get_feature_names_out.return_value = output_feature_names
    return preprocessor


@pytest.fixture
def sample_metadata(mock_preprocessor_object):
    """Sample metadata dictionary."""
    # Metadata should contain info needed for prediction, not the live preprocessor object itself
    return {
        "selected_features": ["feat2_selected", "feat3_selected", "category_feat"],
        "processed_feature_names": mock_preprocessor_object.get_feature_names_out(),
        "class_names": ["Class_0", "Class_1"],
        # 'preprocessor': mock_preprocessor_object # Don't include live object in metadata dict
        # Add other metadata keys as needed
    }


# --- Mocks for File System Operations ---


@pytest.fixture
def mock_os_path(mocker):
    """Mocks os.path functions."""
    mock_isdir = mocker.patch(f"{PREDICTION_PATH}.os.path.isdir")
    mock_isfile = mocker.patch(f"{PREDICTION_PATH}.os.path.isfile")
    mock_exists = mocker.patch(f"{PREDICTION_PATH}.os.path.exists")
    return mock_isdir, mock_isfile, mock_exists


@pytest.fixture
def mock_joblib_load(mocker):
    """Mocks joblib.load."""
    return mocker.patch(f"{PREDICTION_PATH}.joblib.load")


@pytest.fixture
def mock_json_load(mocker):
    """Mocks json.load."""
    return mocker.patch(f"{PREDICTION_PATH}.json.load")


# --- Tests for load_model ---


@patch(
    f"{PREDICTION_PATH}.open", new_callable=mock_open
)  # Keep mock_open for file handle check
def test_load_model_from_directory(
    mock_open_func,
    mock_os_path,
    mock_joblib_load,
    mock_json_load,
    mock_model_object,
    mock_preprocessor_object,
    sample_metadata,
):
    """Test loading model, metadata, and preprocessor from a directory."""
    mock_isdir, mock_isfile, mock_exists = mock_os_path
    mock_isdir.return_value = True
    mock_isfile.return_value = False
    mock_exists.side_effect = lambda path: True  # Simulate all files exist

    # Configure mocks
    mock_joblib_load.side_effect = [
        mock_model_object,
        mock_preprocessor_object,
    ]  # Order matters
    # Mock json.load to return the metadata dict (without the live preprocessor)
    metadata_to_load = sample_metadata.copy()
    # metadata_to_load.pop('preprocessor', None) # Remove if it was accidentally added
    mock_json_load.return_value = metadata_to_load

    model_dir = "/fake/model_dir"
    model, metadata = load_model(model_dir)

    assert model is mock_model_object
    assert "selected_features" in metadata
    assert "preprocessor" in metadata  # Preprocessor loaded via joblib is added here
    assert metadata["preprocessor"] is mock_preprocessor_object
    mock_joblib_load.assert_any_call(os.path.join(model_dir, "model.pkl"))
    mock_joblib_load.assert_any_call(os.path.join(model_dir, "preprocessor.pkl"))
    mock_open_func.assert_called_once_with(
        os.path.join(model_dir, "training_summary.json"), "r"
    )
    mock_json_load.assert_called_once()  # Check json.load was called


def test_load_model_from_pkl(mock_os_path, mock_joblib_load, mock_model_object):
    """Test loading model directly from a .pkl file."""
    mock_isdir, mock_isfile, mock_exists = mock_os_path
    mock_isdir.return_value = False
    mock_isfile.return_value = True
    mock_joblib_load.return_value = mock_model_object

    model_file = "/fake/model.pkl"
    model, metadata = load_model(model_file)

    assert model is mock_model_object
    assert metadata == {}  # No metadata expected when loading only pkl
    mock_joblib_load.assert_called_once_with(model_file)


def test_load_model_path_not_found(mock_os_path):
    """Test loading when path doesn't exist."""
    mock_isdir, mock_isfile, mock_exists = mock_os_path
    mock_isdir.return_value = False
    mock_isfile.return_value = False
    model, metadata = load_model("/non/existent/path")
    assert model is None
    assert metadata == {}


def test_load_model_file_not_found_in_dir(mock_os_path):
    """Test loading from directory when model.pkl is missing."""
    mock_isdir, mock_isfile, mock_exists = mock_os_path
    mock_isdir.return_value = True
    mock_isfile.return_value = False
    mock_exists.side_effect = lambda path: not path.endswith(
        "model.pkl"
    )  # Model file doesn't exist

    model, metadata = load_model("/fake/model_dir")
    assert model is None
    assert metadata == {}


# --- Tests for preprocess_features ---


def test_preprocess_features_with_selection_and_preprocessor(
    sample_features, mock_preprocessor_object, sample_metadata
):
    """Test preprocessing with feature selection and a preprocessor."""
    selected_features = sample_metadata["selected_features"]
    features_processed = preprocess_features(
        sample_features, mock_preprocessor_object, selected_features
    )

    assert features_processed is not None
    # Check preprocessor was called with the selected features
    mock_preprocessor_object.transform.assert_called_once()
    call_args, _ = mock_preprocessor_object.transform.call_args
    transformed_df = call_args[0]
    assert isinstance(transformed_df, pd.DataFrame)
    assert list(transformed_df.columns) == selected_features
    # Check output columns match expected names
    assert (
        list(features_processed.columns)
        == mock_preprocessor_object.get_feature_names_out()
    )


def test_preprocess_features_only_selection(sample_features, sample_metadata):
    """Test preprocessing with only feature selection."""
    selected_features = sample_metadata["selected_features"]
    features_processed = preprocess_features(
        sample_features, preprocessor=None, selected_features=selected_features
    )
    assert features_processed is not None
    assert list(features_processed.columns) == selected_features
    pd.testing.assert_frame_equal(
        features_processed, sample_features[selected_features]
    )


def test_preprocess_features_only_preprocessor(
    sample_features, mock_preprocessor_object
):
    """Test preprocessing with only a preprocessor."""
    features_processed = preprocess_features(
        sample_features, preprocessor=mock_preprocessor_object, selected_features=None
    )
    assert features_processed is not None
    # Check transform was called, but avoid direct DataFrame comparison in assert
    mock_preprocessor_object.transform.assert_called_once()
    call_args, _ = mock_preprocessor_object.transform.call_args
    pd.testing.assert_frame_equal(
        call_args[0], sample_features
    )  # Check input to transform
    assert (
        list(features_processed.columns)
        == mock_preprocessor_object.get_feature_names_out()
    )


def test_preprocess_features_none(sample_features):
    """Test preprocessing with no selection or preprocessor."""
    features_processed = preprocess_features(
        sample_features, preprocessor=None, selected_features=None
    )
    assert features_processed is not None
    pd.testing.assert_frame_equal(features_processed, sample_features)


def test_preprocess_features_missing_input_feature(
    sample_features, mock_preprocessor_object
):
    """Test warning when an expected selected feature is missing from input."""
    selected_features = ["feat2_selected", "feat3_selected", "MISSING"]
    with patch(f"{PREDICTION_PATH}.logger") as mock_logger:
        preprocess_features(
            sample_features, mock_preprocessor_object, selected_features
        )
        mock_logger.warning.assert_any_call(
            "Missing expected features in input data: ['MISSING']. Proceeding with available features."
        )
        # Check preprocessor was called
        mock_preprocessor_object.transform.assert_called_once()  # Should still be called
        call_args, _ = mock_preprocessor_object.transform.call_args
        assert list(call_args[0].columns) == [
            "feat2_selected",
            "feat3_selected",
        ]  # Only available passed


# --- Tests for predict_proba ---


def test_predict_proba_success(
    mock_model_object, sample_features, mock_preprocessor_object, sample_metadata
):  # Use correct fixture name
    """Test successful probability prediction."""
    probs, names = predict_proba(
        mock_model_object,
        sample_features,
        mock_preprocessor_object,
        sample_metadata["selected_features"],
        sample_metadata["class_names"],
    )
    assert probs is not None
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (len(sample_features), 2)  # Based on mock model
    assert names == sample_metadata["class_names"]
    mock_preprocessor_object.transform.assert_called_once()
    mock_model_object.predict_proba.assert_called_once()


def test_predict_proba_no_preprocessor(
    mock_model_object, sample_features
):  # Use correct fixture name
    """Test probability prediction without a preprocessor."""
    probs, names = predict_proba(mock_model_object, sample_features)
    assert probs is not None
    # Check call count and inspect args without direct df comparison
    mock_model_object.predict_proba.assert_called_once()
    call_args, _ = mock_model_object.predict_proba.call_args
    pd.testing.assert_frame_equal(call_args[0], sample_features)


def test_predict_proba_no_predict_proba_method(
    mock_model_object, sample_features
):  # Use correct fixture name
    """Test fallback when model lacks predict_proba."""
    del mock_model_object.predict_proba  # Remove the method
    probs, names = predict_proba(mock_model_object, sample_features)
    assert probs is not None
    assert probs.shape == (len(sample_features), 2)  # Should be one-hot encoded
    assert (probs.sum(axis=1) == 1).all()  # Check one-hot encoding
    # Check call count and inspect args without direct df comparison
    mock_model_object.predict.assert_called_once()
    call_args, _ = mock_model_object.predict.call_args
    pd.testing.assert_frame_equal(call_args[0], sample_features)


# --- Tests for predict_with_threshold ---


def test_predict_with_threshold_binary(sample_eval_data):  # Use correct fixture name
    """Test thresholding for binary classification."""
    _, _, _, y_prob = sample_eval_data
    preds_p5 = predict_with_threshold(y_prob, threshold=0.5, positive_class_index=1)
    preds_p8 = predict_with_threshold(y_prob, threshold=0.8, positive_class_index=1)
    expected_p5 = np.array([0, 1, 0, 0, 0, 1, 1, 0, 1, 1])  # Based on y_prob > 0.5
    expected_p8 = np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 0])  # Based on y_prob > 0.8
    np.testing.assert_array_equal(preds_p5, expected_p5)
    np.testing.assert_array_equal(preds_p8, expected_p8)


def test_predict_with_threshold_multiclass(
    sample_eval_data,
):  # Use correct fixture name
    """Test thresholding for multiclass (should return argmax)."""
    _, _, _, y_prob_binary = sample_eval_data
    # Create dummy multiclass probs
    y_prob_multi = np.array([[0.1, 0.6, 0.3], [0.8, 0.1, 0.1], [0.2, 0.2, 0.6]])
    preds = predict_with_threshold(y_prob_multi, threshold=0.7)  # Threshold ignored
    expected = np.array([1, 0, 2])  # Argmax of each row
    np.testing.assert_array_equal(preds, expected)


# --- Tests for get_confidence_levels ---


def test_get_confidence_levels(sample_eval_data):  # Use correct fixture name
    """Test confidence level calculation."""
    _, _, _, y_prob = sample_eval_data
    confidence = get_confidence_levels(y_prob)
    expected = np.max(y_prob, axis=1)
    np.testing.assert_array_equal(confidence, expected)


# --- Tests for ModelPredictorBase ---


@patch(f"{PREDICTION_PATH}.load_model")
def test_model_predictor_init_success(
    mock_load, mock_model_object, mock_preprocessor_object, sample_metadata
):  # Add preprocessor to args
    """Test successful initialization of ModelPredictorBase."""
    # Adjust metadata to not contain the live preprocessor object
    metadata_loaded = sample_metadata.copy()
    # metadata_loaded.pop('preprocessor', None) # Ensure it's not there
    # load_model should return the preprocessor separately if found
    mock_load.return_value = (mock_model_object, metadata_loaded)
    # Simulate preprocessor being loaded separately by load_model and added back
    metadata_loaded["preprocessor"] = mock_preprocessor_object

    predictor = ModelPredictorBase("/fake/path")
    assert predictor.model is mock_model_object
    assert predictor.preprocessor is mock_preprocessor_object
    assert predictor.selected_features == sample_metadata["selected_features"]
    assert predictor.class_names == sample_metadata["class_names"]
    mock_load.assert_called_once_with("/fake/path")


@patch(f"{PREDICTION_PATH}.load_model")
def test_model_predictor_init_fail(mock_load):
    """Test initialization failure when load_model fails."""
    mock_load.return_value = (None, {})
    with pytest.raises(ValueError, match="Failed to load model"):
        ModelPredictorBase("/fake/path")


@patch(f"{PREDICTION_PATH}.load_model")
@patch(f"{PREDICTION_PATH}.preprocess_features")
def test_model_predictor_predict(
    mock_preprocess,
    mock_load,
    mock_model_object,
    sample_metadata,
    sample_features,
    mock_preprocessor_object,
):  # Add preprocessor
    """Test the predict method of ModelPredictorBase."""
    metadata_loaded = sample_metadata.copy()
    mock_load.return_value = (mock_model_object, metadata_loaded)
    # Add preprocessor back as load_model would
    metadata_loaded["preprocessor"] = mock_preprocessor_object

    # Mock preprocess_features to return a DataFrame with correct columns
    mock_processed_features = pd.DataFrame(
        np.random.rand(10, len(sample_metadata["processed_feature_names"])),
        index=sample_features.index,
        columns=sample_metadata["processed_feature_names"],
    )
    mock_preprocess.return_value = mock_processed_features

    predictor = ModelPredictorBase("/fake/path")
    predictions = predictor.predict(sample_features)

    mock_preprocess.assert_called_once_with(
        sample_features, predictor.preprocessor, predictor.selected_features
    )
    # Check predict called on the *processed* features
    assert mock_model_object.predict.call_count == 1
    call_args, _ = mock_model_object.predict.call_args
    # Check the DataFrame passed to predict has the correct columns and index
    pd.testing.assert_frame_equal(call_args[0], mock_processed_features)
    assert predictions is not None


@patch(f"{PREDICTION_PATH}.load_model")
@patch(f"{PREDICTION_PATH}.predict_proba")
def test_model_predictor_predict_proba(
    mock_predict_proba,
    mock_load,
    mock_model_object,
    sample_metadata,
    sample_features,
    mock_preprocessor_object,
):  # Add preprocessor
    """Test the predict_proba method of ModelPredictorBase."""
    metadata_loaded = sample_metadata.copy()
    mock_load.return_value = (mock_model_object, metadata_loaded)
    metadata_loaded["preprocessor"] = mock_preprocessor_object  # Add preprocessor back

    mock_predict_proba.return_value = (
        np.random.rand(10, 2),
        sample_metadata["class_names"],
    )

    predictor = ModelPredictorBase("/fake/path")
    probs, names = predictor.predict_proba(sample_features)

    mock_predict_proba.assert_called_once_with(
        predictor.model,
        sample_features,
        predictor.preprocessor,
        predictor.selected_features,
        predictor.class_names,
    )
    assert probs is not None
    assert names == sample_metadata["class_names"]


# --- Tests for predict_with_model ---


@patch(f"{PREDICTION_PATH}.ModelPredictorBase")
def test_predict_with_model_success(MockPredictor, sample_features):
    """Test the high-level predict_with_model function."""
    # Mock the predictor instance and its methods
    mock_predictor_instance = MagicMock()
    mock_probs = np.array([[0.1, 0.9], [0.8, 0.2]])
    mock_names = ["Class0", "Class1"]
    mock_predictor_instance.predict_proba.return_value = (mock_probs, mock_names)
    MockPredictor.return_value = mock_predictor_instance

    # Use subset of features matching the mock probability shape
    features_subset = sample_features.iloc[:2]
    result_df = predict_with_model("/fake/path", features_subset)

    MockPredictor.assert_called_once_with("/fake/path")
    # Check predict_proba was called with the correct DataFrame arg
    mock_predictor_instance.predict_proba.assert_called_once()
    call_args, _ = mock_predictor_instance.predict_proba.call_args
    pd.testing.assert_frame_equal(
        call_args[0], features_subset
    )  # Check input to predict_proba

    assert isinstance(result_df, pd.DataFrame)
    assert "prediction" in result_df.columns
    assert "confidence" in result_df.columns
    assert list(result_df.columns) == ["Class0", "Class1", "prediction", "confidence"]
    assert result_df["prediction"].iloc[0] == 1  # Based on mock_probs[0]
    assert result_df["prediction"].iloc[1] == 0  # Based on mock_probs[1]


@patch(f"{PREDICTION_PATH}.ModelPredictorBase")
def test_predict_with_model_load_fail(MockPredictor, sample_features):
    """Test predict_with_model when model loading fails."""
    MockPredictor.side_effect = ValueError("Load failed")
    result_df = predict_with_model("/fake/path", sample_features)
    assert result_df is None


@patch(f"{PREDICTION_PATH}.ModelPredictorBase")
def test_predict_with_model_proba_fail(MockPredictor, sample_features):
    """Test predict_with_model when predict_proba fails."""
    mock_predictor_instance = MagicMock()
    mock_predictor_instance.predict_proba.return_value = (None, [])  # Simulate failure
    MockPredictor.return_value = mock_predictor_instance
    result_df = predict_with_model("/fake/path", sample_features)
    assert result_df is None
