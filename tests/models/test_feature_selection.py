import sys  # Import sys for sys.modules patching
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

# Import sklearn components for mocking specs
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest
from sklearn.preprocessing import StandardScaler

# Import the function to test
from market_ml_model.models.feature_selection import select_features

# Define path for patching
FS_PATH = "market_ml_model.models.feature_selection"

# --- Fixtures ---


@pytest.fixture
def sample_features_target():
    """Provides sample features (X) and target (y)."""
    X = pd.DataFrame(
        {
            "feat1": np.random.rand(50),
            "feat2": np.random.rand(50) * 10,
            "feat3": np.random.rand(50) - 0.5,
            "feat4_useless": np.zeros(50),  # Useless feature
            "feat5": np.random.rand(50),
        }
    )
    y = pd.Series(np.random.randint(0, 2, size=50))
    return X, y


# --- Mocks ---


# Mock sklearn classes used within select_features
@pytest.fixture
def mock_rf(mocker):
    mock = MagicMock(spec=RandomForestClassifier)
    mock.fit.return_value = None
    # Simulate feature importances (make feat4_useless have 0 importance)
    mock.feature_importances_ = np.array([0.3, 0.4, 0.2, 0.0, 0.1])
    return mocker.patch(f"{FS_PATH}.RandomForestClassifier", return_value=mock)


@pytest.fixture
def mock_selectkbest(mocker):
    mock_selector = MagicMock(spec=SelectKBest)
    mock_selector.fit.return_value = None
    # Simulate selecting first 3 features
    mock_selector.get_support.return_value = [0, 1, 2]
    return mocker.patch(f"{FS_PATH}.SelectKBest", return_value=mock_selector)


@pytest.fixture
def mock_mutual_info(mocker):
    # Mock the score function itself if needed, or just SelectKBest
    return mocker.patch(f"{FS_PATH}.mutual_info_classif")  # Just check it's passed


@pytest.fixture
def mock_rfe(mocker):
    mock_selector = MagicMock(spec=RFE)
    mock_selector.fit.return_value = None
    # Simulate selecting features 1, 2, 5
    mock_selector.support_ = np.array([True, True, False, False, True])
    return mocker.patch(f"{FS_PATH}.RFE", return_value=mock_selector)


@pytest.fixture
def mock_logistic_regression(mocker):
    return mocker.patch(f"{FS_PATH}.LogisticRegression")


@pytest.fixture
def mock_pca(mocker):
    mock_pca_instance = MagicMock(spec=PCA)
    # Simulate transforming to 2 components
    mock_pca_instance.fit_transform.return_value = np.random.rand(50, 2)
    return mocker.patch(f"{FS_PATH}.PCA", return_value=mock_pca_instance)


# StandardScaler is imported locally in the function, so we patch it in the test itself
# Removing the fixture as it's not needed globally and patching path was incorrect


@pytest.fixture
def mock_select_from_model(mocker):
    mock_selector = MagicMock(spec=SelectFromModel)
    mock_selector.fit.return_value = None
    # Simulate selecting features 1, 3, 5
    mock_selector.get_support.return_value = np.array([True, False, True, False, True])
    return mocker.patch(f"{FS_PATH}.SelectFromModel", return_value=mock_selector)


@pytest.fixture
def mock_lgbm(mocker):
    # Mock the lightgbm library import itself
    mocker.patch(f"{FS_PATH}.LIGHTGBM_AVAILABLE", True)
    mock_lgbm_module = MagicMock()
    # Mock the classifier class on the mocked module
    mock_lgbm_module.LGBMClassifier.return_value = MagicMock()  # Removed spec
    # Mock the import lookup
    mocker.patch.dict(sys.modules, {"lightgbm": mock_lgbm_module})
    # Patch the 'lgb' alias used in the module *after* the try-except
    mocker.patch(f"{FS_PATH}.lgb", mock_lgbm_module, create=True)
    return mock_lgbm_module  # Return the mocked module


@pytest.fixture
def mock_xgboost(mocker):
    # Mock the xgboost library import itself
    mocker.patch(f"{FS_PATH}.XGBOOST_AVAILABLE", True)
    mock_xgb_module = MagicMock()
    # Mock the classifier class on the mocked module
    mock_xgb_module.XGBClassifier.return_value = MagicMock()  # Removed spec
    # Mock the import lookup
    mocker.patch.dict(sys.modules, {"xgboost": mock_xgb_module})
    # Patch the 'xgb' alias used in the module *after* the try-except
    mocker.patch(f"{FS_PATH}.xgb", mock_xgb_module, create=True)
    return mock_xgb_module  # Return the mocked module


# --- Tests ---


def test_select_features_importance(sample_features_target, mock_rf):
    """Test feature selection using RandomForest importance."""
    X, y = sample_features_target
    n_select = 3
    X_selected, selected_names = select_features(
        X, y, method="importance", params={"n_features": n_select}
    )

    mock_rf.assert_called_once()
    mock_rf().fit.assert_called_once_with(X, y)
    assert len(selected_names) == n_select
    # Based on mocked importances [0.3, 0.4, 0.2, 0.0, 0.1], top 3 are feat2, feat1, feat3
    assert selected_names == ["feat2", "feat1", "feat3"]
    pd.testing.assert_frame_equal(X_selected, X[selected_names])


def test_select_features_mutual_info(
    sample_features_target, mock_selectkbest, mock_mutual_info
):
    """Test feature selection using mutual information."""
    X, y = sample_features_target
    n_select = 3
    X_selected, selected_names = select_features(
        X, y, method="mutual_info", params={"n_features": n_select}
    )

    mock_selectkbest.assert_called_once_with(score_func=mock_mutual_info, k=n_select)
    mock_selectkbest().fit.assert_called_once_with(X, y)
    # Based on mock get_support return [0, 1, 2]
    assert selected_names == ["feat1", "feat2", "feat3"]
    pd.testing.assert_frame_equal(X_selected, X[selected_names])


def test_select_features_rfe(
    sample_features_target, mock_rfe, mock_logistic_regression
):
    """Test feature selection using RFE."""
    X, y = sample_features_target
    n_select = 3
    X_selected, selected_names = select_features(
        X, y, method="rfe", params={"n_features": n_select, "step": 2}
    )

    mock_logistic_regression.assert_called_once()  # Check estimator was created
    mock_rfe.assert_called_once_with(
        estimator=ANY, n_features_to_select=n_select, step=2
    )
    mock_rfe().fit.assert_called_once_with(X, y)
    # Based on mock support_ [True, True, False, False, True]
    assert selected_names == ["feat1", "feat2", "feat5"]
    pd.testing.assert_frame_equal(X_selected, X[selected_names])


# Patch StandardScaler where it's imported (its actual source)
@patch("sklearn.preprocessing.StandardScaler")
def test_select_features_pca(mock_scaler_patch, sample_features_target, mock_pca):
    """Test feature selection (dimensionality reduction) using PCA."""
    X, y = sample_features_target
    n_components = 2
    # Setup mock scaler instance that the patch will return
    mock_scaler_instance = MagicMock(spec=StandardScaler)
    mock_scaler_instance.fit_transform.return_value = X.values  # Return numpy array
    mock_scaler_patch.return_value = (
        mock_scaler_instance  # Assign to the patched object
    )

    X_selected, selected_names = select_features(
        X, y, method="pca", params={"n_components": n_components}
    )

    mock_scaler_patch.assert_called_once()  # Check scaler class was instantiated via patch
    mock_scaler_instance.fit_transform.assert_called_once_with(
        X
    )  # Check method called on instance
    mock_pca.assert_called_once_with(n_components=n_components)
    mock_pca().fit_transform.assert_called_once()

    assert len(selected_names) == n_components
    assert selected_names == ["PC1", "PC2"]
    assert X_selected.shape == (len(X), n_components)
    assert list(X_selected.columns) == selected_names
    pd.testing.assert_index_equal(X_selected.index, X.index)


def test_select_features_model_lgbm(
    sample_features_target, mock_select_from_model, mock_lgbm
):
    """Test feature selection using SelectFromModel with LightGBM."""
    X, y = sample_features_target
    n_select = 3
    X_selected, selected_names = select_features(
        X, y, method="model", params={"model_type": "lightgbm", "n_features": n_select}
    )

    mock_lgbm.LGBMClassifier.assert_called_once()  # Check classifier was instantiated via mocked module
    mock_select_from_model.assert_called_once_with(
        ANY, threshold=0.01, max_features=n_select
    )
    mock_select_from_model().fit.assert_called_once_with(X, y)
    # Based on mock get_support [True, False, True, False, True]
    assert selected_names == ["feat1", "feat3", "feat5"]
    pd.testing.assert_frame_equal(X_selected, X[selected_names])


def test_select_features_model_xgboost(
    sample_features_target, mock_select_from_model, mock_xgboost
):
    """Test feature selection using SelectFromModel with XGBoost."""
    X, y = sample_features_target
    n_select = 3
    X_selected, selected_names = select_features(
        X, y, method="model", params={"model_type": "xgboost", "n_features": n_select}
    )

    mock_xgboost.XGBClassifier.assert_called_once()  # Check classifier was instantiated via mocked module
    mock_select_from_model.assert_called_once_with(
        ANY, threshold=0.01, max_features=n_select
    )
    mock_select_from_model().fit.assert_called_once_with(X, y)
    # Based on mock get_support [True, False, True, False, True]
    assert selected_names == ["feat1", "feat3", "feat5"]
    pd.testing.assert_frame_equal(X_selected, X[selected_names])


@patch(f"{FS_PATH}.logger")
def test_select_features_unsupported_method(mock_logger, sample_features_target):
    """Test fallback when an unsupported method is provided."""
    X, y = sample_features_target
    X_selected, selected_names = select_features(X, y, method="unknown")

    mock_logger.warning.assert_called_with(
        "Unsupported feature selection method: unknown"
    )
    pd.testing.assert_frame_equal(X_selected, X)
    assert selected_names == list(X.columns)


@patch(f"{FS_PATH}.SKLEARN_AVAILABLE", False)
@patch(f"{FS_PATH}.logger")
def test_select_features_sklearn_unavailable(mock_logger, sample_features_target):
    """Test fallback when scikit-learn is unavailable."""
    X, y = sample_features_target
    X_selected, selected_names = select_features(X, y, method="importance")

    mock_logger.error.assert_called_with(
        "scikit-learn not available for feature selection"
    )
    pd.testing.assert_frame_equal(X_selected, X)
    assert selected_names == list(X.columns)
