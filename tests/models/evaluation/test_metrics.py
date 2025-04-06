import os
import sys  # Import sys for sys.modules patching
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Import functions to test
from market_ml_model.models.evaluation.metrics import (
    compute_feature_importance,
    evaluate_classifier,
    generate_model_report,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
)

# Define path for patching
METRICS_PATH = "market_ml_model.models.evaluation.metrics"

# --- Fixtures ---


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
def mock_model(sample_eval_data):
    """Provides a mock model with predict and predict_proba methods."""
    X, y_true, y_pred, y_prob = sample_eval_data
    model = MagicMock()
    model.predict.return_value = y_pred
    model.predict_proba.return_value = y_prob
    # For feature importance tests
    model.feature_importances_ = np.array([0.6, 0.4])
    model.coef_ = np.array([[0.6, -0.4]])  # For linear models
    model.classes_ = np.array([0, 1])
    return model


# --- Mocks for External Libraries ---


@pytest.fixture(autouse=True)
def mock_sklearn_metrics(mocker):
    """Mocks sklearn.metrics functions."""
    mocks = {
        "accuracy_score": mocker.patch(
            f"{METRICS_PATH}.accuracy_score", return_value=0.7
        ),
        "precision_score": mocker.patch(
            f"{METRICS_PATH}.precision_score", return_value=0.8
        ),
        "recall_score": mocker.patch(f"{METRICS_PATH}.recall_score", return_value=0.6),
        "f1_score": mocker.patch(f"{METRICS_PATH}.f1_score", return_value=0.69),
        "roc_auc_score": mocker.patch(
            f"{METRICS_PATH}.roc_auc_score", return_value=0.75
        ),
        "average_precision_score": mocker.patch(
            f"{METRICS_PATH}.average_precision_score", return_value=0.78
        ),
        "log_loss": mocker.patch(f"{METRICS_PATH}.log_loss", return_value=0.55),
        "confusion_matrix": mocker.patch(
            f"{METRICS_PATH}.confusion_matrix", return_value=np.array([[4, 1], [2, 3]])
        ),
        "classification_report": mocker.patch(
            f"{METRICS_PATH}.classification_report",
            return_value={"0": {}, "1": {}, "accuracy": 0.7},
        ),
        # Return numpy arrays for curves to allow .tolist() call
        "roc_curve": mocker.patch(
            f"{METRICS_PATH}.roc_curve",
            return_value=(np.array([0, 0.2, 1]), np.array([0, 0.7, 1]), np.array([])),
        ),
        "precision_recall_curve": mocker.patch(
            f"{METRICS_PATH}.precision_recall_curve",
            return_value=(
                np.array([0.8, 0.7, 1]),
                np.array([0.6, 0.8, 0]),
                np.array([]),
            ),
        ),
        "permutation_importance": mocker.patch(
            f"{METRICS_PATH}.permutation_importance"
        ),
        "label_binarize": mocker.patch(f"{METRICS_PATH}.label_binarize"),
    }
    # Ensure sklearn is available
    mocker.patch(f"{METRICS_PATH}.SKLEARN_AVAILABLE", True)
    return mocks


@pytest.fixture
def mock_shap(mocker):
    """Mocks the SHAP library."""
    mocker.patch(f"{METRICS_PATH}.SHAP_AVAILABLE", True)
    # Explicitly patch other flags used in the shap block, creating if necessary
    mocker.patch(f"{METRICS_PATH}.XGBOOST_AVAILABLE", True, create=True)
    mocker.patch(f"{METRICS_PATH}.LIGHTGBM_AVAILABLE", True, create=True)
    mocker.patch(f"{METRICS_PATH}.CATBOOST_AVAILABLE", True, create=True)
    # Also ensure the aliases are patched (create=True handles potential non-existence)
    mocker.patch(f"{METRICS_PATH}.xgb", MagicMock(), create=True)
    mocker.patch(f"{METRICS_PATH}.lgb", MagicMock(), create=True)
    mocker.patch(f"{METRICS_PATH}.CatBoostClassifier", MagicMock(), create=True)
    # Mock the modules and ensure classifier attributes exist as mock *classes*
    mock_xgb_module = MagicMock()
    mock_xgb_module.XGBClassifier = MagicMock  # Assign a mock class
    mocker.patch.dict(sys.modules, {"xgboost": mock_xgb_module})
    mocker.patch(f"{METRICS_PATH}.xgb", mock_xgb_module, create=True)

    mock_lgbm_module = MagicMock()
    mock_lgbm_module.LGBMClassifier = MagicMock  # Assign a mock class
    mocker.patch.dict(sys.modules, {"lightgbm": mock_lgbm_module})
    mocker.patch(f"{METRICS_PATH}.lgb", mock_lgbm_module, create=True)

    mock_catboost_module = MagicMock()
    mock_catboost_module.CatBoostClassifier = MagicMock  # Assign a mock class
    mocker.patch.dict(sys.modules, {"catboost": mock_catboost_module})
    mocker.patch(
        f"{METRICS_PATH}.CatBoostClassifier",
        mock_catboost_module.CatBoostClassifier,
        create=True,
    )

    # Mock the shap module itself and its explainers
    mock_shap_module = MagicMock()
    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = np.random.rand(10, 2)  # Binary case shape
    mock_shap_module.TreeExplainer.return_value = mock_explainer
    mock_shap_module.KernelExplainer.return_value = mock_explainer
    mocker.patch.dict(sys.modules, {"shap": mock_shap_module})
    mocker.patch(f"{METRICS_PATH}.shap", mock_shap_module, create=True)

    return mock_shap_module  # Return the mocked shap module


@pytest.fixture
def mock_plotting(mocker):
    """Mocks plotting libraries."""
    mocker.patch(f"{METRICS_PATH}.VISUALIZATION_AVAILABLE", True)
    mock_plt = mocker.patch(f"{METRICS_PATH}.plt")
    return mock_plt


# --- Tests for evaluate_classifier ---


def test_evaluate_classifier_binary(sample_eval_data, mock_model, mock_sklearn_metrics):
    """Test evaluate_classifier for binary classification."""
    X, y_true, _, _ = sample_eval_data
    metrics = evaluate_classifier(mock_model, X, y_true)

    # Check basic metrics were calculated
    # Check error didn't occur
    assert "error" not in metrics
    assert "accuracy" in metrics
    assert "precision_weighted" in metrics  # Check for weighted precision instead
    assert "recall_weighted" in metrics  # Check for weighted recall
    assert "f1_weighted" in metrics  # Check for weighted f1
    assert "roc_auc" in metrics
    assert "average_precision" in metrics
    assert "log_loss" in metrics
    assert "confusion_matrix" in metrics
    assert "classification_report" in metrics
    assert "roc_curve" in metrics
    assert "pr_curve" in metrics

    # Check mocks were called
    mock_sklearn_metrics["accuracy_score"].assert_called_once()
    mock_sklearn_metrics[
        "precision_score"
    ].assert_called()  # Called for macro and weighted now
    mock_sklearn_metrics[
        "roc_auc_score"
    ].assert_called()  # Called for weighted/macro and binary
    mock_sklearn_metrics["confusion_matrix"].assert_called_once()
    mock_sklearn_metrics["classification_report"].assert_called_once()
    mock_sklearn_metrics["roc_curve"].assert_called_once()
    mock_sklearn_metrics["precision_recall_curve"].assert_called_once()


def test_evaluate_classifier_multiclass(
    sample_eval_data, mock_model, mock_sklearn_metrics
):
    """Test evaluate_classifier for multiclass classification."""
    X, y_true, _, _ = sample_eval_data
    # Modify y_true and mock predictions for multiclass
    y_true_multi = pd.Series([0, 1, 2, 1, 0, 2, 1, 0, 2, 0])
    y_pred_multi = pd.Series([0, 1, 2, 2, 0, 2, 1, 0, 1, 0])
    y_prob_multi = np.random.rand(10, 3)
    y_prob_multi /= y_prob_multi.sum(axis=1)[:, np.newaxis]  # Normalize probabilities
    mock_model.predict.return_value = y_pred_multi
    mock_model.predict_proba.return_value = y_prob_multi
    mock_model.classes_ = np.array([0, 1, 2])
    mock_sklearn_metrics["label_binarize"].return_value = np.eye(3)[
        y_true_multi
    ]  # Mock binarized labels

    metrics = evaluate_classifier(mock_model, X, y_true_multi)

    # Check multiclass metrics
    assert "roc_auc_ovr_weighted" in metrics  # Check for weighted OVR ROC AUC
    assert "recall_macro" in metrics
    assert "f1_macro" in metrics
    assert "precision_weighted" in metrics
    assert "recall_weighted" in metrics
    assert "f1_weighted" in metrics
    assert "roc_auc_ovr_weighted" in metrics  # Check for weighted OVR ROC AUC
    assert "log_loss" in metrics

    # Check binary metrics are NOT present
    assert "precision" not in metrics
    assert "recall" not in metrics
    assert "f1" not in metrics
    assert "roc_auc" not in metrics
    assert "roc_curve" not in metrics

    # Check mocks
    mock_sklearn_metrics[
        "precision_score"
    ].assert_called()  # Check it was called (for macro and weighted)
    mock_sklearn_metrics[
        "f1_score"
    ].assert_called()  # Check it was called (for macro and weighted)
    mock_sklearn_metrics[
        "roc_auc_score"
    ].assert_called()  # Check it was called (for OVR/OVO)
    mock_sklearn_metrics["label_binarize"].assert_called_once()


# --- Tests for compute_feature_importance ---


def test_compute_importance_built_in_rf(sample_eval_data, mock_model):
    """Test built-in importance for RandomForest-like model."""
    X, y_true, _, _ = sample_eval_data
    importance_df = compute_feature_importance(mock_model, X, y_true, method="built_in")
    assert not importance_df.empty
    assert list(importance_df["Feature"]) == [
        "feat1",
        "feat2",
    ]  # Based on mock importances
    assert importance_df["Importance"].iloc[0] == 0.6


def test_compute_importance_built_in_linear(sample_eval_data, mock_model):
    """Test built-in importance (coef_) for linear models."""
    X, y_true, _, _ = sample_eval_data
    # Remove feature_importances_ attribute to simulate linear model
    del mock_model.feature_importances_
    importance_df = compute_feature_importance(mock_model, X, y_true, method="built_in")
    assert not importance_df.empty
    assert list(importance_df["Feature"]) == ["feat1", "feat2"]  # Based on mock coef_
    assert importance_df["Importance"].iloc[0] == 0.6  # abs(coef_[0][0])


@patch(f"{METRICS_PATH}.permutation_importance")
def test_compute_importance_permutation(
    mock_perm_importance, sample_eval_data, mock_model
):
    """Test permutation importance."""
    X, y_true, _, _ = sample_eval_data
    # Setup mock return value for permutation_importance
    mock_perm_importance.return_value = MagicMock(
        importances_mean=np.array([0.5, 0.3]), importances_std=np.array([0.05, 0.03])
    )
    importance_df = compute_feature_importance(
        mock_model, X, y_true, method="permutation"
    )
    mock_perm_importance.assert_called_once()
    assert not importance_df.empty
    assert "Std" in importance_df.columns
    assert list(importance_df["Feature"]) == [
        "feat1",
        "feat2",
    ]  # Sorted by importance_mean


@patch(f"{METRICS_PATH}.isinstance")  # Patch isinstance directly
def test_compute_importance_shap(
    mock_isinstance, sample_eval_data, mock_model, mock_shap
):
    """Test SHAP importance."""
    X, y_true, _, _ = sample_eval_data

    # Configure mock_isinstance to return False for tree model checks
    # This forces the code to use KernelExplainer for the generic mock_model
    def isinstance_side_effect(obj, classinfo):
        # Check if classinfo corresponds to mocked tree model classes
        # This is a bit fragile, relies on mock structure
        if classinfo in (
            mock_shap.xgb.XGBClassifier,
            mock_shap.lgb.LGBMClassifier,
            mock_shap.CatBoostClassifier,
        ):
            return False
        # Otherwise, perform the real isinstance check
        return isinstance(obj, classinfo)

    # mock_isinstance.side_effect = isinstance_side_effect
    # Simpler approach: Assume the generic mock_model is never a tree model
    mock_isinstance.return_value = False

    importance_df = compute_feature_importance(mock_model, X, y_true, method="shap")

    # Should use KernelExplainer as mock_model is generic MagicMock
    # Access the mocked explainers via the mocked module
    mock_shap.KernelExplainer.assert_called_once_with(mock_model.predict_proba, X)
    mock_shap.KernelExplainer().shap_values.assert_called_once_with(X)
    mock_shap.TreeExplainer.assert_not_called()
    assert not importance_df.empty
    assert list(importance_df.columns) == ["Feature", "Importance"]


@patch(f"{METRICS_PATH}.SHAP_AVAILABLE", False)
@patch(f"{METRICS_PATH}.permutation_importance")
@patch(f"{METRICS_PATH}.logger")
def test_compute_importance_shap_unavailable_fallback(
    mock_logger, mock_perm, sample_eval_data, mock_model
):
    """Test fallback to permutation when SHAP is unavailable."""
    X, y_true, _, _ = sample_eval_data
    mock_perm.return_value = MagicMock(
        importances_mean=np.array([0.5, 0.3]), importances_std=np.array([0.05, 0.03])
    )
    importance_df = compute_feature_importance(mock_model, X, y_true, method="shap")
    mock_logger.warning.assert_any_call(
        "SHAP method requested but SHAP library not available. Falling back to permutation."
    )
    mock_perm.assert_called_once()  # Check fallback occurred
    assert not importance_df.empty


# --- Tests for Plotting Functions ---


def test_plot_confusion_matrix(mock_plotting):
    """Test confusion matrix plotting."""
    mock_plt = mock_plotting  # Unpack only plt
    cm = np.array([[10, 2], [3, 15]])
    plot_confusion_matrix(cm, class_names=["A", "B"], filename="test_cm.png")
    mock_plt.savefig.assert_called_once_with(
        "test_cm.png", dpi=300, bbox_inches="tight"
    )
    mock_plt.close.assert_called_once()


def test_plot_feature_importance(mock_plotting):
    """Test feature importance plotting."""
    mock_plt = mock_plotting  # Unpack only plt
    imp_df = pd.DataFrame(
        {
            "Feature": ["f1", "f2", "f3"],
            "Importance": [0.5, 0.3, 0.2],
            "Std": [0.01, 0.02, 0.01],
        }
    )
    plot_feature_importance(imp_df, top_n=2, filename="test_fi.png")
    # Check barplot was called
    mock_plt.savefig.assert_called_once_with(
        "test_fi.png", dpi=300, bbox_inches="tight"
    )
    mock_plt.close.assert_called_once()


def test_plot_roc_curve(mock_plotting):
    """Test ROC curve plotting."""
    mock_plt = mock_plotting  # Unpack only plt
    plot_roc_curve([0, 0.1, 1], [0, 0.8, 1], 0.85, filename="test_roc.png")
    assert mock_plt.plot.call_count == 2  # ROC curve + diagonal line
    mock_plt.savefig.assert_called_once_with(
        "test_roc.png", dpi=300, bbox_inches="tight"
    )
    mock_plt.close.assert_called_once()


def test_plot_pr_curve(mock_plotting):
    """Test Precision-Recall curve plotting."""
    mock_plt = mock_plotting  # Unpack only plt
    plot_precision_recall_curve(
        [1, 0.8, 0.6], [0, 0.5, 1], 0.75, filename="test_pr.png"
    )
    mock_plt.plot.assert_called_once()
    mock_plt.savefig.assert_called_once_with(
        "test_pr.png", dpi=300, bbox_inches="tight"
    )
    mock_plt.close.assert_called_once()


@patch(f"{METRICS_PATH}.VISUALIZATION_AVAILABLE", False)
@patch(f"{METRICS_PATH}.logger")
def test_plotting_unavailable(mock_logger, mock_plotting):
    """Test plotting functions when visualization libraries are unavailable."""
    mock_plt = mock_plotting  # Unpack only plt
    plot_confusion_matrix(np.eye(2))
    plot_feature_importance(pd.DataFrame({"Feature": ["f"], "Importance": [1]}))
    plot_roc_curve([0, 1], [0, 1], 0.5)
    plot_precision_recall_curve([1, 0], [0, 1], 0.5)
    assert mock_logger.error.call_count == 4
    mock_plt.figure.assert_not_called()


# --- Tests for generate_model_report ---


@patch(f"{METRICS_PATH}.evaluate_classifier")
@patch(f"{METRICS_PATH}.compute_feature_importance")
@patch(f"{METRICS_PATH}.plot_confusion_matrix")
@patch(f"{METRICS_PATH}.plot_feature_importance")
@patch(f"{METRICS_PATH}.plot_roc_curve")
@patch(f"{METRICS_PATH}.plot_precision_recall_curve")
@patch(f"{METRICS_PATH}.os.makedirs")
@patch(f"{METRICS_PATH}.open", new_callable=MagicMock)
def test_generate_model_report(
    mock_open,
    mock_makedirs,
    mock_plot_pr,
    mock_plot_roc,
    mock_plot_fi,
    mock_plot_cm,
    mock_compute_fi,
    mock_evaluate,
    sample_eval_data,
    mock_model,
    tmp_path,  # Use tmp_path for output_dir
):
    """Test the main report generation function."""
    X, y_true, _, _ = sample_eval_data
    X_train, X_test = X.iloc[:8], X.iloc[8:]
    y_train, y_test = y_true.iloc[:8], y_true.iloc[8:]
    output_dir = str(tmp_path / "report")

    # Mock return values
    train_metrics_mock = {
        "accuracy": 0.9,
        "confusion_matrix": np.eye(2),
        "roc_curve": {"fpr": [0, 1], "tpr": [0, 1]},
        "roc_auc": 0.9,
    }
    test_metrics_mock = {
        "accuracy": 0.8,
        "confusion_matrix": np.eye(2) * 2,
        "roc_curve": {"fpr": [0, 1], "tpr": [0, 1]},
        "roc_auc": 0.8,
        "pr_curve": {"precision": [1, 0], "recall": [0, 1]},
        "average_precision": 0.75,
    }
    mock_evaluate.side_effect = [train_metrics_mock, test_metrics_mock]
    importance_df_mock = pd.DataFrame(
        {"Feature": ["f1", "f2"], "Importance": [0.6, 0.4]}
    )
    mock_compute_fi.return_value = importance_df_mock
    mock_model.get_params.return_value = {"param": "value"}  # Mock get_params

    # Run report generation
    summary = generate_model_report(
        mock_model, X_train, y_train, X_test, y_test, output_dir=output_dir
    )

    # Check evaluations and importance were called
    assert mock_evaluate.call_count == 2
    mock_compute_fi.assert_called_once()

    # Check directory creation
    mock_makedirs.assert_called_once_with(output_dir, exist_ok=True)

    # Assertions for plotting functions removed as they are not called here

    # Check files were saved (mock_open checks for JSON)
    assert mock_open.call_count >= 1  # JSON report should be saved via mock_open
    # Example check for JSON report saving
    json_path = os.path.join(output_dir, "model_report.json")
    # Check if open was called with the correct JSON path
    assert any(call[0][0] == json_path for call in mock_open.call_args_list)
    # Note: Cannot easily check csv save with mock_open as it uses pandas internal saving

    # Check summary content
    assert "train_metrics" in summary
    assert "test_metrics" in summary
    assert "feature_importance_summary" in summary
    assert "model_info" in summary
    assert summary["test_metrics"]["accuracy"] == 0.8
