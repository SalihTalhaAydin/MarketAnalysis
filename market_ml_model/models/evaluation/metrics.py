"""
Model evaluation metrics and utilities.
"""

import json  # Added import
import logging
import os  # Added import
from typing import Any, Dict, List, Optional, Tuple  # Added Tuple

import numpy as np
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

# Import scikit-learn components with error handling
try:
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        classification_report,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.preprocessing import label_binarize  # Added import

    SKLEARN_AVAILABLE = True
except ImportError:
    logger.error("scikit-learn not available for model evaluation")
    SKLEARN_AVAILABLE = False

# Try to import SHAP for feature importance
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    logger.warning("SHAP not installed. Advanced feature importance not available.")
    SHAP_AVAILABLE = False

# Try to import visualization libraries # Added block
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib/Seaborn not installed. Visualization unavailable.")
    VISUALIZATION_AVAILABLE = False


def evaluate_classifier(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    class_names: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Dict:
    """
    Evaluate a classification model.

    Args:
        model: Trained model
        X: Feature data
        y: True labels
        class_names: Names of classes
        threshold: Decision threshold for binary classification

    Returns:
        Dictionary of evaluation metrics
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn not available for model evaluation")
        return {}

    try:
        # Get predictions
        y_pred = model.predict(X)

        # Get probabilities if available
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)
        else:
            y_prob = None

        # Basic metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
        }

        # Determine problem type (binary or multiclass)
        unique_classes = np.unique(np.concatenate([y.unique(), np.unique(y_pred)]))
        n_classes = len(unique_classes)

        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]
        else:
            # Ensure we have enough class names
            if len(class_names) < n_classes:
                class_names = list(class_names) + [
                    str(i) for i in range(len(class_names), n_classes)
                ]

        # Classification report
        metrics["classification_report"] = classification_report(
            y, y_pred, output_dict=True, zero_division=0  # Added zero_division
        )

        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(y, y_pred)

        # Additional metrics
        if n_classes == 2:
            # Binary classification
            metrics["precision"] = precision_score(
                y, y_pred, zero_division=0
            )  # Added zero_division
            metrics["recall"] = recall_score(
                y, y_pred, zero_division=0
            )  # Added zero_division
            metrics["f1"] = f1_score(y, y_pred, zero_division=0)  # Added zero_division

            if y_prob is not None:
                # ROC curve
                fpr, tpr, _ = roc_curve(y, y_prob[:, 1])
                metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

                # ROC AUC
                metrics["roc_auc"] = roc_auc_score(y, y_prob[:, 1])

                # Precision-Recall curve
                precision, recall, _ = precision_recall_curve(y, y_prob[:, 1])
                metrics["pr_curve"] = {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                }

                # Average precision
                metrics["average_precision"] = average_precision_score(y, y_prob[:, 1])

                # Log loss
                metrics["log_loss"] = log_loss(y, y_prob)
        else:
            # Multiclass classification
            metrics["precision_macro"] = precision_score(
                y, y_pred, average="macro", zero_division=0
            )  # Added zero_division
            metrics["recall_macro"] = recall_score(
                y, y_pred, average="macro", zero_division=0
            )  # Added zero_division
            metrics["f1_macro"] = f1_score(
                y, y_pred, average="macro", zero_division=0
            )  # Added zero_division

            metrics["precision_weighted"] = precision_score(
                y, y_pred, average="weighted", zero_division=0
            )  # Added zero_division
            metrics["recall_weighted"] = recall_score(
                y, y_pred, average="weighted", zero_division=0
            )  # Added zero_division
            metrics["f1_weighted"] = f1_score(
                y, y_pred, average="weighted", zero_division=0
            )  # Added zero_division

            if y_prob is not None:
                # Convert y to one-hot encoding for multiclass ROC AUC
                y_bin = label_binarize(y, classes=unique_classes)

                if (
                    n_classes > 2 and y_bin.shape[1] == y_prob.shape[1]
                ):  # Check shapes match
                    # Compute ROC AUC for each class
                    try:
                        metrics["roc_auc_ovr"] = roc_auc_score(
                            y_bin, y_prob, multi_class="ovr"
                        )
                        metrics["roc_auc_ovo"] = roc_auc_score(
                            y_bin, y_prob, multi_class="ovo"
                        )
                    except ValueError as roc_e:
                        logger.warning(f"Could not compute multiclass ROC AUC: {roc_e}")

                # Log loss
                metrics["log_loss"] = log_loss(y, y_prob)

        return metrics

    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return {"error": str(e)}


def compute_feature_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "built_in",
    n_repeats: int = 10,
) -> pd.DataFrame:
    """
    Compute feature importance using various methods.

    Args:
        model: Trained model
        X: Feature data
        y: Target data
        method: Method to compute importance ('built_in', 'permutation', 'shap')
        n_repeats: Number of repeats for permutation importance

    Returns:
        DataFrame with feature importance
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn not available for feature importance")
        return pd.DataFrame()

    try:
        if method == "built_in":
            # Use built-in feature importance if available
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
                return pd.DataFrame(
                    {"Feature": X.columns, "Importance": importance}
                ).sort_values("Importance", ascending=False)
            elif hasattr(model, "coef_"):  # For linear models
                importance = np.abs(model.coef_[0])  # Use absolute coefficient values
                return pd.DataFrame(
                    {"Feature": X.columns, "Importance": importance}
                ).sort_values("Importance", ascending=False)
            else:
                logger.warning(
                    "Model does not have built-in feature_importances_ or coef_"
                )
                # Fall back to permutation importance
                method = "permutation"

        if method == "permutation":
            # Use permutation importance
            perm_importance = permutation_importance(
                model,
                X,
                y,
                n_repeats=n_repeats,
                random_state=42,
                n_jobs=-1,  # Added n_jobs
            )

            return pd.DataFrame(
                {
                    "Feature": X.columns,
                    "Importance": perm_importance.importances_mean,
                    "Std": perm_importance.importances_std,
                }
            ).sort_values("Importance", ascending=False)

        elif method == "shap" and SHAP_AVAILABLE:
            # Use SHAP values for feature importance
            try:
                # Create explainer - handle different model types
                # Check availability flags and use module aliases if available
                is_tree_model = False
                if XGBOOST_AVAILABLE and xgb and isinstance(model, xgb.XGBClassifier):
                    is_tree_model = True
                elif (
                    LIGHTGBM_AVAILABLE and lgb and isinstance(model, lgb.LGBMClassifier)
                ):
                    is_tree_model = True
                elif (
                    CATBOOST_AVAILABLE
                    and CatBoostClassifier
                    and isinstance(model, CatBoostClassifier)
                ):
                    is_tree_model = True

                if is_tree_model:
                    explainer = shap.TreeExplainer(model)
                elif hasattr(model, "predict_proba"):
                    explainer = shap.KernelExplainer(
                        model.predict_proba, X
                    )  # Use KernelExplainer for others
                else:
                    explainer = shap.KernelExplainer(model.predict, X)

                shap_values = explainer.shap_values(X)  # Calculate SHAP values

                # Handle different SHAP value structures (multi-class vs binary)
                if isinstance(shap_values, list):  # Multi-class case
                    mean_abs_shap = np.abs(shap_values[1]).mean(
                        axis=0
                    )  # Use class 1 for importance
                else:  # Binary case
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)

                return pd.DataFrame(
                    {"Feature": X.columns, "Importance": mean_abs_shap}
                ).sort_values("Importance", ascending=False)
            except Exception as shap_e:
                logger.warning(
                    f"Could not compute SHAP values: {shap_e}. Falling back to permutation."
                )
                return compute_feature_importance(
                    model, X, y, method="permutation", n_repeats=n_repeats
                )

        else:
            if method == "shap" and not SHAP_AVAILABLE:
                logger.warning(
                    "SHAP method requested but SHAP library not available. Falling back to permutation."
                )
                return compute_feature_importance(
                    model, X, y, method="permutation", n_repeats=n_repeats
                )
            else:
                logger.warning(
                    f"Unsupported importance method: {method}. Falling back to permutation."
                )
                return compute_feature_importance(
                    model, X, y, method="permutation", n_repeats=n_repeats
                )

    except Exception as e:
        logger.error(f"Error computing feature importance: {e}")
        return pd.DataFrame()


# --- Plotting Functions ---


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    figsize: Tuple[int, int] = (10, 8),
    filename: Optional[str] = None,
) -> None:
    """
    Plot a confusion matrix.

    Args:
        conf_matrix: Confusion matrix
        class_names: Names of classes
        title: Plot title
        normalize: Whether to normalize the matrix
        figsize: Figure size
        filename: If provided, save plot to this file
    """
    if not VISUALIZATION_AVAILABLE:
        logger.error("Matplotlib/Seaborn not available for plotting")
        return

    # Normalize if requested
    if normalize:
        conf_matrix_norm = (
            conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
        )
        plot_data = conf_matrix_norm
        fmt = ".2f"
    else:
        plot_data = conf_matrix
        fmt = "d"

    # Create class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(conf_matrix.shape[0])]

    # Create plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        plot_data,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(title)
    plt.tight_layout()

    # Save or show plot
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to {filename}")
    else:
        plt.show()

    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (12, 8),
    filename: Optional[str] = None,
) -> None:
    """
    Plot feature importance.

    Args:
        importance_df: DataFrame with feature importance
        top_n: Number of top features to plot
        title: Plot title
        figsize: Figure size
        filename: If provided, save plot to this file
    """
    if not VISUALIZATION_AVAILABLE:
        logger.error("Matplotlib/Seaborn not available for plotting")
        return

    if importance_df.empty:
        logger.warning("Empty importance DataFrame, nothing to plot")
        return

    # Limit to top N features
    if len(importance_df) > top_n:
        importance_df = importance_df.head(top_n)

    # Create plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        x="Importance",
        y="Feature",
        data=importance_df.sort_values(
            by="Importance", ascending=False
        ),  # Sort for plot
        palette="viridis",
    )

    # Add error bars if available
    if "Std" in importance_df.columns:
        # Need to reorder Std based on sorted Importance for error bars
        sorted_importance = importance_df.sort_values(by="Importance", ascending=False)
        ax.errorbar(
            x=sorted_importance["Importance"],
            y=range(len(sorted_importance)),
            xerr=sorted_importance["Std"],
            fmt="none",
            ecolor="black",
            capsize=3,
        )

    plt.title(title)
    plt.tight_layout()

    # Save or show plot
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved feature importance plot to {filename}")
    else:
        plt.show()

    plt.close()


def plot_roc_curve(
    fpr: List[float],
    tpr: List[float],
    roc_auc: float,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 8),
    filename: Optional[str] = None,
) -> None:
    """
    Plot ROC curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: ROC AUC score
        title: Plot title
        figsize: Figure size
        filename: If provided, save plot to this file
    """
    if not VISUALIZATION_AVAILABLE:
        logger.error("Matplotlib/Seaborn not available for plotting")
        return

    # Create plot
    plt.figure(figsize=figsize)
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save or show plot
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved ROC curve plot to {filename}")
    else:
        plt.show()

    plt.close()


def plot_precision_recall_curve(
    precision: List[float],
    recall: List[float],
    avg_precision: float,
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = (8, 8),
    filename: Optional[str] = None,
) -> None:
    """
    Plot precision-recall curve.

    Args:
        precision: Precision values
        recall: Recall values
        avg_precision: Average precision score
        title: Plot title
        figsize: Figure size
        filename: If provided, save plot to this file
    """
    if not VISUALIZATION_AVAILABLE:
        logger.error("Matplotlib/Seaborn not available for plotting")
        return

    # Create plot
    plt.figure(figsize=figsize)
    plt.plot(
        recall, precision, color="darkorange", lw=2, label=f"AP = {avg_precision:.2f}"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save or show plot
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved Precision-Recall curve plot to {filename}")
    else:
        plt.show()

    plt.close()


# --- Reporting Function ---


def generate_model_report(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Generate a comprehensive model report including metrics and plots.

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        class_names: Names of classes
        output_dir: Directory to save report files (plots, JSON summary)

    Returns:
        Dictionary with report metrics
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn not available for model report")
        return {}

    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Training and test metrics
    logger.info("Evaluating model on training data...")
    train_metrics = evaluate_classifier(model, X_train, y_train, class_names)
    logger.info("Evaluating model on test data...")
    test_metrics = evaluate_classifier(model, X_test, y_test, class_names)

    # Feature importance
    logger.info("Computing feature importance...")
    importance_df = compute_feature_importance(
        model, X_train, y_train, method="built_in"  # Default to built-in first
    )
    if importance_df.empty:  # Try permutation if built-in fails
        importance_df = compute_feature_importance(
            model, X_train, y_train, method="permutation"
        )

    # Generate plots if output directory provided
    if output_dir and VISUALIZATION_AVAILABLE:
        logger.info(f"Generating plots in {output_dir}...")
        # Confusion matrices
        if "confusion_matrix" in train_metrics:
            plot_confusion_matrix(
                train_metrics["confusion_matrix"],
                class_names=class_names,
                title="Training Confusion Matrix",
                filename=os.path.join(output_dir, "train_confusion_matrix.png"),
            )

        if "confusion_matrix" in test_metrics:
            plot_confusion_matrix(
                test_metrics["confusion_matrix"],
                class_names=class_names,
                title="Test Confusion Matrix",
                filename=os.path.join(output_dir, "test_confusion_matrix.png"),
            )

        # Feature importance
        if not importance_df.empty:
            plot_feature_importance(
                importance_df,
                title="Feature Importance",
                filename=os.path.join(output_dir, "feature_importance.png"),
            )

        # ROC curve (binary classification)
        if "roc_curve" in test_metrics and "roc_auc" in test_metrics:
            plot_roc_curve(
                test_metrics["roc_curve"]["fpr"],
                test_metrics["roc_curve"]["tpr"],
                test_metrics["roc_auc"],
                title="ROC Curve (Test Set)",
                filename=os.path.join(output_dir, "roc_curve.png"),
            )

        # Precision-Recall curve (binary classification)
        if "pr_curve" in test_metrics and "average_precision" in test_metrics:
            plot_precision_recall_curve(
                test_metrics["pr_curve"]["precision"],
                test_metrics["pr_curve"]["recall"],
                test_metrics["average_precision"],
                title="Precision-Recall Curve (Test Set)",
                filename=os.path.join(output_dir, "pr_curve.png"),
            )

    # Create summary report (excluding potentially large objects like curves)
    summary = {
        "train_metrics": {
            k: v
            for k, v in train_metrics.items()
            if k
            not in [
                "confusion_matrix",
                "classification_report",
                "roc_curve",
                "pr_curve",
            ]
        },
        "test_metrics": {
            k: v
            for k, v in test_metrics.items()
            if k
            not in [
                "confusion_matrix",
                "classification_report",
                "roc_curve",
                "pr_curve",
            ]
        },
        "feature_importance_summary": (
            importance_df.head(20).to_dict() if not importance_df.empty else {}
        ),
        "model_info": {
            "type": type(model).__name__,
            "params": model.get_params() if hasattr(model, "get_params") else {},
        },
    }

    # Save summary as JSON if output directory provided
    if output_dir:
        report_path = os.path.join(output_dir, "model_report.json")
        try:
            with open(report_path, "w") as f:
                json.dump(
                    summary, f, indent=4, default=str
                )  # Use default=str for non-serializable
            logger.info(f"Saved model report summary to {report_path}")
        except Exception as json_e:
            logger.error(f"Failed to save model report JSON: {json_e}")

        # Save full feature importance as CSV
        if not importance_df.empty:
            importance_path = os.path.join(output_dir, "feature_importance.csv")
            try:
                importance_df.to_csv(importance_path, index=False)
                logger.info(f"Saved feature importance to {importance_path}")
            except Exception as csv_e:
                logger.error(f"Failed to save feature importance CSV: {csv_e}")

    return summary
