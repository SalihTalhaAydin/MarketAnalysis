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
# Try to import XGBoost
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

# Try to import LightGBM
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

# Try to import CatBoost
try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostClassifier = None  # Define as None if not available


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
    threshold: float = 0.5,  # Threshold currently unused in this function
) -> Dict:
    """
    Evaluate a classification model.

    Args:
        model: Trained model
        X: Feature data
        y: True labels
        class_names: Names of classes
        threshold: Decision threshold (unused here, applied during prediction/signal generation)

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

        # Determine problem type (binary or multiclass) and labels
        y_arr = y.values if isinstance(y, pd.Series) else y
        y_pred_arr = y_pred.values if isinstance(y_pred, pd.Series) else y_pred
        # Correctly determine labels present in true and predicted values
        all_labels = sorted(list(set(np.unique(y_arr)) | set(np.unique(y_pred_arr))))
        n_classes = len(all_labels)

        if class_names is None:
            class_names = [str(label) for label in all_labels]
        elif len(class_names) < n_classes:
            logger.warning(
                f"Provided class_names length ({len(class_names)}) is less than detected labels ({n_classes}). Padding names."
            )
            # Pad based on actual labels found, not forced {-1, 0, 1}
            present_labels_str = [str(label) for label in all_labels]
            original_names = list(class_names)
            class_names = original_names + [
                label for label in present_labels_str if label not in original_names
            ]

        # Classification report (handles multi-class correctly)
        metrics["classification_report"] = classification_report(
            y,
            y_pred,
            labels=all_labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(y, y_pred, labels=all_labels)

        # --- Calculate Macro/Weighted Metrics (Always) ---
        metrics["precision_macro"] = precision_score(
            y, y_pred, average="macro", zero_division=0, labels=all_labels
        )
        metrics["recall_macro"] = recall_score(
            y, y_pred, average="macro", zero_division=0, labels=all_labels
        )
        metrics["f1_macro"] = f1_score(
            y, y_pred, average="macro", zero_division=0, labels=all_labels
        )

        metrics["precision_weighted"] = precision_score(
            y, y_pred, average="weighted", zero_division=0, labels=all_labels
        )
        metrics["recall_weighted"] = recall_score(
            y, y_pred, average="weighted", zero_division=0, labels=all_labels
        )
        metrics["f1_weighted"] = f1_score(
            y, y_pred, average="weighted", zero_division=0, labels=all_labels
        )

        # --- Calculate Probability-Based Metrics (If y_prob available) ---
        if y_prob is not None:
            # Log loss (handles multiclass correctly if labels are provided)
            try:
                # Ensure y_prob columns match the number of classes found
                if y_prob.shape[1] == n_classes:
                    metrics["log_loss"] = log_loss(y, y_prob, labels=all_labels)
                else:
                    # Attempt log loss with model's classes if shape mismatch
                    model_classes = getattr(model, "classes_", None)
                    if (
                        model_classes is not None
                        and len(model_classes) == y_prob.shape[1]
                    ):
                        metrics["log_loss"] = log_loss(y, y_prob, labels=model_classes)
                        logger.warning(
                            f"Log loss calculated using model.classes_ due to label mismatch (found {n_classes}, expected {y_prob.shape[1]})"
                        )
                    else:
                        logger.warning(
                            f"Shape mismatch: y_prob columns ({y_prob.shape[1]}) != n_classes ({n_classes}). Cannot calculate log_loss reliably."
                        )

            except Exception as ll_e:
                logger.warning(f"Could not calculate log loss: {ll_e}")

            # Check if shapes are compatible for probability metrics
            if y_prob.shape[1] != n_classes:
                logger.warning(
                    f"Shape mismatch: y_prob columns ({y_prob.shape[1]}) != n_classes ({n_classes}). Skipping probability metrics like ROC/AUC/PR."
                )
            else:
                # ROC/AUC and PR Curve/AP
                # Binarize y for multiclass ROC/PR calculations
                y_bin = label_binarize(y, classes=all_labels)

                # Check if it's effectively binary (even if labels are e.g., -1, 1)
                is_binary_case = n_classes <= 2 or y_bin.shape[1] <= 2

                if is_binary_case:
                    # Use probabilities of the positive class (assume index 1 or the class '1')
                    try:
                        # Determine positive label index robustly
                        pos_label = 1  # Default assumption
                        if 1 in all_labels:
                            pos_label_idx = all_labels.index(1)
                        elif (
                            n_classes == 2
                        ):  # If only 2 classes and 1 isn't one, use the higher label index
                            pos_label_idx = 1
                            pos_label = all_labels[1]  # Get the actual label value
                        else:  # Fallback if unclear
                            pos_label_idx = 1
                            pos_label = (
                                all_labels[pos_label_idx]
                                if pos_label_idx < n_classes
                                else 1
                            )

                        if pos_label_idx < y_prob.shape[1]:  # Ensure index exists
                            y_prob_pos = y_prob[:, pos_label_idx]
                            # ROC
                            fpr, tpr, _ = roc_curve(
                                y, y_prob_pos, pos_label=pos_label
                            )  # Use actual pos_label
                            metrics["roc_curve"] = {
                                "fpr": fpr.tolist(),
                                "tpr": tpr.tolist(),
                            }
                            # Need to handle cases where roc_auc_score fails (e.g., only one class present in y_true)
                            try:
                                metrics["roc_auc"] = roc_auc_score(y, y_prob_pos)
                            except ValueError as roc_auc_e:
                                logger.warning(
                                    f"Could not calculate ROC AUC score: {roc_auc_e}"
                                )
                                metrics["roc_auc"] = np.nan  # Assign NaN or skip

                            # PR
                            precision, recall, _ = precision_recall_curve(
                                y, y_prob_pos, pos_label=pos_label
                            )  # Use actual pos_label
                            metrics["pr_curve"] = {
                                "precision": precision.tolist(),
                                "recall": recall.tolist(),
                            }
                            try:
                                metrics["average_precision"] = average_precision_score(
                                    y, y_prob_pos
                                )
                            except ValueError as ap_e:
                                logger.warning(
                                    f"Could not calculate Average Precision score: {ap_e}"
                                )
                                metrics["average_precision"] = (
                                    np.nan
                                )  # Assign NaN or skip
                        else:
                            logger.warning(
                                f"Positive class index {pos_label_idx} out of bounds for y_prob shape {y_prob.shape}. Skipping binary ROC/PR."
                            )
                    except Exception as bin_prob_e:
                        logger.warning(
                            f"Could not calculate binary ROC/PR/AUC: {bin_prob_e}"
                        )
                elif n_classes > 2:  # Multiclass case
                    try:
                        # Ensure y_bin has same number of columns as y_prob
                        if y_bin.shape[1] == y_prob.shape[1]:
                            metrics["roc_auc_ovr_weighted"] = roc_auc_score(
                                y_bin, y_prob, multi_class="ovr", average="weighted"
                            )
                            metrics["roc_auc_ovo_weighted"] = roc_auc_score(
                                y_bin, y_prob, multi_class="ovo", average="weighted"
                            )
                            # Also calculate macro average if desired
                            metrics["roc_auc_ovr_macro"] = roc_auc_score(
                                y_bin, y_prob, multi_class="ovr", average="macro"
                            )
                            metrics["roc_auc_ovo_macro"] = roc_auc_score(
                                y_bin, y_prob, multi_class="ovo", average="macro"
                            )
                        else:
                            logger.warning(
                                f"Shape mismatch after binarize: y_bin columns ({y_bin.shape[1]}) != y_prob columns ({y_prob.shape[1]}). Skipping multiclass ROC."
                            )
                    except ValueError as roc_e:
                        logger.warning(f"Could not compute multiclass ROC AUC: {roc_e}")
                    # Note: Multiclass PR AUC is more complex, often done per-class or micro/macro averaged.
                    # Skipping average_precision for multiclass for simplicity for now.

        return metrics

    except Exception as e:
        logger.error(f"Error evaluating model: {e}", exc_info=True)  # Log traceback
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
        model,
        X_train,
        y_train,
        method="built_in",  # Default to built-in first
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
