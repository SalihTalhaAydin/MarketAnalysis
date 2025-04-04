"""
Model evaluation metrics and utilities.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any

# Setup logging
logger = logging.getLogger(__name__)

# Import scikit-learn components with error handling
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        precision_recall_curve, roc_curve, roc_auc_score, 
        average_precision_score, log_loss,
        confusion_matrix, classification_report
    )
    from sklearn.inspection import permutation_importance
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


def evaluate_classifier(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    class_names: Optional[List[str]] = None,
    threshold: float = 0.5
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
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)
        else:
            y_prob = None
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
        }
        
        # Determine problem type (binary or multiclass)
        unique_classes = np.unique(np.concatenate([y.unique(), np.unique(y_pred)]))
        n_classes = len(unique_classes)
        
        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]
        else:
            # Ensure we have enough class names
            if len(class_names) < n_classes:
                class_names = list(class_names) + [str(i) for i in range(len(class_names), n_classes)]
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y, y_pred, output_dict=True
        )
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y, y_pred)
        
        # Additional metrics
        if n_classes == 2:
            # Binary classification
            metrics['precision'] = precision_score(y, y_pred)
            metrics['recall'] = recall_score(y, y_pred)
            metrics['f1'] = f1_score(y, y_pred)
            
            if y_prob is not None:
                # ROC curve
                fpr, tpr, _ = roc_curve(y, y_prob[:, 1])
                metrics['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                }
                
                # ROC AUC
                metrics['roc_auc'] = roc_auc_score(y, y_prob[:, 1])
                
                # Precision-Recall curve
                precision, recall, _ = precision_recall_curve(y, y_prob[:, 1])
                metrics['pr_curve'] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist()
                }
                
                # Average precision
                metrics['average_precision'] = average_precision_score(y, y_prob[:, 1])
                
                # Log loss
                metrics['log_loss'] = log_loss(y, y_prob)
        else:
            # Multiclass classification
            metrics['precision_macro'] = precision_score(y, y_pred, average='macro')
            metrics['recall_macro'] = recall_score(y, y_pred, average='macro')
            metrics['f1_macro'] = f1_score(y, y_pred, average='macro')
            
            metrics['precision_weighted'] = precision_score(y, y_pred, average='weighted')
            metrics['recall_weighted'] = recall_score(y, y_pred, average='weighted')
            metrics['f1_weighted'] = f1_score(y, y_pred, average='weighted')
            
            if y_prob is not None:
                # Convert y to one-hot encoding for multiclass ROC AUC
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y, classes=unique_classes)
                
                if n_classes > 2:
                    # Compute ROC AUC for each class
                    metrics['roc_auc_ovr'] = roc_auc_score(y_bin, y_prob, multi_class='ovr')
                    metrics['roc_auc_ovo'] = roc_auc_score(y_bin, y_prob, multi_class='ovo')
                
                # Log loss
                metrics['log_loss'] = log_loss(y, y_prob)
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return {'error': str(e)}


def compute_feature_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    method: str = 'built_in',
    n_repeats: int = 10
) -> pd.DataFrame:
    """
    Compute feature importance using various methods.
    
    Args:
        model: Trained model
        X: Feature data
        y: Target data
        method: Method to compute importance
        n_repeats: Number of repeats for permutation importance
        
    Returns:
        DataFrame with feature importance
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn not available for feature importance")
        return pd.DataFrame()
    
    try:
        if method == 'built_in':
            # Use built-in feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                return pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
            else:
                logger.warning("Model does not have built-in feature_importances_")
                # Fall back to permutation importance
                method = 'permutation'
        
        if method == 'permutation':
            # Use permutation importance
            perm_importance = permutation_importance(
                model, X, y, n_repeats=n_repeats, random_state=42
            )
            
            return pd.DataFrame({
                'Feature': X.columns,
                'Importance': perm_importance.importances_mean,
                'Std': perm_importance.importances_std
            }).sort_values('Importance', ascending=False)
        
        elif method == 'shap' and SHAP_AVAILABLE:
            # Use SHAP values for feature importance
            if hasattr(model, 'predict_proba'):
                # Create explainer
                explainer = shap.Explainer(model, X)
                shap_values = explainer(X)
                
                # Compute mean absolute SHAP values
                mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
                
                return pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': mean_abs_shap
                }).sort_values('Importance', ascending=False)
            else:
                logger.warning("Model does not support SHAP values")
                # Fall back to permutation importance
                return compute_feature_importance(model, X, y, method='permutation')
        
        else:
            logger.warning(f"Unsupported importance method: {method}")
            return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Error computing feature importance: {e}")
        return pd.DataFrame()