"""
Model training orchestration module.
"""

import pandas as pd
import numpy as np
import time
import logging
import os
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import joblib
import uuid

# Import components
from .factory.model_factory import create_model
from .optimization.hyperparameters import optimize_hyperparameters
from .evaluation.metrics import evaluate_classifier, compute_feature_importance

# Import feature selection
from .feature_selection import select_features

# Setup logging
logger = logging.getLogger(__name__)


def train_classification_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_config: Optional[Dict] = None,
    feature_selection: bool = True,
    feature_selection_params: Optional[Dict] = None,
    preprocessing_params: Optional[Dict] = None,
    test_size: float = 0.2,
    optimize_hyperparams: bool = True,
    optimization_params: Optional[Dict] = None,
    output_dir: Optional[str] = None,
    model_id: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    random_state: int = 42
) -> Tuple[Any, pd.DataFrame, Dict]:
    """
    Train a classification model with advanced options.

    Args:
        features: Feature DataFrame
        target: Target Series
        model_config: Model configuration
        feature_selection: Whether to perform feature selection
        feature_selection_params: Parameters for feature selection
        preprocessing_params: Parameters for preprocessing
        test_size: Test set fraction
        optimize_hyperparams: Whether to optimize hyperparameters
        optimization_params: Parameters for hyperparameter optimization
        output_dir: Directory to save model and reports
        model_id: Model identifier (auto-generated if None)
        class_names: Names of target classes
        random_state: Random seed

    Returns:
        Tuple of (trained_model, feature_importance, metrics)
    """
    # Generate model ID if not provided
    if model_id is None:
        model_id = f"model_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"Starting model training with ID: {model_id}")

    # Create model output directory if needed
    if output_dir:
        model_dir = os.path.join(output_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
    else:
        model_dir = None

    # Default model configuration
    default_model_config = {
        'model_type': 'lightgbm',  # Default to LightGBM if available, else Random Forest
        'class_weight': 'balanced'
    }

    # Update with provided config
    model_config = {**default_model_config, **(model_config or {})}

    # Default optimization parameters
    default_optimization_params = {
        'method': 'random',
        'n_iter': 20,
        'cv': 5,
        'scoring': 'f1_weighted'
    }

    # Update with provided params
    optimization_params = {**default_optimization_params, **(optimization_params or {})}

    # Default feature selection parameters
    default_feature_selection_params = {
        'method': 'importance',
        'n_features': min(50, features.shape[1])
    }

    # Update with provided params
    feature_selection_params = {**default_feature_selection_params, **(feature_selection_params or {})}

    # Split data into train and test sets
    try:
        if isinstance(features.index, pd.DatetimeIndex):
            # Time series data - use last n% as test
            split_idx = int(len(features) * (1 - test_size))
            X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]

            logger.info(f"Split data using time series method: {len(X_train)} train, {len(X_test)} test")
        else:
            # Regular split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=test_size, random_state=random_state
            )

            logger.info(f"Split data using random split: {len(X_train)} train, {len(X_test)} test")
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        return None, pd.DataFrame(), {'error': str(e)}

    # Perform feature selection if enabled
    if feature_selection:
        try:
            logger.info(f"Performing feature selection using {feature_selection_params['method']} method")
            
            # Import select_features from the right location
            from .feature_selection import select_features
            
            X_train_selected, selected_features = select_features(
                X_train, y_train,
                method=feature_selection_params['method'],
                params=feature_selection_params
            )

            # Apply same selection to test set
            X_test_selected = X_test[selected_features]

            logger.info(f"Selected {len(selected_features)} features")

            # Save selected features if output directory provided
            if model_dir:
                with open(os.path.join(model_dir, 'selected_features.json'), 'w') as f:
                    json.dump(selected_features, f, indent=4)
        except Exception as e:
            logger.error(f"Error during feature selection: {e}")
            # Fall back to original features
            X_train_selected, X_test_selected = X_train, X_test
            selected_features = list(features.columns)
    else:
        # Use all features
        X_train_selected, X_test_selected = X_train, X_test
        selected_features = list(features.columns)

    # Train model
    try:
        if optimize_hyperparams:
            logger.info(f"Optimizing hyperparameters for {model_config['model_type']} using {optimization_params['method']} method")
            model, best_params = optimize_hyperparameters(
                model_type=model_config['model_type'],
                X_train=X_train_selected,
                y_train=y_train,
                method=optimization_params['method'],
                params=optimization_params,
                cv=optimization_params['cv'],
                scoring=optimization_params['scoring'],
                verbose=1
            )

            logger.info(f"Hyperparameter optimization complete. Best parameters: {best_params}")

            # Save best parameters if output directory provided
            if model_dir:
                with open(os.path.join(model_dir, 'best_params.json'), 'w') as f:
                    json.dump(best_params, f, indent=4, default=str)
        else:
            logger.info(f"Training {model_config['model_type']} with default parameters")
            model = create_model(model_config['model_type'], model_config)
            model.fit(X_train_selected, y_train)

        # Calculate feature importance
        importance_df = compute_feature_importance(
            model, X_train_selected, y_train, method='built_in'
        )

        # Generate model report
        train_metrics = evaluate_classifier(
            model, X_train_selected, y_train, class_names=class_names
        )
        
        test_metrics = evaluate_classifier(
            model, X_test_selected, y_test, class_names=class_names
        )
        
        metrics = {
            'train': train_metrics,
            'test': test_metrics
        }

        # Save model if output directory provided
        if model_dir:
            joblib.dump(model, os.path.join(model_dir, 'model.pkl'))

            # Save training configuration
            config = {
                'model_id': model_id,
                'model_type': model_config['model_type'],
                'feature_selection': feature_selection,
                'feature_selection_params': feature_selection_params,
                'preprocessing_params': preprocessing_params,
                'optimization_params': optimization_params,
                'test_size': test_size,
                'random_state': random_state,
                'training_date': datetime.now().isoformat(),
                'num_features': len(selected_features),
                'num_samples': len(features),
                'class_distribution': target.value_counts().to_dict(),
                'test_accuracy': test_metrics.get('accuracy', 0)
            }

            with open(os.path.join(model_dir, 'training_config.json'), 'w') as f:
                json.dump(config, f, indent=4, default=str)
                
            # Save feature importance if available
            if not importance_df.empty:
                importance_df.to_csv(os.path.join(model_dir, 'feature_importance.csv'), index=False)

        logger.info(f"Model training complete. Test accuracy: {test_metrics.get('accuracy', 0):.4f}")

        return model, importance_df, metrics
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return None, pd.DataFrame(), {'error': str(e)}