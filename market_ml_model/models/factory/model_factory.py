"""
Model factory for creating different types of ML models.
"""

import logging
from typing import Dict, Optional, Any, Callable
import warnings

# Setup logging
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Global registry of model factory functions
MODEL_FACTORY = {}

# Import scikit-learn models with error handling
try:
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
    )
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.error("scikit-learn not available for model creation")
    SKLEARN_AVAILABLE = False

# Try to import advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not installed. XGBoost models unavailable.")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not installed. LightGBM models unavailable.")
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    logger.warning("CatBoost not installed. CatBoost models unavailable.")
    CATBOOST_AVAILABLE = False

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import ( # Updated import block
        Dense, Dropout, LSTM, GRU, Bidirectional,
        BatchNormalization, Input, Concatenate
    )
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not installed. Neural network models unavailable.")
    TENSORFLOW_AVAILABLE = False


def register_model_factory(model_type: str) -> Callable:
    """
    Decorator to register model factory functions.

    Args:
        model_type: Name of the model type

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        MODEL_FACTORY[model_type] = func
        return func
    return decorator


@register_model_factory('random_forest')
def create_random_forest(params: Optional[Dict] = None) -> Any:
    """
    Create a Random Forest classifier with given parameters.

    Args:
        params: Model parameters

    Returns:
        RandomForestClassifier instance
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn not available for Random Forest")
        return None

    params = params or {}

    # Set defaults
    default_params = {
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'bootstrap': True,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }

    # Update with provided params
    default_params.update(params)

    return RandomForestClassifier(**default_params)


@register_model_factory('gradient_boosting')
def create_gradient_boosting(params: Optional[Dict] = None) -> Any:
    """
    Create a Gradient Boosting classifier with given parameters.

    Args:
        params: Model parameters

    Returns:
        GradientBoostingClassifier instance
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn not available for Gradient Boosting")
        return None

    params = params or {}

    # Set defaults
    default_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'subsample': 1.0,
        'random_state': 42
    }

    # Update with provided params
    default_params.update(params)

    return GradientBoostingClassifier(**default_params)


@register_model_factory('xgboost')
def create_xgboost(params: Optional[Dict] = None) -> Any:
    """
    Create an XGBoost classifier with given parameters.

    Args:
        params: Model parameters

    Returns:
        XGBClassifier instance
    """
    if not XGBOOST_AVAILABLE:
        logger.error("XGBoost not available")
        return None

    params = params or {}

    # Set defaults
    default_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'multi:softproba',
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'random_state': 42
    }

    # Update with provided params
    default_params.update(params)

    return xgb.XGBClassifier(**default_params)


@register_model_factory('lightgbm')
def create_lightgbm(params: Optional[Dict] = None) -> Any:
    """
    Create a LightGBM classifier with given parameters.

    Args:
        params: Model parameters

    Returns:
        LGBMClassifier instance
    """
    if not LIGHTGBM_AVAILABLE:
        logger.error("LightGBM not available")
        return None

    params = params or {}

    # Set defaults
    default_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'max_depth': -1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }

    # Update with provided params
    default_params.update(params)

    return lgb.LGBMClassifier(**default_params)


@register_model_factory('catboost')
def create_catboost(params: Optional[Dict] = None) -> Any:
    """
    Create a CatBoost classifier with given parameters.

    Args:
        params: Model parameters

    Returns:
        CatBoostClassifier instance
    """
    if not CATBOOST_AVAILABLE:
        logger.error("CatBoost not available")
        return None

    params = params or {}

    # Set defaults
    default_params = {
        'iterations': 100,
        'learning_rate': 0.1,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'thread_count': -1,
        'verbose': False
    }

    # Update with provided params
    default_params.update(params)

    return CatBoostClassifier(**default_params)


@register_model_factory('logistic_regression')
def create_logistic_regression(params: Optional[Dict] = None) -> Any:
    """
    Create a Logistic Regression classifier with given parameters.

    Args:
        params: Model parameters

    Returns:
        LogisticRegression instance
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn not available for Logistic Regression")
        return None

    params = params or {}

    # Set defaults
    default_params = {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': 42,
        'n_jobs': -1
    }

    # Update with provided params
    default_params.update(params)

    return LogisticRegression(**default_params)


@register_model_factory('neural_network')
def create_neural_network(params: Optional[Dict] = None) -> Any:
    """
    Create a simple neural network classifier with given parameters.

    Args:
        params: Model parameters

    Returns:
        Keras Sequential model
    """
    if not TENSORFLOW_AVAILABLE:
        logger.error("TensorFlow not available for Neural Network")
        return None

    params = params or {}

    # Set defaults
    default_params = {
        'input_dim': 10,  # Must be provided or overridden
        'num_classes': 3,  # Must be provided or overridden
        'hidden_layers': [64, 32],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'activation': 'relu',
        'output_activation': 'softmax'
    }

    # Update with provided params
    default_params.update(params)

    # Build model
    model = Sequential()

    # First hidden layer
    model.add(Dense(
        default_params['hidden_layers'][0],
        input_dim=default_params['input_dim'],
        activation=default_params['activation']
    ))
    model.add(Dropout(default_params['dropout_rate']))

    # Additional hidden layers
    for units in default_params['hidden_layers'][1:]:
        model.add(Dense(units, activation=default_params['activation']))
        model.add(Dropout(default_params['dropout_rate']))

    # Output layer
    model.add(Dense(
        default_params['num_classes'],
        activation=default_params['output_activation']
    ))

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=default_params['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


@register_model_factory('lstm')
def create_lstm_network(params: Optional[Dict] = None) -> Any:
    """
    Create an LSTM network for sequence classification.

    Args:
        params: Model parameters

    Returns:
        Keras Sequential model
    """
    if not TENSORFLOW_AVAILABLE:
        logger.error("TensorFlow not available for LSTM")
        return None

    params = params or {}

    # Set defaults
    default_params = {
        'input_shape': (10, 1),  # (sequence_length, features)
        'num_classes': 3,
        'lstm_units': [64, 32],
        'dropout_rate': 0.2,
        'recurrent_dropout': 0.2,
        'dense_units': [32],
        'learning_rate': 0.001,
        'activation': 'relu',
        'output_activation': 'softmax'
    }

    # Update with provided params
    default_params.update(params)

    # Build model
    model = Sequential()

    # First LSTM layer
    if len(default_params['lstm_units']) == 1:
        # Only one LSTM layer - return sequences False
        model.add(LSTM(
            units=default_params['lstm_units'][0],
            input_shape=default_params['input_shape'],
            dropout=default_params['dropout_rate'],
            recurrent_dropout=default_params['recurrent_dropout']
        ))
    else:
        # Multiple LSTM layers - return sequences True
        model.add(LSTM(
            units=default_params['lstm_units'][0],
            input_shape=default_params['input_shape'],
            dropout=default_params['dropout_rate'],
            recurrent_dropout=default_params['recurrent_dropout'],
            return_sequences=True
        ))

    # Additional LSTM layers
    for i, units in enumerate(default_params['lstm_units'][1:]):
        return_sequences = i < len(default_params['lstm_units']) - 2
        model.add(LSTM(
            units=units,
            dropout=default_params['dropout_rate'],
            recurrent_dropout=default_params['recurrent_dropout'],
            return_sequences=return_sequences
        ))

    # Dense layers
    for units in default_params['dense_units']:
        model.add(Dense(units, activation=default_params['activation']))
        model.add(Dropout(default_params['dropout_rate']))

    # Output layer
    model.add(Dense(
        default_params['num_classes'],
        activation=default_params['output_activation']
    ))

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=default_params['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


@register_model_factory('ensemble')
def create_ensemble(params: Optional[Dict] = None) -> Any:
    """
    Create an ensemble of multiple models.

    Args:
        params: Model parameters

    Returns:
        VotingClassifier instance
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn not available for Ensemble")
        return None

    params = params or {}

    # Set defaults
    default_params = {
        'models': ['random_forest', 'gradient_boosting'],
        'weights': None,
        'voting': 'soft'
    }

    # Update with provided params
    default_params.update(params)

    # Create individual models
    estimators = []
    for model_type in default_params['models']:
        if model_type in MODEL_FACTORY:
            # Pass specific params if provided for sub-models, else use defaults
            model_params = params.get(model_type, {})
            model = MODEL_FACTORY[model_type](model_params)
            if model is not None:
                estimators.append((model_type, model))
        else:
            logger.warning(f"Unsupported model type for ensemble: {model_type}")

    if not estimators:
        logger.error("No valid models for ensemble")
        return None

    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=estimators,
        voting=default_params['voting'],
        weights=default_params['weights']
    )

    return ensemble


def create_model(model_type: str, params: Optional[Dict] = None) -> Any:
    """
    Create a model instance of the specified type.

    Args:
        model_type: Type of model to create
        params: Model parameters

    Returns:
        Model instance
    """
    if model_type in MODEL_FACTORY:
        return MODEL_FACTORY[model_type](params)
    else:
        logger.error(f"Unsupported model type: {model_type}")
        return None