import pandas as pd
import numpy as np
import time
import logging
import os
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import joblib
import random
from sklearn.metrics import confusion_matrix, classification_report
import uuid
import warnings

# Setup logging
logger = logging.getLogger(__name__)

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import scikit-learn components with error handling
try:
    from sklearn.model_selection import (
        train_test_split, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV,
        cross_val_score, cross_validate
    )
    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer,
        OneHotEncoder, LabelEncoder
    )
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier, 
        AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
    )
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        precision_recall_curve, roc_curve, roc_auc_score, 
        average_precision_score, log_loss
    )
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import (
        SelectKBest, f_classif, mutual_info_classif, RFE, RFECV,
        SelectFromModel
    )
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.cluster import KMeans
    from sklearn.exceptions import NotFittedError
    SKLEARN_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing scikit-learn components: {e}")
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

try:
    from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
    from hyperopt.pyll.base import scope
    HYPEROPT_AVAILABLE = True
except ImportError:
    logger.warning("Hyperopt not installed. Bayesian optimization unavailable.")
    HYPEROPT_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    logger.warning("Optuna not installed. Optuna optimization unavailable.")
    OPTUNA_AVAILABLE = False

# Try to import SHAP for model interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    logger.warning("SHAP not installed. Model interpretability will be limited.")
    SHAP_AVAILABLE = False

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib/Seaborn not installed. Visualization unavailable.")
    VISUALIZATION_AVAILABLE = False

# Tensorflow/Keras for neural networks
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Dense, Dropout, LSTM, GRU, Bidirectional,
        BatchNormalization, Input, Concatenate
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    )
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not installed. Neural network models unavailable.")
    TENSORFLOW_AVAILABLE = False


# --- Constants and Configuration ---

# Default hyperparameter spaces for different model types
DEFAULT_RF_PARAMS = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

DEFAULT_XGB_PARAMS = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}

DEFAULT_LGBM_PARAMS = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'num_leaves': [31, 63, 127],
    'max_depth': [-1, 5, 10, 15],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

DEFAULT_CATBOOST_PARAMS = {
    'iterations': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7, 9]
}

DEFAULT_NN_PARAMS = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'epochs': [50, 100, 200],
    'dropout_rate': [0.1, 0.3, 0.5],
    'neurons_layer1': [32, 64, 128],
    'neurons_layer2': [16, 32, 64],
    'activation': ['relu', 'tanh']
}

# Model Factory registry - maps model type strings to creation functions
MODEL_FACTORY = {}


# --- Feature Selection and Engineering ---

def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = 'importance',
    params: Optional[Dict] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select the most important features using various methods.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        method: Method to use ('importance', 'mutual_info', 'rfe', 'pca', 'model')
        params: Additional parameters for the selected method
        
    Returns:
        Tuple of (selected_features_df, selected_feature_names)
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn not available for feature selection")
        return X, list(X.columns)
    
    params = params or {}
    
    if method == 'importance':
        # Use Random Forest feature importance
        n_features = params.get('n_features', min(30, X.shape[1]))
        threshold = params.get('threshold', 0.01)
        
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        try:
            clf.fit(X, y)
            importances = pd.Series(clf.feature_importances_, index=X.columns)
            importances = importances.sort_values(ascending=False)
            
            if n_features:
                selected_features = importances.head(n_features).index.tolist()
            else:
                selected_features = importances[importances >= threshold].index.tolist()
            
            return X[selected_features], selected_features
            
        except Exception as e:
            logger.error(f"Error during feature importance selection: {e}")
            return X, list(X.columns)
    
    elif method == 'mutual_info':
        # Use mutual information for feature selection
        n_features = params.get('n_features', min(30, X.shape[1]))
        
        try:
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
            selector.fit(X, y)
            
            # Get selected feature indices
            selected_indices = selector.get_support(indices=True)
            selected_features = [X.columns[i] for i in selected_indices]
            
            return X[selected_features], selected_features
            
        except Exception as e:
            logger.error(f"Error during mutual information selection: {e}")
            return X, list(X.columns)
    
    elif method == 'rfe':
        # Use Recursive Feature Elimination
        n_features = params.get('n_features', min(30, X.shape[1]))
        step = params.get('step', 1)
        
        # Use LogisticRegression as the base estimator (faster than RF)
        estimator = LogisticRegression(max_iter=1000, random_state=42)
        
        try:
            selector = RFE(
                estimator=estimator,
                n_features_to_select=n_features,
                step=step
            )
            selector.fit(X, y)
            
            # Get selected feature indices
            selected_features = X.columns[selector.support_].tolist()
            
            return X[selected_features], selected_features
            
        except Exception as e:
            logger.error(f"Error during RFE selection: {e}")
            return X, list(X.columns)
    
    elif method == 'pca':
        # Use PCA for dimensionality reduction
        n_components = params.get('n_components', min(X.shape[1], X.shape[0], 30))
        
        try:
            # Standardize data for PCA
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply PCA
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            
            # Create new feature names
            selected_features = [f"PC{i+1}" for i in range(n_components)]
            
            # Convert back to DataFrame
            X_pca_df = pd.DataFrame(
                X_pca,
                index=X.index,
                columns=selected_features
            )
            
            return X_pca_df, selected_features
            
        except Exception as e:
            logger.error(f"Error during PCA selection: {e}")
            return X, list(X.columns)
    
    elif method == 'model':
        # Use model-specific feature selection
        model_type = params.get('model_type', 'lightgbm')
        n_features = params.get('n_features', min(30, X.shape[1]))
        threshold = params.get('threshold', 0.01)
        
        if model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            try:
                lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
                selector = SelectFromModel(
                    lgb_model,
                    threshold='median' if threshold is None else threshold,
                    max_features=n_features
                )
                selector.fit(X, y)
                
                selected_features = X.columns[selector.get_support()].tolist()
                return X[selected_features], selected_features
                
            except Exception as e:
                logger.error(f"Error during LightGBM model selection: {e}")
                return X, list(X.columns)
        
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            try:
                xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
                selector = SelectFromModel(
                    xgb_model,
                    threshold='median' if threshold is None else threshold,
                    max_features=n_features
                )
                selector.fit(X, y)
                
                selected_features = X.columns[selector.get_support()].tolist()
                return X[selected_features], selected_features
                
            except Exception as e:
                logger.error(f"Error during XGBoost model selection: {e}")
                return X, list(X.columns)
        
        else:
            logger.warning(f"Unsupported model type for feature selection: {model_type}")
            return X, list(X.columns)
    
    else:
        logger.warning(f"Unsupported feature selection method: {method}")
        return X, list(X.columns)


def create_feature_pipeline(
    X: pd.DataFrame,
    categorical_features: Optional[List[str]] = None,
    numeric_features: Optional[List[str]] = None,
    scaling_method: str = 'standard',
    handle_missing: bool = True
) -> Tuple[ColumnTransformer, List[str]]:
    """
    Create a scikit-learn pipeline for feature preprocessing.
    
    Args:
        X: Feature DataFrame
        categorical_features: List of categorical features
        numeric_features: List of numeric features
        scaling_method: Method for scaling numeric features
        handle_missing: Whether to handle missing values
        
    Returns:
        Tuple of (preprocessing_pipeline, transformed_feature_names)
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn not available for feature pipeline")
        return None, list(X.columns)
    
    # Identify column types if not provided
    if categorical_features is None and numeric_features is None:
        categorical_features = []
        numeric_features = []
        
        for col in X.columns:
            if X[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X[col]):
                categorical_features.append(col)
            elif pd.api.types.is_numeric_dtype(X[col]):
                numeric_features.append(col)
    
    elif categorical_features is None:
        categorical_features = [col for col in X.columns if col not in numeric_features]
    
    elif numeric_features is None:
        numeric_features = [col for col in X.columns if col not in categorical_features]
    
    # Ensure all features are accounted for
    all_features = categorical_features + numeric_features
    if len(all_features) != len(X.columns):
        missing_features = set(X.columns) - set(all_features)
        logger.warning(f"Some features not classified: {missing_features}")
        
        # Assume numeric for unclassified features
        for feature in missing_features:
            if pd.api.types.is_numeric_dtype(X[feature]):
                numeric_features.append(feature)
            else:
                categorical_features.append(feature)
    
    # Create transformer for numeric features
    numeric_transformer_steps = []
    
    if handle_missing:
        numeric_transformer_steps.append(
            ('imputer', SimpleImputer(strategy='median'))
        )
    
    if scaling_method == 'standard':
        numeric_transformer_steps.append(
            ('scaler', StandardScaler())
        )
    elif scaling_method == 'minmax':
        numeric_transformer_steps.append(
            ('scaler', MinMaxScaler())
        )
    elif scaling_method == 'robust':
        numeric_transformer_steps.append(
            ('scaler', RobustScaler())
        )
    elif scaling_method == 'quantile':
        numeric_transformer_steps.append(
            ('scaler', QuantileTransformer(output_distribution='normal'))
        )
    
    numeric_transformer = Pipeline(steps=numeric_transformer_steps)
    
    # Create transformer for categorical features
    categorical_transformer_steps = []
    
    if handle_missing:
        categorical_transformer_steps.append(
            ('imputer', SimpleImputer(strategy='most_frequent'))
        )
    
    categorical_transformer_steps.append(
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    )
    
    categorical_transformer = Pipeline(steps=categorical_transformer_steps)
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Calculate transformed feature names
    transformed_feature_names = numeric_features.copy()
    
    # Add one-hot encoded feature names
    for cat_feature in categorical_features:
        if cat_feature in X.columns:
            unique_values = X[cat_feature].dropna().unique()
            for value in unique_values:
                transformed_feature_names.append(f"{cat_feature}_{value}")
    
    return preprocessor, transformed_feature_names


# --- Model Factory Functions ---

def register_model_factory(model_type: str):
    """
    Decorator to register model factory functions.
    
    Args:
        model_type: Name of the model type
    """
    def decorator(func):
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
            model = MODEL_FACTORY[model_type]()
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


# --- Hyperparameter Optimization ---

def optimize_hyperparameters_grid(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[Dict] = None,
    cv: int = 5,
    scoring: str = 'f1_weighted',
    verbose: int = 1,
    n_jobs: int = -1
) -> Tuple[Any, Dict]:
    """
    Optimize hyperparameters using grid search.
    
    Args:
        model_type: Type of model to optimize
        X_train: Training features
        y_train: Training target
        param_grid: Parameter grid to search
        cv: Number of cross-validation folds
        scoring: Scoring metric
        verbose: Verbosity level
        n_jobs: Number of jobs to run in parallel
        
    Returns:
        Tuple of (best_model, best_params)
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn not available for grid search")
        return None, {}
    
    # Use default parameter grid if not provided
    if param_grid is None:
        if model_type == 'random_forest':
            param_grid = DEFAULT_RF_PARAMS
        elif model_type == 'xgboost':
            param_grid = DEFAULT_XGB_PARAMS
        elif model_type == 'lightgbm':
            param_grid = DEFAULT_LGBM_PARAMS
        elif model_type == 'catboost':
            param_grid = DEFAULT_CATBOOST_PARAMS
        elif model_type == 'neural_network':
            param_grid = DEFAULT_NN_PARAMS
        else:
            logger.warning(f"No default param_grid for {model_type}")
            param_grid = {}
    
    # Create base model
    base_model = create_model(model_type)
    
    if base_model is None:
        logger.error(f"Failed to create base model for {model_type}")
        return None, {}
    
    # Create cross-validation object
    if isinstance(X_train.index, pd.DatetimeIndex):
        # Use TimeSeriesSplit for time series data
        cv_obj = TimeSeriesSplit(n_splits=cv)
    else:
        # Use regular KFold for non-time series data
        from sklearn.model_selection import KFold
        cv_obj = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Create grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_obj,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )
    
    try:
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        return best_model, best_params
    
    except Exception as e:
        logger.error(f"Error during grid search: {e}")
        return base_model, {}


def optimize_hyperparameters_random(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_distributions: Optional[Dict] = None,
    n_iter: int = 20,
    cv: int = 5,
    scoring: str = 'f1_weighted',
    verbose: int = 1,
    n_jobs: int = -1
) -> Tuple[Any, Dict]:
    """
    Optimize hyperparameters using randomized search.
    
    Args:
        model_type: Type of model to optimize
        X_train: Training features
        y_train: Training target
        param_distributions: Parameter distributions to sample
        n_iter: Number of iterations
        cv: Number of cross-validation folds
        scoring: Scoring metric
        verbose: Verbosity level
        n_jobs: Number of jobs to run in parallel
        
    Returns:
        Tuple of (best_model, best_params)
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn not available for randomized search")
        return None, {}
    
    # Use default parameter distributions if not provided
    if param_distributions is None:
        if model_type == 'random_forest':
            param_distributions = DEFAULT_RF_PARAMS
        elif model_type == 'xgboost':
            param_distributions = DEFAULT_XGB_PARAMS
        elif model_type == 'lightgbm':
            param_distributions = DEFAULT_LGBM_PARAMS
        elif model_type == 'catboost':
            param_distributions = DEFAULT_CATBOOST_PARAMS
        elif model_type == 'neural_network':
            param_distributions = DEFAULT_NN_PARAMS
        else:
            logger.warning(f"No default param_distributions for {model_type}")
            param_distributions = {}
    
    # Create base model
    base_model = create_model(model_type)
    
    if base_model is None:
        logger.error(f"Failed to create base model for {model_type}")
        return None, {}
    
    # Create cross-validation object
    if isinstance(X_train.index, pd.DatetimeIndex):
        # Use TimeSeriesSplit for time series data
        cv_obj = TimeSeriesSplit(n_splits=cv)
    else:
        # Use regular KFold for non-time series data
        from sklearn.model_selection import KFold
        cv_obj = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Create randomized search
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv_obj,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=42
    )
    
    try:
        # Fit randomized search
        random_search.fit(X_train, y_train)
        
        # Get best model and parameters
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {random_search.best_score_:.4f}")
        
        return best_model, best_params
    
    except Exception as e:
        logger.error(f"Error during randomized search: {e}")
        return base_model, {}


def optimize_hyperparameters_bayesian(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_space: Optional[Dict] = None,
    n_trials: int = 50,
    cv: int = 5,
    scoring: str = 'f1_weighted',
    verbose: int = 1
) -> Tuple[Any, Dict]:
    """
    Optimize hyperparameters using Bayesian optimization (Optuna).
    
    Args:
        model_type: Type of model to optimize
        X_train: Training features
        y_train: Training target
        param_space: Parameter space definition
        n_trials: Number of trials
        cv: Number of cross-validation folds
        scoring: Scoring metric
        verbose: Verbosity level
        
    Returns:
        Tuple of (best_model, best_params)
    """
    if not OPTUNA_AVAILABLE:
        logger.error("Optuna not available for Bayesian optimization")
        return None, {}
    
    # Create base model function
    def create_base_model(trial):
        if model_type == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
            return create_model('random_forest', params)
        
        elif model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5)
            }
            return create_model('xgboost', params)
        
        elif model_type == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 200),
                'max_depth': trial.suggest_int('max_depth', -1, 20),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            return create_model('lightgbm', params)
        
        elif model_type == 'catboost':
            params = {
                'iterations': trial.suggest_int('iterations', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10, log=True)
            }
            return create_model('catboost', params)
        
        else:
            logger.warning(f"Unsupported model type for Bayesian optimization: {model_type}")
            return create_model(model_type)
    
    # Create objective function for optimization
    def objective(trial):
        model = create_base_model(trial)
        
        if model is None:
            return float('inf')
        
        try:
            # Create cross-validation object
            if isinstance(X_train.index, pd.DatetimeIndex):
                # Use TimeSeriesSplit for time series data
                cv_obj = TimeSeriesSplit(n_splits=cv)
            else:
                # Use regular KFold for non-time series data
                from sklearn.model_selection import KFold
                cv_obj = KFold(n_splits=cv, shuffle=True, random_state=42)
            
            # Perform cross-validation
            scores = cross_val_score(
                model, X_train, y_train,
                scoring=scoring,
                cv=cv_obj,
                n_jobs=-1
            )
            
            return -scores.mean()  # negative because Optuna minimizes
        
        except Exception as e:
            logger.error(f"Error in Optuna objective: {e}")
            return float('inf')
    
    try:
        # Create study
        study = optuna.create_study(direction='minimize')
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose > 0)
        
        # Get best parameters
        best_params = study.best_params
        
        # Create best model
        if model_type == 'random_forest':
            best_model = create_model('random_forest', best_params)
        elif model_type == 'xgboost':
            best_model = create_model('xgboost', best_params)
        elif model_type == 'lightgbm':
            best_model = create_model('lightgbm', best_params)
        elif model_type == 'catboost':
            best_model = create_model('catboost', best_params)
        else:
            best_model = create_model(model_type)
        
        # Fit best model
        best_model.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {-study.best_value:.4f}")
        
        return best_model, best_params
    
    except Exception as e:
        logger.error(f"Error during Bayesian optimization: {e}")
        return create_model(model_type), {}


def optimize_hyperparameters(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = 'random',
    params: Optional[Dict] = None,
    cv: int = 5,
    scoring: str = 'f1_weighted',
    verbose: int = 1,
    n_jobs: int = -1
) -> Tuple[Any, Dict]:
    """
    Optimize hyperparameters using the specified method.
    
    Args:
        model_type: Type of model to optimize
        X_train: Training features
        y_train: Training target
        method: Optimization method ('grid', 'random', 'bayesian')
        params: Parameters for the optimization method
        cv: Number of cross-validation folds
        scoring: Scoring metric
        verbose: Verbosity level
        n_jobs: Number of jobs to run in parallel
        
    Returns:
        Tuple of (best_model, best_params)
    """
    if method == 'grid':
        return optimize_hyperparameters_grid(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            param_grid=params,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
            n_jobs=n_jobs
        )
    
    elif method == 'random':
        return optimize_hyperparameters_random(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            param_distributions=params,
            n_iter=params.get('n_iter', 20) if params else 20,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
            n_jobs=n_jobs
        )
    
    elif method == 'bayesian':
        return optimize_hyperparameters_bayesian(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            param_space=params,
            n_trials=params.get('n_trials', 50) if params else 50,
            cv=cv,
            scoring=scoring,
            verbose=verbose
        )
    
    else:
        logger.error(f"Unsupported optimization method: {method}")
        return create_model(model_type), {}


# --- Model Evaluation ---

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


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = 'Confusion Matrix',
    normalize: bool = False,
    figsize: Tuple[int, int] = (10, 8),
    filename: Optional[str] = None
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
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(conf_matrix.shape[0])]
    
    # Create plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    
    # Save or show plot
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = 'Feature Importance',
    figsize: Tuple[int, int] = (12, 8),
    filename: Optional[str] = None
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
        x='Importance',
        y='Feature',
        data=importance_df,
        palette='viridis'
    )
    
    # Add error bars if available
    if 'Std' in importance_df.columns:
        ax.errorbar(
            x=importance_df['Importance'],
            y=range(len(importance_df)),
            xerr=importance_df['Std'],
            fmt='none',
            ecolor='black',
            capsize=3
        )
    
    plt.title(title)
    plt.tight_layout()
    
    # Save or show plot
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_roc_curve(
    fpr: List[float],
    tpr: List[float],
    roc_auc: float,
    title: str = 'ROC Curve',
    figsize: Tuple[int, int] = (8, 8),
    filename: Optional[str] = None
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
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save or show plot
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_precision_recall_curve(
    precision: List[float],
    recall: List[float],
    avg_precision: float,
    title: str = 'Precision-Recall Curve',
    figsize: Tuple[int, int] = (8, 8),
    filename: Optional[str] = None
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
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'AP = {avg_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save or show plot
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def generate_model_report(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Generate a comprehensive model report.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        class_names: Names of classes
        output_dir: Directory to save report files
        
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
    train_metrics = evaluate_classifier(model, X_train, y_train, class_names)
    test_metrics = evaluate_classifier(model, X_test, y_test, class_names)
    
    # Feature importance
    importance_df = compute_feature_importance(
        model, X_train, y_train, method='built_in'
    )
    
    # Generate plots if output directory provided
    if output_dir and VISUALIZATION_AVAILABLE:
        # Confusion matrices
        plot_confusion_matrix(
            train_metrics['confusion_matrix'],
            class_names=class_names,
            title='Training Confusion Matrix',
            filename=os.path.join(output_dir, 'train_confusion_matrix.png')
        )
        
        plot_confusion_matrix(
            test_metrics['confusion_matrix'],
            class_names=class_names,
            title='Test Confusion Matrix',
            filename=os.path.join(output_dir, 'test_confusion_matrix.png')
        )
        
        # Feature importance
        if not importance_df.empty:
            plot_feature_importance(
                importance_df,
                title='Feature Importance',
                filename=os.path.join(output_dir, 'feature_importance.png')
            )
        
        # ROC curve (binary classification)
        if 'roc_curve' in test_metrics:
            plot_roc_curve(
                test_metrics['roc_curve']['fpr'],
                test_metrics['roc_curve']['tpr'],
                test_metrics['roc_auc'],
                title='ROC Curve',
                filename=os.path.join(output_dir, 'roc_curve.png')
            )
        
        # Precision-Recall curve (binary classification)
        if 'pr_curve' in test_metrics:
            plot_precision_recall_curve(
                test_metrics['pr_curve']['precision'],
                test_metrics['pr_curve']['recall'],
                test_metrics['average_precision'],
                title='Precision-Recall Curve',
                filename=os.path.join(output_dir, 'pr_curve.png')
            )
    
    # Create summary report
    summary = {
        'train_metrics': {
            k: v for k, v in train_metrics.items()
            if k not in ['confusion_matrix', 'classification_report',
                          'roc_curve', 'pr_curve']
        },
        'test_metrics': {
            k: v for k, v in test_metrics.items()
            if k not in ['confusion_matrix', 'classification_report',
                          'roc_curve', 'pr_curve']
        },
        'feature_importance': importance_df.to_dict() if not importance_df.empty else {},
        'model_info': {
            'type': type(model).__name__,
            'params': model.get_params() if hasattr(model, 'get_params') else {}
        }
    }
    
    # Save summary as JSON if output directory provided
    if output_dir:
        # Remove non-serializable objects
        json_summary = {
            'train_metrics': summary['train_metrics'],
            'test_metrics': summary['test_metrics'],
            'model_info': summary['model_info']
        }
        
        with open(os.path.join(output_dir, 'model_report.json'), 'w') as f:
            json.dump(json_summary, f, indent=4, default=str)
        
        # Save feature importance as CSV
        if not importance_df.empty:
            importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    return summary


# --- Main Training Function ---

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
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn not available for model training")
        return None, pd.DataFrame(), {}
    
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
        'model_type': 'lightgbm' if LIGHTGBM_AVAILABLE else 'random_forest',
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
    
    # Default preprocessing parameters
    default_preprocessing_params = {
        'scaling_method': 'standard',
        'handle_missing': True
    }
    
    # Update with provided params
    preprocessing_params = {**default_preprocessing_params, **(preprocessing_params or {})}
    
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
    
    # Create preprocessing pipeline if needed
    if preprocessing_params:
        try:
            logger.info("Creating preprocessing pipeline")
            # Identify categorical and numeric features
            categorical_features = [
                col for col in selected_features
                if X_train[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X_train[col])
            ]
            
            numeric_features = [
                col for col in selected_features
                if col not in categorical_features and pd.api.types.is_numeric_dtype(X_train[col])
            ]
            
            # Create pipeline
            preprocessor, transformed_feature_names = create_feature_pipeline(
                X_train[selected_features],
                categorical_features=categorical_features,
                numeric_features=numeric_features,
                scaling_method=preprocessing_params['scaling_method'],
                handle_missing=preprocessing_params['handle_missing']
            )
            
            if preprocessor is not None:
                # Apply preprocessing
                X_train_processed = preprocessor.fit_transform(X_train_selected)
                X_test_processed = preprocessor.transform(X_test_selected)
                
                # Convert to DataFrame for better readability
                X_train_processed = pd.DataFrame(
                    X_train_processed,
                    index=X_train_selected.index,
                    columns=transformed_feature_names  # This is simplified; real names would need conversion
                )
                
                X_test_processed = pd.DataFrame(
                    X_test_processed,
                    index=X_test_selected.index,
                    columns=transformed_feature_names
                )
                
                logger.info(f"Applied preprocessing, shape: {X_train_processed.shape}")
                
                # Save preprocessor if output directory provided
                if model_dir:
                    joblib.dump(
                        preprocessor,
                        os.path.join(model_dir, 'preprocessor.pkl')
                    )
                    
                    # Save feature mapping
                    with open(os.path.join(model_dir, 'transformed_features.json'), 'w') as f:
                        json.dump(transformed_feature_names, f, indent=4)
            else:
                # Use selected features without preprocessing
                X_train_processed, X_test_processed = X_train_selected, X_test_selected
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            # Fall back to selected features without preprocessing
            X_train_processed, X_test_processed = X_train_selected, X_test_selected
    else:
        # Use selected features without preprocessing
        X_train_processed, X_test_processed = X_train_selected, X_test_selected
    
    # Train model
    try:
        if optimize_hyperparams:
            logger.info(f"Optimizing hyperparameters for {model_config['model_type']} using {optimization_params['method']} method")
            model, best_params = optimize_hyperparameters(
                model_type=model_config['model_type'],
                X_train=X_train_processed,
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
            model.fit(X_train_processed, y_train)
        
        # Calculate feature importance
        importance_df = compute_feature_importance(
            model, X_train_processed, y_train, method='built_in'
        )
        
        # Generate model report
        metrics = generate_model_report(
            model, X_train_processed, y_train, X_test_processed, y_test,
            class_names=class_names, output_dir=model_dir
        )
        
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
                'test_metrics': metrics['test_metrics']
            }
            
            with open(os.path.join(model_dir, 'training_config.json'), 'w') as f:
                json.dump(config, f, indent=4, default=str)
        
        logger.info(f"Model training complete. Test accuracy: {metrics['test_metrics'].get('accuracy', 'N/A'):.4f}")
        
        return model, importance_df, metrics
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return None, pd.DataFrame(), {'error': str(e)}