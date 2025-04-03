import pandas as pd
# from sklearn.ensemble import RandomForestClassifier # Replaced with LightGBM
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
try:
    import lightgbm as lgb
except ImportError:
    print("Warning: lightgbm not installed (pip install lightgbm). "
          "Model training will fail.")
    lgb = None


def train_classification_model(features: pd.DataFrame, target: pd.Series):
    """Trains a multi-class classification model (LightGBM).

    Uses the provided features and Triple Barrier Method target data (-1, 0, 1)
    to train a model suitable for predicting trade outcomes. Includes
    hyperparameter tuning using GridSearchCV with TimeSeriesSplit.

    Args:
        features: DataFrame of input features for training.
        target: Series containing the multi-class target variable (-1, 0, 1)
                for training.

    Returns:
        The trained scikit-learn model object, or None if training fails.
    """
    print("--- Training Classification Model ---")
    if features.empty or target.empty:
        print("Error: Features or target data is empty for training.")
        return None
    if len(features) != len(target):
        print(f"Error: Features ({len(features)}) and target ({len(target)}) "
              "length mismatch.")
        return None

    print(f"Training with Features shape: {features.shape}")
    print(f"Training with Target shape: {target.shape}")

    # Check for NaNs before training
    if features.isnull().any().any() or target.isnull().any():
        print("Error: NaN values detected in features or target for training.")
        return None

    if lgb is None:
        print("Error: lightgbm library is required but not installed.")
        return None

    # --- Map target labels for LightGBM ---
    # LightGBM expects labels from 0 to num_class-1
    # Mapping: -1 (SL) -> 0, 0 (Time) -> 1, 1 (TP) -> 2
    target_mapped = target.map({-1: 0, 0: 1, 1: 2})
    if target_mapped.isnull().any():
        print("Error: NaN values found after mapping target labels.")
        # This might happen if original target had values other than -1, 0, 1
        return None
    print("Target labels mapped using: {-1: 0, 0: 1, 1: 2}")

    # --- Hyperparameter Tuning with GridSearchCV ---
    print("Starting hyperparameter tuning with GridSearchCV...")

    # Define the parameter grid (start small)
    param_grid = {
        'n_estimators': [50, 100, 150],  # Added 150
        'learning_rate': [0.05, 0.1],
        'num_leaves': [20, 31, 40],  # Added 40
        # 'max_depth': [5, 10, -1], # Keep default for now
        # 'min_child_samples': [10, 20, 30], # Keep default for now
        'reg_alpha': [0.0, 0.1],  # Added L1 regularization
        'reg_lambda': [0.0, 0.1]   # Added L2 regularization
    }

    # Base LGBM model instance
    lgbm = lgb.LGBMClassifier(
        objective='multiclass',
        # For labels -1, 0, 1 (needs mapping if using default LGBM)
        num_class=3,
        metric='multi_logloss',   # Metric for multiclass
        # is_unbalance=True, # Deprecated, handle via metrics/sampling
        random_state=42,
        n_jobs=-1
    )

    # TimeSeriesSplit for cross-validation (important for time series)
    # n_splits=3 is faster, 5 is more robust but slower
    tscv = TimeSeriesSplit(n_splits=3)

    # GridSearchCV setup
    # Scoring: 'roc_auc' is common for binary classification
    grid_search = GridSearchCV(
        estimator=lgbm,
        param_grid=param_grid,
        scoring='f1_weighted',  # Better for potentially imbalanced multiclass
        cv=tscv,
        n_jobs=-1,  # Use all cores for grid search too
        verbose=1  # Show progress
    )

    try:
        # Fit GridSearchCV to find the best parameters
        grid_search.fit(features, target_mapped)

        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best CV score (F1 Weighted): {grid_search.best_score_:.4f}")

        # Get the best model found by the grid search
        best_model = grid_search.best_estimator_
        print("Model training complete (using best GridSearch estimator).")
        return best_model
    except NotFittedError:
        # This check might be redundant if fit() succeeded, but good practice.
        print("Error: Model reported as not fitted after training attempt.")
        return None
    except ValueError as ve:
        print(f"Error during model training (ValueError): {ve}")
        # Common issues: NaNs, inf values, non-numeric data
        return None
    except Exception as e:
        print(f"An unexpected error occurred during model training: {e}")
        return None