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
    """Trains a classification model (e.g., RandomForestClassifier).

    Uses the provided features and target data to train a model suitable
    for predicting market direction or other classification tasks.

    Args:
        features: DataFrame of input features for training.
        target: Series containing the target variable (e.g., 0 or 1) for
                training.

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

    # --- Hyperparameter Tuning with GridSearchCV ---
    print("Starting hyperparameter tuning with GridSearchCV...")

    # Define the parameter grid (start small)
    param_grid = {
        'n_estimators': [50, 100],  # [50, 100, 150],
        'learning_rate': [0.05, 0.1],  # [0.01, 0.05, 0.1],
        'num_leaves': [20, 31],  # [20, 31, 40],
        # 'max_depth': [5, 10, -1], # Keep default for now
        # 'min_child_samples': [10, 20, 30] # Keep default for now
    }

    # Base LGBM model instance
    lgbm = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',  # Can change scoring metric below
        is_unbalance=True,
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
        scoring='roc_auc',  # Or 'accuracy', 'f1', etc.
        cv=tscv,
        n_jobs=-1,  # Use all cores for grid search too
        verbose=1  # Show progress
    )

    try:
        # Fit GridSearchCV to find the best parameters
        grid_search.fit(features, target)

        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best CV score (ROC AUC): {grid_search.best_score_:.4f}")

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