"""
Hyperparameter optimization methods for models.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# Import model factory
from ..factory.model_factory import create_model

# Setup logging
logger = logging.getLogger(__name__)

# Import scikit-learn with error handling
try:
    from sklearn.model_selection import (
        GridSearchCV,
        RandomizedSearchCV,
        TimeSeriesSplit,
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    logger.error("scikit-learn not available for hyperparameter optimization")
    SKLEARN_AVAILABLE = False

# Try to import Optuna for Bayesian optimization
try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    logger.warning("Optuna not installed. Bayesian optimization unavailable.")
    OPTUNA_AVAILABLE = False


# Default hyperparameter spaces for different model types
DEFAULT_RF_PARAMS = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "class_weight": ["balanced", "balanced_subsample", None],
}

DEFAULT_XGB_PARAMS = {
    "n_estimators": [100, 200, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 9],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.2],
}

DEFAULT_LGBM_PARAMS = {
    "n_estimators": [100, 200, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "num_leaves": [31, 63, 127],
    "max_depth": [-1, 5, 10, 15],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
}

DEFAULT_CATBOOST_PARAMS = {
    "iterations": [100, 200, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "depth": [4, 6, 8, 10],
    "l2_leaf_reg": [1, 3, 5, 7, 9],
}


def optimize_hyperparameters_grid(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[Dict] = None,
    cv: int = 5,
    scoring: str = "f1_weighted",
    verbose: int = 1,
    n_jobs: int = -1,
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
        if model_type == "random_forest":
            param_grid = DEFAULT_RF_PARAMS
        elif model_type == "xgboost":
            param_grid = DEFAULT_XGB_PARAMS
        elif model_type == "lightgbm":
            param_grid = DEFAULT_LGBM_PARAMS
        elif model_type == "catboost":
            param_grid = DEFAULT_CATBOOST_PARAMS
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
        verbose=verbose,
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
    scoring: str = "f1_weighted",
    verbose: int = 1,
    n_jobs: int = -1,
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
        if model_type == "random_forest":
            param_distributions = DEFAULT_RF_PARAMS
        elif model_type == "xgboost":
            param_distributions = DEFAULT_XGB_PARAMS
        elif model_type == "lightgbm":
            param_distributions = DEFAULT_LGBM_PARAMS
        elif model_type == "catboost":
            param_distributions = DEFAULT_CATBOOST_PARAMS
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

    # Remove n_iter from distributions if it exists, as it's a direct arg for RandomizedSearchCV
    search_distributions = param_distributions.copy()
    search_distributions.pop("n_iter", None)

    # Create randomized search
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=search_distributions,  # Use cleaned distributions
        n_iter=n_iter,
        cv=cv_obj,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=42,
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
    scoring: str = "f1_weighted",
    verbose: int = 1,
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
        if model_type == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),
            }
            return create_model("random_forest", params)

        elif model_type == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
            }
            return create_model("xgboost", params)

        elif model_type == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 10, 200),
                "max_depth": trial.suggest_int("max_depth", -1, 20),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
            return create_model("lightgbm", params)

        elif model_type == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 50, 500),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "depth": trial.suggest_int("depth", 3, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10, log=True),
            }
            return create_model("catboost", params)

        else:
            logger.warning(
                f"Unsupported model type for Bayesian optimization: {model_type}"
            )
            return create_model(model_type)

    # Create objective function for optimization
    def objective(trial):
        model = create_base_model(trial)

        if model is None:
            return float("inf")

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
            from sklearn.model_selection import cross_val_score

            scores = cross_val_score(
                model, X_train, y_train, scoring=scoring, cv=cv_obj, n_jobs=-1
            )

            return -scores.mean()  # negative because Optuna minimizes

        except Exception as e:
            logger.error(f"Error in Optuna objective: {e}")
            return float("inf")

    try:
        # Create study
        study = optuna.create_study(direction="minimize")

        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose > 0)

        # Get best parameters
        best_params = study.best_params

        # Create best model
        if model_type == "random_forest":
            best_model = create_model("random_forest", best_params)
        elif model_type == "xgboost":
            best_model = create_model("xgboost", best_params)
        elif model_type == "lightgbm":
            best_model = create_model("lightgbm", best_params)
        elif model_type == "catboost":
            best_model = create_model("catboost", best_params)
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
    method: str = "random",
    params: Optional[Dict] = None,
    cv: int = 5,
    scoring: str = "f1_weighted",
    verbose: int = 1,
    n_jobs: int = -1,
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
    if method == "grid":
        return optimize_hyperparameters_grid(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            param_grid=params,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
            n_jobs=n_jobs,
        )

    elif method == "random":
        return optimize_hyperparameters_random(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            param_distributions=params,
            n_iter=params.get("n_iter", 20) if params else 20,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
            n_jobs=n_jobs,
        )

    elif method == "bayesian":
        return optimize_hyperparameters_bayesian(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            param_space=params,
            n_trials=params.get("n_trials", 50) if params else 50,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
        )

    else:
        logger.error(f"Unsupported optimization method: {method}")
        return create_model(model_type), {}
