"""
Feature selection techniques for preprocessing model inputs.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np  # Import numpy
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

# Import scikit-learn components with error handling
try:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import (
        RFE,
        SelectFromModel,
        SelectKBest,
        mutual_info_classif,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler  # Needed for PCA

    SKLEARN_AVAILABLE = True
except ImportError:
    logger.error("scikit-learn not available for feature selection")
    SKLEARN_AVAILABLE = False

# Try to import advanced ML libraries
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    logger.warning(
        "XGBoost not installed. XGBoost models unavailable for feature selection."
    )
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning(
        "LightGBM not installed. LightGBM models unavailable for feature selection."
    )
    LIGHTGBM_AVAILABLE = False


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "importance",
    params: Optional[Dict] = None,
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
    original_columns = list(X.columns)  # Keep original columns for return if needed

    # --- Select only numeric columns for methods that require it ---
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        logger.error(
            "No numeric columns found for feature selection method '{method}'. Returning original features."
        )
        return X, original_columns
    X_numeric = X[numeric_cols]
    logger.info(
        f"Using numeric columns for selection method '{method}': {numeric_cols}"
    )

    # --- Handle NaNs before fitting selectors that require finite values ---
    # Create cleaned versions by dropping rows with any NaNs in numeric columns
    X_numeric_clean = X_numeric.dropna()
    y_clean = y.loc[X_numeric_clean.index]

    if X_numeric_clean.empty:
        logger.warning(
            f"No non-NaN data available for feature selection method '{method}'. Returning original features."
        )
        return X, original_columns

    try:
        if method == "importance":
            # Use Random Forest feature importance on numeric data only
            n_features = params.get("n_features", min(30, X_numeric.shape[1]))
            threshold = params.get("threshold")  # Allow threshold=None

            clf = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            # Fit on cleaned data
            clf.fit(X_numeric_clean, y_clean)
            importances = pd.Series(
                clf.feature_importances_, index=X_numeric_clean.columns
            )  # Use clean columns index
            importances = importances.sort_values(ascending=False)

            # Determine selected features based on importance from cleaned data
            if n_features:
                selected_features = importances.head(n_features).index.tolist()
            elif threshold is not None:
                selected_features = importances[importances >= threshold].index.tolist()
            else:  # Default if neither n_features nor threshold provided
                selected_features = importances.head(
                    min(30, X_numeric_clean.shape[1])
                ).index.tolist()  # Use clean shape

            if not selected_features:
                logger.warning(
                    "No features selected by importance method. Returning original features."
                )
                return X, original_columns

            # Return original DataFrame filtered by selected numeric feature names
            return X[selected_features], selected_features

        elif method == "mutual_info":
            # Use mutual information on numeric data only
            n_features = params.get("n_features", min(30, X_numeric.shape[1]))
            # Ensure k is not greater than the number of features
            # Ensure k is not greater than the number of features in cleaned data
            k = min(n_features, X_numeric_clean.shape[1])
            if k <= 0:
                logger.warning(
                    f"Invalid number of features requested ({n_features}). Returning original features."
                )
                return X, original_columns

            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            # Fit on cleaned data
            selector.fit(X_numeric_clean, y_clean)
            selected_features = X_numeric_clean.columns[
                selector.get_support()
            ].tolist()  # Get names from clean columns

            if not selected_features:
                logger.warning(
                    "No features selected by mutual_info method. Returning original features."
                )
                return X, original_columns

            # Return original DataFrame filtered by selected numeric feature names
            return X[selected_features], selected_features

        elif method == "rfe":
            # Use Recursive Feature Elimination on numeric data only
            n_features = params.get("n_features", min(30, X_numeric.shape[1]))
            step = params.get("step", 1)
            # Ensure n_features_to_select is valid
            # Ensure n_features_to_select is valid based on cleaned data
            n_features_to_select = min(n_features, X_numeric_clean.shape[1])
            if n_features_to_select <= 0:
                logger.warning(
                    f"Invalid n_features_to_select ({n_features_to_select}). Returning original features."
                )
                return X, original_columns

            estimator = LogisticRegression(max_iter=1000, random_state=42)
            selector = RFE(
                estimator=estimator,
                n_features_to_select=n_features_to_select,
                step=step,
            )
            # Fit on cleaned data
            selector.fit(X_numeric_clean, y_clean)
            selected_features = X_numeric_clean.columns[
                selector.support_
            ].tolist()  # Get names from clean columns

            if not selected_features:
                logger.warning(
                    "No features selected by RFE method. Returning original features."
                )
                return X, original_columns

            # Return original DataFrame filtered by selected numeric feature names
            return X[selected_features], selected_features

        elif method == "pca":
            # Use PCA on numeric data only
            # Determine n_components based on cleaned data shape
            n_components = params.get(
                "n_components",
                min(X_numeric_clean.shape[1], X_numeric_clean.shape[0], 30),
            )
            if n_components <= 0:
                logger.warning(
                    f"Invalid n_components ({n_components}) for PCA. Returning original features."
                )
                return X, original_columns

            scaler = StandardScaler()
            # Scale cleaned data
            X_scaled = scaler.fit_transform(X_numeric_clean)
            pca = PCA(n_components=n_components)
            # Fit PCA on scaled cleaned data
            X_pca = pca.fit_transform(X_scaled)
            selected_features = [f"PC{i + 1}" for i in range(n_components)]
            # Create PCA DataFrame using the index from the cleaned data
            X_pca_df = pd.DataFrame(
                X_pca, index=X_numeric_clean.index, columns=selected_features
            )
            # PCA returns a transformed DataFrame, not a subset of original columns
            return X_pca_df, selected_features

        elif method == "model":
            # Use model-based selection on numeric data only
            model_type = params.get("model_type", "lightgbm")
            n_features = params.get(
                "n_features", min(30, X_numeric_clean.shape[1])
            )  # Use clean shape
            threshold = params.get("threshold")  # Allow None

            model_instance = None
            if model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
                model_instance = lgb.LGBMClassifier(n_estimators=100, random_state=42)
            elif model_type == "xgboost" and XGBOOST_AVAILABLE:
                model_instance = xgb.XGBClassifier(n_estimators=100, random_state=42)
            else:
                logger.warning(
                    f"Unsupported or unavailable model type for feature selection: {model_type}. Using RandomForest."
                )
                model_instance = RandomForestClassifier(
                    n_estimators=100, random_state=42
                )

            # Use threshold="median" if threshold param is None, otherwise use the value
            threshold_param = "median" if threshold is None else threshold
            # Ensure max_features is valid
            # Ensure max_features is valid based on cleaned data
            max_features_param = (
                min(n_features, X_numeric_clean.shape[1]) if n_features else None
            )
            if max_features_param is not None and max_features_param <= 0:
                logger.warning(
                    f"Invalid max_features ({max_features_param}). Using threshold only."
                )
                max_features_param = None  # Disable max_features if invalid

            selector = SelectFromModel(
                model_instance,
                threshold=threshold_param,
                max_features=max_features_param,
            )
            # Fit on cleaned data
            selector.fit(X_numeric_clean, y_clean)
            selected_features = X_numeric_clean.columns[
                selector.get_support()
            ].tolist()  # Get names from clean columns

            if not selected_features:
                logger.warning(
                    f"No features selected by model ({model_type}) method. Returning original features."
                )
                return X, original_columns

            # Return original DataFrame filtered by selected numeric feature names
            return X[selected_features], selected_features

        else:
            logger.warning(
                f"Unsupported feature selection method: {method}. Returning original features."
            )
            return X, original_columns

    except Exception as e:
        logger.error(
            f"Error during feature selection method '{method}': {e}", exc_info=True
        )
        return X, original_columns
