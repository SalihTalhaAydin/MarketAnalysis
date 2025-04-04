"""
Feature selection techniques for preprocessing model inputs.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

# Setup logging
logger = logging.getLogger(__name__)

# Import scikit-learn components with error handling
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import (
        SelectKBest, f_classif, mutual_info_classif, RFE, RFECV,
        SelectFromModel
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.error("scikit-learn not available for feature selection")
    SKLEARN_AVAILABLE = False

# Try to import advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not installed. XGBoost models unavailable for feature selection.")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not installed. LightGBM models unavailable for feature selection.")
    LIGHTGBM_AVAILABLE = False


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
            from sklearn.preprocessing import StandardScaler
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