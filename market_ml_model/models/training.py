"""
Model training orchestration module.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from .evaluation.metrics import compute_feature_importance, evaluate_classifier

# Import components
from .factory.model_factory import create_model

# Import feature selection
from .feature_selection import select_features
from .optimization.hyperparameters import optimize_hyperparameters

# Import scikit-learn components for preprocessing and splitting
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        MinMaxScaler,
        OneHotEncoder,
        QuantileTransformer,
        RobustScaler,
        StandardScaler,
    )

    SKLEARN_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing scikit-learn components for training: {e}")
    SKLEARN_AVAILABLE = False


# Setup logging
logger = logging.getLogger(__name__)


def create_feature_pipeline(
    X: pd.DataFrame,
    categorical_features: Optional[List[str]] = None,
    numeric_features: Optional[List[str]] = None,
    scaling_method: str = "standard",
    handle_missing: bool = True,
) -> Tuple[Optional[ColumnTransformer], List[str]]:
    """
    Create a scikit-learn pipeline for feature preprocessing.

    Args:
        X: Feature DataFrame
        categorical_features: List of categorical features
        numeric_features: List of numeric features
        scaling_method: Method for scaling numeric features ('standard', 'minmax', 'robust', 'quantile')
        handle_missing: Whether to handle missing values

    Returns:
        Tuple of (preprocessing_pipeline, transformed_feature_names) or (None, []) if error
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn not available for feature pipeline")
        return None, []

    # Identify column types if not provided
    if categorical_features is None and numeric_features is None:
        categorical_features = []
        numeric_features = []

        for col in X.columns:
            if X[col].dtype == "object" or isinstance(X[col].dtype, pd.CategoricalDtype):
                categorical_features.append(col)
            elif pd.api.types.is_numeric_dtype(X[col]):
                numeric_features.append(col)
            else:
                logger.warning(
                    f"Column '{col}' has unhandled dtype {X[col].dtype}, skipping."
                )

    elif categorical_features is None:
        categorical_features = [col for col in X.columns if col not in numeric_features]

    elif numeric_features is None:
        numeric_features = [col for col in X.columns if col not in categorical_features]

    # Ensure all features are accounted for and exist
    all_input_features = categorical_features + numeric_features
    missing_in_df = set(all_input_features) - set(X.columns)
    if missing_in_df:
        logger.error(f"Features specified but not found in DataFrame: {missing_in_df}")
        return None, []

    unclassified_in_df = set(X.columns) - set(all_input_features)
    if unclassified_in_df:
        logger.warning(
            f"Features in DataFrame but not classified: {unclassified_in_df}. Attempting auto-classification."
        )
        for feature in unclassified_in_df:
            if pd.api.types.is_numeric_dtype(X[feature]):
                numeric_features.append(feature)
                logger.info(f"Auto-classified '{feature}' as numeric.")
            else:
                # Assume categorical for others for now, might need refinement
                categorical_features.append(feature)
                logger.info(f"Auto-classified '{feature}' as categorical.")

    transformers = []
    transformed_feature_names = []

    # Create transformer for numeric features
    if numeric_features:
        numeric_transformer_steps = []
        if handle_missing:
            numeric_transformer_steps.append(
                ("imputer", SimpleImputer(strategy="median"))
            )

        if scaling_method == "standard":
            numeric_transformer_steps.append(("scaler", StandardScaler()))
        elif scaling_method == "minmax":
            numeric_transformer_steps.append(("scaler", MinMaxScaler()))
        elif scaling_method == "robust":
            numeric_transformer_steps.append(("scaler", RobustScaler()))
        elif scaling_method == "quantile":
            numeric_transformer_steps.append(
                ("scaler", QuantileTransformer(output_distribution="normal"))
            )
        elif scaling_method is not None and scaling_method != "none":
            logger.warning(
                f"Unsupported scaling method: {scaling_method}. No scaling applied."
            )

        if numeric_transformer_steps:  # Only add transformer if steps exist
            numeric_transformer = Pipeline(steps=numeric_transformer_steps)
            transformers.append(("num", numeric_transformer, numeric_features))
            transformed_feature_names.extend(numeric_features)
        else:  # If no steps, just pass through
            transformers.append(("num_passthrough", "passthrough", numeric_features))
            transformed_feature_names.extend(numeric_features)

    # Create transformer for categorical features
    if categorical_features:
        categorical_transformer_steps = []
        if handle_missing:
            categorical_transformer_steps.append(
                ("imputer", SimpleImputer(strategy="most_frequent"))
            )

        onehot_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )  # Use sparse_output=False for easier name handling
        categorical_transformer_steps.append(("onehot", onehot_encoder))

        categorical_transformer = Pipeline(steps=categorical_transformer_steps)
        transformers.append(("cat", categorical_transformer, categorical_features))

        # Calculate one-hot encoded feature names after fitting
        # We can't know the names until fit, so return placeholder or fit here?
        # For now, let's just return the original names and note that OHE names need handling later
        # Or, we can try a dummy fit to get names (might be slow/memory intensive)
        # Let's return the original names for now. The caller needs to be aware.
        # Alternatively, return the fitted transformer?
        # Returning the transformer itself is probably best.

    if not transformers:
        logger.error("No features to process in the pipeline.")
        return None, []

    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=transformers, remainder="passthrough"  # Keep other columns if any
    )

    # We cannot reliably get output feature names without fitting.
    # Return the unfitted preprocessor and the original numeric/categorical names.
    # The caller will need to fit and potentially get names from the fitted transformer.
    output_feature_names = numeric_features + categorical_features  # Placeholder

    return preprocessor, output_feature_names


def train_classification_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_config: Optional[Dict] = None,
    feature_selection_config: Optional[Dict] = None,  # Renamed for clarity
    preprocessing_config: Optional[Dict] = None,  # Renamed for clarity
    test_size: float = 0.2,
    optimize_hyperparams: bool = True,
    optimization_config: Optional[Dict] = None,  # Renamed for clarity
    output_dir: Optional[str] = None,
    model_id: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    random_state: int = 42,
) -> Tuple[Optional[Any], Optional[pd.DataFrame], Optional[Dict]]:
    """
    Train a classification model with advanced options.

    Args:
        features: Feature DataFrame
        target: Target Series
        model_config: Model configuration (type, specific params)
        feature_selection_config: Feature selection configuration (method, params)
        preprocessing_config: Preprocessing configuration (scaling, missing handling)
        test_size: Test set fraction
        optimize_hyperparams: Whether to optimize hyperparameters
        optimization_config: Hyperparameter optimization configuration (method, params)
        output_dir: Directory to save model and reports
        model_id: Model identifier (auto-generated if None)
        class_names: Names of target classes
        random_state: Random seed

    Returns:
        Tuple of (trained_model, feature_importance, metrics) or (None, None, None) if error
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn is required for model training.")
        return None, None, None

    # Generate model ID if not provided
    if model_id is None:
        model_id = (
            f"model_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    logger.info(f"Starting model training with ID: {model_id}")

    # Create model output directory if needed
    model_dir = None
    if output_dir:
        model_dir = os.path.join(output_dir, model_id)
        try:
            os.makedirs(model_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {model_dir}: {e}")
            model_dir = None  # Disable saving if directory creation fails

    # --- Configuration Defaults ---
    model_config = model_config or {}
    model_type = model_config.get("model_type", "lightgbm")  # Default to lightgbm
    model_params = model_config.get("params", {})

    feature_selection_config = feature_selection_config or {}
    perform_feature_selection = feature_selection_config.get("enabled", True)
    fs_method = feature_selection_config.get("method", "importance")
    fs_params = feature_selection_config.get(
        "params", {"n_features": min(50, features.shape[1])}
    )

    preprocessing_config = preprocessing_config or {}
    scaling_method = preprocessing_config.get("scaling_method", "standard")
    handle_missing = preprocessing_config.get("handle_missing", True)

    optimization_config = optimization_config or {}
    opt_method = optimization_config.get("method", "random")
    opt_params = optimization_config.get(
        "params", {}
    )  # Params for the method (e.g., n_iter)
    opt_cv = optimization_config.get("cv", 5)
    opt_scoring = optimization_config.get("scoring", "f1_weighted")

    # --- Data Splitting ---
    try:
        if isinstance(features.index, pd.DatetimeIndex):
            # Time series data - use last n% as test
            split_idx = int(len(features) * (1 - test_size))
            X_train_raw, X_test_raw = (
                features.iloc[:split_idx],
                features.iloc[split_idx:],
            )
            y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
            logger.info(
                f"Split data using time series method: {len(X_train_raw)} train, {len(X_test_raw)} test"
            )
        else:
            # Regular split
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                features,
                target,
                test_size=test_size,
                random_state=random_state,
                stratify=target,  # Added stratify
            )
            logger.info(
                f"Split data using random stratified split: {len(X_train_raw)} train, {len(X_test_raw)} test"
            )
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        return None, None, {"error": f"Data splitting failed: {e}"}

    # --- Feature Selection ---
    selected_features = list(features.columns)
    X_train_fs, X_test_fs = X_train_raw, X_test_raw
    if perform_feature_selection:
        try:
            logger.info(f"Performing feature selection using {fs_method} method")
            X_train_fs, selected_features = select_features(
                X_train_raw, y_train, method=fs_method, params=fs_params
            )
            X_test_fs = X_test_raw[selected_features]  # Apply selection to test set
            logger.info(f"Selected {len(selected_features)} features")

            if model_dir:
                fs_path = os.path.join(model_dir, "selected_features.json")
                try:
                    with open(fs_path, "w") as f:
                        json.dump(selected_features, f, indent=4)
                except Exception as e:
                    logger.error(f"Failed to save selected features to {fs_path}: {e}")

        except Exception as e:
            logger.error(f"Error during feature selection: {e}. Using all features.")
            selected_features = list(features.columns)
            X_train_fs, X_test_fs = X_train_raw, X_test_raw
    else:
        logger.info("Skipping feature selection.")

    # --- Preprocessing ---
    try:
        logger.info(
            f"Creating preprocessing pipeline (Scaling: {scaling_method}, Missing: {handle_missing})"
        )
        # Automatically detect numeric/categorical if not specified in config
        num_features = preprocessing_config.get("numeric_features")
        cat_features = preprocessing_config.get("categorical_features")

        # Filter selected features for preprocessing
        current_num_features = (
            [
                f
                for f in selected_features
                if f in X_train_fs.columns
                and pd.api.types.is_numeric_dtype(X_train_fs[f])
            ]
            if num_features is None
            else [f for f in num_features if f in selected_features]
        )
        current_cat_features = (
            [f for f in selected_features if f not in current_num_features]
            if cat_features is None
            else [f for f in cat_features if f in selected_features]
        )

        preprocessor, _ = create_feature_pipeline(
            X_train_fs,  # Pass the feature-selected data
            categorical_features=current_cat_features,
            numeric_features=current_num_features,
            scaling_method=scaling_method,
            handle_missing=handle_missing,
        )

        if preprocessor is None:
            raise ValueError("Failed to create preprocessing pipeline.")

        logger.info("Fitting preprocessing pipeline on training data...")
        X_train_processed = preprocessor.fit_transform(X_train_fs)
        logger.info("Transforming test data...")
        X_test_processed = preprocessor.transform(X_test_fs)

        # Get feature names after transformation (important if OHE used)
        try:
            # Attempt to get feature names from the fitted transformer
            feature_names_out = preprocessor.get_feature_names_out()
        except Exception:
            logger.warning(
                "Could not get feature names from preprocessor. Using original selected names."
            )
            feature_names_out = selected_features  # Fallback

        # Convert processed data back to DataFrame (optional, but good for inspection)
        X_train_processed = pd.DataFrame(
            X_train_processed, index=X_train_fs.index, columns=feature_names_out
        )
        X_test_processed = pd.DataFrame(
            X_test_processed, index=X_test_fs.index, columns=feature_names_out
        )

        if model_dir:
            pipe_path = os.path.join(model_dir, "preprocessor.pkl")
            try:
                joblib.dump(preprocessor, pipe_path)
                logger.info(f"Saved preprocessor pipeline to {pipe_path}")
            except Exception as e:
                logger.error(f"Failed to save preprocessor pipeline: {e}")

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return None, None, {"error": f"Preprocessing failed: {e}"}

    # --- Model Training ---
    model = None
    best_params = model_params  # Start with initial params
    try:
        if optimize_hyperparams:
            logger.info(
                f"Optimizing hyperparameters for {model_type} using {opt_method} method"
            )
            # Pass the processed training data to optimization
            model, best_params = optimize_hyperparameters(
                model_type=model_type,
                X_train=X_train_processed,  # Use processed data
                y_train=y_train,
                method=opt_method,
                params=opt_params,  # Pass optimization specific params (e.g., n_iter, space)
                cv=opt_cv,
                scoring=opt_scoring,
                verbose=1,
            )
            logger.info(
                f"Hyperparameter optimization complete. Best parameters found: {best_params}"
            )

            if model_dir:
                params_path = os.path.join(model_dir, "best_params.json")
                try:
                    # Save only the best hyperparameters found
                    with open(params_path, "w") as f:
                        json.dump(best_params, f, indent=4, default=str)
                    logger.info(f"Saved best hyperparameters to {params_path}")
                except Exception as e:
                    logger.error(f"Failed to save best parameters: {e}")

        else:
            logger.info(f"Training {model_type} with provided/default parameters")
            # Use initial model_params if not optimizing
            model = create_model(model_type, model_params)
            if model is None:
                raise ValueError(f"Failed to create model {model_type}")
            # Fit model on processed data
            model.fit(X_train_processed, y_train)
            best_params = model_params  # Store the params used

        if model is None:
            raise ValueError("Model training failed.")

        logger.info("Model training finished.")

        # --- Evaluation ---
        logger.info("Evaluating model performance...")
        # Evaluate on processed data
        train_metrics = evaluate_classifier(
            model, X_train_processed, y_train, class_names=class_names
        )
        test_metrics = evaluate_classifier(
            model, X_test_processed, y_test, class_names=class_names
        )
        metrics = {"train": train_metrics, "test": test_metrics}
        logger.info(
            f"Test Set Metrics: Accuracy={test_metrics.get('accuracy', 'N/A'):.4f}, F1-Weighted={test_metrics.get('f1_weighted', 'N/A'):.4f}"
        )

        # --- Feature Importance ---
        logger.info("Calculating feature importance...")
        # Calculate importance on processed data
        importance_df = compute_feature_importance(
            model, X_train_processed, y_train, method="built_in"  # Try built-in first
        )
        if importance_df.empty:  # Fallback to permutation
            importance_df = compute_feature_importance(
                model, X_train_processed, y_train, method="permutation"
            )
        logger.info(
            f"Feature importance calculated. Top 5: {importance_df['Feature'].head().tolist()}"
        )

        # --- Saving ---
        if model_dir:
            # Save model
            model_path = os.path.join(model_dir, "model.pkl")
            try:
                joblib.dump(model, model_path)
                logger.info(f"Saved trained model to {model_path}")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")

            # Save training configuration
            final_config = {
                "model_id": model_id,
                "model_type": model_type,
                "model_params_used": best_params,  # Save the actual params used
                "feature_selection_config": feature_selection_config,
                "preprocessing_config": preprocessing_config,
                "optimization_config": (
                    optimization_config if optimize_hyperparams else None
                ),
                "selected_features": selected_features,
                "processed_feature_names": list(X_train_processed.columns),
                "test_size": test_size,
                "random_state": random_state,
                "training_timestamp": datetime.now().isoformat(),
                "num_original_features": features.shape[1],
                "num_selected_features": len(selected_features),
                "num_processed_features": X_train_processed.shape[1],
                "num_train_samples": len(X_train_processed),
                "num_test_samples": len(X_test_processed),
                "target_distribution_train": y_train.value_counts().to_dict(),
                "target_distribution_test": y_test.value_counts().to_dict(),
                "final_test_metrics": test_metrics,  # Include full test metrics
            }
            config_path = os.path.join(model_dir, "training_summary.json")
            try:
                with open(config_path, "w") as f:
                    json.dump(
                        final_config, f, indent=4, default=str
                    )  # Use default=str for non-serializable
                logger.info(f"Saved training summary to {config_path}")
            except Exception as e:
                logger.error(f"Failed to save training summary: {e}")

            # Save feature importance
            if not importance_df.empty:
                imp_path = os.path.join(model_dir, "feature_importance.csv")
                try:
                    importance_df.to_csv(imp_path, index=False)
                    logger.info(f"Saved feature importance to {imp_path}")
                except Exception as e:
                    logger.error(f"Failed to save feature importance: {e}")

        return model, importance_df, metrics

    except Exception as e:
        logger.exception(
            f"Critical error during model training pipeline: {e}"
        )  # Log traceback
        return None, None, {"error": f"Training pipeline failed: {e}"}
