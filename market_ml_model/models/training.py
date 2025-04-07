"""
Model training orchestration module.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np  # Added for importance selection
import pandas as pd

from .evaluation.metrics import compute_feature_importance, evaluate_classifier
from .factory.model_factory import create_model
from .feature_selection import select_features
from .optimization.hyperparameters import optimize_hyperparameters

# Setup logging
logger = logging.getLogger(__name__)

# Import scikit-learn components
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        MinMaxScaler,
        OneHotEncoder,
        RobustScaler,
        StandardScaler,
    )

    SKLEARN_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing scikit-learn components for training: {e}")
    SKLEARN_AVAILABLE = False


def create_feature_pipeline(
    X: pd.DataFrame,
    categorical_features: Optional[List[str]] = None,
    numeric_features: Optional[List[str]] = None,
    scaling_method: str = "standard",
    handle_missing: bool = True,
) -> Tuple[Optional[ColumnTransformer], List[str]]:
    """Creates a scikit-learn pipeline for feature preprocessing."""
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn not available for feature pipeline")
        return None, []

    # Auto-detect features if not provided
    if categorical_features is None and numeric_features is None:
        categorical_features = [
            col
            for col in X.columns
            if X[col].dtype == "object" or isinstance(X[col].dtype, pd.CategoricalDtype)
        ]
        numeric_features = [
            col
            for col in X.columns
            if pd.api.types.is_numeric_dtype(X[col]) and col not in categorical_features
        ]
        unclassified = (
            set(X.columns) - set(categorical_features) - set(numeric_features)
        )
        if unclassified:
            logger.warning(
                f"Columns not classified as numeric or categorical: {unclassified}. Skipping them."
            )

    elif categorical_features is None:
        if numeric_features is None:
            numeric_features = []
        categorical_features = [col for col in X.columns if col not in numeric_features]
    elif numeric_features is None:
        if categorical_features is None:
            categorical_features = []
        numeric_features = [col for col in X.columns if col not in categorical_features]

    # Validate provided/detected features exist
    all_input_features = categorical_features + numeric_features
    missing_in_df = set(all_input_features) - set(X.columns)
    if missing_in_df:
        logger.error(f"Features specified but not found in DataFrame: {missing_in_df}")
        return None, []

    # Filter features to only those present in X before building transformers
    numeric_features = [f for f in numeric_features if f in X.columns]
    categorical_features = [f for f in categorical_features if f in X.columns]

    transformers = []
    original_feature_names_ordered = []

    # Numeric pipeline
    if numeric_features:
        numeric_steps = []
        if handle_missing:
            numeric_steps.append(("imputer", SimpleImputer(strategy="median")))

        normalized_scaling_method = (
            str(scaling_method).lower() if scaling_method is not None else "none"
        )

        # Instantiate only the selected scaler
        if normalized_scaling_method == "standardscaler":
            scaler = StandardScaler()
            numeric_steps.append(("scaler", scaler))
            logger.info("Using StandardScaler for numeric features.")
        elif normalized_scaling_method == "minmaxscaler":
            scaler = MinMaxScaler()
            numeric_steps.append(("scaler", scaler))
            logger.info("Using MinMaxScaler for numeric features.")
        elif normalized_scaling_method == "robustscaler":
            scaler = RobustScaler()
            numeric_steps.append(("scaler", scaler))
            logger.info("Using RobustScaler for numeric features.")
        elif normalized_scaling_method == "none":
            logger.info("No scaling applied to numeric features.")
        else:
            logger.warning(
                f"Unsupported scaling method: '{scaling_method}'. No scaling applied."
            )

        if numeric_steps:
            transformers.append(
                ("num", Pipeline(steps=numeric_steps), numeric_features)
            )
        else:
            transformers.append(("num_passthrough", "passthrough", numeric_features))
        original_feature_names_ordered.extend(numeric_features)

    # Categorical pipeline
    if categorical_features:
        categorical_steps = []
        if handle_missing:
            categorical_steps.append(
                ("imputer", SimpleImputer(strategy="most_frequent"))
            )
        categorical_steps.append(
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=int),
            )
        )
        transformers.append(
            ("cat", Pipeline(steps=categorical_steps), categorical_features)
        )
        original_feature_names_ordered.extend(categorical_features)

    if not transformers:
        logger.error("No features to process in the pipeline.")
        return None, []

    # Use remainder='drop'
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    return preprocessor, original_feature_names_ordered


def train_classification_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_config: Optional[Dict] = None,
    feature_selection_config: Optional[Dict] = None,
    preprocessing_config: Optional[Dict] = None,
    test_size: float = 0.2,
    optimize_hyperparams: bool = True,
    optimization_config: Optional[Dict] = None,
    output_dir: Optional[str] = None,
    model_id: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    random_state: int = 42,
) -> Tuple[Optional[Any], Optional[ColumnTransformer], Optional[Dict], List[str]]:
    """Trains a classification model with advanced options."""
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn is required for model training.")
        return None, None, None, []

    if model_id is None:
        model_id = (
            f"model_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    logger.info(f"Starting model training with ID: {model_id}")

    model_dir = output_dir
    if output_dir:
        try:
            os.makedirs(model_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {model_dir}: {e}")
            model_dir = None

    # --- Config Defaults ---
    model_config = model_config or {}
    model_type = model_config.get("model_type", "lightgbm")  # Use .get() for dict
    model_params = model_config.get("params", {})  # Use .get() for dict
    # Default feature selection to enabled with mutual_info
    feature_selection_config = feature_selection_config or {"enabled": True}
    perform_feature_selection = feature_selection_config.get(
        "enabled", True
    )  # Respect config, use .get() for dict
    fs_method = getattr(feature_selection_config, "method", "mutual_info")
    fs_params = getattr(
        feature_selection_config, "params", {"n_features": min(50, features.shape[1])}
    )
    preprocessing_config = preprocessing_config or {}
    scaling_method = getattr(preprocessing_config, "scaling_method", "standardscaler")
    handle_missing = getattr(preprocessing_config, "handle_missing", True)
    optimization_config = optimization_config or {}
    opt_method = optimization_config.get("method", "random")  # Use .get() for dict
    opt_params = optimization_config.get("params", {})  # Use .get() for dict
    opt_cv = optimization_config.get("cv", 5)  # Use .get() for dict
    opt_scoring = optimization_config.get(
        "scoring", "f1_weighted"
    )  # Use .get() for dict

    # --- Data Splitting ---
    try:
        if isinstance(features.index, pd.DatetimeIndex):
            split_idx = int(len(features) * (1 - test_size))
            X_train_raw, X_test_raw = (
                features.iloc[:split_idx],
                features.iloc[split_idx:],
            )
            y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
        else:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                features,
                target,
                test_size=test_size,
                random_state=random_state,
                stratify=target,
            )
        logger.info(f"Split data: {len(X_train_raw)} train, {len(X_test_raw)} test")
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        logger.error(f"Error splitting data: {e}", exc_info=True)
        # Ensure this returns immediately
        return None, None, {"error": f"Data splitting failed: {e}"}, []

    # --- Feature Selection (On Raw Training Data) ---
    selected_feature_names_raw = list(X_train_raw.columns)
    X_train_selected_raw = X_train_raw
    X_test_selected_raw = X_test_raw

    if (
        perform_feature_selection and fs_method != "importance"
    ):  # Skip initial selection for 'importance'
        try:
            logger.info(
                f"Performing feature selection using {fs_method} method on raw training data (before preprocessing)"
            )
            # Select features based on raw training data
            _, selected_feature_names_raw = select_features(
                X_train_raw, y_train, method=fs_method, params=fs_params
            )
            # Apply selection to raw train and test sets
            X_train_selected_raw = X_train_raw[selected_feature_names_raw]
            X_test_selected_raw = X_test_raw[selected_feature_names_raw]
            logger.info(
                f"Selected {len(selected_feature_names_raw)} raw features (before preprocessing)"
            )

            if model_dir:
                fs_path = os.path.join(model_dir, "selected_features.json")
                try:
                    with open(fs_path, "w") as f:
                        json.dump(selected_feature_names_raw, f, indent=4)
                except Exception as e:
                    logger.error(f"Failed to save selected features to {fs_path}: {e}")

        except Exception as e:
            logger.error(
                f"Error during feature selection: {e}. Using all features.",
                exc_info=True,
            )
            selected_feature_names_raw = list(X_train_raw.columns)
            X_train_selected_raw, X_test_selected_raw = X_train_raw, X_test_raw
    elif perform_feature_selection and fs_method == "importance":
        logger.info(
            "Skipping initial feature selection step; 'importance' method runs after preprocessing."
        )
        # Use all features passed into the function for preprocessing initially
        selected_feature_names_raw = list(X_train_raw.columns)
        X_train_selected_raw = X_train_raw
        X_test_selected_raw = X_test_raw
    else:
        logger.info("Feature selection disabled.")
        selected_feature_names_raw = list(X_train_raw.columns)
        X_train_selected_raw = X_train_raw
        X_test_selected_raw = X_test_raw

    # --- Preprocessing (On Selected Raw Data) ---
    preprocessor = None
    processed_feature_names = (
        selected_feature_names_raw  # Start with selected raw names
    )
    X_train_processed_np = X_train_selected_raw.values
    X_test_processed_np = X_test_selected_raw.values

    try:
        logger.info(
            f"Creating preprocessing pipeline (Scaling: {scaling_method}, Missing: {handle_missing}) on selected features"
        )
        # Detect numeric/categorical from the *selected* raw data
        num_features_in = [
            col
            for col in X_train_selected_raw.columns
            if pd.api.types.is_numeric_dtype(X_train_selected_raw[col])
        ]
        cat_features_in = [
            col for col in X_train_selected_raw.columns if col not in num_features_in
        ]

        preprocessor, _ = create_feature_pipeline(
            X_train_selected_raw,  # Use selected raw data to define pipeline
            categorical_features=cat_features_in,
            numeric_features=num_features_in,
            scaling_method=scaling_method,
            handle_missing=handle_missing,
        )

        if preprocessor is None:
            raise ValueError("Failed to create preprocessing pipeline.")

        logger.info("Fitting preprocessing pipeline on selected training data...")
        X_train_processed_np = preprocessor.fit_transform(X_train_selected_raw)
        logger.info("Transforming selected test data...")
        X_test_processed_np = preprocessor.transform(X_test_selected_raw)

        try:
            processed_feature_names = preprocessor.get_feature_names_out()
        except Exception:
            logger.warning("Could not get feature names from preprocessor.")
            processed_feature_names = [
                f"feature_{i}" for i in range(X_train_processed_np.shape[1])
            ]

        if model_dir:
            pipe_path = os.path.join(model_dir, "preprocessor.pkl")
            try:
                joblib.dump(preprocessor, pipe_path)
                logger.info(f"Saved preprocessor pipeline to {pipe_path}")
            except Exception as e:
                logger.error(f"Failed to save preprocessor pipeline: {e}")

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        # Ensure this returns immediately
        return None, None, {"error": f"Preprocessing failed: {e}"}, []

    # Final data for training/evaluation (NumPy arrays)
    X_train_final_np = X_train_processed_np
    X_test_final_np = X_test_processed_np
    final_feature_names = (
        processed_feature_names  # Names after preprocessing selected features
    )

    # Create DataFrames for evaluation/importance using final names
    X_train_final_df = pd.DataFrame(
        X_train_final_np, columns=final_feature_names, index=X_train_raw.index
    )  # Use original index
    X_test_final_df = pd.DataFrame(
        X_test_final_np, columns=final_feature_names, index=X_test_raw.index
    )  # Use original index

    # --- Importance-Based Feature Selection (After Preprocessing) ---
    num_features_before_importance = X_train_final_np.shape[1]
    if perform_feature_selection and fs_method == "importance":
        try:
            n_features_to_select = fs_params.get("n_features", None)
            if (
                n_features_to_select is None
                or not isinstance(n_features_to_select, int)
                or n_features_to_select <= 0
            ):
                logger.warning(
                    f"Invalid or missing 'n_features' in feature_selection_config.params for 'importance' method. Defaulting to use all {num_features_before_importance} features."
                )
                n_features_to_select = num_features_before_importance
            elif n_features_to_select > num_features_before_importance:
                logger.warning(
                    f"'n_features' ({n_features_to_select}) is greater than available features ({num_features_before_importance}). Using all available features."
                )
                n_features_to_select = num_features_before_importance

            logger.info(
                f"Performing importance-based feature selection. Selecting top {n_features_to_select} features."
            )

            # Train a temporary model (e.g., LightGBM) to get importance scores
            # Use default params or params from model_config if available, but don't optimize here
            temp_model_type = model_type  # Assume same type for importance calc
            temp_model_params = model_params  # Use same base params
            logger.info(
                f"Training temporary {temp_model_type} model for feature importance calculation..."
            )
            temp_model = create_model(temp_model_type, temp_model_params)
            if temp_model is None:
                raise ValueError(
                    f"Failed to create temporary model {temp_model_type} for importance calculation."
                )

            # Check if the model has fit method and feature_importances_ attribute
            if not (
                hasattr(temp_model, "fit")
                and hasattr(temp_model, "feature_importances_")
            ):
                raise TypeError(
                    f"Model type {temp_model_type} does not support 'feature_importances_' needed for importance selection."
                )

            temp_model.fit(X_train_final_np, y_train)  # Fit on processed data
            importances = temp_model.feature_importances_

            if len(importances) != len(final_feature_names):
                raise ValueError(
                    f"Mismatch between number of importances ({len(importances)}) and feature names ({len(final_feature_names)})"
                )

            # Get indices of top N features
            indices = np.argsort(importances)[::-1]
            top_n_indices = indices[:n_features_to_select]

            # Select the corresponding feature names and filter data
            selected_final_feature_names = [
                final_feature_names[i] for i in top_n_indices
            ]
            X_train_final_np = X_train_final_np[:, top_n_indices]
            X_test_final_np = X_test_final_np[:, top_n_indices]

            # Update final feature names and DataFrames
            final_feature_names = selected_final_feature_names
            X_train_final_df = pd.DataFrame(
                X_train_final_np, columns=final_feature_names, index=X_train_raw.index
            )
            X_test_final_df = pd.DataFrame(
                X_test_final_np, columns=final_feature_names, index=X_test_raw.index
            )

            logger.info(
                f"Selected top {len(final_feature_names)} features based on importance: {final_feature_names[:10]}..."
            )  # Log first 10

            # Save selected final features if needed
            if model_dir:
                fs_final_path = os.path.join(
                    model_dir, "selected_features_final_importance.json"
                )
                try:
                    with open(fs_final_path, "w") as f:
                        json.dump(final_feature_names, f, indent=4)
                except Exception as e:
                    logger.error(
                        f"Failed to save final selected features (importance) to {fs_final_path}: {e}"
                    )

        except Exception as e:
            logger.error(
                f"Error during importance-based feature selection: {e}. Proceeding with all {num_features_before_importance} preprocessed features.",
                exc_info=True,
            )
            # Reset to use all features if selection failed
            # Note: X_train_final_np, X_test_final_np, final_feature_names etc. would still hold the values from *before* this block failed.

    # --- Final Model Training ---
    model = None
    best_params = model_params
    try:
        fit_params = {}  # No explicit categorical handling needed for numpy input

        if optimize_hyperparams:
            logger.info(
                f"Optimizing hyperparameters for {model_type} using {opt_method} method"
            )
            # --- Add checks before optimization ---
            logger.debug(
                f"Data shape going into optimization: X={X_train_final_np.shape}, y={y_train.shape}"
            )
            if np.isnan(X_train_final_np).any():
                logger.warning("NaNs detected in X_train_final_np before optimization!")
            if np.isinf(X_train_final_np).any():
                logger.warning("Infs detected in X_train_final_np before optimization!")
            if y_train.isnull().any():
                logger.warning("NaNs detected in y_train before optimization!")
            # Check for sufficient class representation
            unique_classes, counts = np.unique(y_train, return_counts=True)
            logger.debug(
                f"Unique y_train values before optimization: {dict(zip(unique_classes, counts))}"
            )
            # Decide whether to run optimization based on checks
            run_optimization = True
            if len(unique_classes) < 2:
                logger.error(
                    "Only one class present in y_train before optimization! Skipping hyperparameter optimization."
                )
                run_optimization = False
            elif np.any(
                counts < opt_cv
            ):  # Check if any class count is less than CV folds
                logger.warning(
                    f"Some classes in y_train have fewer samples ({counts}) than CV folds ({opt_cv}). Skipping hyperparameter optimization due to potential CV errors."
                )
                run_optimization = False
            # --- End checks ---

            if run_optimization:
                logger.info("Proceeding with hyperparameter optimization...")
                # Pass final numpy array to optimization function
                model, best_params = optimize_hyperparameters(
                    model_type=model_type,
                    X_train=X_train_final_np,
                    y_train=y_train,  # Pass numpy array
                    method=opt_method,
                    params=opt_params,
                    cv=opt_cv,
                    scoring=opt_scoring,
                    verbose=1,
                    # Removed unexpected fit_params argument
                )
            else:
                # Optimization skipped due to data checks
                model = None  # Ensure model is None if optimization is skipped
                best_params = {}  # Indicate optimization was skipped/failed
            # Check if optimization actually returned parameters
            if best_params:
                logger.info(
                    f"Hyperparameter optimization complete. Best parameters found: {best_params}"
                )
                # Create the final model using the best parameters found
                logger.info(
                    f"Creating final {model_type} model with optimized parameters..."
                )
                model = create_model(model_type, best_params)
                if model is None:
                    raise ValueError(
                        f"Failed to create optimized model {model_type} with params {best_params}"
                    )
                # Fit the final optimized model
                logger.info(
                    f"Fitting final optimized model on data with shape: {X_train_final_np.shape}"
                )
                model.fit(X_train_final_np, y_train, **fit_params)  # Pass numpy array
                logger.info("Fitting final optimized model complete.")
            else:
                # Optimization was enabled but failed to find parameters (returned empty dict)
                logger.warning(
                    "Optimization ran but failed to find best parameters. Falling back to default parameters."
                )
                model = create_model(model_type, model_params)  # Use defaults
                if model is None:
                    raise ValueError(
                        f"Failed to create model {model_type} with default parameters after optimization failure."
                    )
                logger.info(
                    f"Fitting model with default parameters on data with shape: {X_train_final_np.shape}"
                )
                model.fit(X_train_final_np, y_train, **fit_params)
                best_params = model_params  # Ensure best_params reflects defaults used

            # Save the parameters that were actually used (either optimized or default)
            if model_dir:
                params_path = os.path.join(model_dir, "best_params.json")
                try:
                    with open(params_path, "w") as f:
                        # Save the potentially updated best_params (could be defaults if opt failed)
                        json.dump(best_params, f, indent=4, default=str)
                    logger.info(
                        f"Saved parameters used for final model to {params_path}"
                    )
                except Exception as e:
                    logger.error(f"Failed to save final parameters: {e}")

        else:  # optimize_hyperparams is False
            logger.info(
                f"Training {model_type} with provided/default parameters (optimization disabled)"
            )
            model = create_model(model_type, model_params)
            if model is None:
                raise ValueError(
                    f"Failed to create model {model_type} (optimization disabled)"
                )
            # Fit model on final numpy array
            logger.info(f"Fitting model on data with shape: {X_train_final_np.shape}")
            model.fit(X_train_final_np, y_train, **fit_params)  # Pass numpy array
            best_params = model_params  # Ensure best_params reflects defaults used

        # Final check: If after all attempts model is still None, something went wrong.
        if model is None:
            raise ValueError("Model is None after training attempts. Critical failure.")
        logger.info("Model training finished.")

        # --- Evaluation ---
        logger.info("Evaluating model performance...")
        # Evaluate on final DataFrames
        train_metrics = evaluate_classifier(
            model, X_train_final_df, y_train, class_names=class_names
        )
        test_metrics = evaluate_classifier(
            model, X_test_final_df, y_test, class_names=class_names
        )
        metrics = {"train": train_metrics, "test": test_metrics}
        acc_metric = test_metrics.get("accuracy", "N/A")
        f1_metric = test_metrics.get("f1_weighted", "N/A")
        acc_str = (
            f"{acc_metric:.4f}"
            if isinstance(acc_metric, (int, float))
            else str(acc_metric)
        )
        f1_str = (
            f"{f1_metric:.4f}"
            if isinstance(f1_metric, (int, float))
            else str(f1_metric)
        )
        logger.info(f"Test Set Metrics: Accuracy={acc_str}, F1-Weighted={f1_str}")

        # --- Feature Importance ---
        logger.info("Calculating feature importance...")
        # Calculate importance on final DataFrame
        importance_df = compute_feature_importance(
            model, X_train_final_df, y_train, method="built_in"
        )
        if importance_df.empty:
            logger.warning(
                "Built-in feature importance failed or not available, trying permutation."
            )
            importance_df = compute_feature_importance(
                model, X_train_final_df, y_train, method="permutation"
            )

        # Log top features only if importance_df is valid
        if not importance_df.empty and "Feature" in importance_df.columns:
            logger.info(
                f"Feature importance calculated. Top 5: {importance_df['Feature'].head().tolist()}"
            )
        elif importance_df.empty:
            logger.warning(
                "Feature importance could not be computed (empty DataFrame)."
            )
        else:  # Importance df exists but missing 'Feature' column (shouldn't happen with current metrics.py)
            logger.warning("Feature importance DataFrame is missing 'Feature' column.")

        # --- Saving ---
        if model_dir:
            model_path = os.path.join(model_dir, "model.pkl")
            try:
                joblib.dump(model, model_path)
                logger.info(f"Attempted to save trained model to {model_path}")
                if not os.path.exists(model_path):
                    logger.error(
                        f"Model file was NOT created at {model_path} despite no save error."
                    )
                else:
                    logger.info(
                        f"Successfully verified model file exists at {model_path}"
                    )
            except Exception as e:
                logger.error(f"Failed to save model due to exception: {e}")

            final_config = {
                "model_id": model_id,
                "model_type": model_type,
                "model_params_used": best_params,
                "feature_selection_config": feature_selection_config,
                "preprocessing_config": preprocessing_config,
                "optimization_config": (
                    optimization_config if optimize_hyperparams else None
                ),
                "selected_features_raw": selected_feature_names_raw,  # Save raw selected names
                "processed_feature_names_final": final_feature_names,  # Save final names after processing
                "test_size": test_size,
                "random_state": random_state,
                "training_timestamp": datetime.now().isoformat(),
                "num_original_features": features.shape[1],
                "num_selected_raw_features": len(
                    selected_feature_names_raw
                ),  # Features selected before preprocessing (if any)
                "num_features_after_preprocessing": num_features_before_importance,  # Features entering importance selection/final training
                "num_processed_features_final": len(
                    final_feature_names
                ),  # Features used in the final model
                "num_train_samples": X_train_final_np.shape[0],
                "num_test_samples": X_test_final_np.shape[0],
                "target_distribution_train": y_train.value_counts().to_dict(),
                "target_distribution_test": y_test.value_counts().to_dict(),
                "final_test_metrics": test_metrics,
            }
            config_path = os.path.join(model_dir, "training_summary.json")
            try:
                with open(config_path, "w") as f:
                    json.dump(final_config, f, indent=4, default=str)
                logger.info(f"Saved training summary to {config_path}")
            except Exception as e:
                logger.error(f"Failed to save training summary: {e}")

            if not importance_df.empty:
                imp_path = os.path.join(model_dir, "feature_importance.csv")
                try:
                    importance_df.to_csv(imp_path, index=False)
                    logger.info(f"Saved feature importance to {imp_path}")
                except Exception as e:
                    logger.error(f"Failed to save feature importance: {e}")

        # Return model, preprocessor, metrics, and the final feature names used by the model
        return model, preprocessor, metrics, final_feature_names

    except Exception as e:
        logger.exception(f"Critical error during model training pipeline: {e}")
        # Ensure this returns immediately
        return None, None, {"error": f"Training pipeline failed: {e}"}, []
