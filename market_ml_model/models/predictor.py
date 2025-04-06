# market_ml_model/models/predictor.py
import json
import logging
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Try importing scikit-learn
try:
    from sklearn.exceptions import NotFittedError

    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not installed. Some functionality will be limited.")
    SKLEARN_AVAILABLE = False

# Try importing SHAP for prediction explanation
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    logger.warning("SHAP not installed. Prediction explanations will be limited.")
    SHAP_AVAILABLE = False

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt

    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib/Seaborn not installed. Visualization disabled.")
    VISUALIZATION_AVAILABLE = False


# --- Utility Functions ---


def load_model(model_path: str) -> Tuple[Optional[Any], Dict]:
    """
    Load a trained model and associated metadata from disk.

    Args:
        model_path: Path to model directory or .pkl file

    Returns:
        Tuple of (model, metadata) or (None, {}) if loading fails
    """
    model = None
    metadata = {}
    preprocessor = None
    selected_features = None

    try:
        # Check if model_path is a directory
        if os.path.isdir(model_path):
            model_file = os.path.join(model_path, "model.pkl")
            metadata_file = os.path.join(
                model_path, "training_summary.json"
            )  # Use summary file
            preprocessor_file = os.path.join(model_path, "preprocessor.pkl")
            features_file = os.path.join(
                model_path, "selected_features.json"
            )  # Kept for backward compat if summary missing

            if not os.path.exists(model_file):
                logger.error(f"Model file not found: {model_file}")
                return None, {}

            # Load model
            model = joblib.load(model_file)
            logger.info(f"Loaded model from {model_file}")

            # Load metadata from summary if available
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata from {metadata_file}")
                # Extract relevant info from metadata
                selected_features = metadata.get("selected_features")
                # Add other relevant metadata keys if needed
                metadata["processed_feature_names"] = metadata.get(
                    "processed_feature_names"
                )
                metadata["class_names"] = metadata.get("class_names")

            # Load preprocessor if available
            if os.path.exists(preprocessor_file):
                preprocessor = joblib.load(preprocessor_file)
                metadata["preprocessor"] = preprocessor  # Add to metadata dict
                logger.info(f"Loaded preprocessor from {preprocessor_file}")

            # Fallback: Load selected features if summary missing but features file exists
            if selected_features is None and os.path.exists(features_file):
                with open(features_file, "r") as f:
                    selected_features = json.load(f)
                metadata["selected_features"] = selected_features
                logger.info(f"Loaded selected features from {features_file}")

            return model, metadata
        elif os.path.isfile(model_path) and model_path.endswith(".pkl"):
            # File - just load the model, no metadata expected
            model = joblib.load(model_path)
            logger.info(f"Loaded model directly from {model_path}")
            return model, {}
        else:
            logger.error(
                f"Invalid model path: {model_path}. Must be directory or .pkl file."
            )
            return None, {}

    except Exception as e:
        logger.exception(f"Error loading model from {model_path}: {e}")  # Log traceback
        return None, {}


def preprocess_features(
    features: pd.DataFrame,
    preprocessor: Optional[Any] = None,
    selected_features: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Preprocess features using the provided preprocessor and feature selection list.

    Args:
        features: Input Feature DataFrame
        preprocessor: Fitted preprocessor object (e.g., ColumnTransformer)
        selected_features: List of selected feature names expected by the preprocessor

    Returns:
        Preprocessed features DataFrame, or None if error
    """
    if features is None or features.empty:
        logger.error("Input features DataFrame is empty or None.")
        return None

    try:
        features_processed = features.copy()
        available_features = None  # Initialize here

        # Apply feature selection if provided
        if selected_features is not None:
            # Check which selected features are actually in the input DataFrame
            available_features = [
                f for f in selected_features if f in features_processed.columns
            ]
            missing_features = list(set(selected_features) - set(available_features))

            if missing_features:
                logger.warning(
                    f"Missing expected features in input data: {missing_features}. Proceeding with available features."
                )

            if not available_features:
                logger.error("None of the selected features found in the input data.")
                return None

            features_processed = features_processed[available_features]
            logger.info(
                f"Applied feature selection. Using {len(available_features)} features."
            )
        else:
            logger.info("No feature selection list provided. Using all input features.")

        # Apply preprocessor if provided
        if preprocessor is not None and SKLEARN_AVAILABLE:
            logger.info("Applying preprocessor...")
            try:
                # Ensure columns match the order expected by the fitted preprocessor
                # Use the list of features that were *actually* available and selected
                if available_features:  # Use the list identified earlier
                    features_to_transform = features_processed[available_features]
                else:  # Should not happen if selection was done, but as fallback use current df
                    features_to_transform = features_processed

                # Transform the data
                processed_array = preprocessor.transform(
                    features_to_transform
                )  # Pass the correctly column-selected df

                # Get feature names after transformation
                try:
                    feature_names_out = preprocessor.get_feature_names_out()
                except AttributeError:
                    # Fallback for older sklearn or complex pipelines without get_feature_names_out
                    logger.warning(
                        "Could not get feature names from preprocessor. Using generic names."
                    )
                    feature_names_out = [
                        f"feat_{i}" for i in range(processed_array.shape[1])
                    ]

                # Convert back to DataFrame
                features_processed = pd.DataFrame(
                    processed_array,
                    index=features_to_transform.index,  # Use index from the data passed to transform
                    columns=feature_names_out,
                )
                logger.info(
                    f"Preprocessing complete. Output shape: {features_processed.shape}"
                )
                return features_processed

            except ValueError as ve:
                logger.error(
                    f"ValueError applying preprocessor: {ve}. Check if input features match training features."
                )
                return None
            except Exception as e:
                logger.exception(f"Error applying preprocessor: {e}")
                return None
        else:
            logger.info("No preprocessor provided or applied.")
            return features_processed  # Return selected (or original) features if no preprocessor

    except Exception as e:
        logger.exception(f"Error preprocessing features: {e}")
        return None


def predict_proba(
    model: Any,
    features: pd.DataFrame,
    preprocessor: Optional[Any] = None,
    selected_features: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Generate class probabilities with a trained model after preprocessing.
    (Standalone version for use outside ModelPredictorBase)

    Args:
        model: Trained model object
        features: Input Feature DataFrame
        preprocessor: Fitted preprocessor object
        selected_features: List of selected feature names
        class_names: Optional list of class names

    Returns:
        Tuple of (probabilities array, class_names list) or (None, []) if error
    """
    if model is None:
        logger.error("Model object is None.")
        return None, []
    if features is None or features.empty:
        logger.error("Input features DataFrame is empty or None.")
        return None, []

    try:
        # Preprocess features
        features_processed = preprocess_features(
            features, preprocessor, selected_features
        )
        if features_processed is None:
            logger.error("Feature preprocessing failed.")
            return None, []

        # Check if model has predict_proba method
        if not hasattr(model, "predict_proba"):
            logger.error("Model does not have predict_proba method.")
            # Attempt fallback using predict if possible (less ideal)
            if hasattr(model, "predict"):
                logger.warning(
                    "Falling back to model.predict() and generating one-hot probabilities."
                )
                predictions = model.predict(features_processed)
                unique_classes = np.unique(predictions)
                n_classes = len(unique_classes)
                probabilities = np.zeros((len(predictions), n_classes))
                class_map = {cls: i for i, cls in enumerate(unique_classes)}
                for i, pred in enumerate(predictions):
                    probabilities[i, class_map[pred]] = 1.0

                final_class_names = (
                    [str(c) for c in unique_classes]
                    if class_names is None
                    else class_names
                )
                return probabilities, final_class_names
            else:
                logger.error("Model does not have predict method either.")
                return None, []

        # Generate probabilities
        try:
            probabilities = model.predict_proba(features_processed)

            # Determine class names
            final_class_names = class_names
            if final_class_names is None:
                if hasattr(model, "classes_"):
                    final_class_names = [str(c) for c in model.classes_]
                else:
                    # Infer from probability shape
                    final_class_names = [str(i) for i in range(probabilities.shape[1])]

            # Validate shape consistency
            if probabilities.shape[1] != len(final_class_names):
                logger.error(
                    f"Mismatch between probability columns ({probabilities.shape[1]}) and class names ({len(final_class_names)})."
                )
                # Attempt to fix class names based on shape
                final_class_names = [str(i) for i in range(probabilities.shape[1])]

            return probabilities, final_class_names

        except NotFittedError:
            logger.error("Model is not fitted.")
            return None, []
        except ValueError as ve:
            logger.error(
                f"ValueError during predict_proba: {ve}. Check feature consistency."
            )
            return None, []
        except Exception as e:
            logger.exception(f"Error generating probabilities: {e}")
            return None, []

    except Exception as e:
        logger.exception(f"Error in predict_proba pipeline: {e}")
        return None, []


def predict_with_threshold(
    probabilities: np.ndarray,
    threshold: float = 0.5,
    positive_class_index: int = 1,  # Index of the positive class (e.g., 1 for [neg, pos])
) -> np.ndarray:
    """
    Convert probabilities to predictions using a threshold (binary classification).
    For multiclass, returns the class with the highest probability.

    Args:
        probabilities: Class probabilities array (n_samples, n_classes)
        threshold: Decision threshold (only used for binary)
        positive_class_index: Index of the positive class for binary thresholding

    Returns:
        Predicted class indices (or labels if ``model.classes_`` was used)
    """
    if probabilities is None or probabilities.ndim != 2:
        logger.error("Invalid probabilities array provided.")
        return np.array([])

    n_classes = probabilities.shape[1]

    if n_classes == 2:
        # Binary classification: Use threshold on the probability of the positive class
        if positive_class_index >= n_classes:
            logger.error(
                f"positive_class_index ({positive_class_index}) out of bounds for {n_classes} classes."
            )
            # Default to predicting the class with highest probability
            return np.argmax(probabilities, axis=1)
        # Predict positive class if prob >= threshold, else predict negative class (index 0 or other)
        negative_class_index = (
            1 - positive_class_index
        )  # Assumes the other index is the negative one
        return np.where(
            probabilities[:, positive_class_index] >= threshold,
            positive_class_index,
            negative_class_index,
        )
    elif n_classes > 2:
        # Multiclass classification: Return the class with the highest probability
        return np.argmax(probabilities, axis=1)
    elif n_classes == 1:
        # Regression or single output case? Return as is or apply threshold?
        # Assuming classification, threshold might apply if it's probability of a single event
        logger.warning(
            "Probabilities array has only one column. Applying threshold >= 0.5."
        )
        return (probabilities[:, 0] >= 0.5).astype(int)
    else:
        logger.error("Probabilities array has zero columns.")
        return np.array([])


def get_confidence_levels(probabilities: np.ndarray) -> np.ndarray:
    """
    Get confidence levels from probabilities (max probability per prediction).

    Args:
        probabilities: Class probabilities array (n_samples, n_classes)

    Returns:
        Confidence levels for each prediction (n_samples,)
    """
    if probabilities is None or probabilities.ndim != 2 or probabilities.shape[1] == 0:
        logger.error("Invalid probabilities array provided for confidence calculation.")
        return np.array([])
    # For each sample, get the highest probability as confidence
    return np.max(probabilities, axis=1)


# --- Prediction Classes ---


class ModelPredictorBase:
    """Base class for loading a model and making predictions."""

    def __init__(self, model_path: str, output_dir: Optional[str] = None):
        """
        Initialize the model predictor by loading the model and metadata.

        Args:
            model_path: Path to the model directory or .pkl file
            output_dir: Optional directory to save results like explanations
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.model, self.metadata = load_model(model_path)

        if self.model is None:
            raise ValueError(f"Failed to load model from {model_path}")

        # Extract relevant info from metadata
        self.preprocessor = self.metadata.get("preprocessor")
        self.selected_features = self.metadata.get("selected_features")
        # Try getting processed feature names if available (newer training format)
        self.processed_feature_names = self.metadata.get("processed_feature_names")
        self.class_names = self.metadata.get("class_names")
        if self.class_names is None and hasattr(self.model, "classes_"):
            self.class_names = [str(c) for c in self.model.classes_]

        # Check if SHAP explainer can be created
        self.explainer = None
        self._create_explainer()  # Attempt to create explainer on init

    def _create_explainer(self):
        """Helper method to create SHAP explainer."""
        if not SHAP_AVAILABLE:
            logger.info("SHAP library not available, cannot create explainer.")
            return

        if self.model is None:
            logger.warning("Model not loaded, cannot create SHAP explainer.")
            return

        try:
            # Choose explainer based on model type
            model_type_name = type(self.model).__name__.lower()
            # Check for pipeline structure first
            if "pipeline" in model_type_name and hasattr(self.model, "steps"):
                final_estimator = self.model.steps[-1][1]  # Get the final estimator
                final_estimator_type = type(final_estimator).__name__.lower()
                logger.info(
                    f"Pipeline detected. Using final estimator type: {final_estimator_type}"
                )
                if (
                    "xgb" in final_estimator_type
                    or "lgbm" in final_estimator_type
                    or "catboost" in final_estimator_type
                ):
                    # Need to wrap the prediction logic for the pipeline
                    # This is complex as SHAP needs the model's direct prediction on processed data
                    logger.warning(
                        "SHAP TreeExplainer on pipelines requires careful handling. Explainer disabled."
                    )
                    self.explainer = None
                    # Example (might need adjustment):
                    # def pipeline_predict_proba(X):
                    #     return self.model.predict_proba(X)
                    # self.explainer = shap.KernelExplainer(pipeline_predict_proba, background_data) # Needs background
                else:
                    logger.warning(
                        f"SHAP explainer not directly supported for final pipeline step {final_estimator_type}"
                    )
                    self.explainer = None

            elif (
                "xgb" in model_type_name
                or "lgbm" in model_type_name
                or "catboost" in model_type_name
            ):
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("Created SHAP TreeExplainer.")
            elif hasattr(self.model, "predict_proba"):
                # Use KernelExplainer as a fallback - requires background data
                # This might be slow and require careful background data selection
                logger.warning(
                    "Creating SHAP KernelExplainer (may be slow). Background data needed for optimal use."
                )
                # self.explainer = shap.KernelExplainer(self.model.predict_proba, background_data) # Needs background data
                self.explainer = (
                    None  # Disable KernelExplainer by default due to complexity
                )
            else:
                logger.warning(
                    f"SHAP explainer not supported for model type {type(self.model).__name__}"
                )

        except Exception as e:
            logger.warning(f"Could not create SHAP explainer: {e}")
            self.explainer = None

    def predict(self, features: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Generate class predictions.

        Args:
            features: Input Feature DataFrame

        Returns:
            Predicted classes array, or None if error
        """
        if self.model is None:
            logger.error("Model is not loaded.")
            return None

        # Preprocess features
        features_processed = preprocess_features(
            features, self.preprocessor, self.selected_features
        )
        if features_processed is None:
            logger.error("Prediction failed due to preprocessing error.")
            return None

        # Ensure columns match processed feature names if available and preprocessor was used
        if (
            self.preprocessor
            is not None  # Only check if preprocessor was actually used
            and self.processed_feature_names
            and list(features_processed.columns) != self.processed_feature_names
        ):
            logger.warning(
                "Input columns after preprocessing do not match expected processed feature names. Reordering/selecting."
            )
            try:
                features_processed = features_processed[self.processed_feature_names]
            except KeyError as e:
                logger.error(
                    f"Missing expected processed features after preprocessing: {e}"
                )
                return None

        # Generate predictions
        try:
            predictions = self.model.predict(features_processed)
            return predictions
        except ValueError as ve:
            logger.error(
                f"ValueError during prediction: {ve}. Check feature consistency."
            )
            return None
        except Exception as e:
            logger.exception(f"Error generating predictions: {e}")
            return None

    def predict_proba(
        self, features: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Generate class probabilities.

        Args:
            features: Input Feature DataFrame

        Returns:
            Tuple of (probabilities array, class_names list) or (None, []) if error
        """
        if self.model is None:
            logger.error("Model is not loaded.")
            return None, []

        # Use the standalone predict_proba function which handles preprocessing
        return predict_proba(
            self.model,
            features,
            self.preprocessor,
            self.selected_features,
            self.class_names,
        )

    def explain(
        self,
        features: pd.DataFrame,
        max_samples: Optional[int] = 100,  # Explain fewer samples by default
    ) -> Optional[Dict]:
        """
        Explain predictions using SHAP values for a subset of samples.

        Args:
            features: Input Feature DataFrame
            max_samples: Maximum number of samples to explain (None for all)

        Returns:
            Dictionary with SHAP values, base values, feature names, etc. or None
        """
        if self.explainer is None:
            logger.warning("SHAP explainer not available or failed to initialize.")
            return None
        if not SHAP_AVAILABLE:
            logger.warning("SHAP library not available.")
            return None

        try:
            # Preprocess features
            features_processed = preprocess_features(
                features, self.preprocessor, self.selected_features
            )
            if features_processed is None:
                logger.error("Cannot explain predictions due to preprocessing error.")
                return None

            # Ensure columns match processed feature names if available and preprocessor was used
            if (
                self.preprocessor is not None  # Only check if preprocessor was used
                and self.processed_feature_names
                and list(features_processed.columns) != self.processed_feature_names
            ):
                logger.warning(
                    "Input columns for explanation do not match expected processed feature names. Reordering/selecting."
                )
                try:
                    features_processed = features_processed[
                        self.processed_feature_names
                    ]
                except KeyError as e:
                    logger.error(
                        f"Missing expected processed features for explanation: {e}"
                    )
                    return None

            # Limit samples if needed
            if max_samples is not None and len(features_processed) > max_samples:
                logger.info(f"Limiting SHAP explanation to {max_samples} samples")
                # Sample consistently if possible
                features_sample = features_processed.sample(
                    max_samples, random_state=42
                )
            else:
                features_sample = features_processed

            if features_sample.empty:
                logger.warning(
                    "No samples available for SHAP explanation after preprocessing/sampling."
                )
                return None

            # Generate SHAP values
            logger.info(
                f"Calculating SHAP values for {len(features_sample)} samples..."
            )
            start_time = time.time()
            # Use explainer directly if possible (newer SHAP versions)
            try:
                shap_object = self.explainer(features_sample)  # Newer interface
                shap_values = shap_object.values
                base_values = shap_object.base_values
            except TypeError:  # Fallback for older interface
                shap_values = self.explainer.shap_values(features_sample)
                base_values = self.explainer.expected_value

            duration = time.time() - start_time
            logger.info(f"SHAP calculation took {duration:.2f} seconds.")

            # Convert to dictionary for easier serialization
            # Handle different SHAP value structures (multi-class vs binary)
            shap_values_list = None
            base_values_list = None
            if isinstance(shap_values, list):  # Multi-class case (TreeExplainer)
                shap_values_list = [arr.tolist() for arr in shap_values]
                base_values_list = (
                    base_values.tolist()
                    if isinstance(base_values, np.ndarray)
                    else base_values
                )  # Base values might be list or array
            elif isinstance(
                shap_values, np.ndarray
            ):  # Binary case (TreeExplainer) or KernelExplainer
                shap_values_list = shap_values.tolist()
                base_values_list = (
                    base_values.tolist()
                    if isinstance(base_values, np.ndarray)
                    else [base_values]
                )  # Ensure list

            shap_dict = {
                "base_values": base_values_list,
                "values": shap_values_list,
                "feature_names": list(
                    features_sample.columns
                ),  # Use columns from processed sample
                "class_names": self.class_names,
                "num_samples_explained": len(features_sample),
            }

            return shap_dict

        except Exception as e:
            logger.exception(f"Error generating SHAP values: {e}")
            return None

    def plot_shap_summary(
        self,
        features: pd.DataFrame,
        max_features: int = 20,
        max_samples: Optional[int] = 1000,  # Allow more samples for summary plot
        plot_type: str = "dot",  # Default SHAP summary plot type
        filename: Optional[str] = None,
    ) -> None:
        """
        Plot SHAP summary plot.

        Args:
            features: Input Feature DataFrame
            max_features: Maximum number of features to include in the plot
            max_samples: Maximum number of samples to use for calculation (None for all)
            plot_type: Type of SHAP summary plot ('dot', 'bar', etc.)
            filename: If provided, save plot to this file path
        """
        if self.explainer is None or not SHAP_AVAILABLE or not VISUALIZATION_AVAILABLE:
            logger.warning(
                "SHAP explainer or visualization libraries not available, cannot plot summary."
            )
            return

        try:
            # Preprocess features
            features_processed = preprocess_features(
                features, self.preprocessor, self.selected_features
            )
            if features_processed is None:
                return

            # Ensure columns match processed feature names if available and preprocessor was used
            if (
                self.preprocessor is not None  # Only check if preprocessor was used
                and self.processed_feature_names
                and list(features_processed.columns) != self.processed_feature_names
            ):
                logger.warning(
                    "Input columns for SHAP plot do not match expected processed feature names. Reordering/selecting."
                )
                try:
                    features_processed = features_processed[
                        self.processed_feature_names
                    ]
                except KeyError as e:
                    logger.error(
                        f"Missing expected processed features for SHAP plot: {e}"
                    )
                    return

            # Limit samples if needed
            if max_samples is not None and len(features_processed) > max_samples:
                logger.info(
                    f"Limiting SHAP calculation for plot to {max_samples} samples"
                )
                features_sample = features_processed.sample(
                    max_samples, random_state=42
                )
            else:
                features_sample = features_processed

            if features_sample.empty:
                logger.warning(
                    "No samples available for SHAP plot after preprocessing/sampling."
                )
                return

            # Generate SHAP values
            logger.info(
                f"Calculating SHAP values for plot ({len(features_sample)} samples)..."
            )
            start_time = time.time()
            # Use explainer directly if possible (newer SHAP versions)
            try:
                shap_object = self.explainer(features_sample)  # Newer interface
                shap_values_for_plot = shap_object.values
                # Handle multi-class output if necessary
                if isinstance(shap_values_for_plot, list):
                    # Default to class 1 for summary plot if binary/multi-class
                    class_index_to_plot = 1 if len(shap_values_for_plot) > 1 else 0
                    shap_values_for_plot = shap_values_for_plot[class_index_to_plot]
            except TypeError:  # Fallback for older interface
                shap_values_for_plot = self.explainer.shap_values(features_sample)
                if isinstance(shap_values_for_plot, list):
                    class_index_to_plot = 1 if len(shap_values_for_plot) > 1 else 0
                    shap_values_for_plot = shap_values_for_plot[class_index_to_plot]

            duration = time.time() - start_time
            logger.info(f"SHAP calculation for plot took {duration:.2f} seconds.")

            # Create plot
            logger.info(f"Generating SHAP summary plot (type='{plot_type}')...")
            shap.summary_plot(
                shap_values_for_plot,
                features_sample,
                max_display=max_features,
                plot_type=plot_type,
                show=False,  # Prevent immediate display
            )

            # Save or show plot
            fig = plt.gcf()  # Get current figure
            if filename:
                try:
                    # Ensure directory exists
                    output_dir = os.path.dirname(filename)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    fig.savefig(filename, dpi=300, bbox_inches="tight")
                    logger.info(f"SHAP summary plot saved to {filename}")
                except Exception as e:
                    logger.error(f"Failed to save SHAP plot to {filename}: {e}")

            else:
                plt.show()

            plt.close(fig)  # Close the figure

        except Exception as e:
            logger.exception(f"Error plotting SHAP summary: {e}")
            plt.close()  # Ensure plot is closed even on error
