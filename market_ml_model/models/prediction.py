import json
import logging
import os
import time
import warnings
from datetime import datetime, timedelta  # Added timedelta
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Try importing scikit-learn
try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.exceptions import NotFittedError
    # Removed Pipeline, StandardScaler, label_binarize

    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not installed. Some functionality will be limited.")
    SKLEARN_AVAILABLE = False

# Try importing SHAP for prediction explanation
try:
    import shap

    # Need to handle different explainer types based on model
    # Removed KernelExplainer, TreeExplainer from specific import

    SHAP_AVAILABLE = True
except ImportError:
    logger.warning("SHAP not installed. Prediction explanations will be limited.")
    SHAP_AVAILABLE = False

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt
    # Removed seaborn import

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

                # Remove original transform call

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
        Predicted class indices (or labels if model.classes_ was used)
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
        # Predict positive class if prob >= threshold, else predict negative class (index 0)
        return np.where(
            probabilities[:, positive_class_index] >= threshold,
            positive_class_index,
            1 - positive_class_index,
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
            if (
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

        # Ensure columns match processed feature names if available
        if (
            self.processed_feature_names
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

            # Ensure columns match processed feature names if available
            if (
                self.processed_feature_names
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
            shap_values = self.explainer.shap_values(features_sample)
            duration = time.time() - start_time
            logger.info(f"SHAP calculation took {duration:.2f} seconds.")

            # Convert to dictionary for easier serialization
            # Handle different SHAP value structures (multi-class vs binary)
            shap_values_list = None
            if isinstance(shap_values, list):  # Multi-class case (TreeExplainer)
                shap_values_list = [arr.tolist() for arr in shap_values]
                base_values = self.explainer.expected_value  # List for multi-class
            elif isinstance(
                shap_values, np.ndarray
            ):  # Binary case (TreeExplainer) or KernelExplainer
                shap_values_list = shap_values.tolist()
                base_values = self.explainer.expected_value
            else:  # Newer SHAP object structure
                shap_values_list = (
                    shap_values.values.tolist()
                    if hasattr(shap_values, "values")
                    else None
                )
                base_values = (
                    shap_values.base_values.tolist()
                    if hasattr(shap_values, "base_values")
                    else None
                )

            shap_dict = {
                "base_values": base_values,
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

            # Ensure columns match processed feature names if available
            if (
                self.processed_feature_names
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
            # Use shap_values object directly if possible (newer SHAP versions)
            try:
                shap_object = self.explainer(features_sample)  # Newer interface
                shap_values_for_plot = shap_object.values
                # Handle multi-class output if necessary
                if isinstance(shap_values_for_plot, list):
                    shap_values_for_plot = shap_values_for_plot[
                        1
                    ]  # Default to class 1 for summary plot
            except TypeError:  # Fallback for older interface
                shap_values_for_plot = self.explainer.shap_values(features_sample)
                if isinstance(shap_values_for_plot, list):
                    shap_values_for_plot = shap_values_for_plot[1]  # Default to class 1

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


class PredictionManager:
    """Class to manage multiple predictors and ensemble their predictions."""

    def __init__(self):
        """Initialize the prediction manager."""
        self.predictors: Dict[str, ModelPredictorBase] = {}
        self.ensemble_weights: Dict[str, float] = {}
        self.class_names: Optional[List[str]] = None  # Store common class names

    def add_predictor(self, name: str, model_path: str, weight: float = 1.0) -> bool:
        """
        Add a predictor to the manager by loading its model.

        Args:
            name: A unique name for this predictor
            model_path: Path to the model directory or .pkl file
            weight: Weight for this predictor in the ensemble (default: 1.0)

        Returns:
            True if successful, False otherwise
        """
        if name in self.predictors:
            logger.warning(f"Predictor with name '{name}' already exists. Overwriting.")

        try:
            predictor = ModelPredictorBase(model_path)
            if predictor.model is None:
                return False  # Loading failed

            # Check for class name consistency
            if self.class_names is None:
                self.class_names = predictor.class_names
            elif (
                predictor.class_names is not None
                and self.class_names != predictor.class_names
            ):
                logger.error(
                    f"Class name mismatch for predictor '{name}'. Expected {self.class_names}, got {predictor.class_names}."
                )
                return False

            self.predictors[name] = predictor
            self.ensemble_weights[name] = weight
            logger.info(
                f"Added predictor '{name}' from {model_path} with weight {weight}."
            )
            return True

        except Exception as e:
            logger.error(f"Failed to add predictor '{name}': {e}")
            return False

    def remove_predictor(self, name: str) -> bool:
        """
        Remove a predictor from the manager.

        Args:
            name: Name of the predictor to remove

        Returns:
            True if successful, False otherwise
        """
        if name in self.predictors:
            del self.predictors[name]
            del self.ensemble_weights[name]
            logger.info(f"Removed predictor '{name}'.")
            # Reset class names if last predictor removed
            if not self.predictors:
                self.class_names = None
            return True
        else:
            logger.warning(f"Predictor '{name}' not found.")
            return False

    def set_weight(self, name: str, weight: float) -> bool:
        """
        Set the ensemble weight for a predictor.

        Args:
            name: Name of the predictor
            weight: New weight

        Returns:
            True if successful, False otherwise
        """
        if name in self.predictors:
            self.ensemble_weights[name] = weight
            logger.info(f"Set weight for predictor '{name}' to {weight}.")
            return True
        else:
            logger.warning(f"Predictor '{name}' not found.")
            return False

    def predict_proba(
        self,
        features: pd.DataFrame,
        ensemble_method: str = "average",  # 'average', 'weighted_average'
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Generate ensemble class probabilities.

        Args:
            features: Input Feature DataFrame
            ensemble_method: Method for combining probabilities

        Returns:
            Tuple of (ensembled probabilities array, class_names list) or (None, [])
        """
        if not self.predictors:
            logger.error("No predictors added to the manager.")
            return None, []

        all_probs = []
        final_class_names = self.class_names or []

        for name, predictor in self.predictors.items():
            probs, names = predictor.predict_proba(features)
            if probs is None:
                logger.warning(
                    f"Predictor '{name}' failed to generate probabilities. Skipping."
                )
                continue
            # Ensure class name consistency was checked during add_predictor
            if not final_class_names:
                final_class_names = names  # Set if first predictor
            all_probs.append(probs * self.ensemble_weights[name])  # Apply weight here

        if not all_probs:
            logger.error("All predictors failed.")
            return None, []

        # Combine probabilities
        if ensemble_method == "average":
            ensembled_probs = np.mean(all_probs, axis=0)
        elif ensemble_method == "weighted_average":
            total_weight = sum(
                self.ensemble_weights[name]
                for name, predictor in self.predictors.items()
                if name in self.ensemble_weights
            )  # Recalculate in case some failed
            if total_weight == 0:
                logger.warning("Total weight is zero, using simple average.")
                ensembled_probs = np.mean(all_probs, axis=0)
            else:
                ensembled_probs = np.sum(all_probs, axis=0) / total_weight
        else:
            logger.error(f"Unsupported ensemble method: {ensemble_method}")
            return None, []

        return ensembled_probs, final_class_names

    def predict(
        self,
        features: pd.DataFrame,
        ensemble_method: str = "average",
        threshold: float = 0.5,
        positive_class_index: int = 1,
    ) -> Optional[np.ndarray]:
        """
        Generate ensemble class predictions.

        Args:
            features: Input Feature DataFrame
            ensemble_method: Method for combining probabilities
            threshold: Decision threshold for binary classification
            positive_class_index: Index of the positive class for binary thresholding

        Returns:
            Predicted classes array, or None if error
        """
        ensembled_probs, _ = self.predict_proba(features, ensemble_method)
        if ensembled_probs is None:
            return None

        return predict_with_threshold(ensembled_probs, threshold, positive_class_index)

    def predict_with_confidence(
        self, features: pd.DataFrame, ensemble_method: str = "average"
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generate ensemble predictions and associated confidence levels.

        Args:
            features: Input Feature DataFrame
            ensemble_method: Method for combining probabilities

        Returns:
            Tuple of (predictions array, confidence array) or (None, None)
        """
        ensembled_probs, _ = self.predict_proba(features, ensemble_method)
        if ensembled_probs is None:
            return None, None

        predictions = predict_with_threshold(
            ensembled_probs
        )  # Use default threshold for prediction
        confidence = get_confidence_levels(ensembled_probs)
        return predictions, confidence


# --- Probability Calibration ---


def calibrate_probabilities(
    model: Any,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    method: str = "isotonic",  # 'isotonic' or 'sigmoid'
) -> Optional[Any]:
    """
    Calibrate model probabilities using a calibration set.

    Args:
        model: Trained (uncalibrated) model
        X_calib: Calibration feature data
        y_calib: Calibration target data
        method: Calibration method ('isotonic' or 'sigmoid')

    Returns:
        Calibrated model object, or None if error
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn needed for probability calibration.")
        return None

    if not hasattr(model, "predict_proba"):
        logger.error("Model does not support predict_proba, cannot calibrate.")
        return None

    try:
        # Check if model is already calibrated (has CalibratedClassifierCV structure)
        if isinstance(model, CalibratedClassifierCV):
            logger.warning("Model appears to be already calibrated.")
            return model

        # Wrap the base estimator if it's not a classifier itself (e.g., pipeline)
        # This requires a dummy classifier that just returns probabilities
        # Note: This approach might not work perfectly for all pipeline structures.
        # A better approach is to calibrate the final estimator within the pipeline if possible.
        # For simplicity here, we assume 'model' is the final classifier or can be wrapped.

        # Create CalibratedClassifierCV
        calibrated_model = CalibratedClassifierCV(
            estimator=model,  # Use the provided model directly
            method=method,
            cv="prefit",  # Use the already trained model
        )

        logger.info(f"Fitting calibration model using '{method}' method...")
        # Fit on the calibration set
        # Note: The base estimator (model) is NOT retrained.
        calibrated_model.fit(X_calib, y_calib)
        logger.info("Calibration complete.")

        return calibrated_model

    except Exception as e:
        logger.exception(f"Error during probability calibration: {e}")
        return None


# --- Prediction Validation ---


def validate_model_predictions(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    preprocessor: Optional[Any] = None,
    selected_features: Optional[List[str]] = None,
    metrics_to_check: List[str] = ["accuracy", "f1_weighted", "roc_auc"],
) -> Dict[str, Any]:
    """
    Validate model performance on a validation set.

    Args:
        model: Trained model object
        X_val: Validation feature data
        y_val: Validation target data
        preprocessor: Fitted preprocessor object
        selected_features: List of selected feature names
        metrics_to_check: List of metrics to compute and return

    Returns:
        Dictionary containing validation metrics
    """
    validation_results = {}
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn needed for validation.")
        return {"error": "scikit-learn not available"}

    try:
        logger.info("Preprocessing validation data...")
        X_val_processed = preprocess_features(X_val, preprocessor, selected_features)
        if X_val_processed is None:
            return {"error": "Validation data preprocessing failed"}

        logger.info("Generating predictions on validation data...")
        # Predictions y_pred and y_prob are calculated within evaluate_classifier
        logger.info("Calculating validation metrics...")
        from market_ml_model.models.evaluation.metrics import (
            evaluate_classifier,
        )  # Avoid circular import if possible

        # Or recalculate specific metrics here
        full_metrics = evaluate_classifier(
            model, X_val_processed, y_val
        )  # Use existing evaluator

        for metric in metrics_to_check:
            if metric in full_metrics:
                validation_results[metric] = full_metrics[metric]
            elif (
                metric == "roc_auc" and "roc_auc_ovr" in full_metrics
            ):  # Handle multiclass AUC
                validation_results[metric] = full_metrics["roc_auc_ovr"]
            else:
                logger.warning(f"Metric '{metric}' not found in evaluation results.")
                validation_results[metric] = None

        logger.info(f"Validation Results: {validation_results}")
        return validation_results

    except Exception as e:
        logger.exception(f"Error during prediction validation: {e}")
        return {"error": f"Validation failed: {e}"}


# --- Signal Generation ---


class SignalGenerator:
    """Class to generate trading signals from model predictions."""

    def __init__(
        self,
        predictor: ModelPredictorBase,
        signal_type: str = "threshold",  # 'threshold', 'probability_weighted'
        threshold: float = 0.6,  # Threshold for 'threshold' type
        neutral_zone: Tuple[float, float] = (0.45, 0.55),  # Zone for neutral signal
        trend_filter_ma: Optional[int] = 50,  # Moving average period for trend filter
        volatility_filter_atr: Optional[int] = 14,  # ATR period for volatility filter
        volatility_threshold_q: float = 0.75,  # Volatility quantile threshold
        cooling_period: int = 3,  # Bars to wait after a signal
    ):
        """
        Initialize the Signal Generator.

        Args:
            predictor: An instance of ModelPredictorBase
            signal_type: How to convert probabilities to signals
            threshold: Confidence threshold for threshold-based signals
            neutral_zone: Probability range considered neutral
            trend_filter_ma: MA period for trend filter (None to disable)
            volatility_filter_atr: ATR period for volatility filter (None to disable)
            volatility_threshold_q: Quantile threshold for high volatility filter
            cooling_period: Minimum bars between signals
        """
        self.predictor = predictor
        self.signal_type = signal_type
        self.threshold = threshold
        self.neutral_zone = neutral_zone
        self.trend_filter_ma = trend_filter_ma
        self.volatility_filter_atr = volatility_filter_atr
        self.volatility_threshold_q = volatility_threshold_q
        self.cooling_period = cooling_period
        self.last_signal_time = {}  # symbol -> timestamp

        if not predictor or not isinstance(predictor, ModelPredictorBase):
            raise ValueError("Invalid predictor provided.")

        # Assume class order is [-1, 0, 1] or [0, 1] or [Neg, Neu, Pos]
        # Need a reliable way to map probabilities to signals
        self.class_names = predictor.class_names
        if self.class_names:
            try:
                # Attempt to map common class names/indices
                self.neg_idx = (
                    self.class_names.index("-1")
                    if "-1" in self.class_names
                    else (
                        self.class_names.index("0")
                        if "0" in self.class_names and len(self.class_names) == 2
                        else 0
                    )
                )
                self.pos_idx = (
                    self.class_names.index("1") if "1" in self.class_names else 1
                )
                logger.info(
                    f"Signal mapping: Negative Class Index={self.neg_idx}, Positive Class Index={self.pos_idx}"
                )
            except ValueError:
                logger.warning(
                    f"Could not automatically map class names {self.class_names} to -1/1 signals. Assuming binary [0, 1] or multiclass [0, 1, ...]."
                )
                self.neg_idx = 0
                self.pos_idx = 1  # Default assumption
        else:
            logger.warning(
                "Class names not found in predictor metadata. Assuming binary [0, 1] output."
            )
            self.neg_idx = 0
            self.pos_idx = 1

    def generate_signals(
        self,
        features: pd.DataFrame,
        ohlc_data: Optional[pd.DataFrame] = None,  # Needed for filters
    ) -> pd.DataFrame:
        """
        Generate trading signals (-1, 0, 1) based on model probabilities.

        Args:
            features: DataFrame of input features for the model
            ohlc_data: DataFrame with OHLC data for filters (must align with features index)

        Returns:
            DataFrame with 'probability_pos', 'probability_neg', 'signal', 'confidence' columns
        """
        if features.empty:
            return pd.DataFrame(
                columns=["probability_pos", "probability_neg", "signal", "confidence"]
            )

        logger.info(f"Generating signals for {len(features)} samples...")
        probabilities, class_names = self.predictor.predict_proba(features)

        if probabilities is None:
            logger.error("Failed to get probabilities from predictor.")
            return pd.DataFrame(
                columns=["probability_pos", "probability_neg", "signal", "confidence"]
            )

        if probabilities.shape[1] < 2:
            logger.error(
                f"Probabilities have unexpected shape: {probabilities.shape}. Need at least 2 classes."
            )
            return pd.DataFrame(
                columns=["probability_pos", "probability_neg", "signal", "confidence"]
            )

        # Ensure class names match if provided earlier
        if self.class_names and self.class_names != class_names:
            logger.warning(
                f"Class names mismatch during prediction. Expected {self.class_names}, got {class_names}. Re-evaluating indices."
            )
            # Re-attempt index finding
            try:
                self.neg_idx = (
                    class_names.index("-1")
                    if "-1" in class_names
                    else (
                        class_names.index("0")
                        if "0" in class_names and len(class_names) == 2
                        else 0
                    )
                )
                self.pos_idx = class_names.index("1") if "1" in class_names else 1
                logger.info(
                    f"Updated Signal mapping: Neg Idx={self.neg_idx}, Pos Idx={self.pos_idx}"
                )
            except ValueError:
                logger.error(
                    f"Could not map new class names {class_names}. Cannot generate signals."
                )
                return pd.DataFrame(
                    columns=[
                        "probability_pos",
                        "probability_neg",
                        "signal",
                        "confidence",
                    ]
                )

        prob_pos = probabilities[:, self.pos_idx]
        # Handle case where negative class might not exist explicitly (e.g., binary [0, 1])
        prob_neg = (
            probabilities[:, self.neg_idx]
            if self.neg_idx < probabilities.shape[1]
            else (1 - prob_pos if probabilities.shape[1] == 2 else 0)
        )

        signals = pd.DataFrame(index=features.index)
        signals["probability_pos"] = prob_pos
        signals["probability_neg"] = prob_neg
        signals["confidence"] = np.max(probabilities, axis=1)
        signals["raw_signal"] = 0  # Initialize

        # --- Apply Signal Generation Logic ---
        if self.signal_type == "threshold":
            signals.loc[prob_pos >= self.threshold, "raw_signal"] = 1
            signals.loc[
                prob_neg >= self.threshold, "raw_signal"
            ] = -1  # Assumes neg prob available
            # Refine for binary case where only pos prob matters vs threshold
            if probabilities.shape[1] == 2:
                signals["raw_signal"] = 0
                signals.loc[prob_pos >= self.threshold, "raw_signal"] = 1
                signals.loc[
                    prob_pos <= (1 - self.threshold), "raw_signal"
                ] = -1  # Signal short if prob_pos is low

        elif self.signal_type == "probability_weighted":
            # Example: Signal = 1 if prob_pos > 0.55, -1 if prob_pos < 0.45
            signals.loc[prob_pos > self.neutral_zone[1], "raw_signal"] = 1
            signals.loc[prob_pos < self.neutral_zone[0], "raw_signal"] = -1
        else:
            logger.warning(
                f"Unsupported signal_type: {self.signal_type}. Using default threshold."
            )
            signals.loc[prob_pos >= self.threshold, "raw_signal"] = 1
            signals.loc[prob_neg >= self.threshold, "raw_signal"] = -1

        # --- Apply Filters ---
        signals["filtered_signal"] = signals["raw_signal"]  # Start with raw signal

        if ohlc_data is not None and not ohlc_data.empty:
            # Align OHLC data index with signals index
            ohlc_data_aligned = ohlc_data.reindex(signals.index)

            # Apply Trend Filter
            if self.trend_filter_ma is not None:
                signals["filtered_signal"] = self._apply_trend_filter(
                    signals, ohlc_data_aligned
                )

            # Apply Volatility Filter
            if self.volatility_filter_atr is not None:
                signals["filtered_signal"] = self._apply_volatility_filter(
                    signals, ohlc_data_aligned
                )
        else:
            if (
                self.trend_filter_ma is not None
                or self.volatility_filter_atr is not None
            ):
                logger.warning(
                    "OHLC data not provided, cannot apply trend/volatility filters."
                )

        # Apply Cooling Period (applied per symbol if multi-symbol features)
        # This requires state and is harder to vectorize simply.
        # We'll apply it conceptually here, assuming single symbol for now.
        signals["signal"] = signals["filtered_signal"]
        self._apply_cooling_period(signals)  # Modifies 'signal' column in place

        logger.info("Signal generation complete.")
        return signals[["probability_pos", "probability_neg", "signal", "confidence"]]

    def _apply_trend_filter(
        self, signals: pd.DataFrame, ohlc: pd.DataFrame
    ) -> pd.Series:
        """Apply trend filter (e.g., only long above MA)."""
        if "close" not in ohlc.columns:
            logger.warning("Trend filter requires 'close' column in OHLC data.")
            return signals["filtered_signal"]

        try:
            ma = ohlc["close"].rolling(window=self.trend_filter_ma).mean()
            is_uptrend = ohlc["close"] > ma
            is_downtrend = ohlc["close"] < ma

            filtered = signals["filtered_signal"].copy()
            # Allow longs only in uptrend, shorts only in downtrend
            filtered.loc[(signals["filtered_signal"] == 1) & (~is_uptrend)] = 0
            filtered.loc[(signals["filtered_signal"] == -1) & (~is_downtrend)] = 0
            logger.info(f"Applied trend filter (MA {self.trend_filter_ma}).")
            return filtered
        except Exception as e:
            logger.error(f"Error applying trend filter: {e}")
            return signals["filtered_signal"]

    def _apply_volatility_filter(
        self, signals: pd.DataFrame, ohlc: pd.DataFrame
    ) -> pd.Series:
        """Apply volatility filter (e.g., avoid trading in high volatility)."""
        req_cols = ["high", "low", "close"]
        if not all(c in ohlc.columns for c in req_cols):
            logger.warning(
                f"Volatility filter requires {req_cols} columns in OHLC data."
            )
            return signals["filtered_signal"]

        try:
            if not SKLEARN_AVAILABLE:  # Need ATR calculation fallback if TA lib missing
                logger.warning(
                    "Cannot calculate ATR for volatility filter without TA-Lib/pandas-ta."
                )
                return signals["filtered_signal"]

            # Calculate ATR (using pandas_ta if available, basic otherwise)
            try:
                import pandas_ta as ta

                atr = ta.atr(
                    ohlc["high"],
                    ohlc["low"],
                    ohlc["close"],
                    length=self.volatility_filter_atr,
                )
            except ImportError:
                # Basic ATR calculation if pandas_ta not found
                tr1 = pd.DataFrame(ohlc["high"] - ohlc["low"])
                tr2 = pd.DataFrame(abs(ohlc["high"] - ohlc["close"].shift()))
                tr3 = pd.DataFrame(abs(ohlc["low"] - ohlc["close"].shift()))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(self.volatility_filter_atr).mean()

            if atr is None or atr.isna().all():
                logger.warning(
                    "ATR calculation failed, cannot apply volatility filter."
                )
                return signals["filtered_signal"]

            # Define high volatility threshold (e.g., top quartile)
            vol_threshold = atr.quantile(self.volatility_threshold_q)
            is_high_vol = atr > vol_threshold

            filtered = signals["filtered_signal"].copy()
            # Filter out signals during high volatility
            filtered.loc[is_high_vol] = 0
            logger.info(
                f"Applied volatility filter (ATR {self.volatility_filter_atr} > {self.volatility_threshold_q:.0%} quantile)."
            )
            return filtered
        except Exception as e:
            logger.error(f"Error applying volatility filter: {e}")
            return signals["filtered_signal"]

    def _apply_cooling_period(self, signals: pd.DataFrame) -> None:
        """Apply cooling period filter (modifies 'signal' column in place)."""
        last_signal_idx = -self.cooling_period - 1
        for i in range(len(signals)):
            if signals["signal"].iloc[i] != 0:
                if i - last_signal_idx > self.cooling_period:
                    last_signal_idx = i  # Allow signal, update last signal time
                else:
                    signals["signal"].iloc[i] = (
                        0  # Suppress signal due to cooling period
                    )
        logger.info(f"Applied cooling period of {self.cooling_period} bars.")

    def plot_signal_history(
        self,
        signals: pd.DataFrame,
        ohlc_data: pd.DataFrame,
        filename: Optional[str] = None,
    ) -> None:
        """
        Plot the generated signals against the price data.

        Args:
            signals: DataFrame generated by generate_signals
            ohlc_data: DataFrame with OHLC data
            filename: Optional path to save the plot
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization libraries not available.")
            return
        if signals.empty or ohlc_data.empty:
            logger.warning("No signals or OHLC data to plot.")
            return

        try:
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(15, 10),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )

            # Align data
            common_index = signals.index.intersection(ohlc_data.index)
            signals_aligned = signals.loc[common_index]
            ohlc_aligned = ohlc_data.loc[common_index]

            if common_index.empty:
                logger.warning("No common index between signals and OHLC data.")
                return

            # Plot Price
            axes[0].plot(
                ohlc_aligned.index,
                ohlc_aligned["close"],
                label="Close Price",
                color="blue",
                alpha=0.8,
            )
            axes[0].set_ylabel("Price")
            axes[0].set_title("Trading Signals vs Price")
            axes[0].grid(True, alpha=0.3)

            # Plot Signals
            long_signals = signals_aligned[signals_aligned["signal"] == 1]
            short_signals = signals_aligned[signals_aligned["signal"] == -1]

            axes[0].scatter(
                long_signals.index,
                ohlc_aligned.loc[long_signals.index, "close"] * 0.99,
                label="Long Signal",
                marker="^",
                color="green",
                s=100,
                alpha=0.9,
                zorder=5,
            )
            axes[0].scatter(
                short_signals.index,
                ohlc_aligned.loc[short_signals.index, "close"] * 1.01,
                label="Short Signal",
                marker="v",
                color="red",
                s=100,
                alpha=0.9,
                zorder=5,
            )
            axes[0].legend()

            # Plot Probabilities
            axes[1].plot(
                signals_aligned.index,
                signals_aligned["probability_pos"],
                label="Prob(Long)",
                color="green",
                alpha=0.7,
            )
            axes[1].plot(
                signals_aligned.index,
                signals_aligned["probability_neg"],
                label="Prob(Short)",
                color="red",
                alpha=0.7,
            )
            axes[1].axhline(0.5, color="grey", linestyle="--", alpha=0.5)
            if self.signal_type == "threshold":
                axes[1].axhline(
                    self.threshold,
                    color="black",
                    linestyle=":",
                    alpha=0.6,
                    label=f"Threshold ({self.threshold:.2f})",
                )
                axes[1].axhline(
                    1 - self.threshold, color="black", linestyle=":", alpha=0.6
                )
            elif self.signal_type == "probability_weighted":
                axes[1].axhline(
                    self.neutral_zone[0],
                    color="orange",
                    linestyle=":",
                    alpha=0.6,
                    label=f"Neutral Zone ({self.neutral_zone[0]:.2f}-{self.neutral_zone[1]:.2f})",
                )
                axes[1].axhline(
                    self.neutral_zone[1], color="orange", linestyle=":", alpha=0.6
                )

            axes[1].set_ylabel("Probability")
            axes[1].set_xlabel("Date")
            axes[1].set_ylim(0, 1)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()

            if filename:
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                logger.info(f"Signal history plot saved to {filename}")
            else:
                plt.show()

            plt.close(fig)

        except Exception as e:
            logger.exception(f"Error plotting signal history: {e}")
            plt.close()  # Ensure plot is closed


# --- Prediction Scheduling ---


class PredictionScheduler:
    """Class to schedule and run predictions at specified intervals."""

    def __init__(
        self,
        predictor: ModelPredictorBase,
        data_loader: Any,  # Should be DataLoader instance from data.loaders
        feature_engineer: Any,  # Should be engineer_features function
        interval_minutes: int = 60,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the Prediction Scheduler.

        Args:
            predictor: Fitted ModelPredictorBase instance
            data_loader: Instance of DataLoader to fetch new data
            feature_engineer: Function to engineer features
            interval_minutes: Prediction interval in minutes
            output_dir: Directory to save prediction results
        """
        self.predictor = predictor
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.interval = timedelta(minutes=interval_minutes)
        self.output_dir = output_dir
        self.last_prediction_time = None
        self.scheduler_active = False

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def predict_once(
        self,
        ticker: str,
        start_date: Optional[
            str
        ] = None,  # For fetching historical data needed for features
        end_date: Optional[str] = None,  # Typically now
    ) -> Optional[pd.DataFrame]:
        """
        Fetch latest data, engineer features, and make a single prediction.

        Args:
            ticker: Asset ticker symbol
            start_date: Start date for historical data fetch
            end_date: End date for data fetch (defaults to now)

        Returns:
            DataFrame with the latest prediction(s), or None if error
        """
        try:
            logger.info(f"Running single prediction for {ticker}...")
            # Fetch latest data (need enough history for feature calculation)
            # Adjust start/end dates based on feature requirements
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Fetch slightly more data than needed for feature lookbacks
            # This needs knowledge of the feature engineer lookback periods
            # Placeholder: fetch last 200 periods assuming max lookback is ~200
            if start_date is None:
                # Heuristic: go back enough days/hours based on interval
                # This is complex, needs better handling based on actual feature lookbacks
                # Defaulting to a fixed lookback for now (Removed unused variable)
                # This needs the data_loader's interval knowledge
                # start_date = (pd.to_datetime(end_date) - timedelta(days=lookback_periods)).strftime('%Y-%m-%d') # Simplistic daily lookback
                logger.warning(
                    "Start date not provided, fetching recent data. Feature accuracy may be affected."
                )
                # Use data_loader's defaults if possible
                start_date = self.data_loader.config.default_start_date

            raw_data = self.data_loader.load_data(
                ticker=ticker, start_date=start_date, end_date=end_date
            )
            if raw_data is None or raw_data.empty:
                logger.error(f"Failed to load data for {ticker}")
                return None

            # Engineer features
            # Assume feature_engineer function takes the raw data DataFrame
            features = self.feature_engineer(raw_data)  # This might need specific args
            if features is None or features.empty:
                logger.error(f"Failed to engineer features for {ticker}")
                return None

            # Get latest features
            latest_features = features.iloc[[-1]]  # Get last row as DataFrame

            # Predict probabilities
            probabilities, class_names = self.predictor.predict_proba(latest_features)
            if probabilities is None:
                logger.error(f"Failed to generate probabilities for {ticker}")
                return None

            # Create result DataFrame
            result_df = pd.DataFrame(
                probabilities, index=latest_features.index, columns=class_names
            )
            result_df["prediction"] = predict_with_threshold(
                probabilities
            )  # Use default threshold
            result_df["confidence"] = get_confidence_levels(probabilities)

            logger.info(
                f"Prediction for {ticker} at {result_df.index[0]}: {result_df.iloc[0].to_dict()}"
            )

            # Save result if output dir specified
            if self.output_dir:
                self._save_prediction_results(ticker, result_df)

            return result_df

        except Exception as e:
            logger.exception(f"Error during single prediction for {ticker}: {e}")
            return None

    def predict_batch(
        self, tickers: List[str], **kwargs
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """Run predict_once for a batch of tickers."""
        results = {}
        for ticker in tickers:
            results[ticker] = self.predict_once(ticker, **kwargs)
        return results

    def start_scheduler(self):
        """Start the prediction scheduler loop."""
        if not self.scheduler_active:
            logger.info(
                f"Starting prediction scheduler with interval {self.interval}..."
            )
            self.scheduler_active = True
            self._run_scheduler()  # Start the loop
        else:
            logger.warning("Scheduler already active.")

    def stop_scheduler(self):
        """Stop the prediction scheduler."""
        if self.scheduler_active:
            logger.info("Stopping prediction scheduler...")
            self.scheduler_active = False
        else:
            logger.warning("Scheduler not active.")

    def _run_scheduler(self):
        """Internal scheduler loop."""
        while self.scheduler_active:
            now = datetime.now()
            if self.last_prediction_time is None or (
                now - self.last_prediction_time >= self.interval
            ):
                logger.info(f"Scheduler triggered at {now}. Running predictions...")
                # --- Add logic here to get list of tickers to predict ---
                tickers_to_predict = ["AAPL", "GOOG"]  # Example list
                self.predict_batch(tickers_to_predict)
                self.last_prediction_time = now
                logger.info("Prediction cycle complete.")

            # Sleep until next check
            # Calculate sleep time more accurately
            if self.last_prediction_time:
                next_run_time = self.last_prediction_time + self.interval
                sleep_seconds = max(0, (next_run_time - datetime.now()).total_seconds())
            else:
                sleep_seconds = 60  # Check every minute initially

            # Check frequently if stopping soon
            check_interval = min(60, sleep_seconds) if sleep_seconds > 0 else 1

            for _ in range(int(sleep_seconds / check_interval)):
                if not self.scheduler_active:
                    break  # Exit loop if stopped
                time.sleep(check_interval)
            if not self.scheduler_active:
                break  # Exit while loop if stopped

    def _save_prediction_results(self, ticker: str, results: pd.DataFrame):
        """Save prediction results to a file."""
        if not self.output_dir or results.empty:
            return
        try:
            timestamp_str = results.index[0].strftime("%Y%m%d_%H%M%S")
            filename = f"{ticker}_prediction_{timestamp_str}.csv"
            filepath = os.path.join(self.output_dir, filename)
            results.to_csv(filepath)
            logger.info(f"Saved prediction results to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save prediction results for {ticker}: {e}")


# --- High-Level Prediction Function ---


def predict_with_model(
    model_path: str, features: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """
    Convenience function to load a model and generate predictions/probabilities.

    Args:
        model_path: Path to the model directory or .pkl file
        features: Input features DataFrame

    Returns:
        DataFrame containing probabilities, prediction, and confidence, or None
    """
    try:
        predictor = ModelPredictorBase(model_path)
        if predictor.model is None:
            return None

        probabilities, class_names = predictor.predict_proba(features)
        if probabilities is None:
            return None

        # Create result DataFrame
        result_df = pd.DataFrame(
            probabilities, index=features.index, columns=class_names
        )
        result_df["prediction"] = predict_with_threshold(
            probabilities
        )  # Use default threshold
        result_df["confidence"] = get_confidence_levels(probabilities)

        return result_df

    except Exception as e:
        logger.exception(f"Error in predict_with_model: {e}")
        return None
