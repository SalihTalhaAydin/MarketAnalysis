import pandas as pd
import numpy as np
import logging
import os
import json
import time
import joblib
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import warnings

# Setup logging
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Try importing scikit-learn
try:
    from sklearn.exceptions import NotFittedError
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
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
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib/Seaborn not installed. Visualization disabled.")
    VISUALIZATION_AVAILABLE = False


# --- Utility Functions ---

def load_model(model_path: str) -> Tuple[Any, Dict]:
    """
    Load a trained model and metadata from disk.
    
    Args:
        model_path: Path to model directory or file
        
    Returns:
        Tuple of (model, metadata)
    """
    try:
        # Check if model_path is a directory or file
        if os.path.isdir(model_path):
            # Directory - load model and metadata
            model_file = os.path.join(model_path, 'model.pkl')
            metadata_file = os.path.join(model_path, 'training_config.json')
            
            if not os.path.exists(model_file):
                logger.error(f"Model file not found: {model_file}")
                return None, {}
            
            # Load model
            model = joblib.load(model_file)
            
            # Load metadata if available
            metadata = {}
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            # Load preprocessor if available
            preprocessor_file = os.path.join(model_path, 'preprocessor.pkl')
            if os.path.exists(preprocessor_file):
                metadata['preprocessor'] = joblib.load(preprocessor_file)
            
            # Load selected features if available
            features_file = os.path.join(model_path, 'selected_features.json')
            if os.path.exists(features_file):
                with open(features_file, 'r') as f:
                    metadata['selected_features'] = json.load(f)
            
            # Load transformed features if available
            transformed_file = os.path.join(model_path, 'transformed_features.json')
            if os.path.exists(transformed_file):
                with open(transformed_file, 'r') as f:
                    metadata['transformed_features'] = json.load(f)
            
            return model, metadata
        else:
            # File - just load the model
            model = joblib.load(model_path)
            return model, {}
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, {}


def preprocess_features(
    features: pd.DataFrame,
    preprocessor: Optional[Any] = None,
    selected_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Preprocess features using the provided preprocessor and feature selection.
    
    Args:
        features: Feature DataFrame
        preprocessor: Preprocessor object
        selected_features: List of selected features
        
    Returns:
        Preprocessed features
    """
    try:
        # Apply feature selection if provided
        if selected_features is not None:
            # Check which selected features are in the input
            available_features = [f for f in selected_features if f in features.columns]
            
            if len(available_features) < len(selected_features):
                missing_features = list(set(selected_features) - set(available_features))
                logger.warning(f"Missing features: {missing_features}")
            
            if not available_features:
                logger.error("No selected features found in input")
                return features
            
            features_selected = features[available_features]
        else:
            features_selected = features
        
        # Apply preprocessor if provided
        if preprocessor is not None and SKLEARN_AVAILABLE:
            try:
                features_processed = preprocessor.transform(features_selected)
                
                # Convert to DataFrame if possible
                if hasattr(preprocessor, 'get_feature_names_out'):
                    # scikit-learn 1.0+ compatibility
                    feature_names = preprocessor.get_feature_names_out()
                    features_processed = pd.DataFrame(
                        features_processed,
                        index=features_selected.index,
                        columns=feature_names
                    )
                else:
                    # Fall back to numpy array
                    features_processed = pd.DataFrame(
                        features_processed,
                        index=features_selected.index
                    )
                
                return features_processed
            
            except Exception as e:
                logger.error(f"Error applying preprocessor: {e}")
                return features_selected
        else:
            return features_selected
    
    except Exception as e:
        logger.error(f"Error preprocessing features: {e}")
        return features


def predict_proba(
    model: Any,
    features: pd.DataFrame,
    preprocessor: Optional[Any] = None,
    selected_features: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate class probabilities with a trained model.
    
    Args:
        model: Trained model
        features: Feature DataFrame
        preprocessor: Preprocessor object
        selected_features: List of selected features
        class_names: Names of classes
        
    Returns:
        Tuple of (probabilities, class_names)
    """
    try:
        # Preprocess features
        features_processed = preprocess_features(
            features, preprocessor, selected_features
        )
        
        # Check if model has predict_proba method
        if not hasattr(model, 'predict_proba'):
            logger.error("Model does not have predict_proba method")
            
            # Fall back to regular predict if available
            if hasattr(model, 'predict'):
                predictions = model.predict(features_processed)
                
                # Convert to one-hot encoding
                unique_classes = np.unique(predictions)
                n_classes = len(unique_classes)
                probabilities = np.zeros((len(predictions), n_classes))
                
                for i, pred in enumerate(predictions):
                    class_idx = np.where(unique_classes == pred)[0][0]
                    probabilities[i, class_idx] = 1.0
                
                # Generate class names if not provided
                if class_names is None:
                    class_names = [str(c) for c in unique_classes]
                
                return probabilities, class_names
            else:
                logger.error("Model does not have predict method")
                return np.array([]), []
        
        # Generate probabilities
        try:
            probabilities = model.predict_proba(features_processed)
            
            # Get class names if not provided
            if class_names is None:
                if hasattr(model, 'classes_'):
                    class_names = [str(c) for c in model.classes_]
                else:
                    class_names = [str(i) for i in range(probabilities.shape[1])]
            
            return probabilities, class_names
        
        except NotFittedError:
            logger.error("Model is not fitted")
            return np.array([]), []
        
        except Exception as e:
            logger.error(f"Error generating probabilities: {e}")
            return np.array([]), []
    
    except Exception as e:
        logger.error(f"Error in predict_proba: {e}")
        return np.array([]), []


def predict_with_threshold(
    probabilities: np.ndarray,
    threshold: float = 0.5,
    positive_class: int = 1
) -> np.ndarray:
    """
    Convert probabilities to predictions using a threshold (binary classification).
    
    Args:
        probabilities: Class probabilities
        threshold: Decision threshold
        positive_class: Index of the positive class
        
    Returns:
        Predicted class indices
    """
    if probabilities.shape[1] == 2:
        # Binary classification
        return (probabilities[:, positive_class] >= threshold).astype(int)
    else:
        # Multiclass classification - use argmax
        return np.argmax(probabilities, axis=1)


def get_confidence_levels(probabilities: np.ndarray) -> np.ndarray:
    """
    Get confidence levels from probabilities.
    
    Args:
        probabilities: Class probabilities
        
    Returns:
        Confidence levels for each prediction
    """
    # For each sample, get the highest probability as confidence
    return np.max(probabilities, axis=1)


# --- Prediction Classes ---

class ModelPredictorBase:
    """Base class for model predictors."""
    
    def __init__(
        self,
        model: Any,
        metadata: Optional[Dict] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the model predictor.
        
        Args:
            model: Trained model
            metadata: Model metadata
            output_dir: Directory to save results
        """
        self.model = model
        self.metadata = metadata or {}
        self.output_dir = output_dir
        
        # Extract preprocessor and features from metadata
        self.preprocessor = self.metadata.get('preprocessor')
        self.selected_features = self.metadata.get('selected_features')
        self.transformed_features = self.metadata.get('transformed_features')
        self.class_names = self.metadata.get('class_names')
        
        # Check if SHAP explainer can be created
        self.explainer = None
        if SHAP_AVAILABLE and hasattr(self.model, 'predict_proba'):
            try:
                self.explainer = shap.Explainer(self.model)
                logger.info("SHAP explainer created successfully")
            except Exception as e:
                logger.warning(f"Could not create SHAP explainer: {e}")
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generate class predictions.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Predicted classes
        """
        # Preprocess features
        features_processed = preprocess_features(
            features, self.preprocessor, self.selected_features
        )
        
        # Generate predictions
        try:
            predictions = self.model.predict(features_processed)
            return predictions
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return np.array([])
    
    def predict_proba(self, features: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Generate class probabilities.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Tuple of (probabilities, class_names)
        """
        return predict_proba(
            self.model, features, self.preprocessor,
            self.selected_features, self.class_names
        )
    
    def explain(
        self,
        features: pd.DataFrame,
        max_samples: int = 100
    ) -> Optional[Dict]:
        """
        Explain predictions using SHAP values.
        
        Args:
            features: Feature DataFrame
            max_samples: Maximum number of samples to explain
            
        Returns:
            Dictionary with SHAP values
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            logger.warning("SHAP explainer not available")
            return None
        
        try:
            # Preprocess features
            features_processed = preprocess_features(
                features, self.preprocessor, self.selected_features
            )
            
            # Limit samples if needed
            if len(features_processed) > max_samples:
                logger.info(f"Limiting SHAP explanation to {max_samples} samples")
                features_sample = features_processed.sample(max_samples)
            else:
                features_sample = features_processed
            
            # Generate SHAP values
            shap_values = self.explainer(features_sample)
            
            # Convert to dictionary for easier serialization
            shap_dict = {
                'base_values': shap_values.base_values.tolist() if hasattr(shap_values, 'base_values') else None,
                'values': shap_values.values.tolist() if hasattr(shap_values, 'values') else None,
                'feature_names': features_sample.columns.tolist() if hasattr(features_sample, 'columns') else None,
                'class_names': self.class_names
            }
            
            return shap_dict
        
        except Exception as e:
            logger.error(f"Error generating SHAP values: {e}")
            return None
    
    def plot_shap_summary(
        self,
        features: pd.DataFrame,
        max_features: int = 20,
        max_samples: int = 100,
        filename: Optional[str] = None
    ) -> None:
        """
        Plot SHAP summary plot.
        
        Args:
            features: Feature DataFrame
            max_features: Maximum number of features to include
            max_samples: Maximum number of samples to explain
            filename: If provided, save plot to this file
        """
        if not SHAP_AVAILABLE or not VISUALIZATION_AVAILABLE or self.explainer is None:
            logger.warning("SHAP and/or visualization libraries not available")
            return
        
        try:
            # Preprocess features
            features_processed = preprocess_features(
                features, self.preprocessor, self.selected_features
            )
            
            # Limit samples if needed
            if len(features_processed) > max_samples:
                logger.info(f"Limiting SHAP explanation to {max_samples} samples")
                features_sample = features_processed.sample(max_samples)
            else:
                features_sample = features_processed
            
            # Generate SHAP values
            shap_values = self.explainer(features_sample)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values, features_sample,
                max_display=max_features,
                show=False
            )
            
            # Save or show plot
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP summary plot saved to {filename}")
            else:
                plt.show()
            
            plt.close()
        
        except Exception as e:
            logger.error(f"Error plotting SHAP summary: {e}")


class PredictionManager:
    """Class to manage multiple predictors and ensemble them."""
    
    def __init__(self):
        """Initialize the prediction manager."""
        self.predictors = {}
        self.ensemble_weights = {}
    
    def add_predictor(
        self,
        name: str,
        model_path: str,
        weight: float = 1.0
    ) -> bool:
        """
        Add a predictor to the manager.
        
        Args:
            name: Predictor name
            model_path: Path to model directory or file
            weight: Ensemble weight
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load model
            model, metadata = load_model(model_path)
            
            if model is None:
                logger.error(f"Failed to load model from {model_path}")
                return False
            
            # Create predictor
            predictor = ModelPredictorBase(model, metadata)
            
            # Add to predictors and weights
            self.predictors[name] = predictor
            self.ensemble_weights[name] = weight
            
            logger.info(f"Added predictor {name} with weight {weight}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding predictor {name}: {e}")
            return False
    
    def remove_predictor(self, name: str) -> bool:
        """
        Remove a predictor from the manager.
        
        Args:
            name: Predictor name
            
        Returns:
            True if successful, False otherwise
        """
        if name in self.predictors:
            del self.predictors[name]
            del self.ensemble_weights[name]
            logger.info(f"Removed predictor {name}")
            return True
        else:
            logger.warning(f"Predictor {name} not found")
            return False
    
    def set_weight(self, name: str, weight: float) -> bool:
        """
        Set the weight of a predictor.
        
        Args:
            name: Predictor name
            weight: Ensemble weight
            
        Returns:
            True if successful, False otherwise
        """
        if name in self.predictors:
            self.ensemble_weights[name] = weight
            logger.info(f"Set weight of predictor {name} to {weight}")
            return True
        else:
            logger.warning(f"Predictor {name} not found")
            return False
    
    def predict_proba(
        self,
        features: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate ensemble probabilities.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Tuple of (probabilities, class_names)
        """
        if not self.predictors:
            logger.error("No predictors available")
            return np.array([]), []
        
        # Generate probabilities for each predictor
        all_probas = {}
        all_classes = {}
        
        for name, predictor in self.predictors.items():
            probas, classes = predictor.predict_proba(features)
            
            if probas.size > 0:
                all_probas[name] = probas
                all_classes[name] = classes
            else:
                logger.warning(f"Predictor {name} returned empty probabilities")
        
        if not all_probas:
            logger.error("All predictors failed")
            return np.array([]), []
        
        # Check if all predictors have the same classes
        # This is a simplification; in reality, you would need to handle
        # predictors with different class sets
        if len(set(tuple(classes) for classes in all_classes.values())) > 1:
            logger.warning("Predictors have different class sets, using first one")
        
        first_classes = list(all_classes.values())[0]
        n_classes = len(first_classes)
        
        # Calculate weighted probabilities
        ensemble_probas = np.zeros((len(features), n_classes))
        total_weight = 0.0
        
        for name, probas in all_probas.items():
            weight = self.ensemble_weights[name]
            ensemble_probas += probas * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            ensemble_probas /= total_weight
        
        return ensemble_probas, first_classes
    
    def predict(
        self,
        features: pd.DataFrame,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Args:
            features: Feature DataFrame
            threshold: Decision threshold for binary classification
            
        Returns:
            Predicted classes
        """
        probas, _ = self.predict_proba(features)
        
        if probas.size > 0:
            return np.argmax(probas, axis=1)
        else:
            return np.array([])
    
    def predict_with_confidence(
        self,
        features: pd.DataFrame,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ensemble predictions with confidence.
        
        Args:
            features: Feature DataFrame
            threshold: Decision threshold for binary classification
            
        Returns:
            Tuple of (predictions, confidence)
        """
        probas, _ = self.predict_proba(features)
        
        if probas.size > 0:
            predictions = np.argmax(probas, axis=1)
            confidence = np.max(probas, axis=1)
            return predictions, confidence
        else:
            return np.array([]), np.array([])


# --- Calibration and Model Validation ---

def calibrate_probabilities(
    probabilities: np.ndarray,
    calibration_type: str = 'isotonic',
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None
) -> np.ndarray:
    """
    Calibrate probability predictions.
    
    Args:
        probabilities: Uncalibrated probabilities
        calibration_type: Type of calibration ('isotonic' or 'sigmoid')
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Calibrated probabilities
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn not available for probability calibration")
        return probabilities
    
    try:
        from sklearn.calibration import CalibratedClassifierCV
        
        # Check if validation data is provided
        if X_val is None or y_val is None:
            logger.warning("Validation data required for calibration")
            return probabilities
        
        # Create a dummy classifier that just returns the input probabilities
        class DummyClassifier:
            def __init__(self, probs):
                self.probs = probs
            
            def predict_proba(self, X):
                # Return the same probabilities for any input
                # Limited to the first len(X) rows
                return self.probs[:len(X)]
        
        # Create classifier with the uncalibrated probabilities
        dummy = DummyClassifier(probabilities)
        
        # Calibrate
        calibrated = CalibratedClassifierCV(
            base_estimator=dummy,
            method=calibration_type,
            cv='prefit'  # Use the already fitted model
        )
        
        # Fit calibration with validation data
        calibrated.fit(X_val, y_val)
        
        # Get calibrated probabilities
        calibrated_probs = calibrated.predict_proba(X_val)
        
        return calibrated_probs
    
    except Exception as e:
        logger.error(f"Error calibrating probabilities: {e}")
        return probabilities


def validate_model_predictions(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    ground_truth: pd.Series,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Validate model predictions against ground truth.
    
    Args:
        predictions: Predicted classes
        probabilities: Class probabilities
        ground_truth: True labels
        class_names: Names of classes
        
    Returns:
        Dictionary with validation metrics
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn not available for prediction validation")
        return {}
    
    try:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report,
            log_loss
        )
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(ground_truth, predictions)
        
        # Check if binary or multiclass
        unique_classes = np.unique(np.concatenate([ground_truth.unique(), np.unique(predictions)]))
        n_classes = len(unique_classes)
        
        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]
        
        # Classification report
        metrics['classification_report'] = classification_report(
            ground_truth, predictions, output_dict=True
        )
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(
            ground_truth, predictions
        ).tolist()
        
        # Precision, recall, F1
        if n_classes == 2:
            # Binary classification
            metrics['precision'] = precision_score(ground_truth, predictions)
            metrics['recall'] = recall_score(ground_truth, predictions)
            metrics['f1'] = f1_score(ground_truth, predictions)
            
            # ROC AUC
            if probabilities.shape[1] >= 2:
                metrics['roc_auc'] = roc_auc_score(ground_truth, probabilities[:, 1])
        else:
            # Multiclass classification
            metrics['precision_macro'] = precision_score(ground_truth, predictions, average='macro')
            metrics['recall_macro'] = recall_score(ground_truth, predictions, average='macro')
            metrics['f1_macro'] = f1_score(ground_truth, predictions, average='macro')
            
            metrics['precision_weighted'] = precision_score(ground_truth, predictions, average='weighted')
            metrics['recall_weighted'] = recall_score(ground_truth, predictions, average='weighted')
            metrics['f1_weighted'] = f1_score(ground_truth, predictions, average='weighted')
            
            # Multiclass ROC AUC
            if probabilities.shape[1] == n_classes:
                try:
                    metrics['roc_auc_ovr'] = roc_auc_score(
                        ground_truth, probabilities, multi_class='ovr'
                    )
                except ValueError:
                    pass
        
        # Log loss
        metrics['log_loss'] = log_loss(ground_truth, probabilities)
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error validating predictions: {e}")
        return {}


# --- Signal Generation ---

class SignalGenerator:
    """Class to generate trading signals from model predictions."""
    
    def __init__(
        self,
        predictor: Union[ModelPredictorBase, PredictionManager],
        signal_config: Optional[Dict] = None
    ):
        """
        Initialize the signal generator.
        
        Args:
            predictor: Model predictor or prediction manager
            signal_config: Signal generation configuration
        """
        self.predictor = predictor
        
        # Default configuration
        default_config = {
            'threshold': 0.65,  # Probability threshold
            'confidence_multiplier': 1.0,  # Position sizing based on confidence
            'min_confidence': 0.5,  # Minimum confidence for any signal
            'trend_filter_enabled': False,  # Use trend filter
            'trend_filter_window': 20,  # Trend filter window size
            'volatility_filter_enabled': False,  # Use volatility filter
            'volatility_filter_window': 20,  # Volatility filter window size
            'volatility_filter_threshold': 2.0,  # Volatility filter threshold
            'signal_cooling_periods': 0,  # Periods to wait after a signal
            'filter_consolidation': 'and',  # How to combine filters ('and' or 'or')
            'signal_mapping': {
                0: 0,  # Neutral/hold
                1: 1,  # Buy/long
                2: -1  # Sell/short
            }
        }
        
        # Update with provided config
        self.config = {**default_config, **(signal_config or {})}
        
        # State tracking
        self.last_signal_time = None
        self.cooling_counter = 0
        
        # Signal history
        self.signal_history = []
    
    def generate_signals(
        self,
        features: pd.DataFrame,
        price_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate trading signals for the given features.
        
        Args:
            features: Feature DataFrame
            price_data: Optional price data for additional filters
            
        Returns:
            DataFrame with signals
        """
        try:
            # Get raw predictions and probabilities
            probas, class_names = self.predictor.predict_proba(features)
            
            if probas.size == 0:
                logger.error("Failed to generate probabilities")
                return pd.DataFrame()
            
            # Create signal DataFrame
            signals = pd.DataFrame(index=features.index)
            
            # Add raw probabilities
            for i, class_name in enumerate(class_names):
                signals[f'prob_{class_name}'] = probas[:, i]
            
            # Add confidence
            signals['confidence'] = np.max(probas, axis=1)
            
            # Apply minimum confidence filter
            min_confidence = self.config['min_confidence']
            signals['high_confidence'] = signals['confidence'] >= min_confidence
            
            # Generate raw signal based on highest probability
            raw_predictions = np.argmax(probas, axis=1)
            signals['raw_prediction'] = raw_predictions
            
            # Apply threshold filter
            threshold = self.config['threshold']
            signals['above_threshold'] = False
            
            for i, class_name in enumerate(class_names):
                if i > 0:  # Skip first class (usually neutral)
                    signals.loc[probas[:, i] >= threshold, 'above_threshold'] = True
            
            # Calculate signal strength based on confidence
            confidence_multiplier = self.config['confidence_multiplier']
            signals['signal_strength'] = signals['confidence'] * confidence_multiplier
            
            # Apply trend filter if enabled and price data available
            if self.config['trend_filter_enabled'] and price_data is not None:
                self._apply_trend_filter(signals, price_data)
            
            # Apply volatility filter if enabled and price data available
            if self.config['volatility_filter_enabled'] and price_data is not None:
                self._apply_volatility_filter(signals, price_data)
            
            # Apply cooling period if enabled
            if self.config['signal_cooling_periods'] > 0:
                self._apply_cooling_period(signals)
            
            # Combine filters based on configuration
            if self.config['filter_consolidation'] == 'and':
                signals['pass_filters'] = signals['high_confidence'] & signals['above_threshold']
                
                if 'trend_filter_pass' in signals.columns:
                    signals['pass_filters'] &= signals['trend_filter_pass']
                
                if 'volatility_filter_pass' in signals.columns:
                    signals['pass_filters'] &= signals['volatility_filter_pass']
                
                if 'cooling_period_pass' in signals.columns:
                    signals['pass_filters'] &= signals['cooling_period_pass']
            else:
                # 'or' consolidation - only need one filter to pass
                # but high_confidence and above_threshold are still required
                signals['pass_filters'] = signals['high_confidence'] & signals['above_threshold']
                
                # Additional filters - any can pass
                additional_pass = False
                
                if 'trend_filter_pass' in signals.columns:
                    additional_pass |= signals['trend_filter_pass']
                
                if 'volatility_filter_pass' in signals.columns:
                    additional_pass |= signals['volatility_filter_pass']
                
                if 'cooling_period_pass' in signals.columns:
                    additional_pass |= signals['cooling_period_pass']
                
                # Only require additional filters if they're enabled
                has_additional_filters = ('trend_filter_pass' in signals.columns or
                                          'volatility_filter_pass' in signals.columns or
                                          'cooling_period_pass' in signals.columns)
                
                if has_additional_filters:
                    signals['pass_filters'] &= additional_pass
            
            # Map predictions to signals
            signal_mapping = self.config['signal_mapping']
            signals['signal'] = 0  # Default to neutral
            
            for pred, sig in signal_mapping.items():
                # Convert pred to int in case it's a string from JSON
                pred_idx = int(pred)
                
                # Set signal for rows that pass filters and have matching prediction
                mask = signals['pass_filters'] & (signals['raw_prediction'] == pred_idx)
                signals.loc[mask, 'signal'] = sig
            
            # Add timestamp for signal tracking
            if isinstance(signals.index, pd.DatetimeIndex):
                signal_time = signals.index[-1]
            else:
                signal_time = datetime.now()
            
            # Update state
            self.last_signal_time = signal_time
            
            # Record non-zero signals in history
            non_zero_signals = signals[signals['signal'] != 0]
            
            for idx, row in non_zero_signals.iterrows():
                self.signal_history.append({
                    'time': idx,
                    'signal': row['signal'],
                    'confidence': row['confidence'],
                    'prediction': row['raw_prediction']
                })
            
            # Limit history length
            max_history = 1000
            if len(self.signal_history) > max_history:
                self.signal_history = self.signal_history[-max_history:]
            
            return signals
        
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return pd.DataFrame()
    
    def _apply_trend_filter(
        self,
        signals: pd.DataFrame,
        price_data: pd.DataFrame
    ) -> None:
        """
        Apply trend filter to signals.
        
        Args:
            signals: Signal DataFrame
            price_data: Price data
        """
        try:
            window = self.config['trend_filter_window']
            
            # Check if required columns exist
            if 'close' not in price_data.columns:
                logger.warning("Close price not available for trend filter")
                signals['trend_filter_pass'] = True
                return
            
            # Calculate simple trend filter (SMA)
            sma = price_data['close'].rolling(window=window).mean()
            trend_up = price_data['close'] > sma
            
            # Align with signals index
            if isinstance(signals.index, pd.DatetimeIndex) and isinstance(trend_up.index, pd.DatetimeIndex):
                trend_up = trend_up.reindex(signals.index, method='ffill')
            
            # Apply filter
            signals['trend_up'] = trend_up
            signals['trend_filter_pass'] = True  # Default to pass
            
            # Only long signals when trend is up, only short signals when trend is down
            for pred, sig in self.config['signal_mapping'].items():
                pred_idx = int(pred)
                
                if sig > 0:  # Long signal
                    # Long signals should respect uptrend
                    mask = (signals['raw_prediction'] == pred_idx) & ~trend_up
                    signals.loc[mask, 'trend_filter_pass'] = False
                
                elif sig < 0:  # Short signal
                    # Short signals should respect downtrend
                    mask = (signals['raw_prediction'] == pred_idx) & trend_up
                    signals.loc[mask, 'trend_filter_pass'] = False
        
        except Exception as e:
            logger.error(f"Error applying trend filter: {e}")
            signals['trend_filter_pass'] = True
    
    def _apply_volatility_filter(
        self,
        signals: pd.DataFrame,
        price_data: pd.DataFrame
    ) -> None:
        """
        Apply volatility filter to signals.
        
        Args:
            signals: Signal DataFrame
            price_data: Price data
        """
        try:
            window = self.config['volatility_filter_window']
            threshold = self.config['volatility_filter_threshold']
            
            # Check if required columns exist
            if 'close' not in price_data.columns:
                logger.warning("Close price not available for volatility filter")
                signals['volatility_filter_pass'] = True
                return
            
            # Calculate price returns
            returns = price_data['close'].pct_change()
            
            # Calculate rolling volatility
            volatility = returns.rolling(window=window).std()
            
            # Calculate current volatility vs historical
            current_volatility = volatility.iloc[-1] if not volatility.empty else 0
            mean_volatility = volatility.mean()
            
            if mean_volatility > 0:
                volatility_ratio = current_volatility / mean_volatility
            else:
                volatility_ratio = 1.0
            
            # Apply filter
            signals['volatility_ratio'] = volatility_ratio
            signals['volatility_filter_pass'] = volatility_ratio <= threshold
        
        except Exception as e:
            logger.error(f"Error applying volatility filter: {e}")
            signals['volatility_filter_pass'] = True
    
    def _apply_cooling_period(self, signals: pd.DataFrame) -> None:
        """
        Apply cooling period to signals.
        
        Args:
            signals: Signal DataFrame
        """
        try:
            cooling_periods = self.config['signal_cooling_periods']
            
            if self.cooling_counter > 0:
                # Still in cooling period
                signals['cooling_period_pass'] = False
                self.cooling_counter -= 1
            else:
                signals['cooling_period_pass'] = True
                
                # Check if we need to start a new cooling period
                if signals['signal'].any():
                    self.cooling_counter = cooling_periods
        
        except Exception as e:
            logger.error(f"Error applying cooling period: {e}")
            signals['cooling_period_pass'] = True
    
    def plot_signal_history(
        self,
        price_data: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (12, 8),
        filename: Optional[str] = None
    ) -> None:
        """
        Plot signal history with price data if available.
        
        Args:
            price_data: Price data
            figsize: Figure size
            filename: If provided, save plot to this file
        """
        if not VISUALIZATION_AVAILABLE or not self.signal_history:
            logger.warning("Visualization not available or no signal history")
            return
        
        try:
            # Create plot
            plt.figure(figsize=figsize)
            
            # Plot price data if available
            if price_data is not None and 'close' in price_data.columns:
                plt.plot(price_data.index, price_data['close'], color='black', label='Price')
                
                # Plot signals
                buy_times = []
                buy_prices = []
                sell_times = []
                sell_prices = []
                
                for signal in self.signal_history:
                    signal_time = signal['time']
                    
                    if signal_time in price_data.index:
                        signal_price = price_data.loc[signal_time, 'close']
                        
                        if signal['signal'] > 0:
                            buy_times.append(signal_time)
                            buy_prices.append(signal_price)
                        elif signal['signal'] < 0:
                            sell_times.append(signal_time)
                            sell_prices.append(signal_price)
                
                if buy_times:
                    plt.scatter(buy_times, buy_prices, color='green', s=100, marker='^', label='Buy Signal')
                
                if sell_times:
                    plt.scatter(sell_times, sell_prices, color='red', s=100, marker='v', label='Sell Signal')
            else:
                # Plot just signal history
                signal_df = pd.DataFrame(self.signal_history)
                
                if 'time' in signal_df.columns and 'signal' in signal_df.columns:
                    plt.scatter(
                        signal_df['time'],
                        signal_df['confidence'],
                        c=signal_df['signal'].apply(lambda x: 'green' if x > 0 else 'red'),
                        s=100,
                        alpha=0.7
                    )
                    plt.ylabel('Confidence')
            
            plt.title('Signal History')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            # Save or show plot
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Signal history plot saved to {filename}")
            else:
                plt.show()
            
            plt.close()
        
        except Exception as e:
            logger.error(f"Error plotting signal history: {e}")


# --- Prediction Scheduler ---

class PredictionScheduler:
    """Class to schedule and run predictions at specified intervals."""
    
    def __init__(
        self,
        predictor: Union[ModelPredictorBase, PredictionManager],
        signal_generator: Optional[SignalGenerator] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the prediction scheduler.
        
        Args:
            predictor: Model predictor or prediction manager
            signal_generator: Signal generator
            output_dir: Directory to save outputs
        """
        self.predictor = predictor
        self.signal_generator = signal_generator
        self.output_dir = output_dir
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # State tracking
        self.last_run_time = None
        self.is_running = False
        self.prediction_history = []
    
    def predict_once(
        self,
        features: pd.DataFrame,
        price_data: Optional[pd.DataFrame] = None,
        save_output: bool = True
    ) -> Dict:
        """
        Run a single prediction cycle.
        
        Args:
            features: Feature DataFrame
            price_data: Optional price data for signal generation
            save_output: Whether to save output files
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Record start time
            start_time = time.time()
            run_timestamp = datetime.now()
            
            # Generate probabilities
            probas, class_names = self.predictor.predict_proba(features)
            
            if probas.size == 0:
                logger.error("Failed to generate probabilities")
                return {'error': "Failed to generate probabilities"}
            
            # Generate signals if signal generator is available
            signals = None
            if self.signal_generator:
                signals = self.signal_generator.generate_signals(features, price_data)
            
            # Calculate confidence
            confidence = np.max(probas, axis=1)
            
            # Create results dictionary
            results = {
                'timestamp': run_timestamp.isoformat(),
                'probabilities': probas.tolist(),
                'class_names': class_names,
                'predictions': np.argmax(probas, axis=1).tolist(),
                'confidence': confidence.tolist(),
                'runtime_seconds': time.time() - start_time
            }
            
            if signals is not None:
                results['signals'] = signals['signal'].tolist()
                
                # Add latest signal
                latest_signal = signals['signal'].iloc[-1] if not signals.empty else 0
                results['latest_signal'] = int(latest_signal)
                
                # Add latest confidence
                latest_confidence = signals['confidence'].iloc[-1] if not signals.empty else 0
                results['latest_confidence'] = float(latest_confidence)
            
            # Save results if requested
            if save_output and self.output_dir:
                self._save_prediction_results(results, signals)
            
            # Update state
            self.last_run_time = run_timestamp
            
            # Add to history
            self.prediction_history.append({
                'timestamp': run_timestamp,
                'latest_prediction': results['predictions'][-1] if results['predictions'] else None,
                'latest_confidence': results['confidence'][-1] if results['confidence'] else None,
                'latest_signal': results.get('latest_signal')
            })
            
            # Limit history length
            max_history = 1000
            if len(self.prediction_history) > max_history:
                self.prediction_history = self.prediction_history[-max_history:]
            
            return results
        
        except Exception as e:
            logger.error(f"Error in predict_once: {e}")
            return {'error': str(e)}
    
    def predict_batch(
        self,
        features_list: List[pd.DataFrame],
        price_data_list: Optional[List[pd.DataFrame]] = None,
        save_output: bool = True
    ) -> List[Dict]:
        """
        Run predictions on a batch of feature sets.
        
        Args:
            features_list: List of feature DataFrames
            price_data_list: Optional list of price data
            save_output: Whether to save output files
            
        Returns:
            List of dictionaries with prediction results
        """
        results = []
        
        # Check if price data is provided
        if price_data_list is None:
            price_data_list = [None] * len(features_list)
        
        # Run predictions for each feature set
        for i, features in enumerate(features_list):
            price_data = price_data_list[i] if i < len(price_data_list) else None
            result = self.predict_once(features, price_data, save_output)
            results.append(result)
        
        return results
    
    def _save_prediction_results(
        self,
        results: Dict,
        signals: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Save prediction results to files.
        
        Args:
            results: Prediction results
            signals: Signal DataFrame
        """
        try:
            # Create timestamped filename
            timestamp_str = results['timestamp'].replace(':', '-').replace('.', '-')
            
            # Save results as JSON
            results_file = os.path.join(self.output_dir, f"prediction_{timestamp_str}.json")
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            # Save signals as CSV if available
            if signals is not None:
                signals_file = os.path.join(self.output_dir, f"signals_{timestamp_str}.csv")
                signals.to_csv(signals_file)
            
            logger.info(f"Saved prediction results to {results_file}")
        
        except Exception as e:
            logger.error(f"Error saving prediction results: {e}")


# --- Main Prediction Function ---

def predict_with_model(
    model: Any,
    features: pd.DataFrame,
    preprocess: bool = False,
    threshold: float = 0.5,
    output_dir: Optional[str] = None,
    model_metadata: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main function to make predictions with a trained model.
    
    Args:
        model: Trained model
        features: Feature DataFrame
        preprocess: Whether to preprocess features
        threshold: Decision threshold for binary classification
        output_dir: Directory to save outputs
        model_metadata: Model metadata
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    try:
        # Create predictor
        predictor = ModelPredictorBase(
            model=model,
            metadata=model_metadata,
            output_dir=output_dir
        )
        
        # Generate probabilities
        probabilities, _ = predictor.predict_proba(features)
        
        if probabilities.size == 0:
            logger.error("Failed to generate probabilities")
            return np.array([]), np.array([])
        
        # Generate predictions
        predictions = np.argmax(probabilities, axis=1)
        
        # Save results if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save probabilities
            np.save(os.path.join(output_dir, 'probabilities.npy'), probabilities)
            
            # Save predictions
            np.save(os.path.join(output_dir, 'predictions.npy'), predictions)
        
        return predictions, probabilities
    
    except Exception as e:
        logger.error(f"Error in predict_with_model: {e}")
        return np.array([]), np.array([])