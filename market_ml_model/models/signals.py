# market_ml_model/models/signals.py
import logging
import os  # Added os import
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import necessary components from other modules if needed
# (predictor might be needed for validation, evaluation for metrics)
from .predictor import SKLEARN_AVAILABLE, preprocess_features

# Import evaluation metrics if validate_model_predictions uses them
try:
    from .evaluation.metrics import evaluate_classifier

    EVALUATION_AVAILABLE = True
except ImportError:
    logging.warning("Evaluation metrics module not found. Validation may be limited.")
    EVALUATION_AVAILABLE = False

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt

    VISUALIZATION_AVAILABLE = True
except ImportError:
    logging.warning("Matplotlib/Seaborn not installed. Visualization unavailable.")
    VISUALIZATION_AVAILABLE = False
    plt = None

# Try importing calibration tools
try:
    from sklearn.calibration import CalibratedClassifierCV

    CALIBRATION_AVAILABLE = SKLEARN_AVAILABLE
except ImportError:
    CALIBRATION_AVAILABLE = False

# Try importing pandas_ta for filters
try:
    import pandas_ta as ta

    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    ta = None


logger = logging.getLogger(__name__)


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
    if not CALIBRATION_AVAILABLE:
        logger.error(
            "scikit-learn CalibratedClassifierCV needed for probability calibration."
        )
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
        # Note: This requires careful handling, assuming 'model' is the final estimator for now.

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
    if not EVALUATION_AVAILABLE:
        logger.error("Evaluation module needed for validation.")
        return {"error": "Evaluation module not available"}

    try:
        logger.info("Preprocessing validation data...")
        X_val_processed = preprocess_features(X_val, preprocessor, selected_features)
        if X_val_processed is None:
            return {"error": "Validation data preprocessing failed"}

        logger.info("Generating predictions on validation data...")
        # Predictions y_pred and y_prob are calculated within evaluate_classifier
        logger.info("Calculating validation metrics...")

        # Calculate metrics using the evaluation function
        full_metrics = evaluate_classifier(model, X_val_processed, y_val)

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
        # Removed predictor dependency, pass config directly
        # predictor: ModelPredictorBase,
        model_config: Optional[Dict] = None,  # Accept config dicts or objects
        feature_config: Optional[Dict] = None,
        trading_config: Optional[Dict] = None,
        # Keep signal generation parameters
        threshold: Optional[float] = None,  # Get from model_config if possible
        neutral_zone: Optional[Tuple[float, float]] = None,  # Get from model_config
        trend_filter_ma: Optional[int] = None,  # Get from model_config
        volatility_filter_atr: Optional[int] = None,  # Get from model_config
        volatility_threshold_q: float = 0.75,  # Default, can be overridden by config
        cooling_period: Optional[int] = None,  # Get from model_config
        class_names: Optional[List[str]] = None,  # Need class names
    ):
        """
        Initialize the Signal Generator.

        Args:
            model_config: Dictionary or object containing model prediction parameters.
            feature_config: Dictionary or object containing feature parameters (for filters).
            trading_config: Dictionary or object containing trading parameters.
            threshold: Minimum probability of the winning class to generate a non-neutral signal.
            neutral_zone: Probability range considered neutral
            trend_filter_ma: MA period for trend filter (None to disable)
            volatility_filter_atr: ATR period for volatility filter (None to disable)
            volatility_threshold_q: Quantile threshold for high volatility filter
            cooling_period: Minimum bars between signals
            class_names: List of class names output by the model.
        """
        # Extract parameters from config dicts/objects or use direct args
        mc = model_config or {}

        self.threshold = (
            threshold if threshold is not None else mc.get("probability_threshold", 0.6)
        )
        self.neutral_zone = (
            neutral_zone
            if neutral_zone is not None
            else mc.get("signal_neutral_zone", (0.45, 0.55))
        )
        self.trend_filter_ma = (
            trend_filter_ma
            if trend_filter_ma is not None
            else mc.get("signal_trend_filter_ma")
        )
        self.volatility_filter_atr = (
            volatility_filter_atr
            if volatility_filter_atr is not None
            else mc.get("signal_volatility_filter_atr")
        )
        self.volatility_threshold_q = (
            volatility_threshold_q  # Keep default or allow override? Check config?
        )
        self.cooling_period = (
            cooling_period
            if cooling_period is not None
            else mc.get("signal_cooling_period", 3)
        )

        self.last_signal_time = (
            {}
        )  # symbol -> timestamp (This state might be better managed outside)

        # --- Class Name to Signal Value Mapping ---
        self.class_names = class_names
        self.signal_map = {}  # Map from class index to signal value (-1, 0, 1)

        if self.class_names:
            # Try to create a mapping based on common conventions
            if set(self.class_names) == {"-1", "1"}:  # Binary -1, 1
                self.signal_map = {
                    self.class_names.index("-1"): -1,
                    self.class_names.index("1"): 1,
                }
            elif set(self.class_names) == {
                "0",
                "1",
            }:  # Binary 0, 1 (map 0 -> -1, 1 -> 1)
                self.signal_map = {
                    self.class_names.index("0"): -1,
                    self.class_names.index("1"): 1,
                }
                logger.warning(
                    "Assuming binary classes '0', '1' map to signals -1, 1 respectively."
                )
            elif set(self.class_names) == {"-1", "0", "1"}:  # Standard ternary
                self.signal_map = {
                    self.class_names.index("-1"): -1,
                    self.class_names.index("0"): 0,
                    self.class_names.index("1"): 1,
                }
            else:
                # Fallback: Assume order corresponds to -1, 0, 1 if 3 classes, or -1, 1 if 2 classes
                logger.warning(
                    f"Could not automatically map class names {self.class_names} to signals. Assuming standard order."
                )
                if len(self.class_names) == 3:
                    self.signal_map = {0: -1, 1: 0, 2: 1}
                elif len(self.class_names) == 2:
                    self.signal_map = {0: -1, 1: 1}  # Map index 0 to -1, index 1 to 1
                else:
                    logger.error(
                        f"Unsupported number of classes ({len(self.class_names)}) for default signal mapping."
                    )
                    # Keep signal_map empty, generation will likely fail

            logger.info(f"Established signal map (index -> signal): {self.signal_map}")
            if len(self.signal_map) != len(self.class_names):
                logger.error("Signal map creation failed or incomplete.")
                self.signal_map = {}  # Reset if failed

        else:
            logger.error(
                "Class names not provided during SignalGenerator initialization. Cannot generate signals reliably."
            )

    def generate_signals(
        self,
        predictions: pd.DataFrame,
        data: Optional[pd.DataFrame] = None,  # OHLC data needed for filters
        current_regime: Optional[int] = None,  # Added for regime-based actions
        regime_actions: Optional[
            Dict[int, str]
        ] = None,  # Added for regime-based actions
    ) -> Optional[pd.DataFrame]:
        """
        Generate trading signals (-1, 0, 1) based on pre-calculated model predictions/probabilities.

        Args:
            predictions: DataFrame containing model outputs (e.g., 'probability_0', 'probability_1', 'confidence')
                         Must align index with ohlc_data if filters are used.
            data: DataFrame with OHLC data for filters (must align with predictions index)
            current_regime: The currently detected market regime (if applicable).
            regime_actions: Dictionary mapping regime index to action (e.g., {0: 'no_trade'}).

        Returns:
            DataFrame with 'probability_pos', 'probability_neg', 'signal', 'confidence' columns, or None if error.
        """
        if predictions.empty:
            logger.warning("Input predictions DataFrame is empty.")
            return pd.DataFrame(
                columns=["probability_pos", "probability_neg", "signal", "confidence"]
            )

        logger.info(
            f"Generating signals from {len(predictions)} predictions using threshold {self.threshold}..."
        )

        if not self.class_names or not self.signal_map:
            logger.error(
                "Cannot generate signals: Class names or signal map missing/invalid."
            )
            return None

        # --- Extract Full Probabilities ---
        prob_cols = [f"probability_{c}" for c in self.class_names]
        missing_cols = [c for c in prob_cols if c not in predictions.columns]
        if missing_cols:
            logger.error(
                f"Missing required probability columns in predictions: {missing_cols}"
            )
            return None

        probabilities_array = predictions[prob_cols].values

        # --- Start Signal DataFrame ---
        signals = pd.DataFrame(index=predictions.index)
        # Store main probabilities for reference (positive and negative if possible)
        try:
            pos_signal_val = 1
            neg_signal_val = -1
            pos_idx = [k for k, v in self.signal_map.items() if v == pos_signal_val][0]
            neg_idx = [k for k, v in self.signal_map.items() if v == neg_signal_val][0]
            signals["probability_pos"] = probabilities_array[:, pos_idx]
            signals["probability_neg"] = probabilities_array[:, neg_idx]
        except IndexError:
            logger.warning(
                "Could not find standard +1/-1 signals in map. Storing max probability as confidence."
            )
            signals["probability_pos"] = np.nan
            signals["probability_neg"] = np.nan

        # --- Apply Argmax + Threshold Logic ---
        winning_class_idx = np.argmax(probabilities_array, axis=1)
        winning_prob = np.max(probabilities_array, axis=1)

        # Map winning index to signal value using the pre-defined map
        signals["raw_signal"] = (
            pd.Series(winning_class_idx, index=signals.index)
            .map(self.signal_map)
            .fillna(0)
            .astype(int)
        )

        # Apply threshold: Set signal to neutral (0) if winning probability is below threshold
        signals.loc[winning_prob < self.threshold, "raw_signal"] = 0

        # Calculate confidence as the probability of the winning class
        signals["confidence"] = winning_prob

        # --- Apply Filters ---
        signals["filtered_signal"] = signals["raw_signal"]  # Start with raw signal

        if data is not None and not data.empty:
            # Align data index with signals index BEFORE applying filters
            data_aligned = data.reindex(signals.index)
            if self.trend_filter_ma is not None:
                signals["filtered_signal"] = self._apply_trend_filter(
                    signals,
                    data_aligned,  # Pass aligned data
                )
            if self.volatility_filter_atr is not None:
                signals["filtered_signal"] = self._apply_volatility_filter(
                    signals,
                    data_aligned,  # Pass aligned data
                )
        elif self.trend_filter_ma is not None or self.volatility_filter_atr is not None:
            logger.warning(
                "OHLC data not provided, cannot apply trend/volatility filters."
            )

        # --- Apply Regime Actions ---
        if current_regime is not None and regime_actions:
            action = regime_actions.get(current_regime)
            if action == "no_trade":
                logger.info(f"Applying 'no_trade' action for regime {current_regime}.")
                signals["filtered_signal"] = 0
            elif action == "long_only":
                logger.info(f"Applying 'long_only' action for regime {current_regime}.")
                signals.loc[signals["filtered_signal"] == -1, "filtered_signal"] = 0
            elif action == "short_only":
                logger.info(
                    f"Applying 'short_only' action for regime {current_regime}."
                )
                signals.loc[signals["filtered_signal"] == 1, "filtered_signal"] = 0
            # Add more actions as needed

        # --- Apply Cooling Period ---
        signals["signal"] = signals["filtered_signal"]
        self._apply_cooling_period(signals)  # Modifies 'signal' column in place

        logger.info("Signal generation complete.")
        # Return standard columns
        return signals[["probability_pos", "probability_neg", "signal", "confidence"]]

    def _apply_trend_filter(
        self, signals: pd.DataFrame, ohlc: pd.DataFrame
    ) -> pd.Series:
        """Apply trend filter (e.g., only long above MA)."""
        if "close" not in ohlc.columns:
            logger.warning("Trend filter requires 'close' column in OHLC data.")
            return signals["filtered_signal"]
        if self.trend_filter_ma is None or self.trend_filter_ma <= 0:
            logger.debug("Trend filter MA period not set or invalid.")
            return signals["filtered_signal"]

        try:
            ma = (
                ohlc["close"].rolling(window=self.trend_filter_ma, min_periods=1).mean()
            )  # Use min_periods
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
        if self.volatility_filter_atr is None or self.volatility_filter_atr <= 0:
            logger.debug("Volatility filter ATR period not set or invalid.")
            return signals["filtered_signal"]

        try:
            if not PANDAS_TA_AVAILABLE:
                logger.warning(
                    "Cannot calculate ATR for volatility filter without pandas-ta."
                )
                return signals["filtered_signal"]

            # Calculate ATR using pandas_ta
            atr = ta.atr(
                ohlc["high"],
                ohlc["low"],
                ohlc["close"],
                length=self.volatility_filter_atr,
            )

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
        if self.cooling_period is None or self.cooling_period <= 0:
            logger.debug("Cooling period not set or invalid.")
            return

        last_signal_idx = -self.cooling_period - 1
        signal_col = signals[
            "signal"
        ].values  # Work on numpy array for potential speedup
        indices_to_reset = []

        for i in range(len(signal_col)):
            if signal_col[i] != 0:
                if i - last_signal_idx > self.cooling_period:
                    last_signal_idx = i  # Allow signal, update last signal time
                else:
                    indices_to_reset.append(i)  # Mark index to reset

        if indices_to_reset:
            signals.iloc[indices_to_reset, signals.columns.get_loc("signal")] = (
                0  # Reset marked indices
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
                plt.close(fig)  # Close the figure if no data
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
            if "probability_pos" in signals_aligned.columns:
                axes[1].plot(
                    signals_aligned.index,
                    signals_aligned["probability_pos"],
                    label="Prob(Long)",
                    color="green",
                    alpha=0.7,
                )
            if "probability_neg" in signals_aligned.columns:
                axes[1].plot(
                    signals_aligned.index,
                    signals_aligned["probability_neg"],
                    label="Prob(Short)",
                    color="red",
                    alpha=0.7,
                )

            axes[1].axhline(0.5, color="grey", linestyle="--", alpha=0.5)
            # Plot threshold lines based on the instance's threshold
            axes[1].axhline(
                self.threshold,
                color="black",
                linestyle=":",
                alpha=0.6,
                label=f"Threshold ({self.threshold:.2f})",
            )
            axes[1].axhline(1 - self.threshold, color="black", linestyle=":", alpha=0.6)
            # Plot neutral zone if defined differently from threshold
            # if self.neutral_zone and (self.neutral_zone[0] != 1 - self.threshold or self.neutral_zone[1] != self.threshold):
            #      axes[1].axhline(
            #          self.neutral_zone[0],
            #          color="orange", linestyle=":", alpha=0.6,
            #          label=f"Neutral Zone ({self.neutral_zone[0]:.2f}-{self.neutral_zone[1]:.2f})"
            #      )
            #      axes[1].axhline(self.neutral_zone[1], color="orange", linestyle=":", alpha=0.6)

            axes[1].set_ylabel("Probability")
            axes[1].set_xlabel("Date")
            axes[1].set_ylim(0, 1)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()

            if filename:
                try:
                    # Ensure directory exists
                    output_dir = os.path.dirname(filename)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    plt.savefig(filename, dpi=300, bbox_inches="tight")
                    logger.info(f"Signal history plot saved to {filename}")
                except Exception as e:
                    logger.error(f"Failed to save signal plot to {filename}: {e}")

            else:
                plt.show()

            plt.close(fig)

        except Exception as e:
            logger.exception(f"Error plotting signal history: {e}")
            plt.close()  # Ensure plot is closed
