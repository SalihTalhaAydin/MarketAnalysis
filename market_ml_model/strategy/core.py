# market_ml_model/strategy/core.py
import concurrent.futures
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Configuration imports
from .config import AssetConfig, StrategyConfig

logger = logging.getLogger(__name__)

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib/Seaborn not installed. Visualization unavailable.")
    VISUALIZATION_AVAILABLE = False
    plt = None
    sns = None

# Import other necessary components using relative paths
# Assume these modules are available when running as part of the package
try:
    # Data Loading
    from ..data.loaders import (
        DataLoader,
        DataLoaderConfig,
    )  # Original import was load_data

    # Feature Engineering
    from ..features.features_engineering import engineer_features
    from ..features.technical.indicators import PANDAS_TA_AVAILABLE  # Import check

    if PANDAS_TA_AVAILABLE:
        import pandas_ta as ta  # Import ta if available
    else:
        ta = None  # Define ta as None if not available

    # Model Training & Prediction
    from ..models.training import (
        train_classification_model,
    )  # Added for direct pipeline creation
    from ..models.predictor import (
        ModelPredictorBase,
        get_confidence_levels,
        predict_with_threshold,
    )
    from ..models.signals import SignalGenerator
    from ..utils.visualization import plot_drawdowns, plot_equity_curve

    # Backtesting & Simulation
    from ..trading.backtest import backtest_strategy

    # Utils

    # Regime Detection (Import the class from the new location)
    from .regime import (
        MarketRegimeDetector,
        SKLEARN_AVAILABLE_FOR_REGIME,
    )  # Import necessary items from regime.py

    MODULES_AVAILABLE = True
except ImportError as e:
    logger.critical(
        f"Error importing strategy components in core.py: {e}. Relative imports failed. Ensure running as part of the package."
    )
    MODULES_AVAILABLE = False
    # Do not include dummy classes here; raise error if modules are missing


# --- Main Strategy Class ---


class EnhancedTradingStrategy:
    """Main class for the enhanced trading strategy."""

    def __init__(self, config: StrategyConfig):
        """Initialize the strategy with configuration."""
        if not MODULES_AVAILABLE:
            raise ImportError(
                "Core strategy modules could not be imported. Cannot initialize strategy."
            )

        self.config = config
        self.data_loader = self._initialize_data_loader()

        # --- Market Regime Configuration (Access via StrategyConfig) ---
        # Initialize MarketRegimeDetector if enabled
        self.market_regime_detector: Optional[MarketRegimeDetector] = None
        if self.config.market_regime_config.enabled:
            logger.info("Initializing Market Regime Detector...")
            # Check dependencies for selected method
            if (
                self.config.market_regime_config.method == "volatility_clustering"
                and not SKLEARN_AVAILABLE_FOR_REGIME
            ):
                logger.error(
                    "Market regime detection uses clustering but scikit-learn is not available. Disabling regime detection."
                )
                self.config.market_regime_config.enabled = (
                    False  # Disable if dependencies missing
                )
            else:
                try:
                    self.market_regime_detector = MarketRegimeDetector(
                        n_regimes=self.config.market_regime_config.params.get(
                            "n_clusters", 3
                        ),
                        lookback_window=self.config.market_regime_config.params.get(
                            "lookback_window", 60
                        ),
                        regime_features=self.config.market_regime_config.params.get(
                            "features"
                        ),  # Pass features from params
                        use_clustering=(
                            self.config.market_regime_config.method
                            == "volatility_clustering"
                        ),
                        random_state=self.config.random_state,
                    )
                    logger.info(
                        f"Market Regime Detector initialized (Method: {self.config.market_regime_config.method})"
                    )
                except Exception as e:
                    logger.exception(f"Failed to initialize MarketRegimeDetector: {e}")
                    self.config.market_regime_config.enabled = (
                        False  # Disable if init fails
                    )

        self.models: Dict[
            str, str
        ] = {}  # Stores paths to saved models (asset_symbol[_regime_X] -> path)
        self.predictors: Dict[str, ModelPredictorBase] = {}  # Stores loaded predictors
        self.signal_generators: Dict[
            str, SignalGenerator
        ] = {}  # Stores signal generators per asset/model_key
        self.results: Dict[
            str, Any
        ] = {}  # Stores results per asset or walk-forward step

        # Set logging level based on debug mode
        log_level = logging.DEBUG if config.debug_mode else logging.INFO
        logger.setLevel(log_level)
        # Set root logger level if needed, or configure specific loggers
        # logging.getLogger().setLevel(log_level)
        logger.info(f"Strategy logging level set to {logging.getLevelName(log_level)}")
        logger.debug(f"Strategy Run ID: {config.strategy_run_id}")

    def _initialize_data_loader(self) -> Optional[DataLoader]:
        """Initialize the data loader based on config."""
        try:
            # Assuming DataLoaderConfig can be created simply or needs more details from self.config
            # Note: DataLoaderConfig might require more specific arguments based on its definition.
            loader_config = DataLoaderConfig(
                start_date=self.config.data_start_date,
                end_date=self.config.data_end_date,
                # Add other necessary params from self.config if needed by DataLoaderConfig
            )
            loader = DataLoader(loader_config)
            logger.info("Data loader initialized successfully.")
            return loader
        except Exception as e:
            logger.exception(f"Error initializing data loader: {e}")
            return None

    def load_data(self, asset_config: AssetConfig) -> Optional[pd.DataFrame]:
        """Load data for a specific asset."""
        if self.data_loader is None:
            logger.error("Data loader not initialized.")
            return None
        try:
            logger.info(
                f"Loading data for {asset_config.symbol} from {asset_config.data_source}..."
            )
            # Call the method on the instance, not the imported function
            data = self.data_loader.load_data(
                ticker=asset_config.symbol,  # Use 'ticker' as expected by DataLoader method
                start_date=self.config.data_start_date,
                end_date=self.config.data_end_date,
                data_source=asset_config.data_source,
                interval=asset_config.timeframe,  # Use 'interval' as expected by DataLoader method
                # Add other potential args like api_keys if needed by load_data
            )
            if data is None or data.empty:
                logger.warning(f"No data loaded for {asset_config.symbol}.")
                return None
            logger.info(f"Data loaded for {asset_config.symbol}. Shape: {data.shape}")
            return data
        except Exception as e:
            logger.exception(f"Error loading data for {asset_config.symbol}: {e}")
            return None

    def engineer_features(
        self, data: pd.DataFrame, asset_config: AssetConfig
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Engineer features and target variable."""
        if data is None or data.empty:
            logger.error("Cannot engineer features with empty data.")
            return None, None
        logger.info(f"Engineering features for {asset_config.symbol}...")
        try:
            # Pass the FeatureConfig object directly
            features_df, target_series = engineer_features(
                data, self.config.feature_config
            )
            if features_df is None or target_series is None:
                logger.error(
                    f"Feature engineering returned None for {asset_config.symbol}"
                )
                return None, None
            logger.info(
                f"Features engineered for {asset_config.symbol}. Features shape: {features_df.shape}, Target shape: {target_series.shape}"
            )
            return features_df, target_series
        except Exception as e:
            logger.exception(
                f"Error engineering features for {asset_config.symbol}: {e}"
            )
            return None, None

    def _detect_regime_for_data(
        self, features_df: pd.DataFrame, asset_symbol: str
    ) -> Optional[int]:
        """Helper to detect regime using the initialized detector."""
        if self.market_regime_detector and self.config.market_regime_config.enabled:
            logger.debug(
                f"Detecting regime for {asset_symbol} using available features..."
            )
            regime_info = self.market_regime_detector.detect_regime(features_df)
            if regime_info:
                logger.info(
                    f"Detected Regime for {asset_symbol}: {regime_info['regime']} ({regime_info['regime_name']})"
                )
                return regime_info["regime"]
            else:
                logger.warning(f"Regime detection failed for {asset_symbol}.")
        return None

    def train_model(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        asset_config: AssetConfig,
        fold_id: str = "full_run",  # Identifier for the training run (e.g., fold number)
        regime: Optional[int] = None,
    ) -> Optional[str]:
        """Train a model for the given features and target."""
        if features is None or target is None or features.empty or target.empty:
            logger.error("Cannot train model with empty features or target.")
            return None

        # Align features and target
        aligned_features, aligned_target = features.align(target, join="inner", axis=0)
        if aligned_features.empty or aligned_target.empty:
            logger.error(
                "Features and target have no overlapping indices after alignment."
            )
            return None

        logger.info(
            f"Training model for {asset_config.symbol} (Fold: {fold_id}, Regime: {regime}). "
            f"Data shape: {aligned_features.shape}"
        )

        # Determine model type based on config and regime
        model_type = self.config.model_config.model_type
        if regime is not None and self.config.model_config.regime_models:
            model_type = self.config.model_config.regime_models.get(regime, model_type)
            logger.info(f"Using regime-specific model type: {model_type}")

        # Define output path for this specific model artifact
        model_key = asset_config.symbol
        if regime is not None:
            model_key += f"_regime_{regime}"
        model_path_key = (
            f"{model_key}_{fold_id}"  # Unique key for this asset/regime/fold
        )
        model_output_dir = os.path.join(
            self.config.output_dir,
            "models",
            asset_config.symbol,
            fold_id,
            f"regime_{regime if regime is not None else 'all'}",
        )
        os.makedirs(model_output_dir, exist_ok=True)

        try:
            # Use the train_classification_model function from models.training
            model, preprocessor, metrics, selected_features = (
                train_classification_model(
                    features=aligned_features,
                    target=aligned_target,
                    model_type=model_type,
                    model_config=self.config.model_config,  # Pass the whole ModelConfig
                    feature_config=self.config.feature_config,  # Pass FeatureConfig for scaling/diff info
                    random_state=self.config.random_state,
                    output_dir=model_output_dir,  # Pass specific dir for this artifact
                    search_space=self.config.search_space,  # Pass search space if defined
                )
            )

            if model is None or preprocessor is None:
                logger.error(
                    f"Model training failed for {asset_config.symbol} (Fold: {fold_id}, Regime: {regime})."
                )
                return None

            # Save the trained model, preprocessor, and selected features
            model_save_path = os.path.join(model_output_dir, "model.pkl")
            preprocessor_save_path = os.path.join(model_output_dir, "preprocessor.pkl")
            features_save_path = os.path.join(
                model_output_dir, "selected_features.json"
            )
            metrics_save_path = os.path.join(model_output_dir, "training_metrics.json")

            joblib.dump(model, model_save_path)
            joblib.dump(preprocessor, preprocessor_save_path)
            with open(features_save_path, "w") as f:
                json.dump(selected_features, f)
            with open(metrics_save_path, "w") as f:
                json.dump(
                    metrics, f, indent=4, default=str
                )  # Use default=str for non-serializable types

            logger.info(f"Model artifacts saved to: {model_output_dir}")
            self.models[model_path_key] = model_output_dir  # Store the directory path

            return model_output_dir  # Return the directory path

        except Exception as e:
            logger.exception(
                f"Error training model for {asset_config.symbol} (Fold: {fold_id}, Regime: {regime}): {e}"
            )
            return None

    def generate_predictions(
        self,
        features: pd.DataFrame,
        asset_config: AssetConfig,
        fold_id: str = "full_run",  # Identifier matching the training run
        regime: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """Generate predictions using a trained model."""
        if features is None or features.empty:
            logger.error("Cannot generate predictions with empty features.")
            return None

        logger.info(
            f"Generating predictions for {asset_config.symbol} (Fold: {fold_id}, Regime: {regime}). "
            f"Features shape: {features.shape}"
        )

        # Construct the key to find the correct model path
        model_key = asset_config.symbol
        if (
            regime is not None and self.config.model_config.regime_adaptation_enabled
        ):  # Check if regime adaptation is on
            model_key += f"_regime_{regime}"
        model_path_key = f"{model_key}_{fold_id}"

        model_artifacts_path = self.models.get(model_path_key)

        if model_artifacts_path is None or not os.path.exists(
            os.path.join(model_artifacts_path, "model.pkl")
        ):
            logger.error(
                f"Trained model artifacts not found for key '{model_path_key}' (Path: {model_artifacts_path}). Cannot generate predictions."
            )
            return None

        try:
            # Load artifacts
            logger.info(f"Loading model artifacts from: {model_artifacts_path}")
            model = joblib.load(os.path.join(model_artifacts_path, "model.pkl"))
            preprocessor = joblib.load(
                os.path.join(model_artifacts_path, "preprocessor.pkl")
            )
            with open(
                os.path.join(model_artifacts_path, "selected_features.json"), "r"
            ) as f:
                selected_features = json.load(f)

            # Preprocess features
            X_processed = preprocessor.transform(features)
            try:
                processed_feature_names = preprocessor.get_feature_names_out()
            except (
                AttributeError
            ):  # Handle older sklearn versions or different preprocessors
                processed_feature_names = list(
                    features.columns
                )  # Fallback, might be incorrect if preprocessor changes columns
            X_processed_df = pd.DataFrame(
                X_processed, index=features.index, columns=processed_feature_names
            )

            # Select features used during training
            available_selected_features = [
                f for f in selected_features if f in X_processed_df.columns
            ]
            if len(available_selected_features) < len(selected_features):
                logger.warning(
                    f"Some selected features were not found in processed prediction data. Using {len(available_selected_features)} features."
                )
                if not available_selected_features:
                    raise ValueError(
                        "No selected features available in prediction data."
                    )
            X_selected_processed_df = X_processed_df[available_selected_features]

            # Predict probabilities
            probabilities = model.predict_proba(X_selected_processed_df)

            # Create predictions DataFrame
            class_names = model.classes_.astype(str)
            prob_col_names = [f"probability_{c}" for c in class_names]
            predictions_df = pd.DataFrame(
                probabilities,
                index=X_selected_processed_df.index,
                columns=prob_col_names,
            )

            # Add prediction based on threshold and confidence
            positive_class_label = "1"  # Assuming '1' is the positive class
            try:
                pos_idx = list(class_names).index(positive_class_label)
            except ValueError:
                logger.warning(
                    f"Positive class label '{positive_class_label}' not found in {class_names}. Defaulting to index 1."
                )
                pos_idx = 1  # Default to the second class if '1' is not found

            predictions_df["prediction"] = predict_with_threshold(
                probabilities,
                threshold=self.config.model_config.probability_threshold,
                positive_class_index=pos_idx,
            )
            predictions_df["confidence"] = get_confidence_levels(probabilities)

            logger.info(
                f"Predictions generated successfully for {asset_config.symbol} (Fold: {fold_id}, Regime: {regime})."
            )
            return predictions_df

        except Exception as e:
            logger.exception(
                f"Error generating predictions for {asset_config.symbol} (Fold: {fold_id}, Regime: {regime}): {e}"
            )
            return None

    def generate_signals(
        self,
        predictions: pd.DataFrame,
        ohlc_data: pd.DataFrame,  # Pass raw OHLCV data for filters
        asset_config: AssetConfig,
        regime: Optional[int] = None,
        regime_actions: Optional[Dict[int, str]] = None,
    ) -> Optional[pd.DataFrame]:
        """Generate trading signals from predictions, applying filters and regime actions."""
        if predictions is None or predictions.empty:
            logger.error("Cannot generate signals with empty predictions.")
            return None
        if ohlc_data is None or ohlc_data.empty:
            logger.error("Cannot generate signals without OHLC data for filters.")
            return None

        logger.info(
            f"Generating signals for {asset_config.symbol} (Regime: {regime})..."
        )

        # Initialize SignalGenerator (could be cached per asset/regime if config is static)
        try:
            signal_gen = SignalGenerator(
                model_config=self.config.model_config,  # Pass ModelConfig for thresholds, filters etc.
                feature_config=self.config.feature_config,  # Pass FeatureConfig if needed for filters
                trading_config=self.config.trading_config,  # Pass TradingConfig if needed
            )

            # Generate signals using the generator
            signals_df = signal_gen.generate_signals(
                predictions=predictions,
                data=ohlc_data,  # Pass OHLC data for filters
                current_regime=regime,
                regime_actions=regime_actions,
            )

            if signals_df is None:
                logger.error(f"SignalGenerator returned None for {asset_config.symbol}")
                return None

            logger.info(
                f"Signals generated for {asset_config.symbol}. Shape: {signals_df.shape}"
            )
            return signals_df

        except Exception as e:
            logger.exception(f"Error generating signals for {asset_config.symbol}: {e}")
            return None

    def generate_dma_signals(
        self, data: pd.DataFrame, asset_config: AssetConfig
    ) -> Optional[pd.DataFrame]:
        """Generate signals for a Dual Moving Average Crossover strategy."""
        logger.info(f"Generating DMA signals for {asset_config.symbol}...")
        short_period = self.config.sma_short_period
        long_period = self.config.sma_long_period

        if not short_period or not long_period:
            logger.error("Short or long SMA period not configured for DMA strategy.")
            return None
        if "close" not in data.columns:
            logger.error("Close price column missing for DMA calculation.")
            return None

        try:
            signals_df = pd.DataFrame(index=data.index)
            signals_df["SMA_Short"] = data["close"].rolling(window=short_period).mean()
            signals_df["SMA_Long"] = data["close"].rolling(window=long_period).mean()

            # Generate signals: 1 for buy (short > long), -1 for sell (short < long)
            signals_df["signal"] = 0
            # Entry signal: Crossover happened in the previous step
            signals_df.loc[
                (signals_df["SMA_Short"].shift(1) > signals_df["SMA_Long"].shift(1)),
                "signal",
            ] = 1
            signals_df.loc[
                (signals_df["SMA_Short"].shift(1) < signals_df["SMA_Long"].shift(1)),
                "signal",
            ] = -1

            # Refine signals: Only signal on the actual crossover day
            # A buy signal occurs when short crosses above long
            buy_signal = (signals_df["SMA_Short"] > signals_df["SMA_Long"]) & (
                signals_df["SMA_Short"].shift(1) <= signals_df["SMA_Long"].shift(1)
            )
            # A sell signal occurs when short crosses below long
            sell_signal = (signals_df["SMA_Short"] < signals_df["SMA_Long"]) & (
                signals_df["SMA_Short"].shift(1) >= signals_df["SMA_Long"].shift(1)
            )

            signals_df["signal"] = 0  # Reset signals
            signals_df.loc[buy_signal, "signal"] = 1
            signals_df.loc[sell_signal, "signal"] = -1

            # Forward fill signals to hold positions until exit signal
            # signals_df['signal'] = signals_df['signal'].replace(0, method='ffill').fillna(0) # Optional: Hold position

            logger.info(
                f"DMA signals generated for {asset_config.symbol}. Signal counts: {signals_df['signal'].value_counts().to_dict()}"
            )
            return signals_df[["signal"]]  # Return only the signal column

        except Exception as e:
            logger.exception(
                f"Error generating DMA signals for {asset_config.symbol}: {e}"
            )
            return None

    def generate_rsi_signals(
        self, data: pd.DataFrame, asset_config: AssetConfig
    ) -> Optional[pd.DataFrame]:
        """Generate signals for an RSI Mean Reversion strategy."""
        logger.info(f"Generating RSI signals for {asset_config.symbol}...")
        rsi_period = self.config.rsi_period
        lower_thresh = self.config.rsi_lower_threshold
        upper_thresh = self.config.rsi_upper_threshold

        if not rsi_period or lower_thresh is None or upper_thresh is None:
            logger.error("RSI parameters (period, lower, upper) not fully configured.")
            return None
        if not PANDAS_TA_AVAILABLE:
            logger.error("pandas_ta library is required for RSI calculation.")
            return None
        if "close" not in data.columns:
            logger.error("Close price column missing for RSI calculation.")
            return None

        try:
            signals_df = pd.DataFrame(index=data.index)
            signals_df["RSI"] = ta.rsi(data["close"], length=rsi_period)

            # Generate signals: 1 for buy (RSI crosses below lower), -1 for sell (RSI crosses above upper)
            signals_df["signal"] = 0
            # Buy signal: RSI crosses below lower threshold
            buy_signal = (signals_df["RSI"] < lower_thresh) & (
                signals_df["RSI"].shift(1) >= lower_thresh
            )
            # Sell signal: RSI crosses above upper threshold
            sell_signal = (signals_df["RSI"] > upper_thresh) & (
                signals_df["RSI"].shift(1) <= upper_thresh
            )

            signals_df.loc[buy_signal, "signal"] = 1
            signals_df.loc[sell_signal, "signal"] = -1

            # Optional: Add exit condition (e.g., RSI crosses back above midline for longs)
            # exit_long = (signals_df['RSI'] > 50) & (signals_df['RSI'].shift(1) <= 50)
            # exit_short = (signals_df['RSI'] < 50) & (signals_df['RSI'].shift(1) >= 50)
            # signals_df.loc[exit_long, 'signal'] = 0 # Close long
            # signals_df.loc[exit_short, 'signal'] = 0 # Close short

            # Forward fill signals to hold positions until exit signal (if exits aren't explicit)
            # signals_df['signal'] = signals_df['signal'].replace(0, method='ffill').fillna(0)

            logger.info(
                f"RSI signals generated for {asset_config.symbol}. Signal counts: {signals_df['signal'].value_counts().to_dict()}"
            )
            return signals_df[["signal"]]  # Return only the signal column

        except Exception as e:
            logger.exception(
                f"Error generating RSI signals for {asset_config.symbol}: {e}"
            )
            return None

    def backtest_signals(
        self,
        signals: pd.DataFrame,
        ohlc_data: pd.DataFrame,
        asset_config: AssetConfig,
        fold_id: str = "full_run",  # Identifier for the backtest run
        regime: Optional[int] = None,
        regime_actions: Optional[Dict[int, str]] = None,
    ) -> Optional[Dict]:
        """Backtest the generated signals."""
        if signals is None or signals.empty:
            logger.error("Cannot backtest with empty signals.")
            return None
        if ohlc_data is None or ohlc_data.empty:
            logger.error("Cannot backtest without OHLC data.")
            return None

        logger.info(
            f"Backtesting signals for {asset_config.symbol} (Fold: {fold_id}, Regime: {regime})..."
        )

        # Align signals with OHLC data
        aligned_signals, aligned_data = signals.align(ohlc_data, join="inner", axis=0)
        if aligned_signals.empty or aligned_data.empty:
            logger.error(
                "Signals and OHLC data have no overlapping indices after alignment."
            )
            return None

        # Define output directory for this specific backtest
        backtest_output_dir = os.path.join(
            self.config.output_dir,
            "backtests",
            asset_config.symbol,
            fold_id,
            f"regime_{regime if regime is not None else 'all'}",
        )
        os.makedirs(backtest_output_dir, exist_ok=True)

        try:
            # Use the backtest_strategy function
            results = backtest_strategy(
                signals=aligned_signals,
                data=aligned_data,
                strategy_config=self.config,  # Pass the main StrategyConfig
                asset_config=asset_config,  # Pass the specific AssetConfig
                output_dir=backtest_output_dir,  # Pass specific dir for results/plots
                # Pass regime info if needed by backtester (e.g., for regime-specific actions)
                current_regime=regime,
                regime_actions=regime_actions,
            )

            if results is None:
                logger.error(
                    f"Backtesting failed for {asset_config.symbol} (Fold: {fold_id}, Regime: {regime})."
                )
                return None

            logger.info(
                f"Backtesting completed for {asset_config.symbol} (Fold: {fold_id}, Regime: {regime})."
            )
            # Add fold/regime info to results if not already present
            results["fold_id"] = fold_id
            results["regime"] = regime
            return results

        except Exception as e:
            logger.exception(
                f"Error backtesting signals for {asset_config.symbol} (Fold: {fold_id}, Regime: {regime}): {e}"
            )
            return None

    def run_walk_forward_validation(self, asset_config: AssetConfig) -> Dict:
        """Run walk-forward validation for a single asset."""
        logger.info(
            f"--- Starting Walk-Forward Validation for {asset_config.symbol} ---"
        )
        wf_config = self.config.walkforward_config
        if not wf_config.enabled:
            logger.warning("Walk-forward validation is disabled in config.")
            return {"status": "skipped", "message": "Walk-forward disabled"}

        full_data = self.load_data(asset_config)
        if full_data is None:
            return {
                "status": "error",
                "message": "Data loading failed for walk-forward",
            }

        # Engineer features on the entire dataset once if strategy is ML-based
        full_features, full_target = None, None
        if (
            self.config.strategy_type != "dma_crossover"
            and self.config.strategy_type != "rsi_mean_reversion"
        ):
            full_features, full_target = self.engineer_features(full_data, asset_config)
            if full_features is None or full_target is None:
                return {
                    "status": "error",
                    "message": "Feature engineering failed for walk-forward",
                }
            # Align features and target with the full data index
            full_features = full_features.reindex(full_data.index)
            full_target = full_target.reindex(full_data.index)

        n_total = len(full_data)
        train_start_idx = 0
        train_end_idx = wf_config.initial_train_periods
        test_end_idx = train_end_idx + wf_config.test_periods

        wf_results_list = []
        all_trades_list = []  # To collect trades from all folds
        step = 0

        while test_end_idx <= n_total:
            step += 1
            fold_id = f"fold_{step}"
            logger.info(
                f"--- Running Walk-Forward Step {step} for {asset_config.symbol} ---"
            )
            logger.info(f"Train Period: Index {train_start_idx} to {train_end_idx-1}")
            logger.info(f"Test Period: Index {train_end_idx} to {test_end_idx-1}")

            # --- Slice Data for Current Step ---
            train_data = full_data.iloc[train_start_idx:train_end_idx]
            test_data = full_data.iloc[train_end_idx:test_end_idx]

            if train_data.empty or test_data.empty:
                logger.warning(f"Empty train or test data for step {step}. Stopping.")
                break

            # --- Initialize variables for this step ---
            model_artifacts_path = None
            current_regime = None
            signals_df = None
            train_features_raw = None  # Keep track if ML features were generated
            test_features_raw = None  # Keep track if ML features were generated

            # --- Strategy-Specific Workflow (DMA, RSI, or ML) ---
            if self.config.strategy_type == "dma_crossover":
                logger.info(f"Generating DMA signals for test period step {step}...")
                # Generate signals on the combined train+test data up to the end of test period
                # to ensure MA calculations are correct up to the point of signaling.
                combined_data_for_signals = full_data.iloc[train_start_idx:test_end_idx]
                dma_signals_full = self.generate_dma_signals(
                    combined_data_for_signals, asset_config
                )
                if dma_signals_full is not None:
                    # Slice signals for the actual test period
                    signals_df = dma_signals_full.loc[test_data.index]
                else:
                    logger.error(f"DMA signal generation failed for step {step}.")
                    # Advance window indices and continue
                    train_start_idx = (
                        train_start_idx + wf_config.step_periods
                        if wf_config.rolling_window
                        else train_start_idx
                    )
                    train_end_idx += wf_config.step_periods
                    test_end_idx += wf_config.step_periods
                    continue
                model_artifacts_path = "DMA_Strategy"  # Placeholder

            elif self.config.strategy_type == "rsi_mean_reversion":
                logger.info(f"Generating RSI signals for test period step {step}...")
                combined_data_for_signals = full_data.iloc[train_start_idx:test_end_idx]
                rsi_signals_full = self.generate_rsi_signals(
                    combined_data_for_signals, asset_config
                )
                if rsi_signals_full is not None:
                    signals_df = rsi_signals_full.loc[test_data.index]
                else:
                    logger.error(f"RSI signal generation failed for step {step}.")
                    train_start_idx = (
                        train_start_idx + wf_config.step_periods
                        if wf_config.rolling_window
                        else train_start_idx
                    )
                    train_end_idx += wf_config.step_periods
                    test_end_idx += wf_config.step_periods
                    continue
                model_artifacts_path = "RSI_Strategy"  # Placeholder

            else:  # ML Workflow
                # --- Slice Features/Target for ML ---
                if full_features is None or full_target is None:
                    logger.error(
                        "Full features/target not available for ML walk-forward. Stopping."
                    )
                    break
                train_features_raw = full_features.iloc[train_start_idx:train_end_idx]
                train_target = full_target.iloc[train_start_idx:train_end_idx]
                test_features_raw = full_features.iloc[train_end_idx:test_end_idx]

                # Align sliced features and target
                train_features_raw, train_target = train_features_raw.align(
                    train_target, join="inner", axis=0
                )

                if (
                    train_features_raw.empty
                    or train_target.empty
                    or test_features_raw.empty
                ):
                    logger.warning(
                        f"Empty features or target after slicing/aligning for step {step}. Skipping."
                    )
                    # Advance window indices
                    train_start_idx = (
                        train_start_idx + wf_config.step_periods
                        if wf_config.rolling_window
                        else train_start_idx
                    )
                    train_end_idx += wf_config.step_periods
                    test_end_idx += wf_config.step_periods
                    continue

                # --- Detect Regime (based on training data end) ---
                current_regime = self._detect_regime_for_data(
                    train_features_raw, asset_config.symbol
                )

                # --- Train or Load Model ---
                # Retrain only if it's the first step or retrain frequency is met
                needs_retrain = (step == 1) or (
                    wf_config.retrain_frequency > 0
                    and (step - 1) % wf_config.retrain_frequency == 0
                )

                if needs_retrain:
                    logger.info(f"Retraining model for step {step}...")
                    model_artifacts_path = self.train_model(
                        train_features_raw,
                        train_target,
                        asset_config,
                        fold_id=fold_id,
                        regime=current_regime,
                    )
                    if model_artifacts_path is None:
                        logger.error(
                            f"Model training failed for step {step}. Skipping step."
                        )
                        # Advance window indices
                        train_start_idx = (
                            train_start_idx + wf_config.step_periods
                            if wf_config.rolling_window
                            else train_start_idx
                        )
                        train_end_idx += wf_config.step_periods
                        test_end_idx += wf_config.step_periods
                        continue
                else:
                    # Find the path of the previously trained model for this asset/regime
                    # Calculate the step number when the last retrain occurred
                    last_retrain_step = (
                        (step - 2) // wf_config.retrain_frequency
                    ) * wf_config.retrain_frequency + 1
                    last_fold_id = f"fold_{last_retrain_step}"
                    model_key = asset_config.symbol
                    if (
                        current_regime is not None
                        and self.config.model_config.regime_adaptation_enabled
                    ):
                        model_key += f"_regime_{current_regime}"
                    model_path_key = f"{model_key}_{last_fold_id}"
                    model_artifacts_path = self.models.get(model_path_key)
                    if model_artifacts_path is None or not os.path.exists(
                        os.path.join(model_artifacts_path, "model.pkl")
                    ):
                        logger.error(
                            f"Model artifacts not found at expected path key '{model_path_key}' (path: {model_artifacts_path}) for step {step}. Retraining might be needed or logic error."
                        )
                        # Advance window indices
                        train_start_idx = (
                            train_start_idx + wf_config.step_periods
                            if wf_config.rolling_window
                            else train_start_idx
                        )
                        train_end_idx += wf_config.step_periods
                        test_end_idx += wf_config.step_periods
                        continue
                    else:
                        logger.info(
                            f"Using previously trained model artifacts from step {last_retrain_step} for step {step}."
                        )

                # --- Generate predictions on test data (ML Path Only) ---
                logger.info(f"Generating predictions for test period step {step}...")
                if test_features_raw.empty:
                    logger.error(
                        f"Raw test features are empty for step {step}. Skipping prediction."
                    )
                    # Advance window indices
                    train_start_idx = (
                        train_start_idx + wf_config.step_periods
                        if wf_config.rolling_window
                        else train_start_idx
                    )
                    train_end_idx += wf_config.step_periods
                    test_end_idx += wf_config.step_periods
                    continue

                predictions_df = None
                if model_artifacts_path:  # Ensure we have a valid path for ML model
                    predictions_df = self.generate_predictions(
                        test_features_raw,
                        asset_config,
                        fold_id=fold_id if needs_retrain else last_fold_id,
                        regime=current_regime,
                    )
                else:  # No valid model_artifacts_path for ML
                    logger.error(
                        f"No valid model path found for prediction in step {step}. Skipping prediction."
                    )

                if predictions_df is None:
                    logger.error(
                        f"Prediction generation failed for step {step}. Skipping step."
                    )
                    # Advance window indices
                    train_start_idx = (
                        train_start_idx + wf_config.step_periods
                        if wf_config.rolling_window
                        else train_start_idx
                    )
                    train_end_idx += wf_config.step_periods
                    test_end_idx += wf_config.step_periods
                    continue

                # --- Generate Signals from Predictions (ML Path Only) ---
                logger.info(
                    f"Generating signals from ML predictions for step {step}..."
                )
                signals_df = self.generate_signals(
                    predictions_df,
                    test_data,  # Pass raw test OHLCV for filters
                    asset_config,
                    regime=current_regime,
                    regime_actions=self.config.market_regime_config.regime_actions,
                )

                if signals_df is None:
                    logger.error(
                        f"Signal generation from ML predictions failed for step {step}. Skipping step."
                    )
                    # Advance window indices
                    train_start_idx = (
                        train_start_idx + wf_config.step_periods
                        if wf_config.rolling_window
                        else train_start_idx
                    )
                    train_end_idx += wf_config.step_periods
                    test_end_idx += wf_config.step_periods
                    continue
                # End of ML-specific workflow path

            # --- Backtest the test period (Common for all strategy types) ---
            if (
                signals_df is None
            ):  # Check if signals were generated successfully by either path
                logger.error(
                    f"No signals available for backtesting in step {step}. Skipping step."
                )
                # Advance window indices
                train_start_idx = (
                    train_start_idx + wf_config.step_periods
                    if wf_config.rolling_window
                    else train_start_idx
                )
                train_end_idx += wf_config.step_periods
                test_end_idx += wf_config.step_periods
                continue

            logger.info(f"Backtesting test period step {step}...")
            backtest_input_data = test_data.copy()  # Start with raw OHLCV for backtest
            trade_cfg = self.config.trading_config
            atr_col_name = trade_cfg.atr_col

            # Add ATR column if needed for dynamic stops, checking engineered features if ML ran
            if trade_cfg.use_dynamic_stops and atr_col_name:
                # Use test_features_raw only if ML path was taken (it's None otherwise)
                if (
                    test_features_raw is not None
                    and atr_col_name in test_features_raw.columns
                ):
                    # If ML ran and features were engineered, use the ATR from there
                    backtest_input_data[atr_col_name] = test_features_raw[atr_col_name]
                    logger.info(
                        f"Using ATR column '{atr_col_name}' from engineered features for backtest stops."
                    )
                else:
                    # If non-ML or ATR wasn't in engineered features, try to calculate it on the fly
                    logger.warning(
                        f"Required ATR column '{atr_col_name}' not found for dynamic stops. Attempting calculation."
                    )
                    try:
                        # Assuming default ATR period (e.g., 14) if not specified elsewhere for stops
                        atr_period_for_stops = 14  # Make this configurable if needed
                        if PANDAS_TA_AVAILABLE and all(
                            c in backtest_input_data.columns
                            for c in ["high", "low", "close"]
                        ):
                            atr_series = ta.atr(
                                backtest_input_data["high"],
                                backtest_input_data["low"],
                                backtest_input_data["close"],
                                length=atr_period_for_stops,
                            )
                            # Calculate ATRr if that's what atr_col_name expects (e.g., 'ATRr_10')
                            if "ATRr" in atr_col_name:
                                close_prices = backtest_input_data["close"]
                                backtest_input_data[atr_col_name] = (
                                    atr_series / close_prices.replace(0, np.nan)
                                ).fillna(0)
                            else:  # Assume it's just ATR_period
                                backtest_input_data[atr_col_name] = atr_series
                            logger.info(
                                f"Calculated '{atr_col_name}' on the fly for backtest stops."
                            )
                        else:
                            logger.error(
                                "Could not calculate ATR on the fly (missing columns or pandas-ta). Dynamic stops might fail."
                            )
                    except Exception as e_atr:
                        logger.error(
                            f"Error calculating ATR on the fly: {e_atr}. Dynamic stops might fail."
                        )

            step_results = self.backtest_signals(
                signals=signals_df[["signal"]],  # Pass only the final signal column
                ohlc_data=backtest_input_data,  # Pass potentially augmented data
                asset_config=asset_config,
                fold_id=fold_id,
                regime=current_regime,  # Pass regime (None for non-ML)
                regime_actions=self.config.market_regime_config.regime_actions,
            )

            # Log signal counts for debugging
            if signals_df is not None:
                signal_counts = signals_df["signal"].value_counts().to_dict()
                logger.info(f"Step {step} Final Signal Counts: {signal_counts}")
            else:
                # This case should be caught earlier, but log just in case
                logger.warning(
                    f"Step {step}: No signals DataFrame available for logging counts."
                )

            if step_results:
                # Add fold info to results dict
                step_results["fold_number"] = step
                # Use train_data index if ML path was taken, otherwise use calculated indices
                train_start_iso = (
                    train_data.index[0].isoformat()
                    if train_features_raw is not None
                    and isinstance(
                        train_data.index, pd.DatetimeIndex
                    )  # Check if train_features_raw exists (ML path)
                    else full_data.index[train_start_idx].isoformat()
                    if isinstance(full_data.index, pd.DatetimeIndex)
                    else train_start_idx
                )
                train_end_iso = (
                    train_data.index[-1].isoformat()
                    if train_features_raw is not None
                    and isinstance(
                        train_data.index, pd.DatetimeIndex
                    )  # Check if train_features_raw exists (ML path)
                    else full_data.index[train_end_idx - 1].isoformat()
                    if isinstance(full_data.index, pd.DatetimeIndex)
                    else train_end_idx - 1
                )
                step_results["train_start"] = train_start_iso
                step_results["train_end"] = train_end_iso
                step_results["test_start"] = (
                    test_data.index[0].isoformat()
                    if isinstance(test_data.index, pd.DatetimeIndex)
                    else train_end_idx
                )
                step_results["test_end"] = (
                    test_data.index[-1].isoformat()
                    if isinstance(test_data.index, pd.DatetimeIndex)
                    else test_end_idx - 1
                )
                wf_results_list.append(step_results)
                logger.info(
                    f"Step {step} Backtest Summary: Return={step_results.get('total_return_pct', 0):.2f}%, Trades={step_results.get('num_trades', 0)}"
                )

                # --- Collect trades from step ---
                step_trades_path = os.path.join(
                    self.config.output_dir,
                    "backtests",
                    asset_config.symbol,
                    fold_id,
                    f"regime_{current_regime if current_regime is not None else 'all'}",
                    "trades.csv",  # Adjusted path
                )
                if os.path.exists(step_trades_path):
                    try:
                        trades_df = pd.read_csv(step_trades_path)
                        trades_df["fold"] = step
                        all_trades_list.append(trades_df)
                    except Exception as e:
                        logger.warning(
                            f"Could not read trades file for step {step}: {e}"
                        )

            else:
                logger.warning(f"Backtest failed for step {step}.")

            # --- Advance window indices ---
            if wf_config.rolling_window:
                train_start_idx += wf_config.step_periods
            train_end_idx += wf_config.step_periods
            test_end_idx += wf_config.step_periods

            # Check if training window is too small (only relevant for expanding window)
            if (
                not wf_config.rolling_window
                and train_end_idx - train_start_idx < wf_config.min_train_periods
            ):
                logger.warning(
                    f"Training window below minimum size ({wf_config.min_train_periods}). Stopping walk-forward."
                )
                break

            # Check for early stopping based on drawdown (optional)
            if wf_config.early_stopping_drawdown is not None and step_results:
                # Need a way to track overall portfolio drawdown across folds if needed
                # For simplicity, check drawdown within the fold for now
                if (
                    step_results.get("max_drawdown_pct", 0)
                    > wf_config.early_stopping_drawdown * 100
                ):
                    logger.warning(
                        f"Fold {step} max drawdown exceeded early stopping threshold. Stopping walk-forward."
                    )
                    break

        logger.info(
            f"--- Walk-Forward Validation Completed for {asset_config.symbol} ---"
        )

        # Combine results
        wf_summary = self.summarize_walk_forward_results(wf_results_list, asset_config)

        # Combine all trades if collected
        combined_trades_df = (
            pd.concat(all_trades_list, ignore_index=True)
            if all_trades_list
            else pd.DataFrame()
        )
        if not combined_trades_df.empty:
            trades_summary_path = os.path.join(
                self.config.output_dir,
                "walk_forward",
                asset_config.symbol,
                "all_wf_trades.csv",
            )
            try:
                os.makedirs(os.path.dirname(trades_summary_path), exist_ok=True)
                combined_trades_df.to_csv(trades_summary_path, index=False)
                logger.info(
                    f"Combined walk-forward trades saved to {trades_summary_path}"
                )
            except Exception as e:
                logger.error(f"Failed to save combined WF trades: {e}")

        # Plot overall WF performance
        wf_plot_path = os.path.join(
            self.config.output_dir,
            "plots",
            f"{asset_config.symbol}_walk_forward_performance.png",
        )
        self.plot_walk_forward_results(
            wf_results_list,
            combined_trades_df,
            asset_config,
            full_data,
            filename=wf_plot_path,
        )

        # Add combined trades path to summary
        wf_summary["combined_trades_path"] = (
            trades_summary_path if not combined_trades_df.empty else None
        )
        wf_summary["performance_plot_path"] = (
            wf_plot_path if VISUALIZATION_AVAILABLE and wf_results_list else None
        )

        return {"walk_forward_summary": wf_summary}  # Return summary nested under a key

    def summarize_walk_forward_results(
        self, wf_results_list: List[Dict], asset_config: AssetConfig
    ) -> Dict:
        """Summarize results from all walk-forward steps."""
        if not wf_results_list:
            return {"message": "No walk-forward results to summarize."}

        summary_df = pd.DataFrame(wf_results_list)

        # Calculate overall metrics
        overall_metrics = {
            "symbol": asset_config.symbol,
            "num_folds": len(summary_df),
            "avg_return_per_fold_pct": summary_df["total_return_pct"].mean(),
            "std_return_per_fold_pct": summary_df["total_return_pct"].std(),
            "avg_sharpe": summary_df["sharpe_ratio"].mean(),
            "avg_trades_per_fold": summary_df["num_trades"].mean(),
            "avg_win_rate_pct": summary_df["win_rate_pct"].mean(),
            "avg_max_drawdown_pct": summary_df["max_drawdown_pct"].mean(),
            "worst_fold_drawdown_pct": summary_df["max_drawdown_pct"].max(),
            "total_trades": summary_df["num_trades"].sum(),
            "profitable_folds": sum(summary_df["total_return_pct"] > 0),
        }
        overall_metrics["profitable_folds_pct"] = (
            (overall_metrics["profitable_folds"] / overall_metrics["num_folds"]) * 100
            if overall_metrics["num_folds"] > 0
            else 0
        )

        # Calculate compounded return across folds
        compounded_return = (1 + summary_df["total_return_pct"] / 100).prod() - 1
        overall_metrics["compounded_return_pct"] = compounded_return * 100

        # Calculate annualized return if possible
        try:
            start_date = pd.to_datetime(summary_df["test_start"].min())
            end_date = pd.to_datetime(summary_df["test_end"].max())
            years = (end_date - start_date).days / 365.25
            if years > 0:
                overall_metrics["annualized_return_pct"] = (
                    (1 + compounded_return) ** (1 / years) - 1
                ) * 100
                overall_metrics["total_years"] = years
        except Exception as e:
            logger.warning(f"Could not calculate annualized return for WF summary: {e}")

        # Save summary dataframe
        summary_path = os.path.join(
            self.config.output_dir,
            "walk_forward",
            asset_config.symbol,
            "walk_forward_summary.csv",
        )
        try:
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Walk-forward fold details saved to {summary_path}")
        except Exception as e:
            logger.error(f"Failed to save walk-forward fold details: {e}")

        # Save overall summary dict
        summary_dict_path = os.path.join(
            self.config.output_dir,
            "walk_forward",
            asset_config.symbol,
            "walk_forward_overall_summary.json",
        )
        try:
            with open(summary_dict_path, "w") as f:
                json.dump(overall_metrics, f, indent=4, default=str)
            logger.info(f"Walk-forward overall summary saved to {summary_dict_path}")
        except Exception as e:
            logger.error(f"Failed to save walk-forward overall summary: {e}")

        # Add paths to saved files in the returned dict
        overall_metrics["fold_details_path"] = summary_path
        overall_metrics["overall_summary_path"] = summary_dict_path

        return overall_metrics

    def plot_walk_forward_results(
        self,
        wf_results_list: List[Dict],
        combined_trades: pd.DataFrame,  # Pass combined trades
        asset_config: AssetConfig,
        full_data: pd.DataFrame,  # Pass full OHLC data
        filename: Optional[str] = None,
    ):
        """Plot overall walk-forward performance including combined equity."""
        if not VISUALIZATION_AVAILABLE:
            return
        if not wf_results_list:
            return

        logger.info(f"Plotting walk-forward results for {asset_config.symbol}...")

        try:
            summary_df = pd.DataFrame(wf_results_list)
            fig, axes = plt.subplots(
                3,
                1,
                figsize=(15, 15),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1, 1]},
            )

            # --- Plot 1: Equity Curve & Drawdowns (from combined trades) ---
            if not combined_trades.empty and "exit_time" in combined_trades.columns:
                # Calculate equity curve from trades
                trades_sorted = combined_trades.sort_values(by="exit_time")
                # Ensure pnl_abs exists or calculate it
                if "pnl_abs" not in trades_sorted.columns:
                    trades_sorted["pnl_abs"] = (
                        (trades_sorted["exit_price"] - trades_sorted["entry_price"])
                        * trades_sorted["size"]
                        * trades_sorted[
                            "direction"
                        ]  # Assumes direction is 1 for long, -1 for short
                    ) - trades_sorted.get(
                        "commission", 0
                    )  # Subtract commission if available

                # Need initial capital to build equity curve accurately
                initial_cap = self.config.trading_config.initial_capital
                trades_sorted["equity"] = (
                    initial_cap + trades_sorted["pnl_abs"].cumsum()
                )
                # Need to handle timestamps correctly
                trades_sorted["exit_time"] = pd.to_datetime(trades_sorted["exit_time"])
                equity_curve = trades_sorted.set_index("exit_time")["equity"]

                # Reindex to full data range covering the WF period and forward fill
                wf_start_date = pd.to_datetime(summary_df["test_start"].min())
                wf_end_date = pd.to_datetime(summary_df["test_end"].max())
                wf_data_index = full_data.loc[wf_start_date:wf_end_date].index

                full_equity = (
                    equity_curve.reindex(wf_data_index).ffill().fillna(initial_cap)
                )

                plot_equity_curve(
                    full_equity,
                    title=f"{asset_config.symbol} Walk-Forward Equity Curve",
                    ax=axes[0],
                )  # Pass ax
                # Also plot drawdowns on the same axis if function supports it or on a twin axis
                plot_drawdowns(
                    full_equity, ax=axes[0].twinx(), color="red", alpha=0.3
                )  # Example on twin axis

            else:
                axes[0].set_title(
                    f"{asset_config.symbol} Walk-Forward Equity (Trades Missing or Invalid)"
                )
                axes[0].text(
                    0.5,
                    0.5,
                    "No trade data available",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axes[0].transAxes,
                )

            # --- Plot 2: Fold Returns ---
            fold_returns = summary_df["total_return_pct"]
            fold_numbers = summary_df["fold_number"]
            colors = ["green" if r > 0 else "red" for r in fold_returns]
            axes[1].bar(fold_numbers, fold_returns, color=colors, alpha=0.7)
            axes[1].set_ylabel("Fold Return (%)")
            axes[1].set_title("Walk-Forward Fold Returns")
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(0, color="black", linewidth=0.5)

            # --- Plot 3: Rolling Sharpe Ratio (example) ---
            # Calculate rolling Sharpe from fold results
            rolling_window_size = (
                self.config.walkforward_config.performance_tracking_window
            )
            rolling_sharpe = (
                summary_df["sharpe_ratio"]
                .rolling(rolling_window_size, min_periods=1)  # Use min_periods=1
                .mean()
            )
            axes[2].plot(
                summary_df["fold_number"],
                summary_df["sharpe_ratio"],
                label="Fold Sharpe",
                marker=".",
                linestyle=":",
                alpha=0.6,
            )
            axes[2].plot(
                summary_df["fold_number"],
                rolling_sharpe,
                label=f"Rolling Sharpe ({rolling_window_size} folds)",
                color="orange",
            )
            axes[2].set_ylabel("Sharpe Ratio")
            axes[2].set_xlabel("Fold Number")
            axes[2].set_title("Rolling Performance")
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()

            plt.tight_layout(
                rect=[0, 0.03, 1, 0.97]
            )  # Adjust layout to prevent title overlap
            fig.suptitle(
                f"Walk-Forward Performance: {asset_config.symbol}", fontsize=16, y=0.99
            )  # Add overall title

            if filename:
                try:
                    # Ensure the directory exists before saving
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    plt.savefig(filename, dpi=300, bbox_inches="tight")
                    logger.info(f"Saved walk-forward plot to {filename}")
                except Exception as e:
                    logger.error(f"Failed to save walk-forward plot to {filename}: {e}")
            else:
                plt.show()
            plt.close(fig)

        except Exception as e:
            logger.exception(
                f"Error plotting walk-forward results for {asset_config.symbol}: {e}"
            )
            plt.close()  # Ensure plot closed on error

    def run_strategy(self) -> Dict:
        """Run the full strategy workflow for all configured assets."""
        logger.info(
            f"--- Running Strategy: {self.config.strategy_name} (ID: {self.config.strategy_run_id}) ---"
        )
        self.config.save_config()  # Save config for this run

        all_results = {}

        # Use ThreadPoolExecutor for parallel asset processing if enabled
        # Note: Parallelism might be complex with shared resources like data cache or log files.
        # Consider potential race conditions or use ProcessPoolExecutor if heavy CPU tasks.
        use_parallel = self.config.parallel_processing and len(self.config.assets) > 1
        max_workers = os.cpu_count() if use_parallel else 1

        logger.info(
            f"Parallel Processing: {'Enabled' if use_parallel else 'Disabled'} (Max Workers: {max_workers})"
        )

        # Define function for single run (used in both parallel and sequential)
        def run_single_asset(asset_cfg):
            logger.info(f"Starting single run for {asset_cfg.symbol}")
            data = self.load_data(asset_cfg)
            if data is None:
                return {"status": "error", "message": "Data loading failed"}

            signals = None
            model_path = None  # Initialize model_path
            current_regime = None  # Initialize regime
            features = None  # Initialize features
            target = None  # Initialize target
            predictions = None  # Initialize predictions
            backtest_results = None  # Initialize backtest_results

            # --- Check Strategy Type and Generate Signals ---
            if self.config.strategy_type == "dma_crossover":
                logger.info(f"Running DMA Crossover strategy for {asset_cfg.symbol}")
                signals = self.generate_dma_signals(data, asset_cfg)
                model_path = "DMA_Strategy"  # Placeholder identifier
                if signals is None:
                    return {
                        "status": "error",
                        "message": "DMA Signal generation failed",
                    }

            elif self.config.strategy_type == "rsi_mean_reversion":
                logger.info(
                    f"Running RSI Mean Reversion strategy for {asset_cfg.symbol}"
                )
                signals = self.generate_rsi_signals(data, asset_cfg)
                model_path = "RSI_Strategy"  # Placeholder identifier
                if signals is None:
                    # Corrected error message for RSI failure
                    return {
                        "status": "error",
                        "message": "RSI Signal generation failed",
                    }

            else:  # Default to ML workflow (or handle other types like 'ml')
                logger.info(f"Running ML strategy for {asset_cfg.symbol}")
                features, target = self.engineer_features(data, asset_cfg)
                # Check both features and target after engineering for ML
                if features is None or target is None:
                    return {"status": "error", "message": "Feature engineering failed"}

                # --- Detect Market Regime (based on full features) ---
                current_regime = self._detect_regime_for_data(
                    features, asset_cfg.symbol
                )

                # --- Train Model (Only for ML workflow) ---
                model_path = self.train_model(
                    features, target, asset_cfg, regime=current_regime
                )
                if model_path is None:
                    return {"status": "error", "message": "Model training failed"}

                # --- Generate Predictions (Only for ML workflow) ---
                predictions = self.generate_predictions(
                    features, asset_cfg, regime=current_regime
                )
                if predictions is None:
                    return {"status": "error", "message": "Prediction failed"}

                # --- Generate Signals from Predictions (Only for ML workflow) ---
                signals = self.generate_signals(
                    predictions,
                    data,  # Pass original OHLCV data for filters
                    asset_cfg,
                    regime=current_regime,
                    regime_actions=self.config.market_regime_config.regime_actions,
                )
                if signals is None:
                    return {
                        "status": "error",
                        "message": "Signal generation from predictions failed",
                    }
                # End of ML-specific workflow path

            # --- Backtest Signals (Common for all strategy types) ---
            if signals is None:
                # This case should ideally be caught earlier, but double-check
                return {
                    "status": "error",
                    "message": "Signals are None before backtesting.",
                }

            # Prepare data for backtest - needs OHLCV and potentially ATR
            # If ML ran, features might contain ATR. If non-ML, data is just OHLCV.
            backtest_input_data = data.copy()
            trade_cfg = self.config.trading_config
            atr_col_name = trade_cfg.atr_col
            if features is not None and atr_col_name in features.columns:
                # Add ATR from ML features if available and needed
                backtest_input_data[atr_col_name] = features[atr_col_name]
            elif trade_cfg.use_dynamic_stops and atr_col_name:
                # If non-ML and dynamic stops are used, ATR needs to be calculated or handled
                logger.warning(
                    f"Dynamic stops enabled but ATR column '{atr_col_name}' may be missing for {self.config.strategy_type} strategy. Attempting calculation."
                )
                # Optionally calculate ATR here if needed by backtester
                try:
                    atr_period_for_stops = 14  # Make this configurable if needed
                    if PANDAS_TA_AVAILABLE and all(
                        c in backtest_input_data.columns
                        for c in ["high", "low", "close"]
                    ):
                        atr_series = ta.atr(
                            backtest_input_data["high"],
                            backtest_input_data["low"],
                            backtest_input_data["close"],
                            length=atr_period_for_stops,
                        )
                        if "ATRr" in atr_col_name:
                            close_prices = backtest_input_data["close"]
                            backtest_input_data[atr_col_name] = (
                                atr_series / close_prices.replace(0, np.nan)
                            ).fillna(0)
                        else:
                            backtest_input_data[atr_col_name] = atr_series
                        logger.info(
                            f"Calculated '{atr_col_name}' on the fly for backtest stops."
                        )
                    else:
                        logger.error(
                            "Could not calculate ATR on the fly (missing columns or pandas-ta). Dynamic stops might fail."
                        )
                except Exception as e_atr:
                    logger.error(
                        f"Error calculating ATR on the fly: {e_atr}. Dynamic stops might fail."
                    )

            backtest_results = self.backtest_signals(
                signals,
                backtest_input_data,  # Use potentially augmented data
                asset_cfg,
                regime=current_regime,  # Pass regime (None for non-ML)
                regime_actions=self.config.market_regime_config.regime_actions,
            )

            return {
                "status": "completed"
                if backtest_results
                else "error",  # Set status based on backtest success
                "backtest_summary": backtest_results,
                "model_path": model_path,
                "detected_regime": (
                    int(current_regime) if current_regime is not None else None
                ),
            }

        if use_parallel:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                future_to_asset = {}
                for asset_config in self.config.assets:
                    logger.info(f"Submitting job for asset: {asset_config.symbol}")
                    # Submit job based on strategy type and walk-forward config
                    if (
                        self.config.walkforward_config.enabled
                        and self.config.strategy_type != "dma_crossover"
                        and self.config.strategy_type != "rsi_mean_reversion"
                    ):
                        # Run walk-forward only for ML-like strategies if enabled
                        future = executor.submit(
                            self.run_walk_forward_validation, asset_config
                        )
                    else:
                        # Run single asset backtest for DMA/RSI or if walk-forward disabled
                        future = executor.submit(run_single_asset, asset_config)
                    future_to_asset[future] = asset_config.symbol

                # Collect results from parallel execution
                for future in concurrent.futures.as_completed(future_to_asset):
                    asset_symbol = future_to_asset[future]
                    try:
                        result = future.result()  # Add timeout?
                        all_results[asset_symbol] = result
                        status = (
                            result.get("status", "completed")
                            if isinstance(result, dict)
                            else "completed"
                        )
                        logger.info(
                            f"Job finished for {asset_symbol}. Status: {status}"
                        )
                    except Exception as exc:
                        logger.exception(
                            f"Asset {asset_symbol} generated an exception during execution: {exc}"
                        )
                        all_results[asset_symbol] = {
                            "status": "error",
                            "message": str(exc),
                        }
        else:
            # Execute sequentially
            for asset_config in self.config.assets:
                asset_symbol = asset_config.symbol
                logger.info(f"Running job sequentially for asset: {asset_symbol}")
                try:
                    # Run job based on strategy type and walk-forward config
                    if (
                        self.config.walkforward_config.enabled
                        and self.config.strategy_type != "dma_crossover"
                        and self.config.strategy_type != "rsi_mean_reversion"
                    ):
                        # Run walk-forward only for ML-like strategies if enabled
                        result = self.run_walk_forward_validation(asset_config)
                    else:
                        # Run single asset backtest for DMA/RSI or if walk-forward disabled
                        result = run_single_asset(asset_config)

                    all_results[asset_symbol] = result
                    status = (
                        result.get("status", "completed")
                        if isinstance(result, dict)
                        else "completed"
                    )
                    logger.info(f"Job finished for {asset_symbol}. Status: {status}")
                except Exception as exc:
                    logger.exception(
                        f"Asset {asset_symbol} generated an exception during sequential execution: {exc}"
                    )
                    all_results[asset_symbol] = {"status": "error", "message": str(exc)}

        # --- Final Summary and Plots ---
        logger.info("--- Strategy Run Summary ---")
        final_summary = self.summarize_strategy_results(all_results)

        # Save final summary
        summary_path = os.path.join(self.config.output_dir, "strategy_summary.json")
        try:
            with open(summary_path, "w") as f:
                json.dump(final_summary, f, indent=4, default=str)
            logger.info(f"Final strategy summary saved to {summary_path}")
        except Exception as e:
            logger.error(f"Failed to save final summary: {e}")

        # Plot overall performance (e.g., combined equity if meaningful)
        self.plot_strategy_performance(final_summary, all_results)

        logger.info(f"--- Strategy Run Finished: {self.config.strategy_name} ---")
        return final_summary

    def summarize_strategy_results(self, all_results: Dict) -> Dict:
        """Aggregate results across all assets/runs."""
        logger.info("Aggregating results across all assets...")
        aggregated_summary = {
            "strategy_name": self.config.strategy_name,
            "strategy_run_id": self.config.strategy_run_id,
            "timestamp": datetime.now().isoformat(),
            "assets": {},
            "overall_summary": {},  # Initialize overall summary dict
        }
        overall_metrics_list = []  # Store metrics per asset for averaging

        for asset, result in all_results.items():
            if isinstance(result, dict):
                # Store individual asset result (could be WF summary or single backtest summary)
                aggregated_summary["assets"][asset] = result
                # Determine if it was a walk-forward run or single backtest
                summary_data = None
                if "walk_forward_summary" in result:
                    summary_data = result["walk_forward_summary"]
                elif "backtest_summary" in result:
                    summary_data = result["backtest_summary"]

                # Extract key performance metrics if the run was successful
                if summary_data and result.get("status") != "error":
                    # Extract metrics, preferring WF metrics if available
                    metrics = {
                        "asset": asset,
                        "return": summary_data.get("annualized_return_pct")
                        or summary_data.get("compounded_return_pct")
                        or summary_data.get("total_return_pct"),
                        "sharpe": summary_data.get("sharpe_ratio")
                        or summary_data.get("avg_sharpe"),
                        "max_dd": summary_data.get("max_drawdown_pct")
                        or summary_data.get("worst_fold_drawdown_pct"),
                        "win_rate": summary_data.get("win_rate_pct")
                        or summary_data.get("avg_win_rate_pct"),
                        "trades": summary_data.get("num_trades")
                        or summary_data.get("total_trades"),
                    }
                    # Add only if essential metrics are present
                    if metrics["return"] is not None and metrics["sharpe"] is not None:
                        overall_metrics_list.append(metrics)
                    else:
                        logger.warning(
                            f"Missing key metrics (return/sharpe) for asset {asset}. Excluding from overall average."
                        )

            else:
                aggregated_summary["assets"][asset] = {
                    "status": "error",
                    "message": "Invalid result format",
                }

        # Calculate overall portfolio metrics if possible
        if overall_metrics_list:
            metrics_df = (
                pd.DataFrame(overall_metrics_list).set_index("asset").dropna()
            )  # Drop assets with NaN metrics for averaging
            if not metrics_df.empty:
                aggregated_summary["overall_summary"] = {
                    "avg_annualized_return_pct": metrics_df["return"].mean(),
                    "avg_sharpe_ratio": metrics_df["sharpe"].mean(),
                    "avg_max_drawdown_pct": metrics_df["max_dd"].mean(),
                    "avg_win_rate_pct": metrics_df["win_rate"].mean(),
                    "total_trades": metrics_df["trades"].sum(),
                    "num_assets_aggregated": len(metrics_df),
                }
                logger.info(
                    f"Overall Average Metrics (across {len(metrics_df)} assets): "
                    f"Return={aggregated_summary['overall_summary']['avg_annualized_return_pct']:.2f}%, "
                    f"Sharpe={aggregated_summary['overall_summary']['avg_sharpe_ratio']:.2f}"
                )
                logger.debug(f"Per-asset metrics used for aggregation:\n{metrics_df}")
            else:
                logger.warning(
                    "No valid metrics found across assets for overall summary."
                )
                aggregated_summary["overall_summary"] = {
                    "message": "No valid metrics to aggregate"
                }

        else:
            logger.warning("Could not calculate overall average metrics.")
            aggregated_summary["overall_summary"] = {
                "message": "No valid asset results"
            }

        return aggregated_summary

    def plot_strategy_performance(self, overall_summary: Dict, all_results: Dict):
        """Plot overall strategy performance."""
        # Placeholder: Requires combined equity curve or aggregated metrics plot
        # Example: Plot distribution of returns across assets
        if not VISUALIZATION_AVAILABLE:
            return
        logger.info("Plotting overall strategy performance distribution...")
        try:
            asset_returns = []
            for asset, res in all_results.items():
                if isinstance(res, dict) and res.get("status") != "error":
                    summary_data = res.get("walk_forward_summary") or res.get(
                        "backtest_summary"
                    )
                    if summary_data:
                        ret = (
                            summary_data.get("annualized_return_pct")
                            or summary_data.get("compounded_return_pct")
                            or summary_data.get("total_return_pct")
                        )
                        if ret is not None:
                            asset_returns.append(ret)

            if not asset_returns:
                logger.warning("No asset returns to plot.")
                return

            plt.figure(figsize=(10, 6))
            sns.histplot(asset_returns, kde=True)
            plt.title(
                f"Distribution of Returns Across Assets ({self.config.strategy_name})"  # Use Annualized if available, otherwise best available
            )
            plt.xlabel("Return (%)")  # Generic label
            plt.ylabel("Frequency")
            plt.grid(alpha=0.3)
            plot_path = os.path.join(
                self.config.output_dir, "plots", "strategy_return_distribution.png"
            )
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved return distribution plot to {plot_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting strategy performance: {e}")
            plt.close()  # Ensure plot closed on error
