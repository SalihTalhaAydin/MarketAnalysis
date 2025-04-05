"""
Enhanced Main Strategy Module

This module orchestrates the entire trading strategy workflow including:
1. Market data loading and preprocessing
2. Feature engineering with regime detection
3. Model training with adaptive parameter optimization
4. Signal generation with adaptive risk management
5. Backtesting with advanced performance analysis
6. Walk-forward validation with regime-adaptive retraining

The implementation supports multiple assets, timeframes, and strategies.
"""

import concurrent.futures
import json
import logging
import os
import uuid
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        # Keep logging to file if desired, adjust path as needed
        # logging.FileHandler("trading_strategy.log"),
        logging.StreamHandler()  # Log to console
    ],
)
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


# Import refactored components using relative paths
try:
    # Data Loading
    from ..data.loaders import DataLoader, DataLoaderConfig, load_data

    # Feature Engineering
    from ..features.features_engineering import engineer_features
    from ..models.evaluation.metrics import (
        generate_model_report,
    )  # Added report generation
    from ..models.prediction import ModelPredictorBase  # Added load_model
    from ..models.prediction import get_confidence_levels  # Added import
    from ..models.prediction import predict_with_threshold  # Added import
    from ..models.prediction import (
        PredictionManager,
        SignalGenerator,
        load_model,
        predict_with_model,
    )

    # Model Training & Prediction
    from ..models.training import (
        train_classification_model,
    )  # Added pipeline creation; Removed unused create_feature_pipeline

    # Backtesting & Simulation
    from ..trading.backtest import backtest_strategy
    from ..trading.simulation import TradeManager

    # Utils (potentially needed for regime detection or other parts)
    from ..utils.metrics import calculate_returns_metrics  # Keep for potential use
    from ..utils.visualization import (  # Keep for potential use
        plot_drawdowns,
        plot_equity_curve,
        plot_monthly_returns,
    )

    # Import clustering if needed for regime detection
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        SKLEARN_AVAILABLE_FOR_REGIME = True
    except ImportError:
        logger.warning("scikit-learn not available for clustering regime detection.")
        SKLEARN_AVAILABLE_FOR_REGIME = False

        class KMeans:
            pass  # Dummy class

        class StandardScaler:
            pass  # Dummy class

    MODULES_AVAILABLE = True
except ImportError as e:
    # Fallback for running script directly (less ideal)
    logger.error(
        f"Error importing strategy components: {e}. Relative imports failed. Ensure running as part of the package."
    )
    MODULES_AVAILABLE = False

    # Add dummy classes/functions if needed for script execution without full package install
    class DataLoader:
        pass

    class DataLoaderConfig:
        pass

    def load_data(*args, **kwargs):
        return None

    def engineer_features(*args, **kwargs):
        return None

    def train_classification_model(*args, **kwargs):
        return None, None, None

    class ModelPredictorBase:
        pass

    class PredictionManager:
        pass

    class SignalGenerator:
        pass

    def predict_with_model(*args, **kwargs):
        return None

    def load_model(*args, **kwargs):
        return None, {}

    def generate_model_report(*args, **kwargs):
        return {}

    def backtest_strategy(*args, **kwargs):
        return {}

    class TradeManager:
        pass

    def calculate_returns_metrics(*args, **kwargs):
        return {}

    def plot_equity_curve(*args, **kwargs):
        pass

    def plot_monthly_returns(*args, **kwargs):
        pass

    def plot_drawdowns(*args, **kwargs):
        pass

    class KMeans:
        pass

    class StandardScaler:
        pass

    SKLEARN_AVAILABLE_FOR_REGIME = False


# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Strategy Configuration Classes ---


class AssetConfig:
    """Configuration for a tradable asset."""

    def __init__(
        self,
        symbol: str,
        timeframe: str = "1d",
        data_source: str = "yahoo",  # Default to yahoo
        commission_pct: float = 0.001,  # 0.1%
        slippage_pct: float = 0.0005,  # 0.05%
        min_position_size: float = 0.01,  # Minimum trade size (e.g., for crypto)
        max_position_size: float = 1.0,  # Max position as fraction of capital
        correlation_group: Optional[str] = None,  # For risk management
    ):
        """
        Initialize asset configuration.

        Args:
            symbol: Asset symbol (e.g., 'AAPL', 'BTC/USDT')
            timeframe: Data timeframe (e.g., '1d', '1h')
            data_source: Data source identifier (e.g., 'yahoo', 'crypto', 'csv')
            commission_pct: Commission percentage per trade side
            slippage_pct: Slippage percentage per trade side
            min_position_size: Minimum position size in units (e.g., shares, contracts)
            max_position_size: Maximum position size as fraction of capital
            correlation_group: Group name for correlated assets
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_source = data_source
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        self.correlation_group = correlation_group

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return self.__dict__  # Simpler conversion

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "AssetConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)


# --- Nested Config Classes for FeatureConfig ---


class DifferencingConfig:
    """Configuration for feature differencing."""

    def __init__(
        self, apply: bool = False, period: int = 1, features: Optional[List[str]] = None
    ):
        self.apply = apply
        self.period = period
        self.features = features or []  # List of feature names to difference

    def to_dict(self) -> Dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "DifferencingConfig":
        return cls(**config_dict)


class ScalingConfig:
    """Configuration for feature scaling."""

    def __init__(self, method: Optional[str] = "StandardScaler"):
        # Validate method?
        valid_methods = ["StandardScaler", "MinMaxScaler", "RobustScaler", "None", None]
        if method not in valid_methods:
            logger.warning(
                f"Invalid scaling method '{method}'. Defaulting to StandardScaler."
            )
            method = "StandardScaler"
        self.method = (
            method if method != "None" else None
        )  # Store None if 'None' string is used

    def to_dict(self) -> Dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ScalingConfig":
        return cls(**config_dict)


# --- Main Feature Configuration ---


class FeatureConfig:
    """Configuration for feature engineering."""

    def __init__(
        self,
        # List of specific technical indicators and their parameters
        technical_indicators: Optional[List[Dict[str, Any]]] = None,
        # Other feature categories (can keep as booleans or make more detailed)
        volatility_features: bool = True,
        momentum_features: bool = True,
        pattern_features: bool = False,
        fractal_features: bool = False,
        time_features: bool = True,
        regime_features: bool = True,
        # Target variable config (if calculated here)
        atr_multiplier_tp: float = 2.0,
        atr_multiplier_sl: float = 1.0,
        max_holding_period: int = 10,
        target_type: str = "triple_barrier",  # 'triple_barrier', 'directional'
        # Feature selection / reduction
        feature_selection_enabled: bool = True,
        feature_selection_method: str = "importance",
        pca_enabled: bool = False,
        max_features: int = 50,
        # New Transformation Configs
        differencing: Optional[Dict] = None,  # Accept dict from YAML
        scaling: Optional[Dict] = None,  # Accept dict from YAML
    ):
        # Store the list of indicator configs, default to empty list if None
        self.technical_indicators = technical_indicators or []
        self.volatility_features = volatility_features
        self.momentum_features = momentum_features
        self.pattern_features = pattern_features
        self.fractal_features = fractal_features
        self.time_features = time_features
        self.regime_features = regime_features
        self.atr_multiplier_tp = atr_multiplier_tp
        self.atr_multiplier_sl = atr_multiplier_sl
        self.max_holding_period = max_holding_period
        self.target_type = target_type
        self.feature_selection_enabled = feature_selection_enabled
        self.feature_selection_method = feature_selection_method
        self.pca_enabled = pca_enabled
        self.max_features = max_features

        # Instantiate nested config objects from input dicts
        self.differencing = DifferencingConfig.from_dict(differencing or {})
        self.scaling = ScalingConfig.from_dict(scaling or {})

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        # Convert nested objects to dicts for serialization
        data = self.__dict__.copy()
        data["differencing"] = (
            self.differencing.to_dict() if self.differencing else None
        )
        data["scaling"] = self.scaling.to_dict() if self.scaling else None
        return data

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "FeatureConfig":
        """Create configuration from dictionary."""
        # Handle nested dicts when creating from dict
        diff_cfg_dict = config_dict.pop("differencing", {})
        scale_cfg_dict = config_dict.pop("scaling", {})

        # Create instance with remaining args
        instance = cls(**config_dict)

        # Manually create nested objects (or pass dicts if __init__ handles it)
        instance.differencing = DifferencingConfig.from_dict(diff_cfg_dict)
        instance.scaling = ScalingConfig.from_dict(scale_cfg_dict)

        return instance


class ModelConfig:
    """Configuration for model training and prediction."""

    def __init__(
        self,
        model_type: str = "lightgbm",  # Default model
        ensemble_models: Optional[
            List[str]
        ] = None,  # List of models for ensemble, None for single model
        optimize_hyperparams: bool = True,
        optimization_method: str = "random",  # 'grid', 'random', 'bayesian'
        optimization_trials: int = 25,  # Number of trials/iterations for random/bayesian
        cv_folds: int = 5,  # Folds for cross-validation during optimization
        validation_size: float = 0.2,  # Hold-out validation set size during training
        scoring_metric: str = "f1_weighted",  # Metric for optimization/evaluation
        early_stopping_rounds: Optional[
            int
        ] = 50,  # For models that support it (e.g., XGB, LGBM)
        # Prediction / Signal Generation
        probability_threshold: float = 0.60,  # Threshold for converting prob to signal
        signal_neutral_zone: Tuple[float, float] = (
            0.45,
            0.55,
        ),  # Prob range for neutral signal
        signal_trend_filter_ma: Optional[int] = 50,  # MA period for trend filter
        signal_volatility_filter_atr: Optional[int] = 14,  # ATR period for vol filter
        signal_cooling_period: int = 3,  # Bars to wait between signals
        # Risk Management (can also be part of BacktestConfig)
        risk_per_trade: float = 0.02,
        use_kelly_sizing: bool = True,
        max_drawdown_pct: float = 0.20,  # Portfolio level drawdown limit
        max_open_trades: int = 5,
        max_correlation_exposure: int = 2,  # Max trades in same correlation group
        # Regime Adaptation
        regime_adaptation_enabled: bool = True,  # Use regime detection?
        regime_models: Optional[
            Dict[int, str]
        ] = None,  # Optional: map regime -> model_type
    ):
        self.model_type = model_type
        self.ensemble_models = ensemble_models or []
        self.optimize_hyperparams = optimize_hyperparams
        self.optimization_method = optimization_method
        self.optimization_trials = optimization_trials
        self.cv_folds = cv_folds
        self.validation_size = validation_size
        self.scoring_metric = scoring_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.probability_threshold = probability_threshold
        self.signal_neutral_zone = signal_neutral_zone
        self.signal_trend_filter_ma = signal_trend_filter_ma
        self.signal_volatility_filter_atr = signal_volatility_filter_atr
        self.signal_cooling_period = signal_cooling_period
        self.risk_per_trade = risk_per_trade
        self.use_kelly_sizing = use_kelly_sizing
        self.max_drawdown_pct = max_drawdown_pct
        self.max_open_trades = max_open_trades
        self.max_correlation_exposure = max_correlation_exposure
        self.regime_adaptation_enabled = regime_adaptation_enabled
        self.regime_models = regime_models or {}

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ModelConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)


class WalkForwardConfig:
    """Configuration for walk-forward validation."""

    def __init__(
        self,
        enabled: bool = True,
        initial_train_periods: int = 1000,  # Number of periods (days, hours)
        test_periods: int = 200,
        step_periods: int = 200,
        min_train_periods: int = 800,
        retrain_frequency: int = 1,  # Retrain every N steps
        rolling_window: bool = False,  # Expanding window by default
        preserve_model_history: bool = True,  # Save models from each step
        early_stopping_drawdown: Optional[
            float
        ] = 0.25,  # Stop WF if portfolio DD exceeds this
        performance_tracking_window: int = 10,  # Steps to track recent performance
    ):
        self.enabled = enabled
        self.initial_train_periods = initial_train_periods
        self.test_periods = test_periods
        self.step_periods = step_periods
        self.min_train_periods = min_train_periods
        self.retrain_frequency = retrain_frequency
        self.rolling_window = rolling_window
        self.preserve_model_history = preserve_model_history
        self.early_stopping_drawdown = early_stopping_drawdown
        self.performance_tracking_window = performance_tracking_window

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "WalkForwardConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)


class TradingConfig:
    """Configuration for trading execution and risk management."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        position_sizing_method: str = "kelly",  # 'kelly' or 'fixed_fractional'
        fixed_fraction_amount: float = 0.02,  # Used if method is 'fixed_fractional'
        max_risk_per_trade: float = 0.02,  # Max risk if using Kelly or other risk-based sizing
        max_capital_per_trade: float = 0.25,  # Max capital allocation per trade
        max_open_trades: int = 5,
        max_drawdown_pct: float = 0.20,  # Portfolio level drawdown limit
        max_correlation_exposure: int = 2,  # Max trades in same correlation group
        # Stop/Profit settings (can be here or in FeatureConfig/ModelConfig depending on use)
        use_dynamic_stops: bool = True,
        atr_multiplier_sl: float = 1.0,
        atr_multiplier_tp: float = 2.0,
        atr_col: str = "ATRr_10",  # Default ATR column name
        trailing_stop_pct: Optional[float] = 0.01,
        trailing_stop_activation_pct: float = 0.01,
        allow_pyramiding: bool = False,
        commission_bps: float = 2.0,  # Basis points (e.g., 2.0 = 0.02%)
        slippage_bps: float = 1.0,  # Basis points (e.g., 1.0 = 0.01%)
    ):
        self.initial_capital = initial_capital
        self.position_sizing_method = position_sizing_method
        self.fixed_fraction_amount = fixed_fraction_amount
        self.max_risk_per_trade = max_risk_per_trade
        self.max_capital_per_trade = max_capital_per_trade
        self.max_open_trades = max_open_trades
        self.max_drawdown_pct = max_drawdown_pct
        self.max_correlation_exposure = max_correlation_exposure
        self.use_dynamic_stops = use_dynamic_stops
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp
        self.atr_col = atr_col
        self.trailing_stop_pct = trailing_stop_pct
        self.trailing_stop_activation_pct = trailing_stop_activation_pct
        self.allow_pyramiding = allow_pyramiding
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.commission_pct = commission_bps / 10000.0  # Convert bps to percentage
        self.slippage_pct = slippage_bps / 10000.0  # Convert bps to percentage

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "TradingConfig":
        """Create configuration from dictionary."""
        # Handle nested position sizing if present in YAML
        pos_sizing_config = config_dict.pop("position_sizing", {})
        config_dict.setdefault(
            "position_sizing_method", pos_sizing_config.get("method", "kelly")
        )
        config_dict.setdefault(
            "fixed_fraction_amount", pos_sizing_config.get("fraction", 0.02)
        )
        # Handle nested stop loss if present in YAML
        stop_loss_config = config_dict.pop("stop_loss", {})
        if isinstance(stop_loss_config, dict) and isinstance(
            stop_loss_config.get("params"), dict
        ):
            sl_multiplier = stop_loss_config["params"].get("multiplier")
            if sl_multiplier is not None:
                config_dict["atr_multiplier_sl"] = sl_multiplier

        # Handle nested take profit if present in YAML
        take_profit_config = config_dict.pop("take_profit", {})
        if isinstance(take_profit_config, dict) and isinstance(
            take_profit_config.get("params"), dict
        ):
            tp_multiplier = take_profit_config["params"].get("multiplier")
            if tp_multiplier is not None:
                config_dict["atr_multiplier_tp"] = tp_multiplier

        return cls(**config_dict)


# --- Market Regime Configuration ---


class MarketRegimeConfig:
    """Configuration for market regime detection and adaptation."""

    def __init__(
        self,
        enabled: bool = False,
        method: Optional[str] = None,  # e.g., 'volatility_clustering'
        params: Optional[Dict[str, Any]] = None,  # Method-specific parameters
        regime_actions: Optional[
            Dict[int, str]
        ] = None,  # Actions per regime (e.g., {0: 'no_trade'})
    ):
        self.enabled = enabled
        self.method = method
        self.params = params or {}
        self.regime_actions = regime_actions or {}

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "MarketRegimeConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)


class StrategyConfig:
    """Main configuration container for the trading strategy."""

    def __init__(
        self,
        strategy_name: str = "Enhanced ML Strategy",
        description: str = "Machine learning strategy with regime adaptation and walk-forward validation",
        data_start_date: str = "2018-01-01",  # Longer history
        data_end_date: Optional[str] = None,  # Default to now
        assets: Optional[List[AssetConfig]] = None,  # List of assets
        feature_config: Optional[FeatureConfig] = None,
        model_config: Optional[
            ModelConfig
        ] = None,  # Model training/prediction specific
        trading_config: Optional[TradingConfig] = None,  # Trading execution specific
        walkforward_config: Optional[WalkForwardConfig] = None,
        market_regime_config: Optional[
            MarketRegimeConfig
        ] = None,  # Added Market Regime Config
        output_dir: str = "strategy_results",
        parallel_processing: bool = True,  # Use multiprocessing where possible
        random_state: int = 42,
        debug_mode: bool = False,  # More verbose logging if True
        search_space: Optional[
            Dict
        ] = None,  # Added for hyperparameter optimization search space
    ):
        self.strategy_name = strategy_name
        # Generate unique ID for this run
        self.strategy_run_id = f"{strategy_name.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:4]}"
        self.description = description
        self.data_start_date = data_start_date
        self.data_end_date = data_end_date or datetime.now().strftime("%Y-%m-%d")
        self.assets = assets or [AssetConfig("SPY")]  # Default to SPY if none provided
        self.feature_config = feature_config or FeatureConfig()
        self.model_config = model_config or ModelConfig()
        self.trading_config = trading_config or TradingConfig()
        self.walkforward_config = walkforward_config or WalkForwardConfig()
        self.market_regime_config = (
            market_regime_config or MarketRegimeConfig()
        )  # Initialize MarketRegimeConfig
        # Ensure output dir includes the unique run ID
        self.output_dir = os.path.join(output_dir, self.strategy_run_id)
        self.parallel_processing = parallel_processing
        self.random_state = random_state
        self.debug_mode = debug_mode
        self.search_space = search_space

        # Set random seeds for reproducibility
        np.random.seed(random_state)
        # Add seeds for other libraries if used (tf, torch, etc.)

        # Create output directory
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Strategy output directory: {self.output_dir}")
        except OSError as e:
            logger.error(f"Failed to create output directory {self.output_dir}: {e}")
            # Potentially raise error or disable saving

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "strategy_run_id": self.strategy_run_id,
            "description": self.description,
            "data_start_date": self.data_start_date,
            "data_end_date": self.data_end_date,
            "assets": [asset.to_dict() for asset in self.assets],
            "feature_config": self.feature_config.to_dict(),
            "model_config": self.model_config.to_dict(),
            "trading_config": self.trading_config.to_dict(),
            "walkforward_config": self.walkforward_config.to_dict(),
            "market_regime": self.market_regime_config.to_dict(),  # Serialize MarketRegimeConfig
            "output_dir": self.output_dir,  # Save the specific run dir
            "parallel_processing": self.parallel_processing,
            "random_state": self.random_state,
            "debug_mode": self.debug_mode,
            "search_space": self.search_space,  # Serialize search_space if needed
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "StrategyConfig":
        """Create configuration from dictionary."""
        # Process nested configurations
        assets_list = config_dict.pop("assets", [])
        assets = [AssetConfig.from_dict(asset) for asset in assets_list]

        feature_config_dict = config_dict.pop("feature_config", {})
        feature_config = (
            FeatureConfig.from_dict(feature_config_dict)
            if feature_config_dict
            else FeatureConfig()
        )

        model_config_dict = config_dict.pop("model_config", {})
        # Extract search_space before passing to ModelConfig
        search_space = model_config_dict.pop("search_space", None)
        model_config = (
            ModelConfig.from_dict(model_config_dict)
            if model_config_dict
            else ModelConfig()
        )

        walkforward_config_dict = config_dict.pop("walkforward_config", {})
        walkforward_config = (
            WalkForwardConfig.from_dict(walkforward_config_dict)
            if walkforward_config_dict
            else WalkForwardConfig()
        )

        trading_config_dict = config_dict.pop("trading", {})
        trading_config = (
            TradingConfig.from_dict(trading_config_dict)
            if trading_config_dict
            else TradingConfig()
        )

        trading_config_dict = config_dict.pop("trading", {})
        trading_config = (
            TradingConfig.from_dict(trading_config_dict)
            if trading_config_dict
            else TradingConfig()
        )

        # Pop market regime section and create config object
        market_regime_dict = config_dict.pop("market_regime", {})
        market_regime_config = (
            MarketRegimeConfig.from_dict(market_regime_dict)
            if market_regime_dict
            else MarketRegimeConfig()
        )

        # Create main configuration, remove run_id if present as it's generated
        config_dict.pop("strategy_run_id", None)

        # Remove other top-level keys from YAML that are not direct __init__ args
        # These were used to structure the YAML but their contents are handled
        # by the nested config objects created above.
        config_dict.pop("features", None)
        config_dict.pop("model", None)  # Note: model_config was handled above
        config_dict.pop(
            "walk_forward", None
        )  # Note: walkforward_config was handled above
        # config_dict.pop("trading", None) # Now handled by TradingConfig above
        # market_regime_config = config_dict.pop("market_regime", {}) # Handled above
        config_dict.pop("log_level", None)  # Handled separately by logging setup
        # 'results_dir' is used to construct self.output_dir, keep it for now if needed, or pop if handled differently
        # Let's assume 'output_dir' in __init__ handles this, so pop 'results_dir'
        config_dict.pop("results_dir", None)  # Pop if 'output_dir' is the intended arg
        # Pop keys used for YAML structure but not direct __init__ args
        config_dict.pop("differencing", None)
        config_dict.pop("scaling", None)

        # Create the main instance first
        instance = cls(
            assets=assets,
            feature_config=feature_config,
            model_config=model_config,
            trading_config=trading_config,
            walkforward_config=walkforward_config,
            market_regime_config=market_regime_config,  # Pass MarketRegimeConfig instance
            search_space=search_space,
            **config_dict,
        )

        # Market regime attributes are now part of instance.market_regime_config

        return instance

    def save_config(self, filename: Optional[str] = None) -> str:
        """
        Save configuration to JSON file within the strategy run directory.

        Args:
            filename: Optional base filename (run_id and .json added automatically)

        Returns:
            Path to saved configuration file
        """
        base_filename = filename or "strategy_config"
        # Ensure filename ends with .json
        if not base_filename.endswith(".json"):
            base_filename += ".json"

        filepath = os.path.join(self.output_dir, base_filename)

        try:
            # Get config dict (to_dict now includes market_regime correctly)
            config_data = self.to_dict()
            with open(filepath, "w") as f:
                json.dump(config_data, f, indent=4)
            logger.info(f"Saved strategy configuration to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save config to {filepath}: {e}")
            return ""

    @classmethod
    def load_config(cls, filepath: str) -> Optional["StrategyConfig"]:
        """
        Load configuration from YAML file.

        Args:
            filepath: Path to configuration file (e.g., 'configs/my_config.yaml')

        Returns:
            StrategyConfig instance or None if loading fails
        """
        if not os.path.exists(filepath):
            logger.error(f"Configuration file not found: {filepath}")
            return None
        try:
            with open(filepath, "r") as f:
                config_dict = yaml.safe_load(f)  # Use safe_load for YAML
            if not config_dict:
                logger.error(f"Configuration file is empty or invalid: {filepath}")
                return None
            logger.info(f"Loaded strategy configuration from {filepath}")
            # Recreate config using from_dict, which handles nested structures
            return cls.from_dict(config_dict)
        except yaml.YAMLError as e:
            logger.exception(f"Error parsing YAML configuration file {filepath}: {e}")
            return None
        except Exception as e:
            logger.exception(f"Failed to load configuration from {filepath}: {e}")
            return None
        # Removed duplicate/incorrect return and instantiation block

        # Removed duplicate save_config method definition

        # Removed duplicate load_config method definition
        try:
            with open(filepath, "r") as f:
                config_dict = yaml.safe_load(f)  # Use safe_load for YAML
            if not config_dict:
                logger.error(f"Configuration file is empty or invalid: {filepath}")
                return None
            logger.info(f"Loaded strategy configuration from {filepath}")
            # Recreate config using from_dict, which handles nested structures
            return cls.from_dict(config_dict)
        except yaml.YAMLError as e:
            logger.exception(f"Error parsing YAML configuration file {filepath}: {e}")
            return None
        except Exception as e:
            logger.exception(f"Failed to load configuration from {filepath}: {e}")
            return None
            # unless the loaded config_dict['output_dir'] is handled specially.
            # For simplicity, we assume loading creates a new run based on the loaded settings.
            return cls.from_dict(config_dict)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Failed to load configuration from {filepath}: {e}")
            return None


# --- Market Regime Detection ---


class MarketRegimeDetector:
    """Detects market regimes using various indicators."""

    def __init__(
        self,
        n_regimes: int = 3,  # e.g., Bull, Bear, Sideways
        lookback_window: int = 60,  # Lookback for calculating features
        regime_features: Optional[List[str]] = None,  # Features used for detection
        use_clustering: bool = True,  # Use KMeans for detection
    ):
        """
        Initialize market regime detector.

        Args:
            n_regimes: Number of regimes to detect
            lookback_window: Lookback window for detection features
            regime_features: List of feature names to use (e.g., ['volatility_50', 'sma_slope'])
            use_clustering: Whether to use clustering (KMeans) or rule-based detection
        """
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        # Default features if none provided
        self.regime_features = regime_features or [
            "volatility_ratio_20_50",
            "efficiency_ratio_20",
            "adf_pvalue",  # Example features
        ]
        self.use_clustering = use_clustering
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE_FOR_REGIME else None
        self.kmeans = (
            KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            if SKLEARN_AVAILABLE_FOR_REGIME
            else None
        )
        self.is_fitted = False
        self.regime_history = []  # Store history for analysis/plotting

    def detect_regime(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect market regime for the latest timestamp in the provided data.
        Requires engineered features to be present in the input DataFrame.

        Args:
            data: DataFrame containing the required regime_features, indexed by time.

        Returns:
            Dictionary with regime information for the latest timestamp, or None if error.
            Example: {'regime': 0, 'regime_name': 'Trending', 'confidence': 0.8, 'timestamp': ...}
        """
        if data is None or data.empty:
            logger.warning("No data provided for regime detection.")
            return None

        missing_features = [f for f in self.regime_features if f not in data.columns]
        if missing_features:
            logger.error(
                f"Missing required features for regime detection: {missing_features}"
            )
            return None

        # Select and clean features for the lookback window ending at the last timestamp
        # Ensure enough data points for lookback
        if len(data) < self.lookback_window:
            logger.warning(
                f"Not enough data ({len(data)}) for regime lookback window ({self.lookback_window})."
            )
            # Optionally fall back to rule-based or return None
            return None  # Cannot reliably detect regime

        # Use data up to the latest point for detection/fitting
        regime_data = data[self.regime_features].dropna()

        # Need sufficient non-NaN data points after dropping NaNs
        if len(regime_data) < self.lookback_window:
            logger.warning(
                f"Not enough non-NaN data ({len(regime_data)}) in lookback window for regime detection."
            )
            return None

        if self.use_clustering:
            regime_info = self._detect_clustering_regime(regime_data)
        else:
            # Implement or call rule-based detection if needed
            logger.warning(
                "Rule-based regime detection not implemented yet. Using clustering."
            )
            regime_info = self._detect_clustering_regime(regime_data)

        # Store history
        if regime_info:
            self.regime_history.append(regime_info)
            # Limit history length
            max_history = 2000  # Keep more history if needed
            if len(self.regime_history) > max_history:
                self.regime_history = self.regime_history[-max_history:]

        return regime_info

    def _detect_clustering_regime(self, regime_data: pd.DataFrame) -> Optional[Dict]:
        """Detect regimes using KMeans clustering for the latest data point."""
        if (
            not SKLEARN_AVAILABLE_FOR_REGIME
            or self.scaler is None
            or self.kmeans is None
        ):
            logger.error("Clustering requires scikit-learn.")
            return None

        try:
            # Scale features (fit scaler only once or periodically, transform here)
            # For simplicity, fit_transform each time, but could be optimized
            scaled_data = self.scaler.fit_transform(regime_data)

            # Fit KMeans model if not already fitted or periodically refit
            if not self.is_fitted:
                logger.info("Fitting KMeans model for regime detection...")
                self.kmeans.fit(scaled_data)
                self.is_fitted = True
                logger.info(
                    f"KMeans model fitted. Cluster centers:\n{self.kmeans.cluster_centers_}"
                )

            # Predict regime for the latest data point
            latest_scaled_features = scaled_data[-1:, :]
            regime = self.kmeans.predict(latest_scaled_features)[0]

            # Calculate confidence (e.g., distance to cluster center)
            distances = self.kmeans.transform(latest_scaled_features)
            min_distance = distances[0, regime]
            # Normalize confidence (optional, simple approach)
            confidence = 1.0 / (1.0 + min_distance)  # Closer = higher confidence

            # Map cluster label to regime name (requires interpretation)
            # This mapping needs to be defined based on cluster properties
            # Example placeholder mapping:
            regime_names = {
                0: "Regime A",
                1: "Regime B",
                2: "Regime C",
            }  # Needs proper mapping
            regime_name = regime_names.get(regime, f"Unknown Regime {regime}")

            regime_info = {
                "regime": int(regime),
                "regime_name": regime_name,
                "confidence": float(confidence),
                "features": regime_data.iloc[
                    -1
                ].to_dict(),  # Features for the detected point
                "timestamp": regime_data.index[-1],
            }
            return regime_info

        except Exception as e:
            logger.exception(f"Error during clustering regime detection: {e}")
            return None

    # Placeholder for rule-based detection
    def _detect_rule_based_regime(self, data: pd.DataFrame) -> Optional[Dict]:
        logger.error("Rule-based regime detection not implemented.")
        # Example structure:
        # latest_features = data.iloc[-1]
        # if latest_features['volatility_ratio'] > 1.5: regime = 2 ...
        # return {'regime': regime, ...}
        return None

    def plot_regime_history(
        self,
        price_data: Optional[pd.Series] = None,  # Pass close price Series
        figsize: Tuple[int, int] = (15, 8),
        filename: Optional[str] = None,
    ):
        """Plot the detected market regimes over time."""
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization libraries not available.")
            return
        if not self.regime_history:
            logger.warning("No regime history to plot.")
            return

        try:
            # Create DataFrame from regime history
            regime_df = pd.DataFrame(self.regime_history)
            if (
                "timestamp" not in regime_df.columns
                or regime_df["timestamp"].isnull().all()
            ):
                logger.warning("No valid timestamps in regime history for plotting.")
                return
            regime_df = regime_df.dropna(subset=["timestamp"]).set_index("timestamp")
            regime_df.index = pd.to_datetime(regime_df.index)  # Ensure datetime index

            plt.figure(figsize=figsize)
            ax1 = plt.gca()
            ax2 = ax1.twinx()  # Create secondary y-axis for regimes

            # Plot price data if available
            if price_data is not None and not price_data.empty:
                # Align price data to regime index
                price_data_aligned = price_data.reindex(regime_df.index).ffill()
                ax1.plot(
                    price_data_aligned.index,
                    price_data_aligned,
                    label="Price",
                    color="black",
                    alpha=0.6,
                    linewidth=1,
                )
                ax1.set_ylabel("Price")
                ax1.legend(loc="upper left")
            else:
                ax1.set_ylabel("Regime Value (if no price)")  # Adjust label if no price

            # Plot regimes as steps
            # colors = plt.cm.viridis(np.linspace(0, 1, self.n_regimes)) # Removed unused variable
            unique_regimes = sorted(regime_df["regime"].unique())

            # Use step plot for regimes
            ax2.plot(
                regime_df.index,
                regime_df["regime"],
                drawstyle="steps-post",
                label="Detected Regime",
                color="blue",
                linewidth=1.5,
            )

            ax2.set_ylabel("Market Regime")
            ax2.set_yticks(unique_regimes)
            # Optional: Add regime names as labels if meaningful mapping exists
            # regime_name_map = {r: info['regime_name'] for r, info in regime_df.groupby('regime').first().iterrows()}
            # ax2.set_yticklabels([regime_name_map.get(r, r) for r in unique_regimes])
            ax2.set_ylim(min(unique_regimes) - 0.5, max(unique_regimes) + 0.5)
            ax2.legend(loc="upper right")

            plt.title("Market Regime Detection History")
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()

            if filename:
                try:
                    # Ensure directory exists
                    output_dir = os.path.dirname(filename)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    plt.savefig(filename, dpi=300, bbox_inches="tight")
                    logger.info(f"Regime history plot saved to {filename}")
                except Exception as e:
                    logger.error(f"Failed to save regime plot to {filename}: {e}")

            else:
                plt.show()

            plt.close()

        except Exception as e:
            logger.exception(f"Error plotting regime history: {e}")
            plt.close()  # Ensure plot is closed


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
        self.market_regime_enabled = getattr(
            self.config.market_regime_config, "enabled", False
        )
        self.market_regime_method = getattr(
            self.config.market_regime_config, "method", None
        )
        self.market_regime_params = getattr(
            self.config.market_regime_config, "params", {}
        )
        self.market_regime_actions = getattr(
            self.config.market_regime_config, "regime_actions", {}
        )

        self.kmeans_regime_model = (
            None  # To store fitted KMeans model per fold if needed
        )
        self.scaler_regime = None  # To store fitted scaler per fold if needed

        logger.info(f"Market Regime Detection Enabled: {self.market_regime_enabled}")
        if self.market_regime_enabled:
            logger.info(f"Market Regime Method: {self.market_regime_method}")
            logger.info(f"Market Regime Params: {self.market_regime_params}")
            logger.info(f"Market Regime Actions: {self.market_regime_actions}")
            # Check dependencies for selected method
            if (
                self.market_regime_method == "volatility_clustering"
                and not SKLEARN_AVAILABLE_FOR_REGIME
            ):
                logger.error(
                    "Market regime detection uses clustering but scikit-learn is not available."
                )
                self.market_regime_enabled = False  # Disable if dependencies missing
        # )
        self.models: Dict[str, str] = (
            {}
        )  # Stores paths to saved models (asset_symbol[_regime_X] -> path)
        self.predictors: Dict[str, ModelPredictorBase] = {}  # Stores loaded predictors
        self.signal_generators: Dict[str, SignalGenerator] = (
            {}
        )  # Stores signal generators per asset/model_key
        self.results: Dict[str, Any] = (
            {}
        )  # Stores results per asset or walk-forward step

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
            # Assuming a default cache location within the strategy output dir
            cache_dir = os.path.join(self.config.output_dir, ".cache")
            loader_config = DataLoaderConfig(
                cache_dir=cache_dir, use_cache=True
            )  # Enable caching
            loader = DataLoader(config=loader_config)
            logger.info(f"DataLoader initialized with cache directory: {cache_dir}")
            return loader
        except Exception:
            logger.exception("Error initializing DataLoader")
            return None

    def load_data(self, asset_config: AssetConfig) -> Optional[pd.DataFrame]:
        """Load and preprocess data for a given asset."""
        if self.data_loader is None:
            logger.error("DataLoader not initialized.")
            return None

        logger.info(
            f"Loading data for {asset_config.symbol} ({asset_config.timeframe}) from {self.config.data_start_date} to {self.config.data_end_date}..."
        )
        try:
            # Use the specific data source from asset config
            raw_data = self.data_loader.load_data(
                ticker=asset_config.symbol,
                start_date=self.config.data_start_date,
                end_date=self.config.data_end_date,
                interval=asset_config.timeframe,
                data_source=asset_config.data_source,
            )
            if raw_data is None or raw_data.empty:
                logger.error(f"Failed to load data for {asset_config.symbol}")
                return None

            # --- Basic Preprocessing ---
            # Standardize column names
            original_cols = raw_data.columns
            raw_data.columns = raw_data.columns.str.lower().str.replace(" ", "_")
            logger.debug(
                f"Standardized columns: {original_cols} -> {list(raw_data.columns)}"
            )

            # Ensure datetime index
            if not isinstance(raw_data.index, pd.DatetimeIndex):
                try:
                    raw_data.index = pd.to_datetime(raw_data.index)
                except Exception as e:
                    logger.error(
                        f"Failed to convert index to DatetimeIndex for {asset_config.symbol}: {e}"
                    )
                    return None  # Cannot proceed without datetime index

            # Handle initial NaNs (e.g., forward fill)
            initial_nans = raw_data.isnull().sum().sum()
            if initial_nans > 0:
                raw_data = raw_data.ffill()
                logger.debug(
                    f"Forward-filled {initial_nans} initial NaNs for {asset_config.symbol}."
                )

            # Drop any remaining NaNs at the beginning
            raw_data = raw_data.dropna()

            if raw_data.empty:
                logger.error(
                    f"Data empty after initial preprocessing for {asset_config.symbol}"
                )
                return None

            logger.info(
                f"Data loaded and preprocessed for {asset_config.symbol}. Shape: {raw_data.shape}"
            )
            return raw_data

        except Exception as e:
            logger.exception(f"Error loading data for {asset_config.symbol}: {e}")
            return None

    def engineer_features(
        self,
        data: pd.DataFrame,
        asset_config: AssetConfig,  # Pass asset config for context
    ) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """
        Engineer features and target variable for the given data.

        Args:
            data: DataFrame with OHLCV data (preprocessed)
            asset_config: Configuration for the specific asset

        Returns:
            Tuple of (features_df, target_series) or (None, None)
        """
        if data is None or data.empty:
            logger.error("Cannot engineer features on empty data.")
            return None, None

        logger.info(f"Engineering features for {asset_config.symbol}...")
        try:
            # --- Construct the feature_config dictionary expected by engineer_features ---
            # The FeatureConfig object now holds the list directly.
            cfg = self.config.feature_config  # Shortcut to the FeatureConfig object

            eng_feature_config = {
                # Pass the list of indicator configurations directly
                "technical_indicators": cfg.technical_indicators,
                "target_config": {
                    "type": cfg.target_type,
                    "atr_multiplier_tp": cfg.atr_multiplier_tp,
                    "atr_multiplier_sl": cfg.atr_multiplier_sl,
                    "max_holding_period": cfg.max_holding_period,
                    # Add other target params if needed from FeatureConfig
                    # 'min_return_threshold': cfg.min_return_threshold,
                    # 'threshold': cfg.directional_threshold,
                    # 'atr_column_name': cfg.atr_column_name # Example if added
                },
                # Add flags for other feature types if engineer_features uses them
                # (Currently, engineer_features doesn't seem to use these flags directly,
                # but relies on the technical_indicators list and hardcoded other features)
                # 'volatility_features': cfg.volatility_features,
                # 'trend_features': cfg.trend_features,
                # ... etc ...
            }

            logger.debug(
                f"Using feature config for engineer_features: {eng_feature_config}"
            )

            # --- Call the centralized feature engineering function ---
            feature_df = engineer_features(
                df=data,
                feature_config=eng_feature_config,  # Pass the correctly structured dict
            )

            if feature_df is None or feature_df.empty:
                logger.error(f"Feature engineering failed for {asset_config.symbol}")
                return None, None

            # Extract target variable
            # Determine target column based on the config used for engineering
            target_type_used = eng_feature_config.get("target_config", {}).get(
                "type", "triple_barrier"
            )
            target_col_map = {
                "triple_barrier": "triple_barrier_label",
                "directional": "directional_label",
            }
            target_col = target_col_map.get(target_type_used)

            if target_col and target_col in feature_df.columns:
                target = feature_df[target_col]
                # Drop target and potentially other label-related columns before returning features
                cols_to_drop = [target_col]
                # Add other potential leakage columns if necessary
                features = feature_df.drop(columns=cols_to_drop, errors="ignore")
                # Select only numeric features for most models (can be configured)
                features = features.select_dtypes(include=np.number)

                # Drop rows with NaNs resulting from feature calculation
                original_len = len(features)
                features = features.dropna()
                target = target.loc[
                    features.index
                ]  # Align target with dropped features
                if len(features) < original_len:
                    logger.warning(
                        f"Dropped {original_len - len(features)} rows due to NaNs in engineered features."
                    )

                if features.empty:
                    logger.error("Features DataFrame empty after dropping NaNs.")
                    return None, None

                logger.info(
                    f"Features engineered. Shape: {features.shape}. Target: '{target_col}' ({len(target)} samples)"
                )
                return features, target
            else:
                logger.error(
                    f"Target column '{target_col}' not found after feature engineering."
                )
                # Check available columns for debugging
                logger.debug(
                    f"Available columns after feature engineering: {list(feature_df.columns)}"
                )
                return None, None

        except Exception as e:
            logger.exception(
                f"Error engineering features for {asset_config.symbol}: {e}"
            )
            return None, None

    def train_model(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        asset_config: AssetConfig,
        regime: Optional[int] = None,  # Optional regime context
        fold_id: Optional[str] = None,  # Optional fold identifier for WF
    ) -> Optional[str]:
        """
        Train a model for a specific asset (and potentially regime). Saves the model.

        Args:
            features: Training features DataFrame
            target: Training target Series
            asset_config: Configuration for the asset
            regime: Optional market regime identifier
            fold_id: Optional identifier for walk-forward fold

        Returns:
            Path to the saved model directory, or None if training fails.
        """
        if features is None or target is None or features.empty or target.empty:
            logger.error("Cannot train model with empty features or target.")
            return None

        model_key = asset_config.symbol
        model_sub_dir = asset_config.symbol  # Subdirectory for the asset
        if regime is not None and self.config.model_config.regime_adaptation_enabled:
            model_key += f"_regime_{regime}"
            model_sub_dir += f"/regime_{regime}"  # Organize regime models
            logger.info(
                f"Training model for {asset_config.symbol} in Regime {regime}..."
            )
        else:
            logger.info(f"Training model for {asset_config.symbol} (no regime)...")

        # Determine model type based on regime if applicable
        model_type = self.config.model_config.model_type
        if regime is not None and regime in self.config.model_config.regime_models:
            model_type = self.config.model_config.regime_models[regime]
            logger.info(f"Using regime-specific model type: {model_type}")

        # Define model output directory structure
        # output_dir / models / asset_symbol / [regime_X] / fold_Y / model.pkl etc.
        model_base_dir = os.path.join(self.config.output_dir, "models", model_sub_dir)
        if fold_id:
            model_output_dir = os.path.join(model_base_dir, fold_id)
        else:
            # For single run, maybe use timestamp or 'latest'
            model_output_dir = os.path.join(
                model_base_dir, f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            )

        try:
            # Prepare configurations for the training function
            train_model_config = {
                "model_type": model_type,
                "params": {},  # Start with empty params, optimization will fill if enabled
            }
            train_fs_config = {
                "enabled": self.config.feature_config.feature_selection_enabled,
                "method": self.config.feature_config.feature_selection_method,
                "params": {"n_features": self.config.feature_config.max_features},
            }
            train_opt_config = {
                "method": self.config.model_config.optimization_method,
                "params": {
                    "n_iter": self.config.model_config.optimization_trials
                },  # Pass n_iter/n_trials
                "cv": self.config.model_config.cv_folds,
                "scoring": self.config.model_config.scoring_metric,
            }
            train_preproc_config = {
                # Get scaling method from FeatureConfig -> ScalingConfig
                "scaling_method": self.config.feature_config.scaling.method,
                "handle_missing": True,  # Keep default missing handling for now
            }

            trained_model, importance, metrics = train_classification_model(
                features=features,
                target=target,
                model_config=train_model_config,
                feature_selection_config=train_fs_config,
                preprocessing_config=train_preproc_config,  # Pass preprocessing config
                test_size=self.config.model_config.validation_size,
                optimize_hyperparams=self.config.model_config.optimize_hyperparams,
                optimization_config=train_opt_config,
                output_dir=model_output_dir,  # Save model artifacts here
                model_id=model_key,  # Use specific key as model ID
                random_state=self.config.random_state,
            )

            if trained_model is None:
                logger.error(f"Model training failed for {model_key}.")
                return None

            # Store the path to the saved model directory
            self.models[model_key] = model_output_dir
            logger.info(
                f"Model training successful for {model_key}. Saved to: {model_output_dir}"
            )
            return model_output_dir

        except Exception as e:
            logger.exception(f"Error training model for {model_key}: {e}")
            return None

    def generate_predictions(
        self,
        features: pd.DataFrame,
        asset_config: AssetConfig,
        regime: Optional[int] = None,
        fold_id: Optional[str] = None,  # Pass fold_id if predicting for WF test set
    ) -> Optional[pd.DataFrame]:
        """
        Generate predictions using the appropriate trained model.

        Args:
            features: DataFrame of features to predict on
            asset_config: Configuration for the asset
            regime: Optional current market regime
            fold_id: Optional identifier for the walk-forward fold model to use

        Returns:
            DataFrame with predictions and probabilities, or None if error.
        """
        if features is None or features.empty:
            logger.error("Cannot generate predictions with empty features.")
            return None

        # Determine which model key to use (handles regimes)
        model_key = asset_config.symbol
        if regime is not None and self.config.model_config.regime_adaptation_enabled:
            regime_key = f"{asset_config.symbol}_regime_{regime}"
            # Check if a specific model was trained for this regime/fold
            if regime_key in self.models:
                model_key = regime_key
            else:
                logger.warning(
                    f"No specific model found for regime {regime}. Using base model for {asset_config.symbol}."
                )

        # Retrieve the correct model path saved during training
        # The path stored in self.models already includes the fold_id if applicable
        model_path = self.models.get(model_key)

        if model_path is None or not os.path.exists(model_path):
            # Handle case where model wasn't trained or path is missing
            # This might happen if training failed for this specific key/fold
            logger.error(
                f"Model path not found or invalid for key '{model_key}'. Expected path: {model_path}"
            )
            # Attempt to find the latest run if not a specific fold? - No, rely on stored path.
            # If fold_id was provided, the path should exist if training succeeded.
            return None

        logger.info(
            f"Generating predictions for {asset_config.symbol} using model from '{model_path}'..."
        )

        # Load predictor (it handles loading model, preprocessor etc from the path)
        predictor_key = model_path  # Use path as unique key for predictor instance
        if predictor_key not in self.predictors:
            try:
                self.predictors[predictor_key] = ModelPredictorBase(model_path)
                logger.info(f"Loaded predictor for '{model_key}' from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load predictor from '{model_path}': {e}")
                return None

        predictor = self.predictors[predictor_key]

        try:
            # Predict probabilities
            probabilities, class_names = predictor.predict_proba(features)
            if probabilities is None:
                logger.error(f"Prediction failed for model at {model_path}.")
                return None

            # Create result DataFrame
            result_df = pd.DataFrame(
                probabilities, index=features.index, columns=class_names
            )
            # Use predict_with_threshold for consistent prediction generation
            # Ensure positive_class_index aligns with class_names
            positive_class_label = "1"  # Assuming '1' represents positive class
            try:
                pos_idx = class_names.index(positive_class_label)
            except ValueError:
                logger.warning(
                    f"Positive class label '{positive_class_label}' not found in {class_names}. Defaulting to index 1."
                )
                pos_idx = 1  # Default assumption

            result_df["prediction"] = predict_with_threshold(
                probabilities,
                threshold=self.config.model_config.probability_threshold,
                positive_class_index=pos_idx,
            )
            result_df["confidence"] = get_confidence_levels(probabilities)

            logger.info(
                f"Predictions generated successfully for {asset_config.symbol}."
            )
            return result_df

        except Exception as e:
            logger.exception(
                f"Error generating predictions using model from {model_path}: {e}"
            )
            return None

    def generate_signals(
        self,
        predictions: pd.DataFrame,
        ohlc_data: pd.DataFrame,  # Pass OHLC for filters
        asset_config: AssetConfig,
        regime: Optional[int] = None,  # Pass regime to potentially adjust signal logic
        regime_actions: Optional[Dict] = None,  # Pass regime actions from config
    ) -> Optional[pd.DataFrame]:
        """
        Generate trading signals from predictions, applying filters.

        Args:
            predictions: DataFrame with 'prediction', 'probability_X', 'confidence' columns
            ohlc_data: DataFrame with OHLC data for filters
            asset_config: Configuration for the asset
            regime: Optional current market regime

        Returns:
            DataFrame with final 'signal' column (-1, 0, 1), or None if error.
        """
        if predictions is None or predictions.empty:
            logger.error("Cannot generate signals from empty predictions.")
            return None

        # Determine the model path that was used to generate these predictions
        # This should match the logic in generate_predictions
        model_key_for_path = asset_config.symbol  # Start with base key
        model_sub_dir = asset_config.symbol
        if regime is not None and self.config.model_config.regime_adaptation_enabled:
            # Check if a specific model was trained for this regime/fold
            regime_key = f"{asset_config.symbol}_regime_{regime}"
            if regime_key in self.models:  # Check if model exists for this regime
                model_key_for_path = regime_key
                model_sub_dir += f"/regime_{regime}"
            # else: use base model_key_for_path and model_sub_dir

        # Retrieve the actual model path used in generate_predictions
        # This relies on self.models containing the correct path after training
        model_path = self.models.get(model_key_for_path)
        if model_path is None:
            logger.error(
                f"Could not find trained model path for key '{model_key_for_path}' in self.models."
            )
            return None

        # Retrieve the corresponding predictor instance using the model_path as the key
        predictor = self.predictors.get(model_path)
        if predictor is None:
            # Attempt to load it if it wasn't loaded during generate_predictions (should not happen ideally)
            logger.warning(
                f"Predictor for path '{model_path}' not found in cache, attempting to load."
            )
            try:
                predictor = ModelPredictorBase(model_path)
                self.predictors[model_path] = predictor  # Cache it
            except Exception as e:
                logger.error(f"Failed to load predictor for path '{model_path}': {e}")
                return None

        # Initialize SignalGenerator if not already done for this asset/config
        # Key might just be asset symbol if signal logic doesn't change per regime model
        signal_gen_key = asset_config.symbol
        if signal_gen_key not in self.signal_generators:
            try:
                # Pass the predictor instance to SignalGenerator if needed for class names etc.
                self.signal_generators[signal_gen_key] = SignalGenerator(
                    predictor=predictor,  # Pass predictor instance
                    threshold=self.config.model_config.probability_threshold,
                    neutral_zone=self.config.model_config.signal_neutral_zone,
                    trend_filter_ma=self.config.model_config.signal_trend_filter_ma,
                    volatility_filter_atr=self.config.model_config.signal_volatility_filter_atr,
                    cooling_period=self.config.model_config.signal_cooling_period,
                )
                logger.info(f"Initialized SignalGenerator for '{signal_gen_key}'.")
            except Exception as e:
                logger.error(
                    f"Failed to initialize SignalGenerator for '{signal_gen_key}': {e}"
                )
                return None

        signal_generator = self.signal_generators[signal_gen_key]

        logger.info(f"Generating signals for {asset_config.symbol}...")
        try:
            # Pass the predictions DataFrame and OHLC data to the generator
            signals_df = signal_generator.generate_signals(predictions, ohlc_data)

            if signals_df is None:
                logger.error(
                    f"Signal generation function returned None for {asset_config.symbol}."
                )
                return None

            # --- Apply Market Regime Actions ---
            # Ensure regime_actions is fetched from self.config.market_regime_config if not passed explicitly
            current_regime_actions = (
                regime_actions
                if regime_actions is not None
                else self.config.market_regime_config.regime_actions
            )

            if regime is not None and current_regime_actions:
                # Convert regime (which might be numpy.int64) to standard int for dict lookup
                regime_key = int(regime)
                action = current_regime_actions.get(regime_key)
                if action == "no_trade":
                    logger.info(
                        f"Regime {regime_key} -> Action: 'no_trade'. Setting all signals to 0."
                    )
                    signals_df["signal"] = 0
                elif action == "reduce_risk":
                    # Risk reduction (e.g., halving size) is handled in backtest_strategy
                    logger.info(
                        f"Regime {regime_key} -> Action: 'reduce_risk'. (Handled in backtest)"
                    )
                    pass  # No signal modification here, handled in backtester
                elif action == "trade_normal":
                    logger.info(
                        f"Regime {regime_key} -> Action: 'trade_normal'. No signal changes."
                    )
                    pass  # No change needed
                elif (
                    action is not None
                ):  # Log warning only if action is defined but unknown
                    logger.warning(
                        f"Unknown regime action '{action}' for regime {regime_key}. Trading normally."
                    )
                # If action is None (regime not in dict), trade normally by default

            logger.info(f"Signals generated successfully for {asset_config.symbol}.")
            # Return only the final signal column, aligned with input index
            return signals_df[["signal"]].reindex(predictions.index)

        except Exception as e:
            logger.exception(f"Error generating signals for {asset_config.symbol}: {e}")
            return None

    def backtest_signals(
        self,
        signals: pd.DataFrame,
        ohlc_data: pd.DataFrame,  # Should contain features needed for backtest (e.g., ATR)
        asset_config: AssetConfig,
        fold_id: Optional[str] = None,  # Add fold_id for output path
        regime: Optional[int] = None,  # Pass regime
        regime_actions: Optional[Dict] = None,  # Pass regime actions from config
    ) -> Optional[Dict]:
        """
        Run a backtest using the generated signals.

        Args:
            signals: DataFrame with 'signal' column (-1, 0, 1)
            ohlc_data: DataFrame with OHLC data and potentially ATR column
            asset_config: Configuration for the asset
            fold_id: Optional identifier for walk-forward fold output

        Returns:
            Dictionary with backtest performance summary, or None if error.
        """
        if signals is None or ohlc_data is None or signals.empty or ohlc_data.empty:
            logger.error("Cannot backtest with empty signals or OHLC data.")
            return None

        # Align data
        common_index = signals.index.intersection(ohlc_data.index)
        if common_index.empty:
            logger.error("No common index between signals and OHLC data for backtest.")
            return None
        signals_aligned = signals.loc[common_index]
        ohlc_aligned = ohlc_data.loc[common_index]

        # Ensure necessary columns (like ATR) are present in ohlc_data
        # Feature engineering should have added the required ATR column based on config
        ohlc_for_backtest = ohlc_aligned.copy()
        # Use ATR column name from trading config
        atr_col_name = self.config.trading_config.atr_col

        if (
            atr_col_name not in ohlc_for_backtest.columns
            and self.config.trading_config.use_dynamic_stops
        ):
            logger.error(
                f"Required ATR column '{atr_col_name}' not found in OHLC data for backtest. "
                f"Ensure it's generated during feature engineering."
            )
            # Attempt calculation if possible (less ideal here)
            # For now, return error if missing and needed
            return None

        # Combine data needed for backtest function
        backtest_data = ohlc_for_backtest.copy()
        backtest_data["prediction"] = signals_aligned[
            "signal"
        ]  # Use 'signal' as prediction input

        logger.info(f"Running backtest for {asset_config.symbol}...")
        try:
            # Prepare backtest config from strategy config
            backtest_output_base = os.path.join(
                self.config.output_dir, "backtests", asset_config.symbol
            )
            if fold_id:
                backtest_output_dir = os.path.join(backtest_output_base, fold_id)
            else:
                backtest_output_dir = os.path.join(
                    backtest_output_base,
                    f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                )

            trades_path = os.path.join(backtest_output_dir, "trades.csv")

            # --- Get Trading Settings from TradingConfig ---
            trade_cfg = self.config.trading_config
            use_kelly = trade_cfg.position_sizing_method.lower() == "kelly"
            # Use fixed fraction amount as risk_per_trade if method is fixed, else use max_risk_per_trade
            risk_setting = (
                trade_cfg.fixed_fraction_amount
                if not use_kelly
                else trade_cfg.max_risk_per_trade
            )

            # Ensure regime_actions is fetched from self.config.market_regime_config if not passed explicitly
            current_regime_actions = (
                regime_actions
                if regime_actions is not None
                else self.config.market_regime_config.regime_actions
            )

            performance_summary = backtest_strategy(
                data_with_predictions=backtest_data,
                transaction_cost_pct=asset_config.commission_pct,  # From AssetConfig
                slippage_pct_per_trade=asset_config.slippage_pct,  # From AssetConfig
                initial_capital=trade_cfg.initial_capital,
                risk_per_trade=risk_setting,  # Use calculated risk setting
                use_kelly_sizing=use_kelly,  # Use boolean based on method
                atr_multiplier_sl=trade_cfg.atr_multiplier_sl,
                atr_multiplier_tp=trade_cfg.atr_multiplier_tp,
                atr_col=trade_cfg.atr_col,  # Use ATR col from trading config
                trailing_stop_pct=trade_cfg.trailing_stop_pct,
                trailing_stop_activation_pct=trade_cfg.trailing_stop_activation_pct,
                max_open_trades=trade_cfg.max_open_trades,
                max_drawdown_pct=trade_cfg.max_drawdown_pct,
                use_dynamic_stops=trade_cfg.use_dynamic_stops,
                signal_threshold=self.config.model_config.probability_threshold,  # From ModelConfig
                allow_pyramiding=trade_cfg.allow_pyramiding,
                benchmark_col="close",  # Example, make configurable if needed
                output_dir=backtest_output_dir,  # Save reports in specific dir
                output_trades_path=trades_path,
                save_detailed_report=True,  # Save plots etc.
                # --- Pass Regime Info ---
                regime=regime,
                regime_actions=current_regime_actions,  # Pass the actions dict
            )
            logger.info(
                f"Backtest completed for {asset_config.symbol} (Regime: {regime})."
            )
            return performance_summary

        except Exception as e:
            logger.exception(f"Error during backtest for {asset_config.symbol}: {e}")
            return None

    def run_walk_forward_validation(self, asset_config: AssetConfig) -> Dict:
        """
        Perform walk-forward validation for a single asset.

        This involves iteratively:
        1. Splitting data into training and testing periods.
        2. Engineering features for both periods.
        3. Detecting the market regime at the end of the training period (if enabled).
        4. Training/retraining the model on the training data (potentially regime-specific).
        5. Generating predictions and signals for the test period using the trained model.
        6. Backtesting the signals on the test period (potentially applying regime actions).

        Args:
            asset_config: Configuration for the asset

        Returns:
            Dictionary containing walk-forward results summary.
        """
        if not self.config.walkforward_config.enabled:
            logger.info("Walk-forward validation disabled.")
            return {"status": "disabled"}

        if not MODULES_AVAILABLE:
            logger.error("Required modules not available for walk-forward.")
            return {"status": "error", "message": "Modules unavailable"}

        logger.info(
            f"--- Starting Walk-Forward Validation for {asset_config.symbol} ---"
        )
        wf_config = self.config.walkforward_config
        wf_results_list = []  # Store results dict from each fold's backtest
        all_trades_list = []  # Store trades DataFrame from each fold
        full_data = self.load_data(asset_config)

        if (
            full_data is None
            or len(full_data) < wf_config.initial_train_periods + wf_config.test_periods
        ):
            logger.error(
                f"Not enough data for walk-forward validation ({len(full_data)} periods). Need at least {wf_config.initial_train_periods + wf_config.test_periods}."
            )
            return {"status": "error", "message": "Insufficient data"}

        # Calculate split points based on periods
        n = len(full_data)
        train_start_idx = 0
        train_end_idx = wf_config.initial_train_periods
        test_end_idx = train_end_idx + wf_config.test_periods
        step = 0

        while test_end_idx <= n:
            step += 1
            fold_id = f"fold_{step}"
            logger.info(f"--- Walk-Forward Step {step} ({fold_id}) ---")
            logger.info(
                f"Train Period Indices: {train_start_idx} to {train_end_idx - 1}"
            )
            logger.info(f"Test Period Indices:  {train_end_idx} to {test_end_idx - 1}")

            # --- Split Raw Data into Train/Test Slices ---
            logger.info(f"Splitting raw data for step {step}...")
            train_data = full_data.iloc[train_start_idx:train_end_idx]
            test_data = full_data.iloc[
                train_end_idx:test_end_idx
            ]  # Raw OHLCV for test period

            if train_data.empty or test_data.empty:
                logger.error(
                    f"Raw train or test data is empty for step {step}. Skipping."
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

            # --- Engineer Features Separately for Train Data ---
            logger.info(f"Engineering features for TRAIN data (Step {step})...")
            train_features, train_target = self.engineer_features(
                train_data, asset_config
            )
            # <<< Added Debug Logging for Fold 2 Train Features >>>
            if step == 2 and train_features is not None:
                logger.info(
                    f"[DEBUG Fold 2] Train Features Columns ({len(train_features.columns)}): {sorted(list(train_features.columns))}"
                )

            if train_features is None or train_target is None:
                logger.error(
                    f"Feature engineering failed for TRAIN data in step {step}. Skipping step."
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

            # --- Engineer Features Separately for Test Data ---

            # --- Detect Market Regime (based on training data end) ---
            current_regime = None
            # Access regime settings via self.config.market_regime_config
            if (
                self.config.market_regime_config.enabled
                and train_features is not None
                and not train_features.empty
            ):
                logger.info(
                    f"Detecting market regime for end of training period (Step {step})..."
                )
                try:
                    if (
                        self.config.market_regime_config.method
                        == "volatility_clustering"
                    ):
                        vol_indicator = self.config.market_regime_config.params.get(
                            "volatility_indicator", "atr"
                        )  # Default to atr
                        n_clusters = self.config.market_regime_config.params.get(
                            "n_clusters", 3
                        )

                        # Find the actual column name for the indicator (e.g., ATRr_14)
                        vol_col = None
                        # Search for column containing indicator name and often period (e.g., atr_14, ATRr_20)
                        indicator_pattern = f"{vol_indicator.lower()}"  # Basic pattern
                        potential_cols = [
                            col
                            for col in train_features.columns
                            if indicator_pattern in col.lower()
                        ]
                        if potential_cols:
                            # If specific period is in params, try to match it
                            vol_period = self.config.market_regime_config.params.get(
                                "volatility_period"
                            )
                            if vol_period:
                                period_pattern = (
                                    f"{indicator_pattern}_{vol_period}"  # e.g., atr_20
                                )
                                exact_match = [
                                    col
                                    for col in potential_cols
                                    if period_pattern in col.lower()
                                ]
                                if exact_match:
                                    vol_col = exact_match[0]
                                else:
                                    vol_col = potential_cols[
                                        0
                                    ]  # Fallback to first match
                                    logger.warning(
                                        f"Exact volatility column for period {vol_period} not found, using {vol_col}"
                                    )
                            else:
                                vol_col = potential_cols[
                                    0
                                ]  # Use first match if no period specified
                            logger.info(
                                f"Using volatility column for regime detection: {vol_col}"
                            )
                        else:
                            logger.warning(
                                f"No column found matching volatility indicator '{vol_indicator}' in train features."
                            )

                        if vol_col and vol_col in train_features.columns:
                            vol_data = train_features[[vol_col]].dropna()
                            if (
                                len(vol_data) > n_clusters
                            ):  # Need more data points than clusters
                                # Fit scaler and kmeans on the training volatility data for this fold
                                self.scaler_regime = StandardScaler()
                                scaled_vol = self.scaler_regime.fit_transform(vol_data)
                                self.kmeans_regime_model = KMeans(
                                    n_clusters=n_clusters,
                                    random_state=self.config.random_state,
                                    n_init=10,
                                )
                                self.kmeans_regime_model.fit(scaled_vol)

                                # Predict regime for the last point of the training data
                                last_scaled_vol = scaled_vol[-1].reshape(1, -1)
                                current_regime = self.kmeans_regime_model.predict(
                                    last_scaled_vol
                                )[0]
                                logger.info(
                                    f"Step {step}: Detected Regime = {current_regime} (based on {vol_col})"
                                )
                            else:
                                logger.warning(
                                    f"Not enough non-NaN data points ({len(vol_data)}) in '{vol_col}' to perform clustering for regime detection (need > {n_clusters})."
                                )
                        else:
                            logger.warning(
                                f"Selected volatility column '{vol_col}' not found or invalid. Cannot detect regime."
                            )
                    else:
                        logger.warning(
                            f"Unsupported market regime method: {self.config.market_regime_config.method}"
                        )
                except Exception as e:
                    logger.exception(
                        f"Error during market regime detection for step {step}: {e}"
                    )
            logger.info(f"Engineering features for TEST data (Step {step})...")
            # Note: We might not need test_target, but engineer_features returns it.
            test_features, _ = self.engineer_features(test_data, asset_config)
            # <<< Added Debug Logging for Fold 2 Test Features >>>
            if step == 2 and test_features is not None:
                logger.info(
                    f"[DEBUG Fold 2] Test Features Columns ({len(test_features.columns)}): {sorted(list(test_features.columns))}"
                )

            if test_features is None:
                logger.error(
                    f"Feature engineering failed for TEST data in step {step}. Skipping step."
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

            # Ensure test_features align with test_data index if needed later (e.g., for evaluation)
            # test_features = test_features.reindex(test_data.index) # Reindex if necessary, depends on engineer_features output

            # test_data (raw OHLCV) is already defined and will be used for backtesting signals

            # --- Retrain model (optional, based on frequency) ---
            model_path = None
            if step == 1 or (step - 1) % wf_config.retrain_frequency == 0:
                logger.info(f"Retraining model for step {step}...")
                # Use the already engineered train_features and train_target
                if train_features is None:
                    logger.error(
                        f"Feature engineering failed for training step {step}. Skipping step."
                    )
                    # Advance window indices based on config
                    train_start_idx = (
                        train_start_idx + wf_config.step_periods
                        if wf_config.rolling_window
                        else train_start_idx
                    )
                    train_end_idx += wf_config.step_periods
                    test_end_idx += wf_config.step_periods
                    continue

                # 2. Train model (pass regime if adaptation is configured)
                model_path = self.train_model(
                    train_features,
                    train_target,
                    asset_config,
                    regime=current_regime,
                    fold_id=fold_id,
                )
                # <<< Added Debug Logging for Fold 2 Selected Features >>>
                if step == 2 and model_path:
                    selected_features_path = os.path.join(
                        model_path, "selected_features.json"
                    )
                    try:
                        # Ensure json is imported if not already available in scope
                        import json

                        with open(selected_features_path, "r") as f:
                            selected_features_data = json.load(f)
                            selected_features_list = selected_features_data.get(
                                "selected_features", []
                            )
                            logger.info(
                                f"[DEBUG Fold 2] Selected Features ({len(selected_features_list)}): {sorted(selected_features_list)}"
                            )
                    except FileNotFoundError:
                        logger.error(
                            f"[DEBUG Fold 2] selected_features.json not found at {selected_features_path}"
                        )
                    except Exception as e:
                        logger.error(
                            f"[DEBUG Fold 2] Error reading selected_features.json: {e}"
                        )
                if model_path is None:
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
                logger.info(f"Using previously trained model for step {step}.")
                # Find the model path from the last retraining step
                last_retrain_step = (
                    (step - 2) // wf_config.retrain_frequency
                ) * wf_config.retrain_frequency + 1
                last_fold_id = f"fold_{last_retrain_step}"
                # Construct path (assuming no regimes for WF simplicity now)
                # model_key = asset_config.symbol # Removed unused variable
                model_sub_dir = asset_config.symbol
                model_base_dir = os.path.join(
                    self.config.output_dir, "models", model_sub_dir
                )
                model_path = os.path.join(model_base_dir, last_fold_id)
                if not os.path.exists(
                    os.path.join(model_path, "model.pkl")
                ):  # Check if model file exists
                    logger.error(
                        f"Model file not found at expected path {model_path} for step {step}. Retraining required?"
                    )
                    # Optionally force retrain here or skip step
                    # Advance window indices
                    train_start_idx = (
                        train_start_idx + wf_config.step_periods
                        if wf_config.rolling_window
                        else train_start_idx
                    )
                    train_end_idx += wf_config.step_periods
                    test_end_idx += wf_config.step_periods
                    continue

            # --- Generate predictions on test data ---
            logger.info(f"Generating predictions for test period step {step}...")
            # Features for test data were already engineered and split above
            if test_features.empty:  # Check if test_features ended up empty after split
                logger.error(
                    f"Feature engineering failed for test step {step}. Skipping step."
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

            # 2. Generate predictions using the model trained/selected for this fold (pass regime)
            predictions_df = self.generate_predictions(
                test_features,
                asset_config,
                regime=current_regime,
                fold_id=fold_id,  # Pass regime here
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

            # --- Generate Signals (pass regime info from this fold) ---
            signals_df = self.generate_signals(
                predictions_df,
                test_data,
                asset_config,
                regime=current_regime,
                regime_actions=self.config.market_regime_config.regime_actions,
            )
            if signals_df is None:
                logger.error(
                    f"Signal generation failed for step {step}. Skipping step."
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

            # --- Backtest the test period ---
            logger.info(f"Backtesting test period step {step}...")

            # Prepare data for backtesting: Start with raw OHLCV, add ATR if available
            backtest_input_data = test_data.copy()
            trade_cfg = self.config.trading_config
            atr_col_name = trade_cfg.atr_col

            if atr_col_name and atr_col_name in test_features.columns:
                # Add the required ATR column from features to the OHLC data
                # Ensure indices align; they should if test_features came from test_data
                backtest_input_data[atr_col_name] = test_features[atr_col_name]
                logger.debug(
                    f"Added ATR column '{atr_col_name}' to backtest input data for step {step}."
                )
            elif atr_col_name and trade_cfg.use_dynamic_stops:
                # Log a warning if dynamic stops are enabled but the ATR column is missing
                logger.warning(
                    f"Required ATR column '{atr_col_name}' not found in test features for step {step}, "
                    f"but dynamic stops are enabled. Backtest may fail."
                )
            # If ATR column is not needed (use_dynamic_stops=False) or not found, proceed with original data

            step_results = self.backtest_signals(
                signals=signals_df,
                ohlc_data=backtest_input_data,  # Use the (potentially augmented) data
                asset_config=asset_config,
                fold_id=fold_id,  # Pass fold_id for output organization
                regime=current_regime,  # Pass detected regime for this fold
                regime_actions=self.config.market_regime_config.regime_actions,  # Pass actions from config
            )

            # Log signal counts for debugging
            if signals_df is not None:
                signal_counts = signals_df["signal"].value_counts().to_dict()
                logger.info(f"Step {step} Signal Counts: {signal_counts}")
            else:
                logger.warning(f"Step {step}: No signals DataFrame generated.")

            if step_results:
                # Add fold info to results dict
                step_results["fold_number"] = step
                step_results["train_start"] = (
                    train_features.index[0].isoformat()
                    if isinstance(train_features.index, pd.DatetimeIndex)
                    else train_start_idx
                )
                step_results["train_end"] = (
                    train_features.index[-1].isoformat()
                    if isinstance(train_features.index, pd.DatetimeIndex)
                    else train_end_idx - 1
                )
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
                # Read trades file saved by backtest_strategy
                step_trades_path = os.path.join(
                    self.config.output_dir,
                    "backtests",
                    asset_config.symbol,
                    fold_id,
                    "trades.csv",
                )
                if os.path.exists(step_trades_path):
                    try:
                        trades_df = pd.read_csv(step_trades_path)
                        trades_df["fold"] = step  # Add fold number
                        all_trades_list.append(trades_df)
                    except Exception as e:
                        logger.warning(
                            f"Could not read trades file for step {step}: {e}"
                        )

                # Check for early stopping based on cumulative drawdown if needed
                # This requires calculating portfolio equity across folds

            else:
                logger.warning(f"Backtest failed for step {step}.")

            # --- Advance window indices ---
            if wf_config.rolling_window:
                train_start_idx += wf_config.step_periods
            # For expanding window, start_train_idx remains 0
            train_end_idx += wf_config.step_periods
            test_end_idx += wf_config.step_periods

            # Ensure minimum training size for expanding window
            if (
                not wf_config.rolling_window
                and train_end_idx - train_start_idx < wf_config.min_train_periods
            ):
                logger.warning(
                    f"Training window below minimum size ({wf_config.min_train_periods}). Stopping walk-forward."
                )
                break

        logger.info(
            f"--- Walk-Forward Validation Completed for {asset_config.symbol} ---"
        )

        # Combine results
        wf_summary = self.summarize_walk_forward_results(
            wf_results_list, asset_config
        )  # Pass list of dicts

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

        return wf_summary

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
                trades_sorted["pnl_abs"] = (
                    (trades_sorted["exit_price"] - trades_sorted["entry_price"])
                    * trades_sorted["size"]
                    * trades_sorted["direction"]
                )
                # Need initial capital to build equity curve accurately
                initial_cap = 100000  # Assume same as backtest start
                trades_sorted["equity"] = (
                    initial_cap + trades_sorted["pnl_abs"].cumsum()
                )
                # Need to handle timestamps correctly
                trades_sorted["exit_time"] = pd.to_datetime(trades_sorted["exit_time"])
                equity_curve = trades_sorted.set_index("exit_time")["equity"]
                # Reindex to full data range and forward fill
                full_equity = (
                    equity_curve.reindex(full_data.index).ffill().fillna(initial_cap)
                )

                plot_equity_curve(
                    full_equity,
                    title=f"{asset_config.symbol} Walk-Forward Equity Curve",
                    ax=axes[0],
                )  # Pass ax
            else:
                axes[0].set_title(
                    f"{asset_config.symbol} Walk-Forward Equity (Trades Missing)"
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
            rolling_sharpe = (
                summary_df["sharpe_ratio"]
                .rolling(self.config.walkforward_config.performance_tracking_window)
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
                label=f"Rolling Sharpe ({self.config.walkforward_config.performance_tracking_window} folds)",
                color="orange",
            )
            axes[2].set_ylabel("Sharpe Ratio")
            axes[2].set_xlabel("Fold Number")
            axes[2].set_title("Rolling Performance")
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()

            plt.tight_layout()
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

        # Ensure the directory exists before saving
        if filename:
            try:
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                logger.info(f"Saved walk-forward plot to {filename}")
            except Exception as e:
                logger.error(f"Failed to save walk-forward plot to {filename}: {e}")
        else:
            plt.show()
        plt.close(fig)

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

            features, target = self.engineer_features(data, asset_cfg)
            if features is None:
                return {"status": "error", "message": "Feature engineering failed"}

            # --- Detect Market Regime (for single run, based on full features) ---
            current_regime = None
            # Access regime settings via self.config.market_regime_config
            if (
                self.config.market_regime_config.enabled
                and features is not None
                and not features.empty
            ):
                logger.info(
                    f"Detecting market regime for {asset_cfg.symbol} (full data)..."
                )
                try:
                    if (
                        self.config.market_regime_config.method
                        == "volatility_clustering"
                    ):
                        vol_indicator = self.config.market_regime_config.params.get(
                            "volatility_indicator", "atr"
                        )
                        n_clusters = self.config.market_regime_config.params.get(
                            "n_clusters", 3
                        )
                        # Find vol column (improved matching)
                        indicator_pattern = f"{vol_indicator.lower()}"
                        potential_cols = [
                            col
                            for col in features.columns
                            if indicator_pattern in col.lower()
                        ]
                        vol_col = None
                        if potential_cols:
                            vol_period = self.config.market_regime_config.params.get(
                                "volatility_period"
                            )
                            if vol_period:
                                period_pattern = f"{indicator_pattern}_{vol_period}"
                                exact_match = [
                                    col
                                    for col in potential_cols
                                    if period_pattern in col.lower()
                                ]
                                if exact_match:
                                    vol_col = exact_match[0]
                                else:
                                    vol_col = potential_cols[0]
                                    logger.warning(
                                        f"Exact vol col for period {vol_period} not found, using {vol_col}"
                                    )
                            else:
                                vol_col = potential_cols[0]
                            logger.info(
                                f"Using volatility column for regime detection: {vol_col}"
                            )
                        else:
                            logger.warning(
                                f"No column found matching vol indicator '{vol_indicator}'."
                            )

                        if vol_col and vol_col in features.columns:
                            vol_data = features[[vol_col]].dropna()
                            if len(vol_data) > n_clusters:
                                scaler = StandardScaler()
                                scaled_vol = scaler.fit_transform(vol_data)
                                kmeans = KMeans(
                                    n_clusters=n_clusters,
                                    random_state=self.config.random_state,
                                    n_init=10,
                                )
                                kmeans.fit(scaled_vol)
                                # Predict regime for the *last* point as representative
                                last_scaled_vol = scaled_vol[-1].reshape(1, -1)
                                current_regime = kmeans.predict(last_scaled_vol)[0]
                                logger.info(
                                    f"Detected Regime = {current_regime} (based on {vol_col} end value)"
                                )
                            else:
                                logger.warning(
                                    f"Not enough data ({len(vol_data)}) in '{vol_col}' for clustering (need > {n_clusters})."
                                )
                        else:
                            logger.warning(
                                f"Selected vol column '{vol_col}' invalid. Cannot detect regime."
                            )
                    else:
                        logger.warning(
                            f"Unsupported regime method: {self.config.market_regime_config.method}"
                        )
                except Exception as e:
                    logger.exception(f"Error detecting regime: {e}")

            # --- Train Model ---
            model_path = self.train_model(
                features, target, asset_cfg, regime=current_regime
            )  # Pass regime
            if model_path is None:
                return {"status": "error", "message": "Model training failed"}

            # --- Generate Predictions ---
            predictions = self.generate_predictions(
                features, asset_cfg, regime=current_regime
            )  # Pass regime
            if predictions is None:
                return {"status": "error", "message": "Prediction failed"}

            # --- Generate Signals ---
            signals = self.generate_signals(
                predictions,
                data,
                asset_cfg,
                regime=current_regime,
                regime_actions=self.config.market_regime_config.regime_actions,
            )  # Pass regime info
            if signals is None:
                return {"status": "error", "message": "Signal generation failed"}

            # --- Backtest Signals ---
            backtest_results = self.backtest_signals(
                signals,
                data,
                asset_cfg,
                regime=current_regime,
                regime_actions=self.config.market_regime_config.regime_actions,
            )  # Pass regime info

            return {
                "backtest_summary": backtest_results,
                "model_path": model_path,
                "detected_regime": (
                    int(current_regime) if current_regime is not None else None
                ),
            }  # Ensure regime is standard int

        if use_parallel:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                future_to_asset = {}
                for asset_config in self.config.assets:
                    logger.info(f"Submitting job for asset: {asset_config.symbol}")
                    # Submit either walk-forward or single backtest based on config
                    if self.config.walkforward_config.enabled:
                        future = executor.submit(
                            self.run_walk_forward_validation, asset_config
                        )
                    else:
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
                    if self.config.walkforward_config.enabled:
                        result = self.run_walk_forward_validation(asset_config)
                    else:
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
                    status = (
                        result.get("status", "completed")
                        if isinstance(result, dict)
                        else "completed"
                    )
                    logger.info(f"Job finished for {asset_symbol}. Status: {status}")

                except Exception as exc:
                    logger.exception(
                        f"Asset {asset_symbol} generated an exception during execution: {exc}"
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
        # self.plot_strategy_performance(final_summary, all_results)

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
        }
        overall_metrics_list = []  # Store metrics per asset for averaging

        for asset, result in all_results.items():
            if isinstance(result, dict):
                # Store individual asset result (could be WF summary or single backtest summary)
                aggregated_summary["assets"][asset] = result
                summary_data = (
                    result.get("walk_forward_summary")
                    or result.get("backtest_summary")
                    or result
                )
                if isinstance(summary_data, dict) and result.get("status") != "error":
                    # Extract key performance metrics for overall summary calculation
                    overall_metrics_list.append(
                        {
                            "asset": asset,
                            "return": summary_data.get(
                                "annualized_return_pct",
                                summary_data.get(
                                    "compounded_return_pct",
                                    summary_data.get("total_return_pct"),
                                ),
                            ),
                            "sharpe": summary_data.get(
                                "sharpe_ratio", summary_data.get("avg_sharpe")
                            ),
                            "max_dd": summary_data.get(
                                "max_drawdown_pct",
                                summary_data.get("worst_fold_drawdown_pct"),
                            ),
                            "win_rate": summary_data.get(
                                "win_rate_pct", summary_data.get("avg_win_rate_pct")
                            ),
                            "trades": summary_data.get(
                                "num_trades", summary_data.get("total_trades")
                            ),
                        }
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
            asset_returns = [
                res.get("walk_forward_summary", {}).get("annualized_return_pct")
                or res.get("backtest_summary", {}).get("annualized_return_pct")
                or 0
                for res in all_results.values()
                if isinstance(res, dict) and res.get("status") != "error"
            ]
            if not asset_returns:
                logger.warning("No asset returns to plot.")
                return

            plt.figure(figsize=(10, 6))
            sns.histplot(asset_returns, kde=True)
            plt.title(
                f"Distribution of Annualized Returns Across Assets ({self.config.strategy_name})"
            )
            plt.xlabel("Annualized Return (%)")
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


# --- Entry Point ---


def run_trading_strategy(config_file: Optional[str] = None) -> Dict:
    """
    Run the trading strategy, loading config from file if provided.

    Args:
        config_file: Path to strategy configuration YAML file.

    Returns:
        Dictionary with strategy summary results.
    """
    if not MODULES_AVAILABLE:
        logger.critical("Core modules failed to import. Strategy cannot run.")
        return {"error": "Core module import failed."}

    try:
        # Load configuration from file or use defaults
        if config_file:
            config = StrategyConfig.load_config(config_file)
            if config is None:
                return {"error": f"Failed to load config from {config_file}"}
        else:
            # Use default configuration
            logger.info(
                "No config file provided, using default strategy configuration."
            )
            config = StrategyConfig()  # Creates default config and output dir

        # Initialize and run the strategy
        strategy = EnhancedTradingStrategy(config)
        results = strategy.run_strategy()

        return results

    except Exception as e:
        logger.exception(
            f"Critical error running trading strategy: {e}"
        )  # Log traceback
        return {"error": str(e)}


if __name__ == "__main__":
    # Example usage: Run with default config or specify a config file
    # Example: python -m market_ml_model.strategy.main --config path/to/strategy_config.json
    import argparse

    parser = argparse.ArgumentParser(description="Run Enhanced Trading Strategy")
    parser.add_argument(
        "--config", type=str, help="Path to strategy configuration JSON file"
    )
    args = parser.parse_args()

    # Configure root logger for command-line execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    run_trading_strategy(config_file=args.config)
