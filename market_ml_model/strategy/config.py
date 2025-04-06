# market_ml_model/strategy/config.py
import json
import logging
import os
import uuid
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Suppress warnings moved here as they are relevant to config loading/defaults
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
        # Make a copy to avoid modifying the original dict during processing
        cfg = config_dict.copy()

        # Handle nested position sizing if present in YAML
        pos_sizing_config = cfg.pop("position_sizing", {})
        cfg.setdefault(
            "position_sizing_method", pos_sizing_config.get("method", "kelly")
        )
        cfg.setdefault("fixed_fraction_amount", pos_sizing_config.get("fraction", 0.02))
        # Handle nested stop loss if present in YAML
        stop_loss_config = cfg.pop("stop_loss", {})
        if isinstance(stop_loss_config, dict):
            # Also check for method: null in stop_loss section
            if stop_loss_config.get("method") is None:
                cfg["use_dynamic_stops"] = False  # Override if method is null
            if isinstance(stop_loss_config.get("params"), dict):
                sl_multiplier = stop_loss_config["params"].get("multiplier")
                if sl_multiplier is not None:
                    cfg["atr_multiplier_sl"] = sl_multiplier

        # Handle nested take profit if present in YAML
        take_profit_config = cfg.pop("take_profit", {})
        if isinstance(take_profit_config, dict):
            # Also check for method: null in take_profit section
            if take_profit_config.get("method") is None:
                # If SL method was null, use_dynamic_stops is already False
                # If SL method was not null, but TP is null, keep use_dynamic_stops as is (might be True for SL)
                pass
            if isinstance(take_profit_config.get("params"), dict):
                tp_multiplier = take_profit_config["params"].get("multiplier")
                if tp_multiplier is not None:
                    cfg["atr_multiplier_tp"] = tp_multiplier

        # Explicitly handle top-level use_dynamic_stops, prioritizing it if present
        # Default to True only if not specified anywhere (top-level or implied by method: null)
        if "use_dynamic_stops" in config_dict:  # Check original dict for top-level key
            cfg["use_dynamic_stops"] = config_dict["use_dynamic_stops"]
        elif (
            "use_dynamic_stops" not in cfg
        ):  # If not set by stop_loss method: null check
            cfg["use_dynamic_stops"] = True  # Default to True if completely absent

        # Remove keys that might have been processed but are not direct __init__ args
        cfg.pop("stop_loss", None)
        cfg.pop("take_profit", None)
        cfg.pop("position_sizing", None)
        # Pop calculated percentage values if they exist in the dict, as __init__ expects bps
        cfg.pop("commission_pct", None)
        cfg.pop("slippage_pct", None)

        return cls(**cfg)


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
        strategy_type: Optional[
            str
        ] = None,  # Type of strategy (e.g., 'ml', 'dma_crossover', 'rsi_mean_reversion')
        # DMA Crossover Params
        sma_short_period: Optional[int] = None,
        sma_long_period: Optional[int] = None,
        # RSI Mean Reversion Params
        rsi_period: Optional[int] = None,
        rsi_lower_threshold: Optional[float] = None,
        rsi_upper_threshold: Optional[float] = None,
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
        self.strategy_type = strategy_type
        self.sma_short_period = sma_short_period
        self.sma_long_period = sma_long_period
        self.rsi_period = rsi_period
        self.rsi_lower_threshold = rsi_lower_threshold
        self.rsi_upper_threshold = rsi_upper_threshold
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
            "strategy_type": self.strategy_type,
            "sma_short_period": self.sma_short_period,
            "sma_long_period": self.sma_long_period,
            "rsi_period": self.rsi_period,
            "rsi_lower_threshold": self.rsi_lower_threshold,
            "rsi_upper_threshold": self.rsi_upper_threshold,
            "data_start_date": self.data_start_date,
            "data_end_date": self.data_end_date,
            "assets": [asset.to_dict() for asset in self.assets],
            "feature_config": self.feature_config.to_dict(),
            "model_config": self.model_config.to_dict(),
            "trading": self.trading_config.to_dict(),  # Use 'trading' key for consistency
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

        # Pop market regime section and create config object
        market_regime_dict = config_dict.pop("market_regime", {})
        market_regime_config = (
            MarketRegimeConfig.from_dict(market_regime_dict)
            if market_regime_dict
            else MarketRegimeConfig()
        )

        # Create main configuration, remove run_id if present as it's generated
        config_dict.pop("strategy_run_id", None)

        # Pop strategy type and DMA params explicitly
        strategy_type = config_dict.pop("strategy_type", None)
        sma_short_period = config_dict.pop("sma_short_period", None)
        sma_long_period = config_dict.pop("sma_long_period", None)
        rsi_period = config_dict.pop("rsi_period", None)
        rsi_lower_threshold = config_dict.pop("rsi_lower_threshold", None)
        rsi_upper_threshold = config_dict.pop("rsi_upper_threshold", None)

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
        # Pop search_space if it exists at the top level, as it's handled explicitly
        config_dict.pop("search_space", None)

        # Create the main instance first
        instance = cls(
            assets=assets,
            feature_config=feature_config,
            model_config=model_config,
            trading_config=trading_config,
            walkforward_config=walkforward_config,
            market_regime_config=market_regime_config,  # Pass MarketRegimeConfig instance
            strategy_type=strategy_type,
            sma_short_period=sma_short_period,
            sma_long_period=sma_long_period,
            rsi_period=rsi_period,
            rsi_lower_threshold=rsi_lower_threshold,
            rsi_upper_threshold=rsi_upper_threshold,
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
