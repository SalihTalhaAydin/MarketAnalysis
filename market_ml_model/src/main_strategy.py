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

import os
import time
import json
import logging
import uuid
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import joblib
import warnings
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns

# Import enhanced modules
from enhanced_data_loader import DataLoader, DataLoaderConfig, load_data
from enhanced_feature_engineering import engineer_features
from enhanced_model_training import train_classification_model, PredictionManager
from enhanced_prediction import predict_with_model, SignalGenerator
from enhanced_backtesting import backtest_strategy, TradeManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
        data_source: str = "yahoo",
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        min_position_size: float = 0.01,
        max_position_size: float = 1.0,
        correlation_group: Optional[str] = None
    ):
        """
        Initialize asset configuration.
        
        Args:
            symbol: Asset symbol
            timeframe: Data timeframe
            data_source: Data source
            commission_pct: Commission percentage
            slippage_pct: Slippage percentage
            min_position_size: Minimum position size
            max_position_size: Maximum position size
            correlation_group: Correlation group for risk management
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
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'data_source': self.data_source,
            'commission_pct': self.commission_pct,
            'slippage_pct': self.slippage_pct,
            'min_position_size': self.min_position_size,
            'max_position_size': self.max_position_size,
            'correlation_group': self.correlation_group
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'AssetConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


class FeatureConfig:
    """Configuration for feature engineering."""
    
    def __init__(
        self,
        technical_indicators: bool = True,
        volatility_features: bool = True,
        trend_features: bool = True,
        pattern_features: bool = False,
        price_action_features: bool = True,
        volume_features: bool = True,
        vwap_features: bool = False,
        support_resistance_features: bool = True,
        time_features: bool = True,
        regime_features: bool = True,
        atr_multiplier_tp: float = 2.0,
        atr_multiplier_sl: float = 1.0,
        max_holding_period: int = 10,
        target_type: str = "triple_barrier",
        feature_selection: bool = True,
        pca_enabled: bool = False,
        max_features: int = 50
    ):
        """
        Initialize feature engineering configuration.
        
        Args:
            technical_indicators: Whether to include technical indicators
            volatility_features: Whether to include volatility features
            trend_features: Whether to include trend features
            pattern_features: Whether to include chart pattern features
            price_action_features: Whether to include price action features
            volume_features: Whether to include volume features
            vwap_features: Whether to include VWAP features
            support_resistance_features: Whether to include support/resistance
            time_features: Whether to include time-based features
            regime_features: Whether to include market regime features
            atr_multiplier_tp: ATR multiplier for take profit (triple barrier)
            atr_multiplier_sl: ATR multiplier for stop loss (triple barrier)
            max_holding_period: Maximum holding period (triple barrier)
            target_type: Type of target variable
            feature_selection: Whether to perform feature selection
            pca_enabled: Whether to use PCA for dimensionality reduction
            max_features: Maximum number of features to keep
        """
        self.technical_indicators = technical_indicators
        self.volatility_features = volatility_features
        self.trend_features = trend_features
        self.pattern_features = pattern_features
        self.price_action_features = price_action_features
        self.volume_features = volume_features
        self.vwap_features = vwap_features
        self.support_resistance_features = support_resistance_features
        self.time_features = time_features
        self.regime_features = regime_features
        self.atr_multiplier_tp = atr_multiplier_tp
        self.atr_multiplier_sl = atr_multiplier_sl
        self.max_holding_period = max_holding_period
        self.target_type = target_type
        self.feature_selection = feature_selection
        self.pca_enabled = pca_enabled
        self.max_features = max_features
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'technical_indicators': self.technical_indicators,
            'volatility_features': self.volatility_features,
            'trend_features': self.trend_features,
            'pattern_features': self.pattern_features,
            'price_action_features': self.price_action_features,
            'volume_features': self.volume_features,
            'vwap_features': self.vwap_features,
            'support_resistance_features': self.support_resistance_features,
            'time_features': self.time_features,
            'regime_features': self.regime_features,
            'atr_multiplier_tp': self.atr_multiplier_tp,
            'atr_multiplier_sl': self.atr_multiplier_sl,
            'max_holding_period': self.max_holding_period,
            'target_type': self.target_type,
            'feature_selection': self.feature_selection,
            'pca_enabled': self.pca_enabled,
            'max_features': self.max_features
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'FeatureConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


class ModelConfig:
    """Configuration for model training and prediction."""
    
    def __init__(
        self,
        model_type: str = "lightgbm",
        ensemble_models: bool = True,
        optimize_hyperparams: bool = True,
        optimization_method: str = "bayesian",
        cv_folds: int = 5,
        validation_size: float = 0.2,
        scoring_metric: str = "f1_weighted",
        early_stopping_rounds: int = 50,
        probability_threshold: float = 0.65,
        risk_per_trade: float = 0.02,
        use_kelly_sizing: bool = True,
        max_drawdown_pct: float = 0.15,
        max_open_trades: int = 5,
        max_correlation_exposure: int = 2,
        regime_adaptation: bool = True
    ):
        """
        Initialize model and prediction configuration.
        
        Args:
            model_type: Type of model to train
            ensemble_models: Whether to use ensemble models
            optimize_hyperparams: Whether to optimize hyperparameters
            optimization_method: Method for hyperparameter optimization
            cv_folds: Number of cross-validation folds
            validation_size: Validation set size
            scoring_metric: Scoring metric for optimization
            early_stopping_rounds: Early stopping rounds
            probability_threshold: Probability threshold for signals
            risk_per_trade: Risk per trade (fraction of capital)
            use_kelly_sizing: Whether to use Kelly criterion
            max_drawdown_pct: Maximum drawdown percentage
            max_open_trades: Maximum number of open trades
            max_correlation_exposure: Maximum correlated positions
            regime_adaptation: Whether to adapt to market regimes
        """
        self.model_type = model_type
        self.ensemble_models = ensemble_models
        self.optimize_hyperparams = optimize_hyperparams
        self.optimization_method = optimization_method
        self.cv_folds = cv_folds
        self.validation_size = validation_size
        self.scoring_metric = scoring_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.probability_threshold = probability_threshold
        self.risk_per_trade = risk_per_trade
        self.use_kelly_sizing = use_kelly_sizing
        self.max_drawdown_pct = max_drawdown_pct
        self.max_open_trades = max_open_trades
        self.max_correlation_exposure = max_correlation_exposure
        self.regime_adaptation = regime_adaptation
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'model_type': self.model_type,
            'ensemble_models': self.ensemble_models,
            'optimize_hyperparams': self.optimize_hyperparams,
            'optimization_method': self.optimization_method,
            'cv_folds': self.cv_folds,
            'validation_size': self.validation_size,
            'scoring_metric': self.scoring_metric,
            'early_stopping_rounds': self.early_stopping_rounds,
            'probability_threshold': self.probability_threshold,
            'risk_per_trade': self.risk_per_trade,
            'use_kelly_sizing': self.use_kelly_sizing,
            'max_drawdown_pct': self.max_drawdown_pct,
            'max_open_trades': self.max_open_trades,
            'max_correlation_exposure': self.max_correlation_exposure,
            'regime_adaptation': self.regime_adaptation
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ModelConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    
    def __init__(
        self,
        enabled: bool = True,
        initial_train_size: int = 1000,
        test_size: int = 200,
        step_size: int = 200,
        min_train_size: int = 800,
        retrain_frequency: int = 1,
        rolling_window: bool = False,
        preserve_model_history: bool = True,
        early_stopping_drawdown: float = 0.15,
        performance_tracking_window: int = 10
    ):
        """
        Initialize walk-forward validation configuration.
        
        Args:
            enabled: Whether to use walk-forward validation
            initial_train_size: Initial training window size
            test_size: Test window size
            step_size: Step size for sliding window
            min_train_size: Minimum training window size
            retrain_frequency: How often to retrain the model
            rolling_window: Whether to use rolling window
            preserve_model_history: Whether to save historical models
            early_stopping_drawdown: Drawdown to stop validation
            performance_tracking_window: Window for tracking performance
        """
        self.enabled = enabled
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.min_train_size = min_train_size
        self.retrain_frequency = retrain_frequency
        self.rolling_window = rolling_window
        self.preserve_model_history = preserve_model_history
        self.early_stopping_drawdown = early_stopping_drawdown
        self.performance_tracking_window = performance_tracking_window
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'enabled': self.enabled,
            'initial_train_size': self.initial_train_size,
            'test_size': self.test_size,
            'step_size': self.step_size,
            'min_train_size': self.min_train_size,
            'retrain_frequency': self.retrain_frequency,
            'rolling_window': self.rolling_window,
            'preserve_model_history': self.preserve_model_history,
            'early_stopping_drawdown': self.early_stopping_drawdown,
            'performance_tracking_window': self.performance_tracking_window
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'WalkForwardConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


class StrategyConfig:
    """Main configuration for the trading strategy."""
    
    def __init__(
        self,
        strategy_name: str = "Enhanced ML Strategy",
        description: str = "Machine learning strategy with regime adaptation",
        data_start_date: str = "2020-01-01",
        data_end_date: Optional[str] = None,
        assets: List[AssetConfig] = None,
        feature_config: Optional[FeatureConfig] = None,
        model_config: Optional[ModelConfig] = None,
        walkforward_config: Optional[WalkForwardConfig] = None,
        output_dir: str = "strategy_results",
        parallel_processing: bool = True,
        random_state: int = 42,
        debug_mode: bool = False
    ):
        """
        Initialize strategy configuration.
        
        Args:
            strategy_name: Name of the strategy
            description: Strategy description
            data_start_date: Start date for data
            data_end_date: End date for data
            assets: List of asset configurations
            feature_config: Feature engineering configuration
            model_config: Model and prediction configuration
            walkforward_config: Walk-forward validation configuration
            output_dir: Directory for strategy outputs
            parallel_processing: Whether to use parallel processing
            random_state: Random seed
            debug_mode: Whether to run in debug mode
        """
        self.strategy_name = strategy_name
        self.strategy_id = f"{strategy_name.replace(' ', '_').lower()}_{uuid.uuid4().hex[:8]}"
        self.description = description
        self.data_start_date = data_start_date
        self.data_end_date = data_end_date or datetime.now().strftime("%Y-%m-%d")
        self.assets = assets or [AssetConfig("SPY")]
        self.feature_config = feature_config or FeatureConfig()
        self.model_config = model_config or ModelConfig()
        self.walkforward_config = walkforward_config or WalkForwardConfig()
        self.output_dir = output_dir
        self.parallel_processing = parallel_processing
        self.random_state = random_state
        self.debug_mode = debug_mode
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'strategy_name': self.strategy_name,
            'strategy_id': self.strategy_id,
            'description': self.description,
            'data_start_date': self.data_start_date,
            'data_end_date': self.data_end_date,
            'assets': [asset.to_dict() for asset in self.assets],
            'feature_config': self.feature_config.to_dict(),
            'model_config': self.model_config.to_dict(),
            'walkforward_config': self.walkforward_config.to_dict(),
            'output_dir': self.output_dir,
            'parallel_processing': self.parallel_processing,
            'random_state': self.random_state,
            'debug_mode': self.debug_mode
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'StrategyConfig':
        """Create configuration from dictionary."""
        # Process nested configurations
        assets = [AssetConfig.from_dict(asset) for asset in config_dict.pop('assets', [])]
        
        feature_config_dict = config_dict.pop('feature_config', {})
        feature_config = FeatureConfig.from_dict(feature_config_dict) if feature_config_dict else None
        
        model_config_dict = config_dict.pop('model_config', {})
        model_config = ModelConfig.from_dict(model_config_dict) if model_config_dict else None
        
        walkforward_config_dict = config_dict.pop('walkforward_config', {})
        walkforward_config = WalkForwardConfig.from_dict(walkforward_config_dict) if walkforward_config_dict else None
        
        # Create main configuration
        return cls(
            assets=assets,
            feature_config=feature_config,
            model_config=model_config,
            walkforward_config=walkforward_config,
            **config_dict
        )
    
    def save_config(self, filename: Optional[str] = None) -> str:
        """
        Save configuration to JSON file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved configuration file
        """
        if filename is None:
            filename = os.path.join(self.output_dir, f"{self.strategy_id}_config.json")
        
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        
        logger.info(f"Saved strategy configuration to {filename}")
        return filename
    
    @classmethod
    def load_config(cls, filename: str) -> 'StrategyConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filename: Path to configuration file
            
        Returns:
            Strategy configuration
        """
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


# --- Market Regime Detection ---

class MarketRegimeDetector:
    """Detects market regimes using various indicators."""
    
    def __init__(
        self,
        n_regimes: int = 3,
        lookback_window: int = 50,
        regime_features: List[str] = None,
        use_clustering: bool = True
    ):
        """
        Initialize market regime detector.
        
        Args:
            n_regimes: Number of regimes to detect
            lookback_window: Lookback window for detection
            regime_features: Features to use for regime detection
            use_clustering: Whether to use clustering for detection
        """
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.regime_features = regime_features or [
            'volatility_20', 'trend_strength', 'range_intensity', 'volume_intensity'
        ]
        self.use_clustering = use_clustering
        self.clustering_model = None
        self.regime_history = []
        
        # Try importing clustering libraries
        try:
            from sklearn.cluster import KMeans
            self.clustering_available = True
        except ImportError:
            logger.warning("scikit-learn not available for clustering")
            self.clustering_available = False
    
    def detect_regime(self, data: pd.DataFrame) -> Dict:
        """
        Detect current market regime.
        
        Args:
            data: Market data
            
        Returns:
            Dictionary with regime information
        """
        if not self.use_clustering or not self.clustering_available:
            return self._detect_rule_based_regime(data)
        else:
            return self._detect_clustering_regime(data)
    
    def _detect_rule_based_regime(self, data: pd.DataFrame) -> Dict:
        """
        Detect regime using rule-based approach.
        
        Args:
            data: Market data
            
        Returns:
            Dictionary with regime information
        """
        try:
            # Calculate key regime indicators
            if len(data) < 20:
                logger.warning("Insufficient data for regime detection")
                return {'regime': 0, 'regime_name': 'Unknown', 'confidence': 0.0}
            
            # Volatility regime
            if 'close' in data.columns:
                returns = data['close'].pct_change()
                current_vol = returns.iloc[-20:].std() * np.sqrt(252)  # Annualized
                historical_vol = returns.iloc[-100:].std() * np.sqrt(252) if len(returns) >= 100 else current_vol
                vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
            else:
                vol_ratio = 1.0
            
            # Trend regime
            if 'close' in data.columns and len(data) >= 50:
                sma20 = data['close'].rolling(20).mean()
                sma50 = data['close'].rolling(50).mean()
                trend_direction = 1 if sma20.iloc[-1] > sma50.iloc[-1] else -1
                trend_intensity = abs(sma20.iloc[-1] / sma50.iloc[-1] - 1)
            else:
                trend_direction = 0
                trend_intensity = 0.0
            
            # Range regime
            if all(col in data.columns for col in ['high', 'low', 'close']):
                high_low_range = (data['high'] - data['low']) / data['close']
                range_intensity = high_low_range.iloc[-20:].mean() / high_low_range.iloc[-100:].mean() if len(high_low_range) >= 100 else 1.0
            else:
                range_intensity = 1.0
            
            # Volume regime
            if 'volume' in data.columns and len(data) >= 50:
                volume_ratio = data['volume'].iloc[-20:].mean() / data['volume'].iloc[-50:].mean()
            else:
                volume_ratio = 1.0
            
            # Determine regime
            # Volatility threshold for high volatility regime
            high_vol_threshold = 1.5
            low_vol_threshold = 0.75
            
            # Trend threshold for trending regime
            high_trend_threshold = 0.05
            
            if vol_ratio > high_vol_threshold:
                regime = 2  # High volatility regime
                regime_name = "High Volatility"
                confidence = min(1.0, (vol_ratio - high_vol_threshold) / high_vol_threshold)
            elif vol_ratio < low_vol_threshold and range_intensity < 0.9:
                regime = 1  # Range-bound regime
                regime_name = "Range-Bound"
                confidence = min(1.0, (low_vol_threshold - vol_ratio) / low_vol_threshold)
            elif trend_intensity > high_trend_threshold:
                regime = 0  # Trending regime
                regime_name = "Trending" + (" Up" if trend_direction > 0 else " Down")
                confidence = min(1.0, trend_intensity / high_trend_threshold)
            else:
                # Mixed signals
                regime = 3  # Mixed regime
                regime_name = "Mixed"
                confidence = 0.5
            
            # Record regime information
            regime_info = {
                'regime': regime,
                'regime_name': regime_name,
                'confidence': confidence,
                'vol_ratio': vol_ratio,
                'trend_intensity': trend_intensity,
                'trend_direction': trend_direction,
                'range_intensity': range_intensity,
                'volume_ratio': volume_ratio,
                'timestamp': data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else None
            }
            
            self.regime_history.append(regime_info)
            
            # Limit history length
            max_history = 1000
            if len(self.regime_history) > max_history:
                self.regime_history = self.regime_history[-max_history:]
            
            return regime_info
            
        except Exception as e:
            logger.error(f"Error detecting rule-based regime: {e}")
            return {'regime': 0, 'regime_name': 'Unknown', 'confidence': 0.0}
    
    def _detect_clustering_regime(self, data: pd.DataFrame) -> Dict:
        """
        Detect regime using clustering.
        
        Args:
            data: Market data
            
        Returns:
            Dictionary with regime information
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Check if enough data for clustering
            if len(data) < self.lookback_window:
                logger.warning("Insufficient data for clustering-based regime detection")
                return self._detect_rule_based_regime(data)
            
            # Extract features for clustering
            feature_data = []
            for feature in self.regime_features:
                if feature in data.columns:
                    feature_data.append(data[feature].iloc[-self.lookback_window:].values)
                elif feature == 'volatility' and 'close' in data.columns:
                    returns = data['close'].pct_change()
                    volatility = returns.rolling(20).std().iloc[-self.lookback_window:].values
                    feature_data.append(volatility)
                elif feature == 'trend_strength' and 'close' in data.columns:
                    sma20 = data['close'].rolling(20).mean()
                    sma50 = data['close'].rolling(50).mean()
                    trend = ((sma20 / sma50) - 1).iloc[-self.lookback_window:].values
                    feature_data.append(trend)
            
            if not feature_data:
                logger.warning("No valid features for clustering")
                return self._detect_rule_based_regime(data)
            
            # Prepare data for clustering
            X = np.column_stack(feature_data)
            
            # Handle NaNs
            X = np.nan_to_num(X, nan=0.0)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train clustering model if not already trained
            if self.clustering_model is None:
                self.clustering_model = KMeans(n_clusters=self.n_regimes, random_state=42)
                self.clustering_model.fit(X_scaled)
            
            # Get current regime
            current_features = X_scaled[-1:, :]
            regime = self.clustering_model.predict(current_features)[0]
            
            # Calculate confidence (distance to cluster center)
            distances = self.clustering_model.transform(current_features)
            min_distance = distances[0, regime]
            max_distance = np.max(distances)
            confidence = 1.0 - (min_distance / max_distance) if max_distance > 0 else 0.5
            
            # Get regime name based on properties
            regime_names = ["Trending", "Range-Bound", "High Volatility", "Mixed"]
            
            # Record regime information
            regime_info = {
                'regime': int(regime),
                'regime_name': regime_names[regime % len(regime_names)],
                'confidence': float(confidence),
                'features': {
                    feature: float(X[-1, i]) for i, feature in enumerate(self.regime_features)
                    if i < X.shape[1]
                },
                'timestamp': data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else None
            }
            
            self.regime_history.append(regime_info)
            
            # Limit history length
            max_history = 1000
            if len(self.regime_history) > max_history:
                self.regime_history = self.regime_history[-max_history:]
            
            return regime_info
            
        except Exception as e:
            logger.error(f"Error detecting clustering-based regime: {e}")
            return self._detect_rule_based_regime(data)
    
    def plot_regime_history(
        self,
        price_data: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (12, 8),
        filename: Optional[str] = None
    ) -> None:
        """
        Plot regime history with price data if available.
        
        Args:
            price_data: Price data
            figsize: Figure size
            filename: If provided, save plot to this file
        """
        if not self.regime_history:
            logger.warning("No regime history to plot")
            return
        
        try:
            # Create DataFrame from regime history
            regime_df = pd.DataFrame(self.regime_history)
            
            # Create plot
            plt.figure(figsize=figsize)
            
            # Plot price data if available
            ax1 = plt.subplot(2, 1, 1)
            if price_data is not None and 'close' in price_data.columns:
                ax1.plot(price_data.index, price_data['close'], color='black', label='Price')
                
                # Highlight regimes
                if 'timestamp' in regime_df.columns and 'regime' in regime_df.columns:
                    # Ensure regimes align with price data
                    regime_df = regime_df[regime_df['timestamp'].isin(price_data.index)]
                    
                    for regime in regime_df['regime'].unique():
                        regime_periods = regime_df[regime_df['regime'] == regime]
                        regime_color = f'C{regime % 10}'
                        
                        for i in range(len(regime_periods) - 1):
                            start_time = regime_periods.iloc[i]['timestamp']
                            end_time = regime_periods.iloc[i + 1]['timestamp']
                            
                            if start_time in price_data.index and end_time in price_data.index:
                                start_idx = price_data.index.get_loc(start_time)
                                end_idx = price_data.index.get_loc(end_time)
                                
                                ax1.axvspan(
                                    price_data.index[start_idx],
                                    price_data.index[end_idx],
                                    alpha=0.3,
                                    color=regime_color
                                )
            
            ax1.set_title('Price and Regime History')
            ax1.grid(alpha=0.3)
            ax1.legend()
            
            # Plot regime classifications
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            ax2.plot(regime_df['timestamp'], regime_df['regime'], color='blue', label='Regime')
            ax2.set_yticks(sorted(regime_df['regime'].unique()))
            ax2.set_yticklabels([regime_df[regime_df['regime'] == r]['regime_name'].iloc[0] 
                                for r in sorted(regime_df['regime'].unique())])
            ax2.grid(alpha=0.3)
            ax2.set_title('Market Regimes')
            
            plt.tight_layout()
            
            # Save or show plot
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Regime history plot saved to {filename}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting regime history: {e}")


# --- Strategy Class ---

class EnhancedTradingStrategy:
    """Main class for the enhanced trading strategy."""
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize the trading strategy.
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.data_loader = None
        self.regime_detector = MarketRegimeDetector()
        self.models = {}
        self.feature_selectors = {}
        self.preprocessors = {}
        self.signal_generators = {}
        self.backtester = None
        self.current_regime = None
        self.performance_metrics = {}
        
        # Create output directories
        self.model_dir = os.path.join(config.output_dir, "models")
        self.data_dir = os.path.join(config.output_dir, "data")
        self.results_dir = os.path.join(config.output_dir, "results")
        self.plots_dir = os.path.join(config.output_dir, "plots")
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Save configuration
        self.config.save_config(os.path.join(self.config.output_dir, "strategy_config.json"))
        
        # Initialize data loader
        self._initialize_data_loader()
    
    def _initialize_data_loader(self) -> None:
        """Initialize the data loader."""
        try:
            # Create data loader configuration
            loader_config = DataLoaderConfig(
                data_source=self.config.assets[0].data_source,  # Using first asset's source
                cache_dir=self.data_dir,
                use_cache=True
            )
            
            # Create data loader
            self.data_loader = DataLoader(loader_config)
            
            logger.info("Data loader initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing data loader: {e}")
    
    def load_data(self, asset_config: AssetConfig) -> pd.DataFrame:
        """
        Load data for an asset.
        
        Args:
            asset_config: Asset configuration
            
        Returns:
            DataFrame with asset data
        """
        try:
            # Use data loader if available
            if self.data_loader is not None:
                data = self.data_loader.load_data(
                    ticker=asset_config.symbol,
                    start_date=self.config.data_start_date,
                    end_date=self.config.data_end_date,
                    interval=asset_config.timeframe
                )
            else:
                # Fall back to direct loading
                data = load_data(
                    ticker=asset_config.symbol,
                    start_date=self.config.data_start_date,
                    end_date=self.config.data_end_date,
                    interval=asset_config.timeframe,
                    data_source=asset_config.data_source
                )
            
            if data is None or data.empty:
                logger.error(f"Failed to load data for {asset_config.symbol}")
                return pd.DataFrame()
            
            logger.info(f"Loaded data for {asset_config.symbol}: {len(data)} rows")
            return data
        
        except Exception as e:
            logger.error(f"Error loading data for {asset_config.symbol}: {e}")
            return pd.DataFrame()
    
    def engineer_features(
        self,
        data: pd.DataFrame,
        asset_config: AssetConfig,
        regime_info: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Engineer features for an asset.
        
        Args:
            data: Asset data
            asset_config: Asset configuration
            regime_info: Current market regime information
            
        Returns:
            DataFrame with engineered features
        """
        try:
            # Prepare additional feature parameters
            feature_params = {}
            
            # Add regime information if available
            if regime_info:
                feature_params['market_regime'] = regime_info['regime']
                feature_params['regime_confidence'] = regime_info['confidence']
            
            # Get feature configuration
            feature_config = self.config.feature_config
            
            # Prepare list of feature groups to include
            additional_features = []
            
            if feature_config.technical_indicators:
                additional_features.append('basic_ta')
            
            if feature_config.volatility_features:
                additional_features.append('volatility')
            
            if feature_config.trend_features:
                additional_features.append('trend')
            
            if feature_config.pattern_features:
                additional_features.append('patterns')
            
            if feature_config.price_action_features:
                additional_features.append('price_action')
            
            if feature_config.volume_features and 'volume' in data.columns:
                additional_features.append('volume_indicators')
            
            if feature_config.vwap_features and 'volume' in data.columns:
                additional_features.append('vwap')
            
            if feature_config.support_resistance_features:
                additional_features.append('support_resistance')
            
            if feature_config.time_features:
                additional_features.append('time')
            
            if feature_config.regime_features:
                additional_features.append('market_regime')
            
            # Engineer features
            featured_data = engineer_features(
                df=data,
                atr_multiplier_tp=feature_config.atr_multiplier_tp,
                atr_multiplier_sl=feature_config.atr_multiplier_sl,
                max_holding_period=feature_config.max_holding_period,
                target_type=feature_config.target_type,
                additional_features=additional_features,
                feature_config=feature_params
            )
            
            if featured_data is None or featured_data.empty:
                logger.error(f"Failed to engineer features for {asset_config.symbol}")
                return pd.DataFrame()
            
            logger.info(f"Engineered features for {asset_config.symbol}: {featured_data.shape}")
            return featured_data
        
        except Exception as e:
            logger.error(f"Error engineering features for {asset_config.symbol}: {e}")
            return pd.DataFrame()
    
    def train_model(
        self,
        features: pd.DataFrame,
        asset_config: AssetConfig,
        fold_id: Optional[str] = None,
        regime_model: bool = False
    ) -> Tuple[Any, pd.DataFrame, Dict]:
        """
        Train a model for an asset.
        
        Args:
            features: Feature DataFrame
            asset_config: Asset configuration
            fold_id: Walk-forward fold ID
            regime_model: Whether this is a regime-specific model
            
        Returns:
            Tuple of (trained_model, feature_importance, metrics)
        """
        try:
            # Check if target column exists
            target_column = 'triple_barrier_label'
            if target_column not in features.columns:
                logger.error(f"Target column '{target_column}' not found in features")
                return None, pd.DataFrame(), {}
            
            # Separate features and target
            X = features.drop(columns=[target_column])
            y = features[target_column]
            
            # Get model configuration
            model_config = self.config.model_config
            
            # Configure model parameters
            model_params = {
                'model_type': model_config.model_type,
                'feature_selection': self.config.feature_config.feature_selection,
                'feature_selection_params': {
                    'method': 'importance',
                    'n_features': self.config.feature_config.max_features
                },
                'test_size': model_config.validation_size,
                'optimize_hyperparams': model_config.optimize_hyperparams,
                'optimization_params': {
                    'method': model_config.optimization_method,
                    'n_iter': 20,
                    'cv': model_config.cv_folds,
                    'scoring': model_config.scoring_metric
                },
                'random_state': self.config.random_state
            }
            
            # Determine model ID
            model_id = fold_id or f"{asset_config.symbol}_{uuid.uuid4().hex[:8]}"
            
            if regime_model:
                regime = self.current_regime['regime'] if self.current_regime else 0
                model_id = f"{model_id}_regime{regime}"
            
            # Create model output directory
            model_output_dir = os.path.join(self.model_dir, model_id)
            os.makedirs(model_output_dir, exist_ok=True)
            
            # Train model
            model, importance_df, metrics = train_classification_model(
                features=X,
                target=y,
                model_config=model_params,
                output_dir=model_output_dir,
                model_id=model_id
            )
            
            if model is None:
                logger.error(f"Failed to train model for {asset_config.symbol}")
                return None, pd.DataFrame(), {}
            
            logger.info(f"Trained model for {asset_config.symbol}: {model_id}")
            
            # Return trained model
            return model, importance_df, metrics
        
        except Exception as e:
            logger.error(f"Error training model for {asset_config.symbol}: {e}")
            return None, pd.DataFrame(), {}
    
    def generate_predictions(
        self,
        model: Any,
        features: pd.DataFrame,
        asset_config: AssetConfig,
        regime_info: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Generate predictions for an asset.
        
        Args:
            model: Trained model
            features: Feature DataFrame
            asset_config: Asset configuration
            regime_info: Current market regime information
            
        Returns:
            DataFrame with predictions
        """
        try:
            # Target column name
            target_column = 'triple_barrier_label'
            
            # Prepare features
            if target_column in features.columns:
                X = features.drop(columns=[target_column])
            else:
                X = features
            
            # Generate predictions
            predictions, probabilities = predict_with_model(
                model=model,
                features=X,
                threshold=self.config.model_config.probability_threshold
            )
            
            if len(predictions) == 0:
                logger.error(f"Failed to generate predictions for {asset_config.symbol}")
                return pd.DataFrame()
            
            # Create results DataFrame
            results = pd.DataFrame(index=X.index)
            results['prediction'] = predictions
            
            # Add probabilities if available
            if probabilities.size > 0:
                for i in range(probabilities.shape[1]):
                    results[f'probability_{i}'] = probabilities[:, i]
                
                # Add confidence
                results['confidence'] = np.max(probabilities, axis=1)
            
            # Add regime information if available
            if regime_info:
                results['regime'] = regime_info['regime']
                results['regime_name'] = regime_info['regime_name']
                results['regime_confidence'] = regime_info['confidence']
            
            logger.info(f"Generated predictions for {asset_config.symbol}: {len(results)} rows")
            return results
        
        except Exception as e:
            logger.error(f"Error generating predictions for {asset_config.symbol}: {e}")
            return pd.DataFrame()
    
    def generate_signals(
        self,
        predictions: pd.DataFrame,
        asset_config: AssetConfig,
        regime_info: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Generate trading signals for an asset.
        
        Args:
            predictions: Prediction DataFrame
            asset_config: Asset configuration
            regime_info: Current market regime information
            
        Returns:
            DataFrame with trading signals
        """
        try:
            # Create signal generator if not exists
            if asset_config.symbol not in self.signal_generators:
                # Configure signal generator
                signal_config = {
                    'threshold': self.config.model_config.probability_threshold,
                    'confidence_multiplier': 1.0,
                    'min_confidence': 0.5,
                    'trend_filter_enabled': True,
                    'volatility_filter_enabled': True,
                    'signal_cooling_periods': 0,
                    'filter_consolidation': 'and',
                    'signal_mapping': {
                        0: 0,  # Neutral
                        1: 1,  # Long
                        2: -1  # Short
                    }
                }
                
                # Adjust signal parameters based on regime
                if regime_info:
                    if regime_info['regime'] == 0:  # Trending
                        # Trending regime - focus on trend following
                        signal_config['trend_filter_enabled'] = True
                        signal_config['volatility_filter_enabled'] = False
                        signal_config['threshold'] = 0.6  # Lower threshold
                    elif regime_info['regime'] == 1:  # Range-bound
                        # Range regime - focus on mean reversion
                        signal_config['trend_filter_enabled'] = False
                        signal_config['volatility_filter_enabled'] = True
                        signal_config['threshold'] = 0.7  # Higher threshold
                    elif regime_info['regime'] == 2:  # High volatility
                        # Volatile regime - more conservative
                        signal_config['trend_filter_enabled'] = True
                        signal_config['volatility_filter_enabled'] = True
                        signal_config['threshold'] = 0.75  # Higher threshold
                        signal_config['min_confidence'] = 0.6  # Higher confidence
                
                # Create dummy predictor for signal generator
                class DummyPredictor:
                    def predict_proba(self, features):
                        # Extract probabilities from predictions DataFrame
                        prob_columns = [col for col in predictions.columns if col.startswith('probability_')]
                        
                        if not prob_columns:
                            # If no probability columns, use prediction column
                            probs = np.zeros((len(predictions), 3))
                            for i, pred in enumerate(predictions['prediction']):
                                probs[i, int(pred)] = 1.0
                            return probs, ['0', '1', '2']
                        
                        # Extract probabilities
                        probs = predictions[prob_columns].values
                        class_names = [col.split('_')[1] for col in prob_columns]
                        
                        return probs, class_names
                
                # Create signal generator
                self.signal_generators[asset_config.symbol] = SignalGenerator(
                    predictor=DummyPredictor(),
                    signal_config=signal_config
                )
            
            # Generate signals
            signals = pd.DataFrame(index=predictions.index)
            signals['prediction'] = predictions['prediction']
            
            # Add probability columns if available
            prob_columns = [col for col in predictions.columns if col.startswith('probability_')]
            for col in prob_columns:
                signals[col] = predictions[col]
            
            # Add confidence if available
            if 'confidence' in predictions.columns:
                signals['confidence'] = predictions['confidence']
            
            # Add regime information if available
            if regime_info:
                signals['regime'] = regime_info['regime']
                signals['regime_name'] = regime_info['regime_name']
            
            # Add signal column
            signal_generator = self.signal_generators[asset_config.symbol]
            signal_df = signal_generator.generate_signals(signals, None)
            
            if 'signal' in signal_df.columns:
                signals['signal'] = signal_df['signal']
            else:
                # Fall back to prediction if signal generation fails
                signals['signal'] = signals['prediction'].map({0: 0, 1: 1, 2: -1})
            
            logger.info(f"Generated signals for {asset_config.symbol}: {len(signals)} rows")
            return signals
        
        except Exception as e:
            logger.error(f"Error generating signals for {asset_config.symbol}: {e}")
            return pd.DataFrame()
    
    def backtest_signals(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        asset_config: AssetConfig,
        regime_info: Optional[Dict] = None,
        fold_id: Optional[str] = None
    ) -> Dict:
        """
        Backtest trading signals for an asset.
        
        Args:
            data: Asset data
            signals: Signal DataFrame
            asset_config: Asset configuration
            regime_info: Current market regime information
            fold_id: Walk-forward fold ID
            
        Returns:
            Dictionary with backtest results
        """
        try:
            # Prepare data with predictions
            data_with_predictions = data.copy()
            
            # Align signals with data
            signals_aligned = signals.reindex(data_with_predictions.index)
            
            # Add prediction column
            if 'signal' in signals_aligned.columns:
                data_with_predictions['prediction'] = signals_aligned['signal']
            elif 'prediction' in signals_aligned.columns:
                data_with_predictions['prediction'] = signals_aligned['prediction']
            else:
                logger.error("No signal or prediction column in signals DataFrame")
                return {}
            
            # Fill missing predictions with 0 (neutral)
            data_with_predictions['prediction'] = data_with_predictions['prediction'].fillna(0)
            
            # Define backtesting parameters
            backtest_params = {
                'transaction_cost_pct': asset_config.commission_pct,
                'slippage_pct_per_trade': asset_config.slippage_pct,
                'initial_capital': 10000.0,
                'risk_per_trade': self.config.model_config.risk_per_trade,
                'use_kelly_sizing': self.config.model_config.use_kelly_sizing,
                'atr_multiplier_sl': self.config.feature_config.atr_multiplier_sl,
                'atr_multiplier_tp': self.config.feature_config.atr_multiplier_tp,
                'trailing_stop_pct': 0.01,  # 1% trailing stop
                'max_open_trades': self.config.model_config.max_open_trades,
                'max_drawdown_pct': self.config.model_config.max_drawdown_pct,
                'signal_threshold': self.config.model_config.probability_threshold,
                'save_detailed_report': True
            }
            
            # Define output directory for backtest results
            output_id = fold_id or asset_config.symbol
            if regime_info:
                output_id = f"{output_id}_regime{regime_info['regime']}"
            
            output_dir = os.path.join(self.results_dir, output_id)
            os.makedirs(output_dir, exist_ok=True)
            
            backtest_params['output_dir'] = output_dir
            backtest_params['output_trades_path'] = os.path.join(output_dir, "trades.csv")
            
            # Run backtest
            backtest_results = backtest_strategy(
                data_with_predictions=data_with_predictions,
                **backtest_params
            )
            
            if not backtest_results:
                logger.error(f"Failed to backtest signals for {asset_config.symbol}")
                return {}
            
            # Add regime information if available
            if regime_info:
                backtest_results['regime'] = regime_info['regime']
                backtest_results['regime_name'] = regime_info['regime_name']
            
            # Add fold information if available
            if fold_id:
                backtest_results['fold_id'] = fold_id
            
            # Add timestamp
            backtest_results['timestamp'] = datetime.now().isoformat()
            
            # Save backtest results
            with open(os.path.join(output_dir, "backtest_results.json"), "w") as f:
                json.dump(backtest_results, f, indent=4, default=str)
            
            logger.info(f"Completed backtest for {asset_config.symbol}: "
                       f"Return {backtest_results.get('total_return_pct', 0):.2f}%")
            
            return backtest_results
        
        except Exception as e:
            logger.error(f"Error backtesting signals for {asset_config.symbol}: {e}")
            return {}
    
    def run_walk_forward_validation(self, asset_config: AssetConfig) -> Dict:
        """
        Run walk-forward validation for an asset.
        
        Args:
            asset_config: Asset configuration
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Get walk-forward configuration
            wf_config = self.config.walkforward_config
            
            if not wf_config.enabled:
                logger.info(f"Walk-forward validation disabled for {asset_config.symbol}")
                return {}
            
            # Load data
            data = self.load_data(asset_config)
            
            if data is None or data.empty:
                logger.error(f"No data available for {asset_config.symbol}")
                return {}
            
            # Detect initial market regime
            self.current_regime = self.regime_detector.detect_regime(data)
            
            # Engineer features for the entire dataset
            features = self.engineer_features(data, asset_config, self.current_regime)
            
            if features is None or features.empty:
                logger.error(f"Failed to engineer features for {asset_config.symbol}")
                return {}
            
            # Set up walk-forward validation
            n_samples = len(features)
            
            if n_samples < wf_config.initial_train_size + wf_config.test_size:
                logger.error(f"Insufficient data for walk-forward validation: {n_samples} samples")
                return {}
            
            # Initialize walk-forward tracking
            all_fold_performances = []
            all_signals = pd.DataFrame()
            fold_number = 0
            start_train_idx = 0
            
            # Prepare output directory for this asset
            asset_output_dir = os.path.join(self.results_dir, asset_config.symbol)
            os.makedirs(asset_output_dir, exist_ok=True)
            
            # Walk-forward validation loop
            while start_train_idx + wf_config.initial_train_size + wf_config.test_size <= n_samples:
                fold_number += 1
                fold_id = f"{asset_config.symbol}_fold{fold_number}"
                
                # Calculate window indices
                end_train_idx = start_train_idx + wf_config.initial_train_size
                start_test_idx = end_train_idx
                end_test_idx = start_test_idx + wf_config.test_size
                
                # Ensure minimum train size
                if end_train_idx - start_train_idx < wf_config.min_train_size:
                    logger.info(f"Minimum train size not met for fold {fold_number}. Stopping.")
                    break
                
                logger.info(f"--- WFV Fold {fold_number} ---")
                logger.info(f"Train window: {start_train_idx} - {end_train_idx-1} "
                           f"({end_train_idx-start_train_idx} samples)")
                logger.info(f"Test window: {start_test_idx} - {end_test_idx-1} "
                           f"({end_test_idx-start_test_idx} samples)")
                
                # Slice data for the current fold
                train_features = features.iloc[start_train_idx:end_train_idx]
                test_features = features.iloc[start_test_idx:end_test_idx]
                
                train_data = data.iloc[start_train_idx:end_train_idx]
                test_data = data.iloc[start_test_idx:end_test_idx]
                
                # Check for empty slices
                if train_features.empty or test_features.empty:
                    logger.warning(f"Empty train/test features for fold {fold_number}. Skipping.")
                    
                    # Move to next fold
                    start_train_idx += wf_config.step_size
                    
                    # Update training window based on rolling/expanding window setting
                    if wf_config.rolling_window:
                        # Rolling window: start_train_idx moves forward by step_size
                        pass
                    else:
                        # Expanding window: initial_train_size increases by step_size
                        wf_config.initial_train_size += wf_config.step_size
                    
                    continue
                
                # Detect market regime for the test period
                test_regime = self.regime_detector.detect_regime(test_data)
                self.current_regime = test_regime
                
                # Train a model for this fold
                model, importance_df, metrics = self.train_model(
                    features=train_features,
                    asset_config=asset_config,
                    fold_id=fold_id,
                    regime_model=True
                )
                
                if model is None:
                    logger.warning(f"Failed to train model for fold {fold_number}. Skipping.")
                    
                    # Move to next fold
                    start_train_idx += wf_config.step_size
                    
                    # Update training window
                    if wf_config.rolling_window:
                        pass
                    else:
                        wf_config.initial_train_size += wf_config.step_size
                    
                    continue
                
                # Generate predictions for test set
                predictions = self.generate_predictions(
                    model=model,
                    features=test_features,
                    asset_config=asset_config,
                    regime_info=test_regime
                )
                
                if predictions.empty:
                    logger.warning(f"Failed to generate predictions for fold {fold_number}. Skipping.")
                    
                    # Move to next fold
                    start_train_idx += wf_config.step_size
                    
                    # Update training window
                    if wf_config.rolling_window:
                        pass
                    else:
                        wf_config.initial_train_size += wf_config.step_size
                    
                    continue
                
                # Generate signals
                signals = self.generate_signals(
                    predictions=predictions,
                    asset_config=asset_config,
                    regime_info=test_regime
                )
                
                if signals.empty:
                    logger.warning(f"Failed to generate signals for fold {fold_number}. Skipping.")
                    
                    # Move to next fold
                    start_train_idx += wf_config.step_size
                    
                    # Update training window
                    if wf_config.rolling_window:
                        pass
                    else:
                        wf_config.initial_train_size += wf_config.step_size
                    
                    continue
                
                # Backtest signals
                backtest_results = self.backtest_signals(
                    data=test_data,
                    signals=signals,
                    asset_config=asset_config,
                    regime_info=test_regime,
                    fold_id=fold_id
                )
                
                if not backtest_results:
                    logger.warning(f"Failed to backtest signals for fold {fold_number}. Skipping.")
                    
                    # Move to next fold
                    start_train_idx += wf_config.step_size
                    
                    # Update training window
                    if wf_config.rolling_window:
                        pass
                    else:
                        wf_config.initial_train_size += wf_config.step_size
                    
                    continue
                
                # Save fold results
                backtest_results['fold_number'] = fold_number
                backtest_results['train_start'] = train_features.index[0].isoformat() if isinstance(train_features.index, pd.DatetimeIndex) else start_train_idx
                backtest_results['train_end'] = train_features.index[-1].isoformat() if isinstance(train_features.index, pd.DatetimeIndex) else end_train_idx
                backtest_results['test_start'] = test_features.index[0].isoformat() if isinstance(test_features.index, pd.DatetimeIndex) else start_test_idx
                backtest_results['test_end'] = test_features.index[-1].isoformat() if isinstance(test_features.index, pd.DatetimeIndex) else end_test_idx
                
                all_fold_performances.append(backtest_results)
                
                # Store signals for this fold
                signals['fold_number'] = fold_number
                all_signals = pd.concat([all_signals, signals])
                
                # Check for early stopping due to drawdown
                if backtest_results.get('max_drawdown_pct', 0) > wf_config.early_stopping_drawdown:
                    logger.warning(f"Stopping walk-forward validation due to excessive drawdown "
                                  f"({backtest_results['max_drawdown_pct']:.2f}%)")
                    break
                
                # Move to the next fold
                start_train_idx += wf_config.step_size
                
                # Update training window based on rolling/expanding window setting
                if wf_config.rolling_window:
                    # Rolling window: size stays the same
                    pass
                else:
                    # Expanding window: size increases by step_size
                    wf_config.initial_train_size += wf_config.step_size
            
            # Save all walk-forward results
            if all_fold_performances:
                # Save performance summary
                with open(os.path.join(asset_output_dir, "walk_forward_results.json"), "w") as f:
                    json.dump(all_fold_performances, f, indent=4, default=str)
                
                # Save signals
                if not all_signals.empty:
                    all_signals.to_csv(os.path.join(asset_output_dir, "walk_forward_signals.csv"))
                
                # Generate summary metrics
                summary = self.summarize_walk_forward_results(all_fold_performances, asset_config)
                
                # Save summary
                with open(os.path.join(asset_output_dir, "walk_forward_summary.json"), "w") as f:
                    json.dump(summary, f, indent=4, default=str)
                
                # Plot results
                self.plot_walk_forward_results(
                    all_fold_performances=all_fold_performances,
                    all_signals=all_signals,
                    asset_config=asset_config,
                    data=data,
                    filename=os.path.join(self.plots_dir, f"{asset_config.symbol}_walk_forward.png")
                )
                
                logger.info(f"Completed walk-forward validation for {asset_config.symbol}: "
                           f"{len(all_fold_performances)} folds")
                
                return summary
            else:
                logger.warning(f"No successful walk-forward folds for {asset_config.symbol}")
                return {}
        
        except Exception as e:
            logger.error(f"Error running walk-forward validation for {asset_config.symbol}: {e}")
            return {}
    
    def summarize_walk_forward_results(
        self,
        all_fold_performances: List[Dict],
        asset_config: AssetConfig
    ) -> Dict:
        """
        Summarize walk-forward validation results.
        
        Args:
            all_fold_performances: List of fold performance dictionaries
            asset_config: Asset configuration
            
        Returns:
            Dictionary with summary metrics
        """
        try:
            # Extract key metrics across all folds
            returns = [perf.get('total_return_pct', 0) for perf in all_fold_performances]
            win_rates = [perf.get('win_rate_pct', 0) for perf in all_fold_performances]
            sharpe_ratios = [perf.get('sharpe_ratio', 0) for perf in all_fold_performances]
            max_drawdowns = [perf.get('max_drawdown_pct', 0) for perf in all_fold_performances]
            num_trades = [perf.get('num_trades', 0) for perf in all_fold_performances]
            
            # Calculate aggregate metrics
            summary = {
                'symbol': asset_config.symbol,
                'num_folds': len(all_fold_performances),
                'total_return_pct': sum(returns),
                'avg_return_per_fold_pct': np.mean(returns) if returns else 0,
                'std_return_per_fold_pct': np.std(returns) if returns else 0,
                'avg_win_rate_pct': np.mean(win_rates) if win_rates else 0,
                'avg_sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0,
                'avg_max_drawdown_pct': np.mean(max_drawdowns) if max_drawdowns else 0,
                'worst_drawdown_pct': max(max_drawdowns) if max_drawdowns else 0,
                'total_trades': sum(num_trades),
                'avg_trades_per_fold': np.mean(num_trades) if num_trades else 0,
                'best_fold': max(range(len(returns)), key=lambda i: returns[i]) + 1 if returns else 0,
                'worst_fold': min(range(len(returns)), key=lambda i: returns[i]) + 1 if returns else 0,
                'best_fold_return_pct': max(returns) if returns else 0,
                'worst_fold_return_pct': min(returns) if returns else 0,
                'profitable_folds': sum(1 for r in returns if r > 0),
                'profitable_folds_pct': (sum(1 for r in returns if r > 0) / len(returns) * 100) if returns else 0
            }
            
            # Calculate compounded return
            compounded_return = 1.0
            for r in returns:
                compounded_return *= (1 + r / 100)
            
            summary['compounded_return_pct'] = (compounded_return - 1) * 100
            
            # Calculate annualized return if we have datetime information
            first_fold = all_fold_performances[0] if all_fold_performances else {}
            last_fold = all_fold_performances[-1] if all_fold_performances else {}
            
            if 'test_start' in first_fold and 'test_end' in last_fold:
                try:
                    start_date = pd.to_datetime(first_fold['test_start'])
                    end_date = pd.to_datetime(last_fold['test_end'])
                    
                    years = (end_date - start_date).days / 365.25
                    
                    if years > 0:
                        summary['years'] = years
                        summary['annualized_return_pct'] = ((compounded_return) ** (1 / years) - 1) * 100
                except Exception as e:
                    logger.warning(f"Error calculating annualized return: {e}")
            
            # Record regime performance if available
            regime_returns = {}
            
            for perf in all_fold_performances:
                if 'regime' in perf:
                    regime = perf['regime']
                    regime_name = perf.get('regime_name', f"Regime {regime}")
                    
                    if regime_name not in regime_returns:
                        regime_returns[regime_name] = []
                    
                    regime_returns[regime_name].append(perf.get('total_return_pct', 0))
            
            # Calculate performance by regime
            if regime_returns:
                summary['regime_performance'] = {}
                
                for regime_name, returns in regime_returns.items():
                    summary['regime_performance'][regime_name] = {
                        'num_folds': len(returns),
                        'total_return_pct': sum(returns),
                        'avg_return_pct': np.mean(returns) if returns else 0,
                        'std_return_pct': np.std(returns) if returns else 0
                    }
            
            return summary
        
        except Exception as e:
            logger.error(f"Error summarizing walk-forward results: {e}")
            return {
                'symbol': asset_config.symbol,
                'num_folds': len(all_fold_performances),
                'error': str(e)
            }
    
    def plot_walk_forward_results(
        self,
        all_fold_performances: List[Dict],
        all_signals: pd.DataFrame,
        asset_config: AssetConfig,
        data: pd.DataFrame,
        filename: Optional[str] = None
    ) -> None:
        """
        Plot walk-forward validation results.
        
        Args:
            all_fold_performances: List of fold performance dictionaries
            all_signals: DataFrame with signals from all folds
            asset_config: Asset configuration
            data: Asset data
            filename: Output filename
        """
        try:
            if not all_fold_performances or all_signals.empty:
                logger.warning("No data to plot walk-forward results")
                return
            
            # Create figure
            plt.figure(figsize=(16, 12))
            
            # Plot 1: Price with signals
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            
            # Plot price
            if 'close' in data.columns:
                ax1.plot(data.index, data['close'], color='black', label='Price')
                
                # Plot signals
                buy_signals = all_signals[all_signals['signal'] > 0]
                sell_signals = all_signals[all_signals['signal'] < 0]
                
                if not buy_signals.empty:
                    buy_prices = data.loc[buy_signals.index, 'close']
                    ax1.scatter(buy_signals.index, buy_prices, color='green', s=100, marker='^', label='Buy Signal')
                
                if not sell_signals.empty:
                    sell_prices = data.loc[sell_signals.index, 'close']
                    ax1.scatter(sell_signals.index, sell_prices, color='red', s=100, marker='v', label='Sell Signal')
                
                # Highlight fold test periods
                for i, perf in enumerate(all_fold_performances):
                    if 'test_start' in perf and 'test_end' in perf:
                        try:
                            test_start = pd.to_datetime(perf['test_start'])
                            test_end = pd.to_datetime(perf['test_end'])
                            
                            color = f'C{i % 10}'
                            alpha = 0.2
                            
                            # Highlight test period
                            ax1.axvspan(test_start, test_end, alpha=alpha, color=color)
                            
                            # Add fold number
                            mid_point = test_start + (test_end - test_start) / 2
                            ax1.text(mid_point, ax1.get_ylim()[1] * 0.95, f"F{i+1}", 
                                    horizontalalignment='center', color=color)
                        except Exception as e:
                            logger.warning(f"Error plotting fold {i+1}: {e}")
            
            ax1.set_title(f"{asset_config.symbol} - Walk-Forward Validation", fontsize=14)
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # Plot 2: Cumulative returns by fold
            ax2 = plt.subplot2grid((3, 1), (2, 0))
            
            # Prepare data for cumulative returns
            cum_returns = []
            fold_numbers = []
            fold_returns = []
            
            for perf in all_fold_performances:
                fold_numbers.append(perf.get('fold_number', len(fold_numbers) + 1))
                fold_returns.append(perf.get('total_return_pct', 0))
            
            # Calculate cumulative returns
            cum_return = 0
            for fold_return in fold_returns:
                cum_return += fold_return
                cum_returns.append(cum_return)
            
            # Plot cumulative returns
            ax2.plot(fold_numbers, cum_returns, marker='o', linestyle='-', color='blue', 
                     label=f"Cumulative Return: {cum_return:.2f}%")
            
            # Plot individual fold returns
            for i, (fold, ret) in enumerate(zip(fold_numbers, fold_returns)):
                color = 'green' if ret > 0 else 'red'
                ax2.bar(fold, ret, color=color, alpha=0.6)
            
            ax2.set_title("Fold Returns and Cumulative Performance", fontsize=12)
            ax2.set_xlabel("Fold Number")
            ax2.set_ylabel("Return (%)")
            ax2.grid(alpha=0.3)
            ax2.legend()
            
            # Add summary statistics
            summary = self.summarize_walk_forward_results(all_fold_performances, asset_config)
            
            summary_text = (
                f"Total Return: {summary.get('total_return_pct', 0):.2f}% | "
                f"Avg Return/Fold: {summary.get('avg_return_per_fold_pct', 0):.2f}% | "
                f"Win Rate: {summary.get('avg_win_rate_pct', 0):.1f}% | "
                f"Trades: {summary.get('total_trades', 0)}"
            )
            
            plt.figtext(0.5, 0.01, summary_text, ha='center', fontsize=12, 
                       bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 5})
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            
            # Save or show plot
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Saved walk-forward plot to {filename}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting walk-forward results: {e}")
    
    def run_strategy(self) -> Dict:
        """
        Run the complete trading strategy.
        
        Returns:
            Dictionary with strategy results
        """
        try:
            logger.info(f"Starting strategy: {self.config.strategy_name}")
            
            # Run for each asset
            all_results = {}
            
            if self.config.parallel_processing and len(self.config.assets) > 1:
                # Run assets in parallel
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = {}
                    
                    for asset_config in self.config.assets:
                        future = executor.submit(self.run_walk_forward_validation, asset_config)
                        futures[asset_config.symbol] = future
                    
                    for symbol, future in futures.items():
                        try:
                            result = future.result(timeout=3600)  # 1 hour timeout
                            all_results[symbol] = result
                        except Exception as e:
                            logger.error(f"Error processing {symbol} in parallel: {e}")
                            all_results[symbol] = {'error': str(e)}
            else:
                # Run assets sequentially
                for asset_config in self.config.assets:
                    result = self.run_walk_forward_validation(asset_config)
                    all_results[asset_config.symbol] = result
            
            # Summarize all results
            overall_summary = self.summarize_strategy_results(all_results)
            
            # Save overall summary
            with open(os.path.join(self.config.output_dir, "strategy_summary.json"), "w") as f:
                json.dump(overall_summary, f, indent=4, default=str)
            
            # Create overall performance plot
            self.plot_strategy_performance(all_results, overall_summary)
            
            logger.info(f"Strategy complete: {self.config.strategy_name}")
            
            return overall_summary
        
        except Exception as e:
            logger.error(f"Error running strategy: {e}")
            return {'error': str(e)}
    
    def summarize_strategy_results(self, all_results: Dict) -> Dict:
        """
        Summarize results for all assets.
        
        Args:
            all_results: Dictionary with results for each asset
            
        Returns:
            Dictionary with overall summary
        """
        try:
            # Prepare summary metrics
            summary = {
                'strategy_name': self.config.strategy_name,
                'strategy_id': self.config.strategy_id,
                'num_assets': len(all_results),
                'assets': list(all_results.keys()),
                'total_return_pct': 0,
                'compounded_return_pct': 0,
                'average_sharpe_ratio': 0,
                'average_win_rate_pct': 0,
                'total_trades': 0,
                'max_drawdown_pct': 0,
                'profitable_assets': 0,
                'asset_returns': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate strategy-wide metrics
            asset_returns = []
            asset_sharpes = []
            asset_win_rates = []
            asset_drawdowns = []
            
            for symbol, result in all_results.items():
                if not result or 'error' in result:
                    summary['asset_returns'][symbol] = 0
                    continue
                
                # Extract key metrics
                total_return = result.get('total_return_pct', 0)
                sharpe_ratio = result.get('avg_sharpe_ratio', 0)
                win_rate = result.get('avg_win_rate_pct', 0)
                max_drawdown = result.get('worst_drawdown_pct', 0)
                total_trades = result.get('total_trades', 0)
                
                # Store asset return
                summary['asset_returns'][symbol] = total_return
                
                # Update totals
                asset_returns.append(total_return)
                asset_sharpes.append(sharpe_ratio)
                asset_win_rates.append(win_rate)
                asset_drawdowns.append(max_drawdown)
                
                summary['total_trades'] += total_trades
                
                # Count profitable assets
                if total_return > 0:
                    summary['profitable_assets'] += 1
            
            # Calculate averages
            if asset_returns:
                summary['total_return_pct'] = sum(asset_returns)
                summary['avg_asset_return_pct'] = np.mean(asset_returns)
                
                # Compounded return (if treating assets as a portfolio)
                compounded_return = 1.0
                for ret in asset_returns:
                    compounded_return *= (1 + ret / 100)
                
                summary['compounded_return_pct'] = (compounded_return - 1) * 100
            
            if asset_sharpes:
                summary['average_sharpe_ratio'] = np.mean(asset_sharpes)
            
            if asset_win_rates:
                summary['average_win_rate_pct'] = np.mean(asset_win_rates)
            
            if asset_drawdowns:
                summary['max_drawdown_pct'] = max(asset_drawdowns)
            
            # Calculate percentage of profitable assets
            if all_results:
                summary['profitable_assets_pct'] = (summary['profitable_assets'] / len(all_results)) * 100
            
            return summary
        
        except Exception as e:
            logger.error(f"Error summarizing strategy results: {e}")
            return {
                'strategy_name': self.config.strategy_name,
                'error': str(e)
            }
    
    def plot_strategy_performance(
        self,
        all_results: Dict,
        overall_summary: Dict,
        filename: Optional[str] = None
    ) -> None:
        """
        Plot overall strategy performance.
        
        Args:
            all_results: Dictionary with results for each asset
            overall_summary: Overall strategy summary
            filename: Output filename
        """
        try:
            # Create figure
            plt.figure(figsize=(16, 10))
            
            # Plot 1: Asset Returns
            ax1 = plt.subplot2grid((2, 2), (0, 0))
            
            # Extract asset returns
            symbols = []
            returns = []
            
            for symbol, result in all_results.items():
                if not result or 'error' in result:
                    continue
                
                symbols.append(symbol)
                returns.append(result.get('total_return_pct', 0))
            
            # Sort by return
            sorted_indices = np.argsort(returns)
            sorted_symbols = [symbols[i] for i in sorted_indices]
            sorted_returns = [returns[i] for i in sorted_indices]
            
            # Plot bars
            bars = ax1.barh(sorted_symbols, sorted_returns, color=['red' if r < 0 else 'green' for r in sorted_returns])
            
            # Add values to bars
            for bar, value in zip(bars, sorted_returns):
                text_color = 'white' if abs(value) > 20 else 'black'
                ax1.text(
                    value * 0.95 if value < 0 else value * 0.05,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}%",
                    ha='right' if value < 0 else 'left',
                    va='center',
                    color=text_color,
                    fontweight='bold'
                )
            
            ax1.set_title("Asset Returns", fontsize=12)
            ax1.grid(alpha=0.3)
            ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)
            
            # Plot 2: Return Distribution
            ax2 = plt.subplot2grid((2, 2), (0, 1))
            
            # Create histogram
            if returns:
                ax2.hist(returns, bins=10, alpha=0.7, color='blue')
                
                # Add statistics
                mean_return = np.mean(returns)
                median_return = np.median(returns)
                std_return = np.std(returns)
                
                ax2.axvline(mean_return, color='red', linestyle='--', label=f"Mean: {mean_return:.1f}%")
                ax2.axvline(median_return, color='green', linestyle='-.', label=f"Median: {median_return:.1f}%")
                
                # Add normal distribution fit
                x = np.linspace(min(returns), max(returns), 100)
                y = 0.4 * len(returns) * np.exp(-(x - mean_return)**2 / (2 * std_return**2)) / (std_return * np.sqrt(2 * np.pi))
                ax2.plot(x, y, 'r-', linewidth=2)
            
            ax2.set_title("Return Distribution", fontsize=12)
            ax2.set_xlabel("Return (%)")
            ax2.set_ylabel("Frequency")
            ax2.grid(alpha=0.3)
            ax2.legend()
            
            # Plot 3: Performance Metrics
            ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
            
            # Create metrics table
            metrics = [
                ['Total Return', f"{overall_summary.get('total_return_pct', 0):.2f}%"],
                ['Compounded Return', f"{overall_summary.get('compounded_return_pct', 0):.2f}%"],
                ['Avg Sharpe Ratio', f"{overall_summary.get('average_sharpe_ratio', 0):.2f}"],
                ['Avg Win Rate', f"{overall_summary.get('average_win_rate_pct', 0):.1f}%"],
                ['Max Drawdown', f"{overall_summary.get('max_drawdown_pct', 0):.2f}%"],
                ['Total Trades', f"{overall_summary.get('total_trades', 0)}"],
                ['Profitable Assets', f"{overall_summary.get('profitable_assets', 0)}/{overall_summary.get('num_assets', 0)} ({overall_summary.get('profitable_assets_pct', 0):.1f}%)"]
            ]
            
            # Hide axes
            ax3.axis('off')
            
            # Create table
            table = ax3.table(
                cellText=metrics,
                colLabels=['Metric', 'Value'],
                cellLoc='center',
                loc='center',
                colWidths=[0.4, 0.4]
            )
            
            # Style table
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            
            # Title and subtitle
            plt.suptitle(f"Strategy Performance: {self.config.strategy_name}", fontsize=16)
            plt.figtext(0.5, 0.01, f"Period: {self.config.data_start_date} to {self.config.data_end_date}", 
                        ha='center', fontsize=10)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save or show plot
            if filename:
                output_path = filename or os.path.join(self.plots_dir, "strategy_performance.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved strategy performance plot to {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting strategy performance: {e}")


# --- Main Function ---

def run_trading_strategy(config_file: Optional[str] = None) -> Dict:
    """
    Run the trading strategy from configuration.
    
    Args:
        config_file: Path to strategy configuration file
        
    Returns:
        Dictionary with strategy results
    """
    try:
        # Load configuration from file or use defaults
        if config_file:
            config = StrategyConfig.load_config(config_file)
        else:
            # Use default configuration
            config = StrategyConfig()
        
        # Create and run strategy
        strategy = EnhancedTradingStrategy(config)
        results = strategy.run_strategy()
        
        return results
    
    except Exception as e:
        logger.error(f"Error running trading strategy: {e}")
        return {'error': str(e)}


if __name__ == "__main__":
    # Run with default configuration
    results = run_trading_strategy()
    print(results)