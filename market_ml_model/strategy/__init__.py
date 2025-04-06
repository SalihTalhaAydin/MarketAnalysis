# market_ml_model/strategy/__init__.py
"""
Strategy Module Initialization

This module provides the core components for defining, configuring, and running
trading strategies, including ML-based strategies with regime adaptation and
walk-forward validation.
"""

# Import key configuration classes
from .config import (
    AssetConfig,
    DifferencingConfig,
    FeatureConfig,
    MarketRegimeConfig,
    ModelConfig,
    ScalingConfig,
    StrategyConfig,
    TradingConfig,
    WalkForwardConfig,
)

# Import core strategy class
from .core import EnhancedTradingStrategy

# Import regime detection class
from .regime import MarketRegimeDetector

# Import the main runner function
from .runner import run_trading_strategy

# Define what gets imported with 'from market_ml_model.strategy import *'
__all__ = [
    "AssetConfig",
    "DifferencingConfig",
    "FeatureConfig",
    "MarketRegimeConfig",
    "ModelConfig",
    "ScalingConfig",
    "StrategyConfig",
    "TradingConfig",
    "WalkForwardConfig",
    "EnhancedTradingStrategy",
    "MarketRegimeDetector",
    "run_trading_strategy",
]
