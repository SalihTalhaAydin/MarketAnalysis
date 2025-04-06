"""
Market ML Model - Automated Trading Framework

A comprehensive machine learning framework for developing,
testing and deploying trading strategies.
"""

__version__ = "0.1.0"

# Import key components for easy access
from .data import DataLoader, load_data
from .features import engineer_features
from .models.factory.model_factory import create_model
from .models.utils import predict_with_model
from .models.training import train_classification_model
from .strategy.config import StrategyConfig
from .strategy.runner import run_trading_strategy
from .trading import TradeManager
from .trading.backtest import backtest_strategy
from .utils import calculate_returns_metrics  # Example utility import

__all__ = [
    # Version
    "__version__",
    # Data
    "DataLoader",
    "load_data",
    # Features
    "engineer_features",
    # Models
    "train_classification_model",
    "predict_with_model",
    "create_model",
    # Trading
    "backtest_strategy",
    "TradeManager",
    # Strategy
    "run_trading_strategy",
    "StrategyConfig",
    # Utils
    "calculate_returns_metrics",
]
