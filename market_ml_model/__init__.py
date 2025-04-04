"""
Market ML Model - Automated Trading Framework

A comprehensive machine learning framework for developing,
testing and deploying trading strategies.
"""

__version__ = "0.1.0"

# Import key components for easy access
from .data import DataLoader, load_data
from .features import engineer_features
from .models import train_classification_model
from .trading import backtest_strategy, TradeManager
from .strategy import run_trading_strategy, StrategyConfig