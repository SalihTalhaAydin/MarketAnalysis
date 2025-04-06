# market_ml_model/trading/__init__.py
"""
Trading Module Initialization

This module provides classes and functions for trade simulation,
management, position sizing, and backtesting.
"""

# Import key classes from submodules
from .trade import Trade
from .manager import TradeManager
# Import other components like backtest, position sizing if they exist and are needed at this level
# from .backtest import backtest_strategy
# from .position import calculate_position_size

# Define what gets imported with 'from market_ml_model.trading import *'
__all__ = [
    "Trade",
    "TradeManager",
    # "backtest_strategy",
    # "calculate_position_size",
]
