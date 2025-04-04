"""
Trading simulation and backtesting module.
"""

from .backtest import backtest_strategy
from .position.position_sizing import calculate_position_size
from .simulation import Trade, TradeManager

__all__ = [
    "TradeManager",
    "Trade",
    "backtest_strategy",
    "calculate_position_size",
]
