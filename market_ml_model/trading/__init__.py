"""
Trading simulation and backtesting module.
"""

from .simulation import TradeManager, Trade
from .backtest import backtest_strategy
from .position.position_sizing import calculate_position_size

__all__ = [
    "TradeManager",
    "Trade",
    "backtest_strategy",
    "calculate_position_size",
]