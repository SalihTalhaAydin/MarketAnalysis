"""
Trading simulation and signal generation module.
"""

from .simulation import TradeManager, Trade
from .backtest import backtest_strategy
from .signals.signal_generator import SignalGenerator
from .position.position_sizing import calculate_position_size