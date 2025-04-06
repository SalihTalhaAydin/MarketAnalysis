# market_ml_model/trading/simulation.py
"""
DEPRECATED: Original simulation module.

Functionality has been refactored into:
- market_ml_model.trading.trade (contains Trade class)
- market_ml_model.trading.manager (contains TradeManager class)

Please update imports accordingly. See market_ml_model/trading/__init__.py
"""

import warnings

warnings.warn(
    "market_ml_model.trading.simulation is deprecated. Use components from market_ml_model.trading.* submodules.",
    DeprecationWarning,
    stacklevel=2,
)

# Optionally re-export main components for backward compatibility (use with caution)
# from .trade import Trade
# from .manager import TradeManager
#
# __all__ = ["Trade", "TradeManager"]
