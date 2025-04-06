# market_ml_model/strategy/main.py
"""
DEPRECATED: Original main strategy module.

Functionality has been refactored into:
- market_ml_model.strategy.config
- market_ml_model.strategy.regime
- market_ml_model.strategy.core
- market_ml_model.strategy.runner

Please update imports accordingly. See market_ml_model/strategy/__init__.py
"""

import warnings

warnings.warn(
    "market_ml_model.strategy.main is deprecated. Use components from market_ml_model.strategy.* submodules.",
    DeprecationWarning,
    stacklevel=2,
)

# Optionally re-export main components for backward compatibility (use with caution)
# from .config import StrategyConfig
# from .core import EnhancedTradingStrategy
# from .runner import run_trading_strategy
#
# __all__ = ["StrategyConfig", "EnhancedTradingStrategy", "run_trading_strategy"]
