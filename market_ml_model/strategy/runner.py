# market_ml_model/strategy/runner.py
import argparse
import logging
from typing import Dict, Optional

# Import necessary components from the refactored modules
from .config import StrategyConfig
from .core import (
    MODULES_AVAILABLE,  # Import MODULES_AVAILABLE check
    EnhancedTradingStrategy,
)

logger = logging.getLogger(__name__)


def run_trading_strategy(config_file: Optional[str] = None) -> Dict:
    """
    Run the trading strategy, loading config from file if provided.

    Args:
        config_file: Path to strategy configuration YAML file.

    Returns:
        Dictionary with strategy summary results.
    """
    if not MODULES_AVAILABLE:
        logger.critical("Core modules failed to import. Strategy cannot run.")
        return {"error": "Core module import failed."}

    try:
        # Load configuration from file or use defaults
        if config_file:
            config = StrategyConfig.load_config(config_file)
            if config is None:
                return {"error": f"Failed to load config from {config_file}"}
        else:
            # Use default configuration
            logger.info(
                "No config file provided, using default strategy configuration."
            )
            config = StrategyConfig()  # Creates default config and output dir

        # Initialize and run the strategy
        strategy = EnhancedTradingStrategy(config)
        results = strategy.run_strategy()

        return results

    except Exception as e:
        logger.exception(
            f"Critical error running trading strategy: {e}"
        )  # Log traceback
        return {"error": str(e)}


if __name__ == "__main__":
    # Example usage: Run with default config or specify a config file
    # Example: python -m market_ml_model.strategy.runner --config path/to/strategy_config.yaml
    parser = argparse.ArgumentParser(description="Run Enhanced Trading Strategy")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to strategy configuration YAML file",  # Changed help text to YAML
    )
    args = parser.parse_args()

    # Configure root logger for command-line execution
    # Basic config should be set up higher level if running as package,
    # but this is okay for direct script execution.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],  # Log to console by default
    )

    run_trading_strategy(config_file=args.config)
