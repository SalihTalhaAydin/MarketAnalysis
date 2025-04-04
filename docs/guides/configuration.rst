# Configuring Strategies

Trading strategies within the MarketAnalysis Framework are configured using YAML files. This allows for flexible and readable parameterization without modifying the core code.

## Configuration File Location

-   Strategy configuration files should typically reside in the `configs/` directory at the root of the project.
-   An example configuration is provided in `configs/example_strategy_config.yaml`.

## Loading Configuration

The framework loads configuration using the `StrategyConfig.load_config` method, which parses the specified YAML file. The main entry point `run_trading_strategy` accepts a `--config` argument (or similar mechanism, TBD) to specify the configuration file path.

```python
# Example usage within a script or main entry point
from market_ml_model.strategy.main import StrategyConfig, run_trading_strategy

# Load config from a specific file
config_path = "configs/my_strategy.yaml"
config = StrategyConfig.load_config(config_path)

if config:
    # Run the strategy using the loaded configuration
    results = run_trading_strategy(config_file=config_path)
    print("Strategy Results:", results)
else:
    print(f"Failed to load configuration from {config_path}")

# If no config file is provided, run_trading_strategy uses default settings
# results_default = run_trading_strategy()
```

## Configuration Structure

The YAML configuration file mirrors the structure of the configuration classes defined in `market_ml_model/strategy/main.py` (`StrategyConfig`, `AssetConfig`, `FeatureConfig`, `ModelConfig`, `WalkForwardConfig`, `TradingConfig`, `MarketRegimeConfig`).

Refer to `configs/example_strategy_config.yaml` for a detailed breakdown of available parameters and their structure. Key sections include:

-   `strategy_name`, `description`
-   `assets`: List of assets to trade, including symbol, dates, data source.
-   `features`: Configuration for technical indicators, target labeling, scaling.
-   `model`: Model type, parameters, feature selection, hyperparameter optimization settings.
-   `walk_forward`: Parameters for walk-forward validation (window sizes, step, retraining).
-   `trading`: Backtesting parameters like initial capital, commission, slippage, position sizing, stop-loss/take-profit methods.
-   `market_regime` (Optional): Settings for market regime detection and adaptation.

## Customizing Strategies

To run a different strategy:

1.  Copy `configs/example_strategy_config.yaml` to a new file (e.g., `configs/my_awesome_strategy.yaml`).
2.  Modify the parameters within the new YAML file according to your desired strategy settings (e.g., change assets, indicators, model type, backtesting rules).
3.  Run the main strategy execution script, pointing it to your new configuration file.

*(Note: The exact command-line interface for specifying the config file in `run_trading_strategy` might need further implementation using libraries like `argparse` or `click`.)*
