Configuration Guide
===================

The MarketAnalysis Framework uses a central YAML configuration file to define all aspects of a trading strategy, from data loading and feature engineering to model training and backtesting. This guide explains the structure and key parameters of this configuration file.

You can find a complete example at ``configs/example_strategy_config.yaml``.

.. contents:: Sections
   :local:
   :depth: 2

General Strategy Settings
-------------------------

This section defines high-level properties of the strategy run.

*   ``strategy_name``: (String) A unique name for this strategy run. Used for organizing results.
*   ``description``: (String) A brief description of the strategy.
*   ``log_level``: (String) Logging verbosity. Options: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``. Default: ``INFO``.
*   ``results_dir``: (String) Path to the directory where results (logs, plots, models) will be saved. Default: ``strategy_results``.
*   ``data_start_date``: (String, YYYY-MM-DD) The earliest date for which to load historical data.
*   ``data_end_date``: (String, YYYY-MM-DD) The latest date for which to load historical data.

Asset Configuration
-------------------

Defines the financial instruments the strategy will trade.

*   ``assets``: (List of Objects) Each object represents an asset.
    *   ``symbol``: (String) The ticker symbol (e.g., "AAPL", "BTC/USDT"). Format depends on the ``data_source``.
    *   ``data_source``: (String) The source for historical data (e.g., "yahoo", "ccxt"). Must match a registered data loader.
    *   ``exchange``: (String, Optional) Required for some sources like "ccxt" (e.g., "binance").
    *   ``timeframe``: (String, Optional) Data frequency (e.g., "1d", "1h", "5m").

Feature Engineering (`feature_config`)
--------------------------------------

This section configures how raw market data is transformed into features for the machine learning model. It corresponds to the ``FeatureConfig`` object in the code.

Technical Indicators
~~~~~~~~~~~~~~~~~~~~
*   ``technical_indicators``: (List of Objects) Defines standard technical indicators to calculate. Uses `pandas-ta <https://github.com/twopirllc/pandas-ta>`_ or `ta <https://github.com/bukosabino/ta>`_ conventions.
    *   ``indicator``: (String) Name of the indicator (e.g., "rsi", "sma", "macd").
    *   *Other keys*: Parameters specific to the indicator (e.g., ``length``, ``fast_period``, ``slow_period``).

Additional Feature Categories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Boolean flags to enable/disable calculation of custom feature sets:

*   ``volatility_features``: (Boolean) Enable/disable volatility-based features.
*   ``regime_features``: (Boolean) Enable/disable market regime-based features.
*   ``momentum_features``: (Boolean) Enable/disable momentum-based features.
*   ``fractal_features``: (Boolean) Enable/disable fractal dimension features.
*   ``fractal_window``: (Integer, Optional) Window size if ``fractal_features`` is true. Default: 5.

Target Variable
~~~~~~~~~~~~~~~
Configures the prediction target for the ML model.

*   ``target_type``: (String) Method for generating the target label.
    *   ``'triple_barrier'``: Uses the triple barrier method (profit take, stop loss, time limit).
    *   ``'directional'``: Predicts simple price direction (up/down).
*   *Triple Barrier Parameters* (used if ``target_type: 'triple_barrier'``):
    *   ``atr_multiplier_tp``: (Float) Profit-take barrier distance as a multiple of ATR.
    *   ``atr_multiplier_sl``: (Float) Stop-loss barrier distance as a multiple of ATR.
    *   ``max_holding_period``: (Integer) Maximum number of bars to hold a position before the time barrier is hit.

Feature Selection
~~~~~~~~~~~~~~~~~
Configures optional feature selection methods to reduce dimensionality.

*   ``feature_selection``: (Object) Contains feature selection settings.
    *   ``enabled``: (Boolean) Set to ``true`` to perform feature selection.
    *   ``method``: (String) The selection algorithm to use. Options
        *   ``'importance'``: Use RandomForest feature importance.
        *   ``'mutual_info'``: Use mutual information (SelectKBest).
        *   ``'rfe'``: Use Recursive Feature Elimination (with Logistic Regression).
        *   ``'pca'``: Use Principal Component Analysis (transforms features).
        *   ``'model'``: Use feature importance from a specified model (LightGBM/XGBoost via SelectFromModel).
        *   ``null`` or ``'none'``: Disable feature selection.

    *   ``params``: (Object) Parameters specific to the chosen ``method``.
        *   ``n_features``: (Integer) Target number of features for most methods.
        *   ``threshold``: (Float or String) Importance/coefficient threshold (alternative for 'importance', 'model').
        *   ``step``: (Integer) Step size for 'rfe'.
        *   ``n_components``: (Integer) Number of components for 'pca'.
        *   ``model_type``: (String) Model for 'model' method ('lightgbm' or 'xgboost').

Feature Differencing (`differencing`)
-------------------------------------

Optional configuration to apply differencing to features (often used for stationarity).

*   ``apply``: (Boolean) Set to ``true`` to enable differencing.
*   ``period``: (Integer) Differencing period (usually 1).
*   ``features``: (List of Strings, Optional) List of feature names to difference. If empty or null, no differencing is applied.

Feature Scaling (`scaling`)
---------------------------

Optional configuration to scale features. Applied *after* differencing if both are enabled.

*   ``method``: (String) Scaling method. Options: ``'StandardScaler'``, ``'MinMaxScaler'``, ``'RobustScaler'``, ``'None'`` or ``null``.

Model Configuration (`model_config`)
------------------------------------

Configures the machine learning model used for prediction. Corresponds to the ``ModelConfig`` object.

*   ``model_type``: (String) The type of model to use. Must match a key registered in ``market_ml_model.models.model_factory.py`` (e.g., "lightgbm", "random_forest", "logistic_regression").
*   ``probability_threshold``: (Float) Minimum predicted probability required to generate a trading signal (for binary classification models). Default: 0.5.
*   ``signal_trend_filter_ma``: (Integer or null) If set to an integer, requires price to be above the SMA of this length for long signals (and below for short). If ``null``, the filter is disabled.
*   ``signal_volatility_filter_atr``: (Float or null) If set to a float, requires ATR (14) to be above this threshold to generate signals. If ``null``, the filter is disabled.

Hyperparameter Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Optional configuration for automatic hyperparameter tuning using Optuna.

*   ``optimize_hyperparams``: (Boolean) Set to ``true`` to enable optimization.
*   ``optimization_method``: (String) Tuning algorithm (e.g., "bayesian", "random"). Default: "bayesian".
*   ``optimization_trials``: (Integer) Number of optimization trials to run. Default: 100.
*   ``search_space``: (Object) Defines the parameters and ranges to search. Structure depends on Optuna's trial suggestion methods (e.g., ``suggest_int``, ``suggest_float``).
    *   Each key is a hyperparameter name (e.g., ``n_estimators``).
    *   The value is an object specifying ``type`` ("int", "float", "categorical"), range (``low``, ``high``), ``step`` (for int/float), ``log`` (boolean, for float), or ``choices`` (list, for categorical).

Walk-Forward Validation (`walk_forward`)
----------------------------------------

Configures the walk-forward validation process for more realistic backtesting.

*   ``enabled``: (Boolean) Set to ``true`` to use walk-forward validation. If ``false``, a single train-test split is used based on ``data_start_date`` and ``data_end_date``.
*   ``train_window_size``: (Integer) Number of periods (e.g., days) in each training window.
*   ``test_window_size``: (Integer) Number of periods in each testing (out-of-sample) window.
*   ``step_size``: (Integer) Number of periods to slide the windows forward in each step.
*   ``retrain_every_step``: (Boolean) If ``true``, the model is retrained at the beginning of each walk-forward step. If ``false``, it's trained only once initially.
*   ``purge_window_size``: (Integer) Number of periods to remove between the end of the train set and the start of the test set to prevent lookahead bias.
*   ``expanding_window``: (Boolean) If ``true``, the training window grows with each step. If ``false``, it remains a fixed size (sliding window).

Trading Simulation (`trading`)
------------------------------

Configures the backtesting engine and trading logic. Corresponds to the ``TradingConfig`` object.

*   ``initial_capital``: (Float) Starting capital for the backtest.
*   ``commission_bps``: (Float) Trading commission per trade in basis points (1 bps = 0.01%).
*   ``slippage_bps``: (Float) Estimated slippage per trade in basis points.

Position Sizing
~~~~~~~~~~~~~~~
*   ``position_sizing``: (Object) Defines how trade sizes are determined.
    *   ``method``: (String) Sizing strategy. Options
        *   ``'fixed_fractional'``: Risk a fixed fraction of current equity.
        *   ``'fixed_amount'``: Allocate a fixed monetary amount per trade.
        *   *Other custom methods...*

    *   ``params``: (Object) Parameters for the chosen ``method``.
        *   ``fraction``: (Float) Fraction of equity for 'fixed_fractional'.
        *   ``amount``: (Float) Fixed amount for 'fixed_amount'.

Stop Loss
~~~~~~~~~
*   ``stop_loss``: (Object) Defines how stop-loss levels are set.
    *   ``method``: (String or null) Stop-loss calculation method. Options
        *   ``'atr'``: Based on a multiple of the Average True Range (ATR).
        *   ``'fixed_percentage'``: Fixed percentage below entry price.
        *   ``null``: No explicit stop-loss logic (relies on model signals or other exit conditions).

*   ``params``: (Object) Parameters for the chosen ``method``.
        *   ``multiplier``: (Float) ATR multiplier for 'atr'.
        *   ``percentage``: (Float) Percentage for 'fixed_percentage' (e.g., 0.05 for 5%).

Take Profit
~~~~~~~~~~~
*   ``take_profit``: (Object) Defines how take-profit levels are set.
    *   ``method``: (String or null) Take-profit calculation method. Options similar to ``stop_loss`` ('atr', 'fixed_percentage', null)
    *   ``params``: (Object) Parameters for the chosen ``method``.
        *   ``multiplier``: (Float) ATR multiplier for 'atr'.
        *   ``percentage``: (Float) Percentage for 'fixed_percentage'.

Market Regime Detection (`market_regime`)
-----------------------------------------

Optional configuration to detect market regimes (e.g., trending, mean-reverting, high/low volatility) and potentially adapt the strategy.

*   ``enabled``: (Boolean) Set to ``true`` to enable regime detection.
*   ``method``: (String) Algorithm for regime detection (e.g., "volatility_clustering", "rule_based_sma"). Must match a registered method.
*   ``params``: (Object) Parameters specific to the chosen ``method``.
*   ``regime_actions``: (Object, Optional) Defines how the strategy behaves in different detected regimes (e.g., ``regime_0: "trade_long_only"``). Implementation depends on the strategy logic integrating with regime information.
