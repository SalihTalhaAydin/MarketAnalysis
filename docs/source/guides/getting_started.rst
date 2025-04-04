Getting Started with MarketAnalysis
===================================

This guide provides instructions on setting up and running the market analysis and trading strategy backtesting framework.

1. Setup
--------

Prerequisites
~~~~~~~~~~~~~

*   Python (tested with 3.13)
*   ``pip`` (Python package installer)

Installation
~~~~~~~~~~~~

1. **Clone the Repository:**

   .. code-block:: bash

      git clone <repository_url>
      cd MarketAnalysis

2. **Create and Activate Virtual Environment:** (Recommended)

   *   On Windows:

       .. code-block:: bash

          python -m venv venv
          .\venv\Scripts\activate

   *   On macOS/Linux:

       .. code-block:: bash

          python3 -m venv venv
          source venv/bin/activate

3. **Install Dependencies:** Install the required Python packages using the provided ``requirements.txt`` file. Make sure your virtual environment is active.

   .. code-block:: bash

      pip install -r requirements.txt

   *Note: During debugging, it was found necessary to also install ``lightgbm`` separately, as it was used by the code but missing from requirements:*

   .. code-block:: bash

      pip install lightgbm

   *(Consider adding ``lightgbm`` to ``requirements.txt`` or ``setup.py``)*

2. Configuration
----------------

The trading strategy is configured using YAML files located in the ``configs/`` directory.

*   **``example_strategy_config.yaml``**: Provides a template and example settings.
*   **Key Sections:**

    *   ``strategy_name``, ``description``: Metadata for the run.
    *   ``data_start_date``, ``data_end_date``: Global date range for data loading.
    *   ``assets``: A list of assets to process, each specifying ``symbol`` and ``data_source`` (e.g., "yahoo").
    *   ``feature_config``: Controls feature engineering.

        *   ``technical_indicators``: List of indicators to calculate (requires ``pandas-ta``).
        *   ``target_type``: Method for creating the target variable (e.g., "triple_barrier").
        *   ``atr_multiplier_tp``, ``atr_multiplier_sl``, ``max_holding_period``: Parameters for Triple Barrier labeling.

    *   ``model_config``: Controls model training and signal generation.

        *   ``model_type``: The key for the model to use (e.g., "random_forest", "lightgbm", registered in ``model_factory.py``).
        *   ``optimize_hyperparams``: Boolean flag to enable/disable hyperparameter tuning.
        *   ``probability_threshold``: Threshold for converting model probability to a trading signal (1 or -1).
        *   ``signal_trend_filter_ma``, ``signal_volatility_filter_atr``: Set periods (e.g., 50) to enable filters or ``null`` to disable.

    *   ``walk_forward``: Configures walk-forward validation parameters (window sizes, step, retraining).
    *   ``trading``: Configures backtest simulation parameters (capital, costs, position sizing, stops).

*   **Important Note:** The structure in the YAML file must precisely match the expected arguments for the corresponding Python configuration classes (``StrategyConfig``, ``AssetConfig``, ``FeatureConfig``, ``ModelConfig``, ``WalkForwardConfig``) defined in ``market_ml_model/strategy/main.py``. Mismatches will cause loading errors.

3. Running the Strategy
-----------------------

The main entry point for running a backtest or walk-forward validation is ``market_ml_model/strategy/main.py``.

*   **Activate Virtual Environment:** Ensure your virtual environment (``venv``) is active.
*   **Run Command:** Execute the script as a module, providing the path to your configuration file.

    *   On Windows:

        .. code-block:: bash

           .\venv\Scripts\python.exe -m market_ml_model.strategy.main --config configs/example_strategy_config.yaml

    *   On macOS/Linux:

        .. code-block:: bash

           python -m market_ml_model.strategy.main --config configs/example_strategy_config.yaml

4. Output and Results
---------------------

*   Results for each run are saved in a unique timestamped subdirectory within the directory specified by ``results_dir`` in the configuration file (defaults to ``strategy_results/``).
*   **Key Outputs:**

    *   ``strategy_config.json``: A copy of the configuration used for the run.
    *   ``models/``: Contains saved models, preprocessors, and training summaries for each asset and walk-forward fold.
    *   ``backtests/``: Contains detailed backtest reports (plots, metrics, trades CSV) for each asset and fold.
    *   ``walk_forward/``: Contains aggregated walk-forward results (summary CSV, overall JSON) per asset.
    *   ``plots/``: Contains overall strategy plots (e.g., combined equity curve if implemented, return distribution).
    *   ``strategy_summary.json``: The final aggregated summary for the entire strategy run across all assets.

5. Making Predictions (Future Work)
-----------------------------------

To predict future market movements:

1.  Identify the path to the desired trained model (usually the latest fold) within the ``strategy_results`` output directory.
2.  Load the model using ``ModelPredictorBase`` from ``market_ml_model.models.prediction``.
3.  Fetch the latest market data using ``DataLoader``.
4.  Engineer features using ``engineer_features``.
5.  Call the predictor's ``predict_proba`` method with the latest features.
6.  Interpret the resulting probabilities.

*(A dedicated script for this prediction workflow could be created).*
