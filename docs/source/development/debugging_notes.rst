Debugging Notes (April 4, 2025)
===============================

This document summarizes the debugging process undertaken to get the ``market_ml_model/strategy/main.py`` script running successfully with the ``configs/example_strategy_config.yaml`` configuration.

Initial Goal
------------

Run the strategy script, evaluate its performance, and understand its workings.

Issues Encountered and Fixes Applied
------------------------------------

1.  **Configuration Loading (``TypeError: AssetConfig.__init__``):**

    *   **Issue:** ``start_date`` and ``end_date`` were incorrectly placed under the ``assets`` list in the YAML, while ``StrategyConfig`` expected them at the top level as ``data_start_date`` and ``data_end_date``.
    *   **Fix:** Moved dates to the top level in ``configs/example_strategy_config.yaml`` and renamed them.

2.  **Configuration Loading (``TypeError: ... got an unexpected keyword argument ...``):**

    *   **Issue:** The ``StrategyConfig.from_dict`` method in ``main.py`` didn't remove processed top-level keys (like ``features``, ``model``, ``walk_forward``, ``trading``, ``results_dir``, ``log_level``) from the dictionary before passing it to ``StrategyConfig.__init__``, causing TypeErrors. Similar issues existed for ``FeatureConfig`` (unexpected ``labeling``, ``scaling``) and ``ModelConfig`` (unexpected ``params``, ``feature_selection``).
    *   **Fix:**

        *   Modified ``StrategyConfig.from_dict`` in ``main.py`` to ``pop`` unused keys before calling ``cls()``.
        *   Restructured ``feature_config`` in the YAML to match ``FeatureConfig`` attributes (moved ``atr_multiplier_tp/sl``, ``max_holding_period``, added ``target_type``, removed ``labeling`` and ``scaling``).
        *   Restructured ``model_config`` in the YAML to match ``ModelConfig`` attributes (removed ``params`` and ``feature_selection``).

3.  **Execution Logic (``AttributeError: ... SequentialExecutor``):**

    *   **Issue:** ``main.py`` attempted to use ``concurrent.futures.SequentialExecutor`` (which doesn't exist) for non-parallel execution.
    *   **Fix:** Modified ``run_strategy`` in ``main.py`` to use a standard loop for sequential execution instead of trying to use a non-existent executor.

4.  **Data Loading (``Unsupported data source: yfinance``):**

    *   **Issue:** ``market_ml_model/data/loaders.py`` expected the data source identifier "yahoo", but the config specified "yfinance".
    *   **Fix:** Changed ``data_source`` in ``configs/example_strategy_config.yaml`` from "yfinance" to "yahoo".

5.  **Data Loading (``AttributeError: Can only use .str accessor with Index, not MultiIndex``):**

    *   **Issue:** ``market_ml_model/data/sources/yahoo.py`` assumed ``yfinance.download`` always returned columns with a simple Index, causing an error when a MultiIndex was returned. The initial fix incorrectly appended the ticker to column names.
    *   **Fix:** Modified the column processing in ``yahoo.py`` to correctly handle both Index and MultiIndex outputs, extracting the primary name (e.g., 'close') and converting to lowercase. Added logic to prefer 'adj close' and rename it to 'close' if ``adjust_prices`` was true.

6.  **Feature Engineering (``pandas-ta not available``):**

    *   **Issue:** The script consistently failed to import ``pandas-ta`` when run with the default system Python, even though it was installed (likely in user site-packages). This prevented indicator calculation.
    *   **Fix:** Executed the script using the Python interpreter within the project's virtual environment (``venv/Scripts/python.exe``), which resolved the import issue.

7.  **Model Training (``ValueError: Model training failed.`` due to ``LightGBM not available``):**

    *   **Issue:** The default model (``lightgbm``) was used by the code but was not installed in the virtual environment or listed in ``requirements.txt``.
    *   **Fix:** Installed ``lightgbm`` into the virtual environment (``venv/Scripts/pip.exe install lightgbm``).

8.  **Hyperparameter Optimization (``TypeError: Parameter grid ... is not iterable``):**

    *   **Issue:** The ``n_iter`` parameter was incorrectly included in the ``param_distributions`` dictionary passed to ``RandomizedSearchCV``.
    *   **Fix:** Modified ``optimize_hyperparameters_random`` in ``hyperparameters.py`` to remove ``n_iter`` from the distribution dictionary before passing it to the search function.

9.  **Model Evaluation (``ValueError: Unknown format code 'f' for object of type 'str'``):**

    *   **Issue:** The logging statement in ``training.py`` tried to format metrics using ``:.4f``, which failed when evaluation returned 'N/A' (due to the model not being fitted after the hyperparameter optimization error).
    *   **Fix:** Modified the logging statement in ``training.py`` to conditionally apply float formatting only if the metric is numeric.

10. **Feature Engineering (``IntCastingNaNError``):**

    *   **Issue:** Attempting to convert the ``triple_barrier_label`` column (which contained NaNs generated by the labeling function) to ``int`` failed.
    *   **Fix:** Modified ``features_engineering.py`` to replace ``inf``/``-inf`` with ``NaN`` and then fill remaining ``NaN``s with 0 *before* converting the column type to ``int``.

11. **Model Evaluation (``AttributeError: 'numpy.ndarray' object has no attribute 'unique'``):**

    *   **Issue:** The code in ``metrics.py`` assumed ``y`` and ``y_pred`` were always pandas Series and used ``.unique()``.
    *   **Fix:** Modified ``metrics.py`` to consistently use ``np.unique()`` after ensuring the inputs are treated as arrays.

12. **Model Evaluation (``ValueError: y_true and y_pred contain different number of classes``):**

    *   **Issue:** Evaluation metrics failed because the true labels (``y_true`` = -1, 1) didn't include all classes predicted by the model (``y_pred`` could include 0 due to NaN filling).
    *   **Fix:** Modified ``evaluate_classifier`` in ``metrics.py`` to explicitly pass the full set of possible labels (``[-1, 0, 1]``) to ``classification_report`` and ``confusion_matrix``. Removed binary-specific metric calculations that were incompatible with the 3-class scenario.

13. **Model Loading (``Model file not found``):**

    *   **Issue:** The model predictor couldn't find the saved ``model.pkl`` file because ``training.py`` was saving it one directory level deeper than expected (appending ``model_id`` to the path).
    *   **Fix:** Corrected the path construction in ``training.py`` to save artifacts directly into the provided ``model_dir`` (fold directory). *(Self-correction: Initial fix was correct, the subsequent error was due to a different issue)*. The final fix involved correcting the path retrieval logic in ``main.py``'s ``generate_predictions`` function to use the path stored in ``self.models``.

14. **Backtesting (``ValueError: Length of values ... does not match length of index ...``):**

    *   **Issue:** The ``TradeManager`` in ``simulation.py`` had an off-by-one error between the ``equity_curve`` list and the ``equity_timestamps`` list.
    *   **Fix:** Modified the ``update`` method in ``simulation.py`` to correctly handle the initialization of the timestamp list relative to the initial capital entry.

15. **Backtesting (``Missing required columns ... ['ATRr_10']``):**

    *   **Issue:** The backtester required the ``ATRr_10`` column for dynamic stops, but it wasn't calculated on the test data slice passed to it.
    *   **Fix:** Modified ``backtest_signals`` in ``main.py`` to explicitly calculate ``ATRr_10`` on the aligned OHLC data before passing it to the backtester.

16. **Plot Saving (``FileNotFoundError``):**

    *   **Issue:** The ``plot_walk_forward_results`` function in ``main.py`` attempted to save the plot before ensuring the target ``plots`` subdirectory existed.
    *   **Fix:** Added ``os.makedirs(os.path.dirname(filename), exist_ok=True)`` before the ``plt.savefig`` call.

Final Status
------------

*   The script runs end-to-end without errors using the virtual environment.
*   The configured strategy (RandomForest or LightGBM with default features and adjusted thresholds) **does not generate any trades**.
*   Further work is needed on **strategy development and parameter tuning** to achieve trading activity and evaluate performance. The underlying code infrastructure is now functional.


.. rubric:: Debugging Walk-Forward Execution (2025-04-04)

This section details the debugging process specifically related to getting the walk-forward execution logic in ``market_ml_model/strategy/main.py`` to function correctly after the initial setup debugging.

1.  **Initial Prediction Feature Error (``ValueError: None of the selected features found...``):**

    *   **Issue:** The prediction logic in ``market_ml_model/models/prediction.py`` used hardcoded feature names, while the training logic used features derived from the configuration file (``configs/example_strategy_config.yaml``). This mismatch caused errors when the prediction component tried to find features that weren't generated.
    *   **Fix:** Modified the prediction feature generation process to be driven by the same configuration used during training, ensuring consistency.

2.  **Persistent Feature Error (Inconsistent Engineering):**

    *   **Issue:** Even after the initial fix, feature-related errors persisted. The root cause was identified as inconsistent feature engineering application across different train/test splits within the walk-forward folds.
    *   **Attempt 1:** Refactored feature engineering to occur *before* the train/test split within a fold. This failed due to a ``NameError``.
    *   **Attempt 2:** Reverted to engineering features *after* the split. The error persisted, indicating a deeper issue.

3.  **Refactoring Error (``NameError: name 'train_data' is not defined``):**

    *   **Issue:** During attempts to fix the feature engineering inconsistency, refactoring introduced a simple variable naming error where ``train_data`` was referenced before assignment.
    *   **Fix:** Corrected the variable reference in the relevant scope.

4.  **Persistent Feature Error (Train vs. Predict Path & Selection):**

    *   **Issue:** The feature errors were ultimately traced to two core problems:
        a) Subtle inconsistencies in how features were generated between the training path and the prediction path, even when using a seemingly shared component.
        b) Incorrect application of feature *selection* results during the prediction phase. The prediction logic wasn't using the subset of features selected during the corresponding training fold.
    *   **Fix:** Further refactored the feature engineering logic into a truly shared component. Ensured that the prediction step explicitly loaded and used the selected features identified during the training phase of that specific fold. *Note: This initially still failed, pointing to an issue within the signal generation step.*

5.  **Feature Error Pinpointed (``SignalGenerator`` Issue):**

    *   **Issue:** The final feature-related error was located in the ``SignalGenerator`` class within ``market_ml_model/strategy/main.py``. It was incorrectly re-applying feature preprocessing and selection steps to the data *after* predictions had already been made, leading to a mismatch with the expected input columns for signal generation.
    *   **Fix:** Modified ``SignalGenerator`` to directly use the prediction outputs (probabilities and features) without reapplying preprocessing or selection.

6.  **Probability Column Error (``KeyError: "Positive probability column... not found"``):**

    *   **Issue:** After fixing the feature pipeline, a new error emerged because the ``SignalGenerator`` expected specific column names for the predicted class probabilities (e.g., ``'positive_probability'``), but the prediction component was outputting columns based on the raw class labels (e.g., ``'1'``, ``'-1'``).
    *   **Fix:** Updated ``SignalGenerator`` to correctly reference the probability column names as generated by the prediction model (e.g., using ``f"probability_{config.model_config.positive_label}"``).

7.  **Plotting Error (``TypeError: plot_equity_curve() missing 1 required positional argument: 'ax'``):**

    *   **Issue:** The final error occurred during results visualization. The ``plot_equity_curve`` function in ``market_ml_model/utils/visualization.py`` required a Matplotlib ``Axes`` object (``ax``) to be passed, but the calling code in ``main.py`` was not providing it.
    *   **Fix:** Updated the call to ``plot_equity_curve`` in ``main.py`` to pass the created Matplotlib ``ax`` object.
