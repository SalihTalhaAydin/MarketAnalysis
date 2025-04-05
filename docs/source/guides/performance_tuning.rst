Performance Tuning Experiments
============================

Introduction
------------

Based on initial poor performance results using the default configuration, several changes were implemented in the example strategy configuration (`configs/example_strategy_config.yaml`) to improve the model's predictive power and overall strategy returns. The following modifications were tested:

Implemented Changes
-------------------

*   **Expanded Feature Set:** Added the following technical indicators to provide the model with more market context:
    *   MACD (Moving Average Convergence Divergence)
    *   Bollinger Bands
    *   ADX (Average Directional Index)
    *   OBV (On-Balance Volume)
    *   MFI (Money Flow Index)
*   **Switched Model:** Changed the underlying prediction model from the default to LightGBM, known for its speed and efficiency.
*   **Enabled Hyperparameter Optimization:** Utilized Optuna with the Bayesian sampler for 100 trials to automatically find better model parameters.
*   **Increased Signal Probability Threshold:** Raised the minimum probability required to generate a trading signal to 0.65, aiming for higher confidence trades.
*   **Adjusted Triple Barrier Parameters:** Modified the profit-taking (TP) multiplier to 1.5 and the maximum holding period (max_hold) to 15 bars.

Results
-------

After applying these changes collectively, the backtest yielded the following performance metrics:

*   **Cumulative Return:** -6.04%
*   **Sharpe Ratio:** -8.83

While these changes represent an attempt at optimization, the results indicate that further tuning and potentially different approaches are necessary to achieve profitability.
