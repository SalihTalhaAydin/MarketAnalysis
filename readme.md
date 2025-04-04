# Automated Trading Framework

## Overview
This project is a framework for developing and testing an automated trading strategy. It uses machine learning to analyze historical stock data and generate trading signals—"Buy," "Sell," or "Do Nothing." The goal is to identify patterns that predict short-term price movements.

## Project Components
### 1. Data Collection (data_loader.py)
- **Concept:** Stock trading data includes OHLCV (Open, High, Low, Close, Volume).
- **Programming:** Uses `yfinance` to fetch historical stock data based on a ticker symbol, date range, and time interval.

### 2. Data Preprocessing (preprocessing.py)
- **Concept:** Cleans raw stock data to ensure consistency and handle missing values.
- **Programming:**
  - Converts column names to lowercase.
  - Fills missing values using forward fill.
  - Ensures numerical formatting for price and volume data.

### 3. Feature Engineering (feature_engineering.py)
- **Concept:** Computes technical indicators like:
  - **SMA/EMA:** Moving averages to smooth price fluctuations.
  - **RSI:** Measures stock momentum.
  - **ATR:** Estimates price volatility.
- **Programming:** Uses `pandas-ta` to add these indicators to the dataset.
- **Target Labeling (Triple Barrier Method):**
  - Assigns labels (1: Profit, -1: Loss, 0: Timeout) based on predefined thresholds.

### 4. Model Training (model_training.py)
- **Concept:** Trains a machine learning model to predict trade outcomes based on historical data.
- **Programming:**
  - Uses `RandomForestClassifier` from `scikit-learn`.
  - Implements `TimeSeriesSplit` for proper time-sensitive validation.
  - Optimizes hyperparameters using `GridSearchCV`.

### 5. Prediction (prediction.py)
- **Concept:** Applies the trained model to new data to generate trade signals.
- **Programming:**
  - Uses `predict_proba` to output probability scores for trade outcomes.

### 6. Backtesting (backtesting.py)
- **Concept:** Simulates trades based on historical data to evaluate strategy performance.
- **Programming:**
  - Iterates through data to simulate buying/selling decisions.
  - Applies stop-loss and take-profit rules.
  - Calculates performance metrics including profitability, trade count, and win rate.

### 7. Strategy Execution (main_strategy.py)
- **Concept:** Walk-Forward Validation to adapt to market changes over time.
- **Programming:**
  - Splits data into training and testing segments.
  - Iteratively trains and evaluates the model on different time periods.
  - Aggregates results to assess long-term strategy effectiveness.

## Summary
This framework systematically:
1. Downloads and cleans stock data.
2. Extracts meaningful features from price movements.
3. Defines a structured trading decision problem.
4. Trains a machine learning model to identify profitable patterns.
5. Rigorously tests the model using walk-forward validation.
6. Simulates real-world performance via backtesting.

By following this approach, traders can evaluate and refine automated strategies before deploying them in live markets.

