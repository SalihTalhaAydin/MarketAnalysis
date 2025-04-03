"""
Main script for orchestrating the trading strategy analysis workflow.

This script coordinates the following steps:
1. Loads market data using the data_loader module.
2. Preprocesses the data using the preprocessing module.
3. Engineers features using the feature_engineering module.
4. Trains a predictive model using the model_training module.
5. Makes predictions using the prediction module.
6. Backtests the strategy using the backtesting module.
"""

# --- Imports ---
# import pandas as pd # No longer directly used here
# import numpy as np # No longer directly used here

# Import functions from the new modules
from .data_loader import load_data
from .preprocessing import preprocess_data
from .feature_engineering import engineer_features
from .model_training import train_classification_model
from .prediction import predict_with_model
from .backtesting import backtest_strategy

# --- Configuration ---
# Default parameters for analysis, can be overridden or read from config
DEFAULT_SYMBOL = "AAPL"  # Example stock symbol
DEFAULT_TIMEFRAME = "1d"  # Example timeframe (e.g., '1m', '5m', '1h', '1d')
# Dates for data loading (adjust as needed)
DEFAULT_START_DATE = "2022-01-01"
DEFAULT_END_DATE = "2024-01-01"


# --- Main Workflow ---

def run_trading_strategy_analysis(
    symbol: str,
    timeframe: str,  # May not be used if load_data handles it internally
    start_date: str,
    end_date: str
):
    """
    Orchestrates the entire analysis workflow for a given symbol and timeframe.
    """
    print(f"\n=== Running Analysis for {symbol} ({timeframe}) ===")
    print(f"Data Period: {start_date} to {end_date}")

    # 1. Load Data
    raw_data = load_data(symbol, start_date, end_date)
    if raw_data is None or raw_data.empty:
        print(f"Failed to load data for {symbol}. Exiting analysis.")
        return

    # 2. Preprocess Data
    preprocessed_data = preprocess_data(raw_data, symbol)
    if preprocessed_data is None or preprocessed_data.empty:
        print("Failed initial preprocessing. Exiting analysis.")
        return

    # 3. Engineer Features
    featured_data = engineer_features(preprocessed_data)
    if featured_data is None or featured_data.empty:
        print("Failed to engineer features. Exiting analysis.")
        return

    # 4. Separate features (X) and target (y)
    if 'target_direction' not in featured_data.columns:
        print("Error: 'target_direction' column not found after features.")
        return

    # Define columns to exclude from features (target, raw OHLCV, etc.)
    # Adjust based on actual features engineered and model requirements.
    cols_to_drop_for_features = [
        'open', 'high', 'low', 'close', 'volume',  # Raw prices often dropped
        'target_direction'                         # Target variable itself
        # Add other columns not for features (e.g., intermediate 'return')
        # 'return' # Example if 'return' was only intermediate
    ]
    # Ensure all columns exist before dropping
    cols_to_drop_for_features = [
        col for col in cols_to_drop_for_features
        if col in featured_data.columns
    ]
    features = featured_data.drop(columns=cols_to_drop_for_features)
    target = featured_data['target_direction']

    # --- Implement proper Train/Test Split (Chronological) ---
    # Crucial for time series to avoid lookahead bias.
    split_ratio = 0.8
    split_index = int(len(features) * split_ratio)

    if split_index == 0 or split_index == len(features):
        print(f"Error: Cannot split data. Train or test set would be empty. "
              f"Total rows: {len(features)}, Split index: {split_index}")
        return

    train_features = features.iloc[:split_index]
    train_target = target.iloc[:split_index]
    test_features = features.iloc[split_index:]
    # Keep original data + target for backtest evaluation
    test_data_with_target = featured_data.iloc[split_index:].copy()

    print(f"Data split: Train {len(train_features)} rows, "
          f"Test {len(test_features)} rows")
    print("Warning: Using simple chronological split. "
          "Consider more robust validation (e.g., walk-forward).")

    if train_features.empty or test_features.empty:
        print("Error: Not enough data for train/test split after processing.")
        return

    # 5. Train Model
    model = train_classification_model(train_features, train_target)
    if model is None:
        print("Model training failed. Exiting.")
        return

    # 6. Generate Probabilities (on test data)
    probabilities = predict_with_model(model, test_features)
    if probabilities.shape[0] != len(test_features):
        print(f"Error: Probability length ({probabilities.shape[0]}) mismatch "
              f"with test features ({len(test_features)}).")
        return
    if probabilities.shape[1] < 2:
        print(f"Error: Probabilities array has unexpected shape: "
              f"{probabilities.shape}. Expected >= 2 columns.")
        return

    # --- Generate Predictions from Probabilities ---
    # Example: Predict class 1 if prob > 0.5 (adjust threshold?)
    prediction_threshold = 0.5
    # Assuming column 1 is the probability of the positive class (e.g., 'up')
    predicted_classes = (probabilities[:, 1] > prediction_threshold)\
        .astype(int)

    # Add predictions to the test data subset for backtesting
    # Ensure index alignment if test_features index was reset or changed
    if len(predicted_classes) == len(test_data_with_target):
        test_data_with_target['prediction'] = predicted_classes
    else:
        print(f"Error: Length mismatch between predictions "
              f"({len(predicted_classes)}) and test data "
              f"({len(test_data_with_target)}). Cannot add column.")
        return

    # 7. Backtest Strategy
    performance = backtest_strategy(test_data_with_target)

    # 8. Display Results
    print("\n--- Analysis Complete ---")
    if performance:
        print("Backtest Performance Summary:")
        for key, value in performance.items():
            if isinstance(value, float):
                print(f"- {key}: {value:.2f}")
            else:
                print(f"- {key}: {value}")
    else:
        print("Backtesting did not produce results.")

    # --- TBD: Add further analysis, visualization, or saving results ---


# --- Script Execution ---
if __name__ == "__main__":
    # Example usage when running the script directly
    run_trading_strategy_analysis(
        symbol=DEFAULT_SYMBOL,
        timeframe=DEFAULT_TIMEFRAME,
        start_date=DEFAULT_START_DATE,
        end_date=DEFAULT_END_DATE
    )