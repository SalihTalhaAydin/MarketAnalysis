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
import datetime
from datetime import timedelta
import numpy as np
import pandas as pd  # noqa: F401 Needed for WFV slicing and ATR access

# Import functions from the new modules
from market_ml_model.src.data_loader import load_data
from market_ml_model.src.preprocessing import preprocess_data
from market_ml_model.src.feature_engineering import engineer_features
from market_ml_model.src.model_training import train_classification_model
from market_ml_model.src.prediction import predict_with_model
from market_ml_model.src.backtesting import backtest_strategy

# --- Configuration ---
# Default parameters for analysis, can be overridden or read from config
DEFAULT_SYMBOL = "AAPL"  # Example stock symbol
DEFAULT_TIMEFRAME = "1m"  # Changed to 1-minute for scalping
# Dates for data loading (last 7 days for 1m data)
today = datetime.date.today()
start_date_obj = today - timedelta(days=7)  # 7 days (yfinance 1m limit)
DEFAULT_START_DATE = start_date_obj.strftime("%Y-%m-%d")
DEFAULT_END_DATE = today.strftime("%Y-%m-%d")


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
    raw_data = load_data(symbol, start_date, end_date, interval=timeframe)
    if raw_data is None or raw_data.empty:
        print(f"Failed to load data for {symbol}. Exiting analysis.")
        return

    # 2. Preprocess Data
    preprocessed_data = preprocess_data(raw_data, symbol)
    if preprocessed_data is None or preprocessed_data.empty:
        print("Failed initial preprocessing. Exiting analysis.")
        return

    # 3. Engineer Features
    # Pass SL/TP multipliers and max hold period to feature engineering
    # These are defined later in the WFV setup section
    # Scalping-specific Triple Barrier parameters
    atr_multiplier_sl = 1.0  # Tighter SL (was 2.0)
    atr_multiplier_tp = 1.5  # Tighter TP (was 3.0)
    max_holding_period = 3   # Shorter hold period (was 5)

    featured_data = engineer_features(
        preprocessed_data,
        atr_multiplier_tp=atr_multiplier_tp,
        atr_multiplier_sl=atr_multiplier_sl,
        max_holding_period=max_holding_period
    )
    if featured_data is None or featured_data.empty:
        print("Failed to engineer features. Exiting analysis.")
        return

    # 4. Separate features (X) and target (y)
    target_col_name = 'triple_barrier_label'  # New target column
    if target_col_name not in featured_data.columns:
        print(f"Error: '{target_col_name}' column not found after features.")
        return

    # Define columns to exclude from features (target, raw OHLCV, etc.)
    # Adjust based on actual features engineered and model requirements.
    cols_to_drop_for_features = [
        'open', 'high', 'low', 'close', 'volume',  # Raw prices often dropped
        target_col_name                            # Target variable itself
        # Add other columns not for features (e.g., intermediate 'return')
        # 'return' # Example if 'return' was only intermediate
    ]
    # Ensure all columns exist before dropping
    cols_to_drop_for_features = [
        col for col in cols_to_drop_for_features
        if col in featured_data.columns
    ]
    features = featured_data.drop(columns=cols_to_drop_for_features)
    target = featured_data[target_col_name]

    # --- Walk-Forward Validation Setup ---
    print("\n--- Setting up Walk-Forward Validation ---")
    n_samples = len(features)

    # WFV Parameters (adjust as needed)
    initial_train_size = 1000  # Number of bars for the first training window
    test_size = 200          # Number of bars for each test window
    step_size = 200          # How much to slide the window forward each time
    min_train_size = 800     # Minimum bars required for training

    # ATR SL/TP Parameters
    atr_col = 'ATRr_10'      # Column name for ATR from feature engineering
    # Scalping-specific Triple Barrier parameters (defined above, used here)
    # atr_multiplier_sl = 1.0 # Defined earlier
    # atr_multiplier_tp = 1.5 # Defined earlier
    prob_threshold = 0.65    # Increased confidence threshold (was 0.52)
    slippage_pct = 0.0005   # Example: 0.05% slippage per trade side
    # max_holding_period defined earlier before calling engineer_features

    if atr_col not in featured_data.columns:
        print(f"Error: ATR column '{atr_col}' not found in featured data. "
              "Cannot use ATR for SL/TP.")
        return

    if n_samples < initial_train_size + test_size:
        print(f"Error: Not enough data ({n_samples} rows) for initial train "
              f"({initial_train_size}) + test ({test_size}) windows.")
        return

    all_fold_performances = []
    fold_number = 0
    start_train_idx = 0

    # --- Walk-Forward Validation Loop ---
    while start_train_idx + initial_train_size + test_size <= n_samples:
        fold_number += 1
        end_train_idx = start_train_idx + initial_train_size
        start_test_idx = end_train_idx
        end_test_idx = start_test_idx + test_size

        # Ensure train size doesn't shrink below minimum
        current_train_size = end_train_idx - start_train_idx
        if current_train_size < min_train_size:
            print(f"Skipping Fold {fold_number}: Training size "
                  f"({current_train_size}) < minimum ({min_train_size}).")
            break  # Stop if training window gets too small

        print(f"\n--- WFV Fold {fold_number} ---")
        print(f"Train Window: {start_train_idx} - {end_train_idx-1} "
              f"({current_train_size} rows)")
        print(f"Test Window:  {start_test_idx} - {end_test_idx-1} "
              f"({test_size} rows)")

        # Slice data for the current fold
        train_features = features.iloc[start_train_idx:end_train_idx]
        train_target = target.iloc[start_train_idx:end_train_idx]
        test_features = features.iloc[start_test_idx:end_test_idx]
        # Keep original data + target + ATR for backtest evaluation
        test_data_fold = featured_data.iloc[start_test_idx:end_test_idx].copy()
        # train_data_fold was previously used for ATR calculation, now unused.
        # train_data_fold = featured_data.iloc[start_train_idx:end_train_idx]

        if train_features.empty or test_features.empty:
            print(
                f"Warning: Empty train/test features for Fold {fold_number}. "
                "Skipping."
            )
            start_train_idx += step_size  # Move to next potential window
            # Adjust for next fold (using expanding window logic here)
            initial_train_size = end_train_idx + step_size - start_train_idx
            continue

        # 5. Train Model (on current training fold)
        print("Training model...")
        model = train_classification_model(train_features, train_target)
        if model is None:
            print(f"Model training failed for Fold {fold_number}. Skipping.")
            start_train_idx += step_size
            initial_train_size = end_train_idx - start_train_idx
            continue

        # 6. Generate Probabilities (on current test fold)
        print("Generating predictions...")
        probabilities = predict_with_model(model, test_features)
        if probabilities.shape[0] != len(test_features):
            print(
                f"Error: Probability length mismatch for Fold {fold_number}. "
                "Skipping."
            )
            start_train_idx += step_size
            initial_train_size = end_train_idx - start_train_idx
            continue
        # Expect probabilities for 3 classes: -1 (SL), 0 (Time), 1 (TP)
        if probabilities.shape[1] < 3:
            print(
                f"Error: Probabilities shape mismatch (expected 3 classes) "
                f"Fold {fold_number}. Got shape {probabilities.shape}. "
                f"Skipping."
            )
            start_train_idx += step_size
            # Recalculate for expanding window logic
            initial_train_size = end_train_idx + step_size - start_train_idx
            continue

        # Generate Predictions (Signals) from Probabilities
        # Generate Predictions (Signals) based on Triple Barrier probabilities
        # Assumes model.predict_proba() returns probs for classes [-1, 0, 1]
        # Index 0: P(SL = -1)
        # Index 1: P(Time = 0)
        # Index 2: P(TP = 1)
        # Generate Predictions (Signals) based on Triple Barrier probabilities
        # Index 0: P(SL = -1), Index 1: P(Time = 0), Index 2: P(TP = 1)
        predicted_classes = []
        for prob_sl, prob_time, prob_tp in probabilities:
            if prob_tp > prob_threshold:
                predicted_classes.append(1)  # Signal Long
            elif prob_sl > prob_threshold:
                predicted_classes.append(-1)  # Signal Short
            else:
                predicted_classes.append(0)  # Signal Hold

        # Add predictions to the test data subset for backtesting
        if len(predicted_classes) == len(test_data_fold):
            test_data_fold['prediction'] = predicted_classes
        else:
            print(
                f"Error: Prediction len mismatch Fold {fold_number}. Skipping."
            )
            start_train_idx += step_size
            initial_train_size = end_train_idx - start_train_idx
            continue

        # 7. SL/TP calculation is now handled dynamically in backtest_strategy.
        #    based on ATR multipliers.
        # 8. Backtest Strategy (on current test fold with dynamic SL/TP)
        print("Running backtest for fold...")
        # Define unique output path per fold (optional)
        trades_output_filename = (
             f"market_ml_model/data/trades_{symbol}_WFVfold{fold_number}.csv"
        )

        performance = backtest_strategy(
            data_with_predictions=test_data_fold,  # Pass the fold data
            # Pass ATR multipliers directly now
            atr_multiplier_sl=atr_multiplier_sl,
            atr_multiplier_tp=atr_multiplier_tp,
            atr_col=atr_col,  # Pass the ATR column name
            slippage_pct_per_trade=slippage_pct,  # Pass slippage
            output_trades_path=trades_output_filename  # Or None
        )

        if performance:
            print(
                f"Fold {fold_number} Result: Return="
                f"{performance.get('total_strategy_return_pct', 'N/A'):.2f}%"
            )
            all_fold_performances.append(performance)
        else:
            print(f"Backtest failed or no results for Fold {fold_number}.")

        # Slide the window
        start_train_idx += step_size
        # The next training window starts where the previous one did,
        # but its size increases by step_size (expanding window)
        # Or, for a purely sliding window of fixed size:
        # initial_train_size = initial_train_size # Keep it fixed
        # For expanding window:
        initial_train_size = end_train_idx + step_size - start_train_idx

    # --- Aggregate and Display WFV Results ---
    print("\n--- Walk-Forward Validation Complete ---")
    if not all_fold_performances:
        print("No successful backtest folds completed.")
        return  # Exit if no results

    # Example Aggregation (more sophisticated analysis is possible)
    total_returns = [
        p.get('total_strategy_return_pct', 0) for p in all_fold_performances
    ]
    num_trades = [p.get('num_trades', 0) for p in all_fold_performances]
    win_rates = [
        p.get('win_rate_pct (based on trades)', 0)
        for p in all_fold_performances
    ]

    print("\n=== Aggregated Walk-Forward Performance ===")
    print(f"Number of Folds Tested: {len(all_fold_performances)}")
    if total_returns:
        print(f"Average Fold Return: {np.mean(total_returns):.2f}%")
        print(f"Std Dev Fold Return: {np.std(total_returns):.2f}%")
        # Note: Simple averaging doesn't give true compounded return.
        # A proper aggregation would link returns chronologically.
    print(f"Average Trades per Fold: {np.mean(num_trades):.1f}")
    if win_rates:  # Check if win rates were calculated
        valid_win_rates = [
            wr for wr in win_rates if wr is not None
        ]  # Filter None
        if valid_win_rates:
            print(
                f"Average Win Rate (Trades): {np.mean(valid_win_rates):.2f}%"
            )

    # --- TBD: Calculate overall Sharpe, Drawdown etc. across all folds ---
    # This requires stitching the equity curves or returns series together.

    # --- (Code from lines 116-153 is now inside the WFV loop) ---

    # --- (Code from lines 154-219 is replaced by WFV loop and aggregation) ---

    # --- TBD: Add further analysis, visualization, or saving results ---


# --- Script Execution ---
if __name__ == "__main__":
    print("--- Script starting execution ---")
    # Example usage when running the script directly
    run_trading_strategy_analysis(
        symbol=DEFAULT_SYMBOL,
        timeframe=DEFAULT_TIMEFRAME,
        start_date=DEFAULT_START_DATE,
        end_date=DEFAULT_END_DATE
    )