"""
Main script for defining, analyzing, and backtesting a trading strategy.

This script orchestrates the workflow:
1. Loads market data using the data_loader module.
2. Preprocesses the data and engineers features (e.g., technical indicators).
3. (Placeholder) Trains a predictive model.
4. (Placeholder) Makes predictions using the trained model.
5. (Placeholder) Backtests the strategy based on the predictions.
"""

# --- Imports ---
import pandas as pd
import numpy as np
from .data_loader import load_data
# from .data_loader import preprocess_data as pproc # Keep if needed later

# Using pandas_ta for technical indicators. Install: pip install pandas-ta
try:
    import pandas_ta as ta
except ImportError:
    print("Warning: pandas-ta not installed (pip install pandas-ta). "
          "Feature engineering using it will be skipped.")
    ta = None

# Machine Learning Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

# --- Configuration ---
# Default parameters for analysis, can be overridden or read from config
DEFAULT_SYMBOL = "AAPL"  # Example stock symbol
DEFAULT_TIMEFRAME = "1d"  # Example timeframe (e.g., '1m', '5m', '1h', '1d')
# Dates for data loading (adjust as needed)
DEFAULT_START_DATE = "2022-01-01"
DEFAULT_END_DATE = "2024-01-01"


# --- 1. Data Acquisition (Using data_loader) ---

# Note: The original load_market_data function is now replaced by
# using the dedicated load_data function from data_loader.py


# --- 2. Data Preprocessing and Feature Engineering ---

def preprocess_and_engineer_features(
    df: pd.DataFrame, ticker: str
) -> pd.DataFrame | None:
    """
    Cleans data and adds technical indicators and other features.
    Leverages the preprocessing steps from data_loader and adds more.

    Args:
        df: Raw OHLCV DataFrame (potentially MultiIndex from yfinance).
        ticker: The ticker symbol for which data is being processed.

    Returns:
        DataFrame with added features, or None if processing fails.
    """
    print("--- Preprocessing and Engineering Features ---")
    if df is None or df.empty:
        print("Error: Input DataFrame is empty/None for feature engineering.")
        return None

    # --- Initial Preprocessing (Leverage from data_loader if desired) ---
    # Option 1: Assume df is already somewhat preprocessed by data_loader
    # Option 2: Call preprocessor from data_loader again (might be redundant)
    # processed_df = preprocess_data_from_loader(df.copy(), ticker) # Example
    # if processed_df is None:
    #     print("Error during initial preprocessing step.")
    #     return None
    # For this example, we assume the input df is the raw data from load_data
    processed_df = df.copy()

    # --- Standardize Column Names (lowercase for pandas_ta) ---
    # Handle potential MultiIndex from yfinance if multiple tickers were loaded
    if isinstance(processed_df.columns, pd.MultiIndex):
        print(f"Attempting to extract data for ticker '{ticker}' "
              f"from MultiIndex...")
        # Check if ticker is at level 1 (standard yfinance format)
        if ticker in processed_df.columns.get_level_values(1):
            try:
                # Select columns where level 1 matches the ticker
                idx = pd.IndexSlice
                processed_df = processed_df.loc[:, idx[:, ticker]].copy()
                # Columns like ('Open','AAPL'). Drop ticker level (level 1).
                processed_df.columns = processed_df.columns.droplevel(1)
                processed_df.columns = processed_df.columns.str.lower()
                print(f"Successfully extracted data for {ticker} "
                      f"from column level 1.")
            except Exception as e:
                print(f"Error extracting ticker {ticker} from level 1: {e}")
                return None
        # Check if ticker is at level 0 (current test fixture format)
        elif ticker in processed_df.columns.get_level_values(0):
            try:
                # Use xs for level 0 selection
                processed_df = processed_df.xs(ticker, level=0, axis=1).copy()
                processed_df.columns = processed_df.columns.str.lower()
                print(f"Successfully extracted data for {ticker} "
                      f"from column level 0.")
            except KeyError:  # Should not happen if check passed, but safe
                print(f"Error: Ticker {ticker} in level 0 but xs failed.")
                return None
            except Exception as e:
                print(f"Error extracting ticker {ticker} from level 0: {e}")
                return None
        else:
            print(f"Error: Ticker {ticker} not found in MultiIndex columns "
                  f"at level 0 or 1.")
            print(f"Available columns: {processed_df.columns}")
            return None
    else:
        # Assume single index columns, convert to lowercase
        processed_df.columns = processed_df.columns.str.lower()

    # --- Preprocessing Steps ---
    # 1. Handle missing values (example: forward fill)
    initial_nan_count = processed_df.isnull().sum().sum()
    processed_df.ffill(inplace=True)
    # Handle any remaining NaNs at the beginning (e.g., drop rows)
    processed_df.dropna(inplace=True)
    final_nan_count = processed_df.isnull().sum().sum()
    if initial_nan_count > 0:
        print(f"Handled NaNs (Initial: {initial_nan_count}, "
              f"Final: {final_nan_count})")

    # 2. Ensure correct data types (e.g., numeric for OHLCV)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(
                processed_df[col], errors='coerce'
            )
        else:
            print(f"Warning: Column '{col}' not found for type conversion.")
    processed_df.dropna(inplace=True)  # Drop rows if coercion failed

    # --- Feature Engineering Steps ---
    if ta:
        print("Calculating indicators using pandas_ta...")
        # Example Indicators using pandas_ta:
        try:
            # Simple Moving Average (SMA)
            processed_df.ta.sma(length=20, append=True)  # Appends 'SMA_20'
            processed_df.ta.sma(length=50, append=True)  # Appends 'SMA_50'

            # Relative Strength Index (RSI)
            processed_df.ta.rsi(length=14, append=True)  # Appends 'RSI_14'

            # Bollinger Bands
            # Appends 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0'
            processed_df.ta.bbands(length=20, std=2, append=True)

            # Average True Range (ATR) - useful for stops/volatility
            processed_df.ta.atr(length=14, append=True)  # Appends 'ATRr_14'
            print("pandas_ta indicators calculated.")
        except Exception as e:
            print(f"Error calculating pandas_ta indicators: {e}")
            # Decide if you want to continue without indicators or stop
            # return None # Or just continue

    else:
        print("Warning: pandas_ta not available. Skipping indicators.")

    # Example: Calculate simple return (log return might be better sometimes)
    # Example: Calculate simple return (log return might be better sometimes)
    if 'close' in processed_df.columns:
        processed_df['return'] = processed_df['close'].pct_change()
    else:
        print("Warning: 'close' column not found, skipping returns.")

    # Example: Lagged features (previous period's close)
    if 'close' in processed_df.columns:
        processed_df['close_lag1'] = processed_df['close'].shift(1)
    else:
        print("Warning: 'close' column not found, skipping close_lag1.")

    # --- Define Target Variable (Example: Predict next period's direction) ---
    # Target = 1 if next period's close > current close, 0 otherwise
    # Important: This creates a dependency on future data (close price of t+1).
    # When training, ensure features (X) only use data up to time t,
    # and the target (y) is based on data from time t+1.
    if 'close' in processed_df.columns:
        target_condition = (processed_df['close'].shift(-1) >
                            processed_df['close'])
        processed_df['target_direction'] = target_condition.astype(int)
    else:
        print("Warning: 'close' column not found, skipping target_direction.")

    # --- Clean up ---
    # Drop rows with NaNs from indicators/shifts (start/end of series)
    initial_rows = len(processed_df)
    processed_df.dropna(inplace=True)
    rows_dropped = initial_rows - len(processed_df)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows due to NaNs from "
              f"feature/target calculation.")

    print("Preprocessing and feature engineering complete. "
          f"DataFrame shape: {processed_df.shape}")
    return processed_df


# --- 3. Model Training ---


def train_classification_model(features: pd.DataFrame, target: pd.Series):
    """Trains a classification model (e.g., RandomForestClassifier).

    Uses the provided features and target data to train a model suitable
    for predicting market direction or other classification tasks.

    Args:
        features: DataFrame of input features for training.
        target: Series containing the target variable (e.g., 0 or 1) for
                training.

    Returns:
        The trained scikit-learn model object, or None if training fails.
    """
    print("--- Training Classification Model ---")
    if features.empty or target.empty:
        print("Error: Features or target data is empty for training.")
        return None
    if len(features) != len(target):
        print(f"Error: Features ({len(features)}) and target ({len(target)}) "
              "length mismatch.")
        return None

    print(f"Training with Features shape: {features.shape}")
    print(f"Training with Target shape: {target.shape}")

    # Check for NaNs before training
    if features.isnull().any().any() or target.isnull().any():
        print("Error: NaN values detected in features or target for training.")
        return None

    # Instantiate the model (example hyperparameters)
    # Consider tuning these parameters later (e.g., using GridSearchCV)
    model = RandomForestClassifier(
        n_estimators=100,     # Number of trees in the forest
        max_depth=10,         # Maximum depth of the trees
        min_samples_split=50,    # Min samples required to split internal node
        min_samples_leaf=25,     # Min samples required to be at a leaf node
        class_weight='balanced',   # Adjust weights for class imbalance
        random_state=42,      # For reproducibility
        n_jobs=-1             # Use all available CPU cores
    )

    try:
        # Train the model
        model.fit(features, target)
        print("Model training complete.")
        # Optional: Check if the model is fitted (basic sanity check)
        model.predict(features.iloc[[0]])  # Try predicting on one sample
        return model
    except NotFittedError:
        # This check might be redundant if fit() succeeded, but good practice.
        print("Error: Model reported as not fitted after training attempt.")
        return None
    except ValueError as ve:
        print(f"Error during model training (ValueError): {ve}")
        # Common issues: NaNs, inf values, non-numeric data
        return None
    except Exception as e:
        print(f"An unexpected error occurred during model training: {e}")
        return None


# --- 4. Prediction (Placeholder) ---

# --- 4. Prediction ---

def predict_with_model(model, features_live: pd.DataFrame) -> np.ndarray:
    """
    Generates class probabilities on new/live data using the trained model.

    Args:
        model: The trained scikit-learn compatible model object.
        features_live: DataFrame of features for which to make predictions.
                       Must have the same columns as the training features.

    Returns:
        Numpy array of shape (n_samples, n_classes) with probabilities
        for each class, or an empty array if prediction fails.
        Column 0: Probability of class 0 (e.g., down/flat)
        Column 1: Probability of class 1 (e.g., up)
    """
    print("--- Making Predictions ---")
    if model is None:
        print("Error: Invalid model object provided for prediction.")
        return np.array([])
    if not hasattr(model, 'predict_proba'):
        print("Error: Model object does not have a 'predict_proba' method.")
        return np.array([])
    if features_live.empty:
        print("Warning: No live features provided for prediction.")
        return np.array([])

    print(f"Predicting on live features shape: {features_live.shape}")

    try:
        # Ensure the model is fitted before predicting
        # (predict will raise NotFittedError if not)
        # Use predict_proba to get probabilities for each class
        probabilities = model.predict_proba(features_live)
        print(f"Generated probabilities for {len(probabilities)} samples.")
        return probabilities
    except NotFittedError:
        print("Error: Cannot predict probabilities, the model is not fitted.")
        return np.array([])
    except ValueError as ve:
        print(f"Error during probability prediction (ValueError): {ve}")
        # Common issue: features_live columns mismatch training columns, NaNs
        return np.array([])
    except Exception as e:
        print(f"Unexpected error during probability prediction: {e}")
        return np.array([])


# --- 5. Evaluation and Backtesting (Placeholder) ---

# --- 5. Evaluation and Backtesting ---

def backtest_strategy(
    data_with_predictions: pd.DataFrame,
    # Note: SL/TP implementation adds complexity, omitted in this basic version
    # stop_loss_pct: float | None = 0.02,
    # take_profit_pct: float | None = 0.04
):
    """
    Performs a basic vectorized backtest for a simple long-only strategy.

    Assumptions:
    - Buys/Holds when prediction is 1, sells/stays out when prediction is 0.
    - Trades occur at the 'close' price of the signal bar.
    - No transaction costs or slippage included.
    - Stop-loss / Take-profit are NOT implemented in this basic version.

    Args:
        data_with_predictions: DataFrame with OHLC data ('close') and
                               'prediction' column (0 or 1).

    Returns:
        A dictionary summarizing the basic backtest performance.
    """
    print("--- Backtesting Strategy (Basic Vectorized) ---")
    required_cols = ['close', 'prediction']
    if not all(col in data_with_predictions.columns for col in required_cols):
        print(f"Error: Missing required columns ({required_cols}) "
              "for backtesting.")
        return {}
    if data_with_predictions.empty:
        print("Warning: No data provided for backtesting.")
        return {}

    print(f"Backtesting data shape: {data_with_predictions.shape}")
    # sl_info = f"{stop_loss_pct*100}%" if stop_loss_pct else "None" # Omitted
    # Note: SL/TP info display omitted for basic backtest
    # print(f"Using SL={sl_info}, TP={tp_info}") # Omitted

    # Calculate daily returns of the asset
    df = data_with_predictions  # Use shorter alias
    df['daily_return'] = df['close'].pct_change(fill_method=None)

    # Calculate strategy returns (hold asset when signal is 1, else 0 return)
    # Shift prediction by 1 (trade based on previous day's signal)
    df['signal'] = df['prediction'].shift(1)
    df['strategy_return'] = df['signal'] * df['daily_return']

    # Drop initial NaN rows from pct_change and shift
    subset_cols = ['daily_return', 'signal', 'strategy_return']
    df.dropna(subset=subset_cols, inplace=True)

    if data_with_predictions.empty:
        print("Warning: No valid data remaining after calculating "
              "returns/signals.")
        return {}

    # Calculate cumulative returns
    df['cumulative_market_return'] = (
        (1 + df['daily_return']).cumprod() - 1
    )
    df['cumulative_strategy_return'] = (
        (1 + df['strategy_return']).cumprod() - 1
    )

    # Calculate basic performance metrics
    total_strategy_return = df['cumulative_strategy_return'].iloc[-1]
    total_market_return = df['cumulative_market_return'].iloc[-1]

    # Count trades (entry points: where signal changes from 0 to 1)
    # Ensure signal is integer type first
    df['signal'] = df['signal'].fillna(0).astype(int)
    # Count entries: where signal changes 0->1 OR first valid signal is 1
    entries = ((df['signal'] == 1) & (df['signal'].shift(1) == 0))
    # Check if the very first signal (after NaNs are dropped) is 1
    if not df.empty and df['signal'].iloc[0] == 1:
        # Manually add 1 trade if the first action is a buy
        num_trades = entries.sum() + 1
    else:
        num_trades = entries.sum()

    print("Basic backtesting logic complete.")
    performance_summary = {
        "total_strategy_return_pct": total_strategy_return * 100,
        "total_market_return_pct": total_market_return * 100,  # Buy & Hold
        "num_trades": num_trades
        # --- TBD: Add more metrics (Sharpe, Drawdown, Win Rate) ---
        # Requires more complex calculations (e.g., tracking equity curve)
    }
    return performance_summary


# --- 6. Optimization (Placeholder) ---
# TBD: Functions for optimizing strategy parameters (e.g., indicator lengths,
# SL/TP levels) by running multiple backtests with different parameter sets.
# Often involves libraries like Optuna or grid search.


# --- 7. Main Workflow ---

def run_trading_strategy_analysis(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str
):
    """
    Orchestrates the entire analysis workflow for a given symbol and timeframe.
    """
    print(f"\n=== Running Analysis for {symbol} ({timeframe}) ===")
    print(f"Data Period: {start_date} to {end_date}")

    # 1. Load Data using data_loader
    raw_data = load_data(symbol, start_date, end_date)
    if raw_data is None or raw_data.empty:
        print(f"Failed to load data for {symbol}. Exiting analysis.")
        return

    # 2. Preprocess and Engineer Features
    featured_data = preprocess_and_engineer_features(raw_data, symbol)
    if featured_data is None or featured_data.empty:
        print("Failed to process features. Exiting analysis.")
        return

    # 3. Separate features (X) and target (y)
    if 'target_direction' not in featured_data.columns:
        print("Error: 'target_direction' column not found after features.")
        return

    # Define columns to exclude from features (target, raw OHLCV, etc.)
    # Adjust based on actual features engineered and model requirements.
    cols_to_drop_for_features = [
        'open', 'high', 'low', 'close', 'volume',  # Raw prices often dropped
        'target_direction'                         # Target variable itself
    ]
    # Ensure all columns exist before dropping
    cols_to_drop_for_features = [
        col for col in cols_to_drop_for_features
        if col in featured_data.columns
    ]
    features = featured_data.drop(columns=cols_to_drop_for_features)
    target = featured_data['target_direction']

    # --- TBD: Implement proper Train/Test Split (Chronological) ---
    # Crucial for time series to avoid lookahead bias.
    # Example: Split data into 80% train, 20% test based on date.
    split_ratio = 0.8
    split_index = int(len(features) * split_ratio)
    train_features = features.iloc[:split_index]
    train_target = target.iloc[:split_index]
    test_features = features.iloc[split_index:]
    # test_target = target.iloc[split_index:] # For evaluation (unused now)

    print(f"Data split: Train {len(train_features)} rows, "
          f"Test {len(test_features)} rows")
    print("Warning: Using simple chronological split. "
          "Consider more robust validation (e.g., walk-forward).")

    if train_features.empty or test_features.empty:
        print("Error: Not enough data for train/test split after processing.")
        return

    # 4. Train Model
    model = train_classification_model(train_features, train_target)
    if model is None:
        print("Model training failed. Exiting.")
        return

    # 5. Generate Probabilities (on test data)
    probabilities = predict_with_model(model, test_features)
    if len(probabilities) != len(test_features):
        print(f"Error: Probability length ({len(probabilities)}) mismatch "
              f"with test features ({len(test_features)}).")
        return
    # Display probability info (temporary - replace with actual use)
    print(f"Probabilities shape: {probabilities.shape}")
    if len(probabilities) > 0:
        print("Sample Probabilities (Class 0, Class 1):")
        print(probabilities[:5])  # Print first 5 probability pairs

    # --- TODO: Use probabilities in backtesting or decision logic ---
    # For now, we still need a binary prediction for the basic backtester.
    # We can derive this from probabilities, e.g., predict 1 if P(1) > 0.5
    # Note: This threshold (0.5) could be optimized later.
    # Get class 1 prediction (e.g., if P(1) > 0.5)
    predictions = (probabilities[:, 1] > 0.5).astype(int)

    # 6. Add derived binary predictions back for the *basic* backtester
    # Ensure alignment using the index
    data_for_backtest = featured_data.loc[test_features.index].copy()
    data_for_backtest['prediction'] = predictions  # Use derived prediction
    # Store probabilities as well for potential future use/analysis
    data_for_backtest['probability_class_0'] = probabilities[:, 0]
    data_for_backtest['probability_class_1'] = probabilities[:, 1]

    # 7. Backtest Strategy (using placeholder)
    # --- TBD: Define SL/TP rules (potentially optimize them later) ---
    # Omitted SL/TP args for basic backtest
    performance = backtest_strategy(data_for_backtest)

    # 8. Display Results
    print("\n--- Performance Summary (Basic Backtest) ---")
    if performance:
        for metric, value in performance.items():
            # Format percentages nicely
            if "pct" in metric:
                print(f"{metric}: {value:.2f}%")
            elif isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
    else:
        print("Backtesting did not produce results.")

    print(f"\n=== Analysis Complete for {symbol} ({timeframe}) ===")


# --- Execution Example ---

if __name__ == "__main__":
    print("Starting Trading Strategy Analysis Workflow...")
    # Example: Run analysis for a default symbol and timeframe
    run_trading_strategy_analysis(
        symbol=DEFAULT_SYMBOL,
        timeframe=DEFAULT_TIMEFRAME,
        start_date=DEFAULT_START_DATE,
        end_date=DEFAULT_END_DATE
    )

    # --- TBD: Add logic to loop through multiple symbols/timeframes ---
    # print("\n--- Running Batch Analysis (Example) ---")
    # symbols_to_run = ['MSFT', 'GOOGL'] # Add more symbols
    # timeframes_to_run = ['1d'] # Add more timeframes if needed
    # for sym in symbols_to_run:
    #     for tf in timeframes_to_run:
    #         run_trading_strategy_analysis(
    #               symbol=sym, timeframe=tf,
    #               start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE
    #         )

    print("\nWorkflow finished.")