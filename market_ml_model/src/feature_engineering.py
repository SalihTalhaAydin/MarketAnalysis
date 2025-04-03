import pandas as pd

# Using pandas_ta for technical indicators. Install: pip install pandas-ta
try:
    import pandas_ta as ta
except ImportError:
    print("Warning: pandas-ta not installed (pip install pandas-ta). "
          "Feature engineering using it will be skipped.")
    ta = None


def engineer_features(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Adds technical indicators, returns, lagged features, and target variable.

    Assumes the input DataFrame has already been preprocessed
    (e.g., NaNs handled for OHLCV, columns are lowercase).
    for OHLCV, columns are lowercase).

    Args:
        df: Preprocessed DataFrame with 'open', 'high', 'low', 'close',
            'volume'.

    Returns:
        DataFrame with added features, or None if processing fails.
    """
    print("--- Engineering Features ---")
    if df is None or df.empty:
        print("Error: Input DataFrame is empty/None for feature engineering.")
        return None

    processed_df = df.copy()  # Work on a copy

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

    print("Feature engineering complete. "
          f"DataFrame shape: {processed_df.shape}")
    return processed_df