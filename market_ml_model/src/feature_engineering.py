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

    if not isinstance(df.index, pd.DatetimeIndex):
        print("Error: Index must be DatetimeIndex for time features.")
        # Attempt conversion if possible, otherwise return None
        try:
            df.index = pd.to_datetime(df.index)
            print("Converted index to DatetimeIndex.")
        except Exception as e:
            print(f"Failed to convert index to DatetimeIndex: {e}")
            return None

    processed_df = df.copy()  # Work on a copy

    # --- Feature Engineering Steps ---

    # --- Feature Engineering Steps ---

    if ta:
        print("Calculating indicators using pandas_ta...")
        # Example Indicators using pandas_ta:
        try:
            # Simple Moving Average (SMA)
            # Shorter SMAs for scalping
            processed_df.ta.sma(length=5, append=True)   # Appends 'SMA_5'
            processed_df.ta.sma(length=10, append=True)  # Appends 'SMA_10'

            # Relative Strength Index (RSI)
            # Relative Strength Index (RSI)
            # Shorter RSI period
            processed_df.ta.rsi(length=9, append=True)   # Appends 'RSI_9'

            # Bollinger Bands
            # Appends 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0'
            # Shorter Bollinger Bands period
            # Appends 'BBL_10_2.0', 'BBM_10_2.0', 'BBU_10_2.0', 'BBB_10_2.0'
            processed_df.ta.bbands(length=10, std=2, append=True)

            # Average True Range (ATR) - useful for stops/volatility
            # Shorter ATR period
            processed_df.ta.atr(length=10, append=True)  # Appends 'ATRr_10'

            # --- Add More Scalping-Relevant Indicators ---
            # MACD (Defaults: 12, 26, 9)
            # Appends MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            processed_df.ta.macd(append=True)

            # Stochastic Oscillator (%K, %D) - Use shorter periods
            # Appends STOCHk_14_3_3, STOCHd_14_3_3 (default) -> Let's shorten
            # Stochastic Oscillator (%K, %D) - Use shorter periods
            # Appends STOCHk_5_3_3, STOCHd_5_3_3
            processed_df.ta.stoch(k=5, d=3, smooth_k=3, append=True)

            # Volume SMA (Simple Moving Average of Volume)
            if 'volume' in processed_df.columns:
                processed_df.ta.sma(close='volume', length=20, prefix='VOL',
                                    append=True)  # Appends VOL_SMA_20
            else:
                print("Warning: 'volume' col not found, skipping Volume SMA.")

            print("pandas_ta indicators calculated.")
        except Exception as e:
            print(f"Error calculating pandas_ta indicators: {e}")
            # Stop processing if indicators fail
            return None
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

    # Example: Volume Rate of Change (ROC) - 5 period
    if 'volume' in processed_df.columns:
        processed_df['volume_roc'] = processed_df['volume'] \
                                        .pct_change(periods=5)
    else:
        print("Warning: 'volume' col not found, skipping volume_roc.")

    # --- Add Time-Based Features ---
    # Ensure index is DatetimeIndex (checked at the beginning)
    processed_df['hour'] = processed_df.index.hour
    processed_df['minute'] = processed_df.index.minute
    # Consider dayofweek if running over multiple days:
    # processed_df['dayofweek'] = processed_df.index.dayofweek

    # --- Define Target Variable (Predict direction 5 periods ahead) ---
    # Target = 1 if close 5 periods later > current close, 0 otherwise
    # Important: Ensure features (X) only use data up to time t.
    target_shift = -5
    if 'close' in processed_df.columns:
        target_condition = (processed_df['close'].shift(target_shift) >
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