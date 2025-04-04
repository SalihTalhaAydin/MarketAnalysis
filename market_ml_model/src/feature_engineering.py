import pandas as pd
import numpy as np

# Using pandas_ta for technical indicators. Install: pip install pandas-ta
try:
    import pandas_ta as ta
except ImportError:
    print("Warning: pandas-ta not installed (pip install pandas-ta). "
          "Feature engineering using it will be skipped.")
    ta = None


# --- Triple Barrier Calculation ---

def get_triple_barrier_labels(
    prices: pd.Series,
    highs: pd.Series,
    lows: pd.Series,
    atr: pd.Series,
    atr_multiplier_tp: float,
    atr_multiplier_sl: float,
    max_holding_period: int
) -> pd.Series:
    """
    Calculates Triple Barrier labels for each timestamp.

    Labels:
        1: Take Profit hit first.
       -1: Stop Loss hit first.
        0: Neither hit within max_holding_period (Time Barrier).

    Args:
        prices: Series of closing prices (used for entry reference).
        highs: Series of high prices (used to check TP).
        lows: Series of low prices (used to check SL).
        atr: Series of ATR values at entry time.
        atr_multiplier_tp: Multiplier for Take Profit distance from entry.
        atr_multiplier_sl: Multiplier for Stop Loss distance from entry.
        max_holding_period: Maximum number of bars to hold the hypothetical
                            trade.

    Returns:
        pd.Series containing the outcome labels (1, -1, 0) or NaN if
        calculation is not possible (e.g., near the end of the series).
    """
    n = len(prices)
    # Initialize labels with NaN (easier to handle end cases)
    labels = pd.Series(np.nan, index=prices.index)
    # Calculate target levels for all points upfront
    tp_levels = prices + atr * atr_multiplier_tp
    sl_levels = prices - atr * atr_multiplier_sl

    # Iterate through each potential entry point `i`
    for i in range(n - 1):  # Stop before the last row
        entry_idx = prices.index[i]
        tp = tp_levels.iloc[i]
        sl = sl_levels.iloc[i]

        # Determine the lookahead window (up to max_holding_period or end
        # of data)
        lookahead_end_idx = min(i + 1 + max_holding_period, n)
        window_highs = highs.iloc[i + 1: lookahead_end_idx]
        window_lows = lows.iloc[i + 1: lookahead_end_idx]

        # Find the first time TP or SL is hit within the window
        tp_hit_times = window_highs[window_highs >= tp].index
        sl_hit_times = window_lows[window_lows <= sl].index

        tp_hits = tp_hit_times.min() if not tp_hit_times.empty else pd.NaT
        sl_hits = sl_hit_times.min() if not sl_hit_times.empty else pd.NaT
        first_tp_hit_time = tp_hits
        first_sl_hit_time = sl_hits

        # Determine the outcome based on which barrier was hit first
        outcome = 0  # Default to time barrier
        if not pd.isna(first_tp_hit_time) and not pd.isna(first_sl_hit_time):
            # Both hit, choose the earliest
            if first_tp_hit_time <= first_sl_hit_time:
                outcome = 1  # TP hit first or simultaneously
            else:
                outcome = -1  # SL hit first
        elif not pd.isna(first_tp_hit_time):
            outcome = 1  # Only TP hit
        elif not pd.isna(first_sl_hit_time):
            outcome = -1  # Only SL hit
        # else: outcome remains 0 (neither hit within the window)

        # Assign the calculated outcome to the entry point's index
        labels.loc[entry_idx] = outcome

    # The last few rows where the full lookahead window wasn't available
    # will remain NaN, which is desired.
    return labels


# --- Main Feature Engineering Function ---
def engineer_features(
    df: pd.DataFrame,
    atr_multiplier_tp: float = 2.0,  # Changed to 2:1 R:R with SL
    atr_multiplier_sl: float = 1.0,  # Default for 5m
    max_holding_period: int = 10     # Default for 5m
) -> pd.DataFrame | None:
    """
    Adds technical indicators, returns, lagged features, and Triple Barrier
    target.

    Assumes the input DataFrame has 'open', 'high', 'low', 'close', 'volume'
    and has been preprocessed (e.g., NaNs handled for OHLCV).

    Args:
        df: Preprocessed DataFrame.
        atr_multiplier_tp: Multiplier for Take Profit distance based on ATR.
        atr_multiplier_sl: Multiplier for Stop Loss distance based on ATR.
        max_holding_period: Maximum number of bars for the vertical barrier.

    Returns:
        DataFrame with added features and 'triple_barrier_label', or None if
        fails.
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
            # Ensure ATR is calculated as it's needed for barriers
            if 'ATRr_10' not in processed_df.columns:
                # Appends 'ATRr_10'
                processed_df.ta.atr(length=10, append=True)
            else:  # If already exists (e.g. from preprocessing)
                pass  # ATRr_10 already present

            # --- Add More Scalping-Relevant Indicators ---
            # Exponential Moving Averages (EMA) - Faster reaction than SMA
            processed_df.ta.ema(length=5, append=True)  # Appends 'EMA_5'
            processed_df.ta.ema(length=8, append=True)  # Appends 'EMA_8'

            # MACD (Defaults: 12, 26, 9) - Keep commented out for now
            # Appends MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            # processed_df.ta.macd(append=True)

            # Faster MACD for Scalping
            # Appends MACD_5_10_5, MACDh_5_10_5, MACDs_5_10_5
            processed_df.ta.macd(fast=5, slow=10, signal=5, append=True)

            # Stochastic Oscillator (%K, %D) - Removed as potentially noisy
            # # Appends STOCHk_14_3_3, STOCHd_14_3_3 (default) -> Let's shorten
            # # Stochastic Oscillator (%K, %D) - Use shorter periods
            # # Appends STOCHk_5_3_3, STOCHd_5_3_3
            # processed_df.ta.stoch(k=5, d=3, smooth_k=3, append=True)

            # Volume SMA (Simple Moving Average of Volume)
            if 'volume' in processed_df.columns:
                processed_df.ta.sma(close='volume', length=20, prefix='VOL',
                                    append=True)  # Appends VOL_SMA_20
            else:
                print("Warning: 'volume' col not found, skipping Volume SMA.")

            # Price Rate of Change (ROC) - Removed as potentially noisy
            # processed_df.ta.roc(length=3, append=True)  # Appends 'ROC_3'

            # --- Add Custom Features Based on Indicators ---
            # EMA 5/8 Crossover Signal
            if ('EMA_5' in processed_df.columns and
                    'EMA_8' in processed_df.columns):
                ema5 = processed_df['EMA_5']
                ema8 = processed_df['EMA_8']
                processed_df['EMA_5_8_cross'] = 0
                # Cross above: EMA5 was below EMA8, now is above
                cross_above_mask = ((ema5 > ema8) &
                                    (ema5.shift(1) < ema8.shift(1)))
                processed_df.loc[cross_above_mask, 'EMA_5_8_cross'] = 1
                # Cross below: EMA5 was above EMA8, now is below
                cross_below_mask = ((ema5 < ema8) &
                                    (ema5.shift(1) > ema8.shift(1)))
                processed_df.loc[cross_below_mask, 'EMA_5_8_cross'] = -1
            else:
                print("Warning: EMA_5 or EMA_8 missing, skipping EMA "
                      "crossover feature.")

            # Distance from SMA10, normalized by ATR
            if ('SMA_10' in processed_df.columns and
                    'ATRr_10' in processed_df.columns):
                # Avoid division by zero or near-zero ATR
                atr_safe = processed_df['ATRr_10'].replace(
                    0, np.nan
                ).ffill()  # Use ffill() directly
                dist_sma_atr = (
                    (processed_df['close'] - processed_df['SMA_10']) / atr_safe
                )
                # Fill initial NaNs and assign back
                processed_df['dist_from_SMA10_atr'] = dist_sma_atr.fillna(0)
            else:
                print("Warning: SMA_10 or ATRr_10 missing, skipping distance "
                      "from SMA feature.")

            print("pandas_ta indicators and derived features calculated.")
        except Exception as e:
            print("Error calculating pandas_ta indicators or derived "
                  f"features: {e}")
            return None  # Stop processing if indicators fail
    elif 'ATRr_10' not in processed_df.columns:
        # If pandas_ta failed or wasn't available, we MUST have ATRr_10
        print("Error: ATRr_10 is required but is missing.")  # Still need ATR
        return None
    # else: pandas_ta not available, but ATRr_10 exists, proceed.

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

    # --- Add Time-Based Features --- # Keep hour/minute commented out
    # Ensure index is DatetimeIndex (checked at the beginning)
    # processed_df['hour'] = processed_df.index.hour
    # processed_df['minute'] = processed_df.index.minute
    # Consider dayofweek if running over multiple days:
    # processed_df['dayofweek'] = processed_df.index.dayofweek

    # --- Define Target Variable (Triple Barrier Method) ---
    print(f"--- Calculating Triple Barrier Labels "
          f"(TP: {atr_multiplier_tp}*ATR, SL: {atr_multiplier_sl}*ATR, "
          f"Max Hold: {max_holding_period} bars) ---")
    required_cols = ['close', 'high', 'low', 'ATRr_10']
    if not all(col in processed_df.columns for col in required_cols):
        missing = [c for c in required_cols if c not in processed_df.columns]
        print(f"Error: Missing required columns for Triple Barrier: {missing}")
        return None

    # Drop NaNs from features *before* calculating labels to ensure alignment
    # Keep track of original index if needed later, but labels are based on
    # clean data
    feature_cols = processed_df.columns.difference(['triple_barrier_label'])
    # Drop rows with NaNs in feature columns before label calculation
    processed_df.dropna(subset=feature_cols, inplace=True)
    if processed_df.empty:
        print("Error: DataFrame empty after dropping NaNs from features.")
        return None

    # Calculate labels on the feature-cleaned data
    processed_df['triple_barrier_label'] = get_triple_barrier_labels(
        prices=processed_df['close'],
        highs=processed_df['high'],
        lows=processed_df['low'],
        atr=processed_df['ATRr_10'],
        atr_multiplier_tp=atr_multiplier_tp,
        atr_multiplier_sl=atr_multiplier_sl,
        max_holding_period=max_holding_period
    )

    # --- Clean up ---
    # Drop rows where the label could not be calculated (NaNs from barrier
    # func, e.g., near the end of the data)
    initial_rows = len(processed_df)
    processed_df.dropna(subset=['triple_barrier_label'], inplace=True)
    rows_dropped = initial_rows - len(processed_df)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows due to NaNs from "
              f"Triple Barrier label calculation (e.g., end of series).")

    # Convert label to integer type after dropping NaNs
    if 'triple_barrier_label' in processed_df.columns:
        processed_df['triple_barrier_label'] = processed_df[
            'triple_barrier_label'
        ].astype(int)

    # --- Sanitize Column Names ---
    # Replace spaces and potentially other problematic characters with
    # underscores to avoid issues with models like LightGBM.
    original_cols = processed_df.columns
    processed_df.columns = (
        processed_df.columns.str.replace(r'[\s\(\)\%\.]+', '_', regex=True)
        .str.replace(r'_+', '_', regex=True)  # Consolidate underscores
        .str.strip('_')  # Remove leading/trailing underscores
    )
    cleaned_cols = processed_df.columns
    if list(original_cols) != list(cleaned_cols):
        print("Sanitized column names (example changes):")
        for orig, clean in zip(original_cols, cleaned_cols):
            if orig != clean:
                print(f"  '{orig}' -> '{clean}'")
                break  # Just show one example

    print("Feature engineering complete (with Triple Barrier). "
          f"DataFrame shape: {processed_df.shape}")
    return processed_df