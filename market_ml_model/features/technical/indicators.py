"""
Technical indicators for market data analysis.
"""

import logging
from typing import Any, Dict, List, Tuple  # Add Any here

import numpy as np
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

# pandas-ta import removed. Indicators relying on it will be skipped or need manual implementation.
# Try importing advanced stats libraries
try:
    from arch import arch_model
    from hurst import compute_Hc
    from statsmodels.tsa.stattools import adfuller

    ADVANCED_STATS = True
except ImportError:
    logger.warning(
        "Advanced statistical packages not installed. Some features will be disabled."
    )
    ADVANCED_STATS = False


# Helper function for ATR calculation using pandas/numpy
def _calculate_atr_pandas(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.Series:
    """Calculates Average True Range (ATR) using pandas."""
    if not all(isinstance(s, pd.Series) for s in [high, low, close]):
        # Log error instead of raising, return empty series
        logger.error("ATR calculation inputs must be pandas Series.")
        return pd.Series(
            dtype=float, index=close.index
        )  # Return empty series matching index
    if (
        high.empty
        or low.empty
        or close.empty
        or len(high) < length
        or len(low) < length
        or len(close) < length
    ):
        # Return NaN series if input is empty or too short
        logger.warning(
            f"ATR calculation input data is empty or shorter than length {length}."
        )
        return pd.Series(np.nan, index=close.index)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    # Ensure alignment before max operation, fill initial NaN in prev_close
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=False)
    tr.iloc[0] = np.nan  # First TR is undefined

    # Calculate ATR using Exponential Moving Average (EMA) - Wilder's Smoothing (RMA)
    # pandas ewm adjust=False matches Wilder's smoothing.
    # Use min_periods=length to ensure enough data before starting calculation.
    atr = tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    logger.debug(f"Calculated ATR with length {length} using pandas/numpy.")
    return atr


# Helper function for SMA calculation using pandas/numpy
def _calculate_sma_pandas(series: pd.Series, length: int) -> pd.Series:
    """Calculates Simple Moving Average (SMA) using pandas."""
    if not isinstance(series, pd.Series):
        logger.error("SMA calculation input must be a pandas Series.")
        return pd.Series(
            dtype=float, index=series.index if hasattr(series, "index") else None
        )
    if series.empty or len(series) < length:
        logger.warning(
            f"SMA calculation input data is empty or shorter than length {length}."
        )
        return pd.Series(np.nan, index=series.index)
    sma = series.rolling(window=length, min_periods=length).mean()
    logger.debug(f"Calculated SMA with length {length} using pandas/numpy.")
    return sma


# Helper function for EMA calculation using pandas/numpy
def _calculate_ema_pandas(series: pd.Series, length: int) -> pd.Series:
    """Calculates Exponential Moving Average (EMA) using pandas."""
    if not isinstance(series, pd.Series):
        logger.error("EMA calculation input must be a pandas Series.")
        return pd.Series(
            dtype=float, index=series.index if hasattr(series, "index") else None
        )
    if series.empty or len(series) < length:
        logger.warning(
            f"EMA calculation input data is empty or shorter than length {length}."
        )
        return pd.Series(np.nan, index=series.index)
    # Standard EMA uses span
    ema = series.ewm(span=length, adjust=False, min_periods=length).mean()
    logger.debug(f"Calculated EMA with length {length} using pandas/numpy.")
    return ema


# Helper function for RSI calculation using pandas/numpy
def _calculate_rsi_pandas(close: pd.Series, length: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI) using pandas."""
    if not isinstance(close, pd.Series):
        logger.error("RSI calculation input must be a pandas Series.")
        return pd.Series(
            dtype=float, index=close.index if hasattr(close, "index") else None
        )
    if close.empty or len(close) < length + 1:  # Need length+1 for diff
        logger.warning(
            f"RSI calculation input data is empty or shorter than length+1 ({length+1})."
        )
        return pd.Series(np.nan, index=close.index)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Use Wilder's smoothing (alpha = 1 / length) like pandas-ta
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    # Handle division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)  # Avoid division by zero, result is NaN
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Fill initial NaNs from calculation start
    rsi[:length] = np.nan

    logger.debug(f"Calculated RSI with length {length} using pandas/numpy.")
    return rsi


# Helper function for MACD calculation using pandas/numpy
def _calculate_macd_pandas(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Calculates Moving Average Convergence Divergence (MACD) using pandas."""
    if not isinstance(close, pd.Series):
        logger.error("MACD calculation input must be a pandas Series.")
        return pd.DataFrame(index=close.index if hasattr(close, "index") else None)
    min_len = max(slow, fast) + signal - 1  # Approximate minimum length needed
    if close.empty or len(close) < min_len:
        logger.warning(
            f"MACD calculation input data is empty or shorter than required length (~{min_len})."
        )
        return pd.DataFrame(
            columns=[
                f"MACD_{fast}_{slow}",
                f"MACDs_{signal}",
                f"MACDh_{fast}_{slow}_{signal}",
            ],
            index=close.index,
        )

    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line

    # Match common pandas-ta column naming structure
    macd_df = pd.DataFrame(
        {
            f"MACD_{fast}_{slow}": macd_line,
            f"MACDs_{signal}": signal_line,  # Signal line suffix often includes signal period
            f"MACDh_{fast}_{slow}_{signal}": histogram,  # Histogram often includes all periods
        },
        index=close.index,
    )

    logger.debug(f"Calculated MACD ({fast},{slow},{signal}) using pandas/numpy.")
    return macd_df


# Helper function for Bollinger Bands calculation using pandas/numpy
def _calculate_bbands_pandas(
    close: pd.Series, length: int = 20, std: float = 2.0
) -> pd.DataFrame:
    """Calculates Bollinger Bands (BBands) using pandas."""
    if not isinstance(close, pd.Series):
        logger.error("BBands calculation input must be a pandas Series.")
        return pd.DataFrame(index=close.index if hasattr(close, "index") else None)
    if close.empty or len(close) < length:
        logger.warning(
            f"BBands calculation input data is empty or shorter than length {length}."
        )
        return pd.DataFrame(
            columns=[
                f"BBL_{length}_{std}",
                f"BBM_{length}_{std}",
                f"BBU_{length}_{std}",
            ],
            index=close.index,
        )

    sma = close.rolling(window=length, min_periods=length).mean()
    rolling_std = close.rolling(window=length, min_periods=length).std()
    upper_band = sma + (rolling_std * std)
    lower_band = sma - (rolling_std * std)

    # Match common pandas-ta column naming structure
    bbands_df = pd.DataFrame(
        {
            f"BBL_{length}_{std}": lower_band,  # Lower Band
            f"BBM_{length}_{std}": sma,  # Middle Band (SMA)
            f"BBU_{length}_{std}": upper_band,  # Upper Band
        },
        index=close.index,
    )

    logger.debug(f"Calculated Bollinger Bands ({length},{std}) using pandas/numpy.")
    return bbands_df


# ADX Implementation
def _calculate_adx_pandas(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.DataFrame:
    """Calculates Average Directional Index (ADX) using pandas."""
    if not all(isinstance(s, pd.Series) for s in [high, low, close]):
        logger.error("ADX calculation inputs must be pandas Series.")
        return pd.DataFrame(index=close.index if hasattr(close, "index") else None)
    min_len = length + 1  # Need length + 1 for diff
    if (
        high.empty
        or low.empty
        or close.empty
        or len(high) < min_len
        or len(low) < min_len
        or len(close) < min_len
    ):
        logger.warning(
            f"ADX calculation input data is empty or shorter than required length (~{min_len})."
        )
        return pd.DataFrame(
            columns=[f"ADX_{length}", f"DMP_{length}", f"DMN_{length}"],
            index=close.index,
        )

    # Calculate True Range (TR) - reusing ATR logic components
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=False)
    tr.iloc[0] = np.nan  # First TR is undefined

    # Calculate Directional Movement (+DM, -DM)
    move_up = high.diff()
    move_down = -low.diff()  # Note the negative sign

    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)

    plus_dm[(move_up > move_down) & (move_up > 0)] = move_up[
        (move_up > move_down) & (move_up > 0)
    ]
    minus_dm[(move_down > move_up) & (move_down > 0)] = move_down[
        (move_down > move_up) & (move_down > 0)
    ]

    # Use Wilder's smoothing (EMA with alpha = 1 / length)
    atr = tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    plus_di = (
        100
        * plus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
        / atr.replace(0, np.nan)
    )
    minus_di = (
        100
        * minus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
        / atr.replace(0, np.nan)
    )

    # Calculate DX
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    dx.iloc[: length - 1] = np.nan  # DX calculation starts after initial smoothing

    # Calculate ADX (Smoothed DX)
    adx = dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    adx.iloc[: 2 * length - 2] = (
        np.nan
    )  # ADX requires DX to be smoothed, so more NaNs initially

    # Match common pandas-ta column naming structure (approximate)
    adx_df = pd.DataFrame(
        {
            f"ADX_{length}": adx,
            f"DMP_{length}": plus_di,  # +DI
            f"DMN_{length}": minus_di,  # -DI
        },
        index=close.index,
    )

    logger.debug(f"Calculated ADX ({length}) using pandas/numpy.")
    return adx_df


# OBV Implementation
def _calculate_obv_pandas(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculates On Balance Volume (OBV) using pandas."""
    if not isinstance(close, pd.Series) or not isinstance(volume, pd.Series):
        logger.error("OBV calculation inputs must be pandas Series.")
        return pd.Series(
            dtype=float, index=close.index if hasattr(close, "index") else None
        )
    if close.empty or volume.empty or len(close) != len(volume):
        logger.warning("OBV calculation input data is empty or lengths mismatch.")
        return pd.Series(np.nan, index=close.index)

    # Calculate OBV
    signed_volume = volume * np.sign(close.diff()).fillna(
        0
    )  # Volume signed by price change direction
    obv = signed_volume.cumsum()

    logger.debug("Calculated OBV using pandas/numpy.")
    return obv


# MFI Implementation
def _calculate_mfi_pandas(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int = 14,
) -> pd.Series:
    """Calculates Money Flow Index (MFI) using pandas."""
    if not all(isinstance(s, pd.Series) for s in [high, low, close, volume]):
        logger.error("MFI calculation inputs must be pandas Series.")
        return pd.Series(
            dtype=float, index=close.index if hasattr(close, "index") else None
        )
    min_len = length + 1  # Need length + 1 for diff
    if high.empty or low.empty or close.empty or volume.empty or len(close) < min_len:
        logger.warning(
            f"MFI calculation input data is empty or shorter than required length (~{min_len})."
        )
        return pd.Series(np.nan, index=close.index)

    # Calculate Typical Price
    typical_price = (high + low + close) / 3

    # Calculate Raw Money Flow
    raw_money_flow = typical_price * volume

    # Calculate Positive and Negative Money Flow
    typical_price_diff = typical_price.diff()
    positive_money_flow = raw_money_flow.where(typical_price_diff > 0, 0)
    negative_money_flow = raw_money_flow.where(typical_price_diff < 0, 0)

    # Calculate Money Flow Ratio
    positive_mf_sum = positive_money_flow.rolling(
        window=length, min_periods=length
    ).sum()
    negative_mf_sum = negative_money_flow.rolling(
        window=length, min_periods=length
    ).sum()

    # Avoid division by zero
    money_flow_ratio = positive_mf_sum / negative_mf_sum.replace(0, np.nan)

    # Calculate MFI
    mfi = 100.0 - (100.0 / (1.0 + money_flow_ratio))
    # Handle cases where negative_mf_sum is 0 (MFI should be 100)
    mfi[negative_mf_sum == 0] = 100.0
    # Handle cases where both are 0 (or ratio is NaN due to division by zero) - typically set to 50 or previous value? Let's use NaN for now.
    mfi[money_flow_ratio.isna()] = (
        np.nan
    )  # Or potentially 50? Check standard definitions. Using NaN is safer.

    # Fill initial NaNs from rolling sum
    mfi.iloc[: length - 1] = np.nan

    logger.debug(f"Calculated MFI with length {length} using pandas/numpy.")
    return mfi


def calculate_technical_indicators(
    df: pd.DataFrame, indicator_configs: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Calculate technical indicators based on a list of configurations using pandas-ta.

    Handles common indicator name variations and parameter passing. Ensures ATRr_10
    is calculated if ATR is requested.

    Args:
        df: DataFrame with OHLCV data. Must contain 'open', 'high', 'low', 'close', 'volume'.
        indicator_configs: A list of dictionaries, where each dictionary defines
                           an indicator and its parameters. Example:
                           ``[{'indicator': 'sma', 'length': 50}, {'indicator': 'rsi', 'length': 14}]``

    Returns:
        DataFrame with added technical indicator columns.
    """
    # Removed PANDAS_TA_AVAILABLE check. ATR is implemented locally. Others are skipped.

    # Mapping from common config names to pandas-ta function names
    indicator_mapping = {
        "bollinger_bands": "bbands",
        "atr": "atr",
        "rsi": "rsi",
        "sma": "sma",
        "ema": "ema",
        "macd": "macd",
        "stoch": "stoch",
        "mfi": "mfi",
        "adx": "adx",
        "obv": "obv",
        "vwap": "vwap",
        # Add more mappings as needed
    }

    # Make a copy of the dataframe to avoid modifying the original
    result_df = df.copy()

    # Check for required columns (case-insensitive check)
    required_cols_base = ["open", "high", "low", "close", "volume"]
    df_cols_lower = [col.lower() for col in result_df.columns]
    required_cols = []
    col_mapping = {}  # Map required base name to actual df column name
    for req_col in required_cols_base:
        found = False
        for i, df_col_lower in enumerate(df_cols_lower):
            if df_col_lower == req_col:
                required_cols.append(result_df.columns[i])
                col_mapping[req_col] = result_df.columns[i]
                found = True
                break
        if not found:
            logger.error(
                f"Missing required column '{req_col}' (case-insensitive) for technical indicators. Returning original DataFrame."
            )
            return df  # Stop if essential columns are missing

    # Extract standard OHLCV columns using the mapping
    high_col = col_mapping["high"]
    low_col = col_mapping["low"]
    close_col = col_mapping["close"]
    volume_col = col_mapping["volume"]

    try:
        calculated_count = 0
        for config in indicator_configs:
            config_indicator_name = config.get("indicator")
            if not config_indicator_name:
                logger.warning(f"Skipping config without 'indicator' key: {config}")
                continue

            # Get the actual pandas-ta function name from mapping, default to config name
            ta_indicator_name = indicator_mapping.get(
                config_indicator_name.lower(), config_indicator_name
            )

            # Extract parameters, excluding the 'indicator' key
            params = {k: v for k, v in config.items() if k != "indicator"}

            try:
                # --- Refactored Indicator Calculation ---
                indicator_result = None  # Initialize

                if ta_indicator_name == "atr":
                    # Use the custom pandas/numpy ATR implementation
                    atr_length = params.get(
                        "length", 14
                    )  # Get length from params or default
                    indicator_result = _calculate_atr_pandas(
                        high=result_df[high_col],
                        low=result_df[low_col],
                        close=result_df[close_col],
                        length=atr_length,
                    )
                    if indicator_result is None or indicator_result.empty:
                        logger.warning(
                            f"Custom ATR calculation failed for length {atr_length}. Skipping."
                        )
                        continue
                    # Ensure the result is a Series (ATR returns a Series)
                    if not isinstance(indicator_result, pd.Series):
                        logger.error(
                            f"Custom ATR calculation returned unexpected type: {type(indicator_result)}. Skipping."
                        )
                        continue

                # --- Call appropriate helper function ---
                elif ta_indicator_name == "sma":
                    sma_length = params.get("length", 50)  # Default SMA length
                    indicator_result = _calculate_sma_pandas(
                        series=result_df[close_col], length=sma_length
                    )
                elif ta_indicator_name == "ema":
                    ema_length = params.get("length", 50)  # Default EMA length
                    indicator_result = _calculate_ema_pandas(
                        series=result_df[close_col], length=ema_length
                    )
                elif ta_indicator_name == "rsi":
                    rsi_length = params.get("length", 14)  # Default RSI length
                    indicator_result = _calculate_rsi_pandas(
                        close=result_df[close_col], length=rsi_length
                    )
                elif ta_indicator_name == "macd":
                    macd_fast = params.get("fast", 12)
                    macd_slow = params.get("slow", 26)
                    macd_signal = params.get("signal", 9)
                    indicator_result = _calculate_macd_pandas(
                        close=result_df[close_col],
                        fast=macd_fast,
                        slow=macd_slow,
                        signal=macd_signal,
                    )
                elif ta_indicator_name == "bbands":
                    bb_length = params.get("length", 20)
                    bb_std = params.get("std", 2.0)
                    indicator_result = _calculate_bbands_pandas(
                        close=result_df[close_col], length=bb_length, std=bb_std
                    )
                elif ta_indicator_name == "adx":
                    adx_length = params.get("length", 14)  # Default ADX length
                    indicator_result = _calculate_adx_pandas(
                        high=result_df[high_col],
                        low=result_df[low_col],
                        close=result_df[close_col],
                        length=adx_length,
                    )
                elif ta_indicator_name == "obv":
                    # OBV typically doesn't have parameters like length
                    indicator_result = _calculate_obv_pandas(
                        close=result_df[close_col], volume=result_df[volume_col]
                    )
                elif ta_indicator_name == "mfi":
                    mfi_length = params.get("length", 14)  # Default MFI length
                    indicator_result = _calculate_mfi_pandas(
                        high=result_df[high_col],
                        low=result_df[low_col],
                        close=result_df[close_col],
                        volume=result_df[volume_col],
                        length=mfi_length,
                    )
                # --- Indicators still not implemented ---
                elif ta_indicator_name in ["stoch", "vwap"]:  # Removed mfi, adx, obv
                    logger.warning(
                        f"Indicator '{ta_indicator_name}' (from config '{config_indicator_name}') "
                        f"is not yet implemented without pandas-ta. Skipping."
                    )
                    continue
                else:
                    # Log warning for any other unknown/unhandled indicator
                    logger.warning(
                        f"Unknown or unhandled indicator '{ta_indicator_name}' (from config '{config_indicator_name}'). Skipping."
                    )
                    continue
                # --- End Refactored Indicator Calculation ---

                # --- Add results to DataFrame ---
                if indicator_result is None:
                    logger.warning(
                        f"Indicator '{ta_indicator_name}' with params {params} returned None. Skipping."
                    )
                    continue

                # Construct base column name using the original config name for clarity
                param_str = "_".join(f"{k}{v}" for k, v in params.items())
                base_col_name = (
                    f"{config_indicator_name}_{param_str}"
                    if params
                    else config_indicator_name
                )

                if isinstance(indicator_result, pd.DataFrame):
                    # Handle multi-column results (like MACD, BBands, ADX, Stoch)
                    # Use the column names returned by the helper functions directly.
                    # The helper functions are designed to return names similar to pandas-ta.
                    # Prefixing with base_col_name might be redundant or incorrect if helpers already include parameters.
                    # Example: _calculate_bbands returns 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0'
                    # Example: _calculate_macd returns 'MACD_12_26', 'MACDs_9', 'MACDh_12_26_9'
                    # We just need to assign these columns to the result_df.
                    for col in indicator_result.columns:
                        # Construct the final column name. Use the original config name and the suffix from the result.
                        # This aims for clarity, e.g., 'bollinger_bands_BBL_20_2.0' if config was {'indicator': 'bollinger_bands', 'length': 20, 'std': 2.0}
                        # However, the original code's base_col_name already includes params: 'bollinger_bands_length20_std2.0'
                        # Let's stick to the original logic's base_col_name + cleaned suffix for consistency.
                        # Clean the suffix coming from the helper function (e.g., remove 'MACD_12_26' -> '')
                        # This cleaning logic might need refinement based on actual helper outputs.
                        # Let's try a simpler approach: just use the columns as returned by the helpers, prefixed by the *original config name* only.
                        # e.g., if config is {'indicator': 'macd', 'fast': 12, ...}, columns will be 'macd_MACD_12_26', 'macd_MACDs_9', etc.
                        # This seems less prone to complex cleaning logic.

                        # Alternative: Use the base_col_name (which includes params) and add the specific suffix part.
                        # Example: base_col_name = 'macd_fast12_slow26_signal9'
                        # Helper col = 'MACDs_9' -> final col = 'macd_fast12_slow26_signal9_MACDs_9' (a bit long)

                        # Let's revert to the original logic's intention: base_col_name + cleaned suffix
                        # base_col_name = f"{config_indicator_name}_{param_str}"
                        # Example: base_col_name = "bollinger_bands_length20_std2.0"
                        # Helper col = "BBL_20_2.0"
                        # Clean suffix: "BBL_20_2.0".replace("BBANDS_","").replace("bbands_","") -> "BBL_20_2.0" (assuming ta_indicator_name is 'bbands')
                        # Final col: "bollinger_bands_length20_std2.0_BBL_20_2.0" (Still very long)

                        # --- Let's simplify the naming ---
                        # Use the original config name + the column name from the helper function's DataFrame.
                        # This keeps the indicator type clear and uses the specific component name from the helper.
                        final_col_name = f"{config_indicator_name}_{col}"
                        result_df[final_col_name] = indicator_result[col]

                        # Example: config={'indicator': 'macd', ...} -> result_df['macd_MACD_12_26'], result_df['macd_MACDs_9'], ...
                        # Example: config={'indicator': 'bollinger_bands', ...} -> result_df['bollinger_bands_BBL_20_2.0'], ...
                        # Example: config={'indicator': 'adx', ...} -> result_df['adx_ADX_14'], result_df['adx_DMP_14'], ...
                elif isinstance(indicator_result, pd.Series):
                    # Handle single-column results (like SMA, RSI, ATR, OBV, MFI)
                    # For OBV, base_col_name will just be 'obv' as it has no params
                    result_df[base_col_name] = indicator_result
                else:
                    logger.warning(
                        f"Indicator '{ta_indicator_name}' returned unexpected type: {type(indicator_result)}. Skipping."
                    )
                    continue

                # --- Special Handling for ATRr_10 ---
                # --- Special Handling for ATRr (using the calculated ATR result) ---
                # This block now runs *after* the ATR indicator_result has been calculated and added to result_df
                if config_indicator_name.lower() == "atr":
                    atr_length = params.get("length", 14)
                    # Use the base_col_name which was just assigned to the ATR result series
                    # Example: base_col_name might be 'atr_length14'
                    if base_col_name in result_df.columns:
                        atr_col = result_df[base_col_name]
                        close_prices = result_df[close_col]

                        # Calculate general ATRr (ATR / Close)
                        atrr_col_name = f"ATRr_{atr_length}"
                        result_df[atrr_col_name] = (
                            atr_col / close_prices.replace(0, np.nan)
                        ).fillna(0)
                        logger.debug(f"Calculated {atrr_col_name} using custom ATR.")

                        # Calculate ATRr_10 specifically if length is 10
                        # Note: The triple barrier might specifically look for 'ATRr_10'
                        if atr_length == 10:
                            # Ensure the specific name 'ATRr_10' is used if length is 10
                            # This might overwrite the general 'ATRr_10' if calculated above, which is fine.
                            result_df["ATRr_10"] = (
                                atr_col / close_prices.replace(0, np.nan)
                            ).fillna(0)
                            logger.debug(
                                "Ensured ATRr_10 column exists using custom ATR."
                            )
                    else:
                        logger.warning(
                            f"Could not find calculated ATR column '{base_col_name}' to compute ATRr."
                        )

                calculated_count += 1

            except Exception as e_ind:
                logger.error(
                    f"Error calculating indicator '{config_indicator_name}' (mapped to '{ta_indicator_name}') with params {params}: {e_ind}",
                    exc_info=True,  # Log traceback for debugging
                )
                # Continue to the next indicator

        # --- Explicit ATRr_10 calculation block removed ---
        # This is now handled within the main indicator loop when 'atr' with length 10 is processed.

        logger.info(f"Successfully calculated {calculated_count} technical indicators.")
        return result_df

    except Exception as e:
        logger.error(f"General error calculating indicators: {e}")
        return df


# --- Additional Feature Engineering Functions ---


def calculate_fractals(
    high: pd.Series, low: pd.Series, window: int = 5
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Williams' Fractals for identifying potential support/resistance.

    Args:
        high: Series of high prices
        low: Series of low prices
        window: Window size for fractal calculation (default: 5)

    Returns:
        Tuple of Series (bullish_fractals, bearish_fractals)
    """
    bullish_fractals = pd.Series(0, index=low.index)
    bearish_fractals = pd.Series(0, index=high.index)

    # Need at least window*2+1 data points - adjust window logic slightly
    half_window = window // 2
    if len(high) < window:
        logger.warning(f"Not enough data ({len(high)}) for fractal window {window}")
        return bullish_fractals, bearish_fractals

    # Bullish fractals (low point with higher lows on both sides)
    for i in range(half_window, len(low) - half_window):
        is_bullish = True
        for j in range(1, half_window + 1):
            if low.iloc[i] >= low.iloc[i - j] or low.iloc[i] >= low.iloc[i + j]:
                is_bullish = False
                break
        if is_bullish:
            bullish_fractals.iloc[i] = 1

    # Bearish fractals (high point with lower highs on both sides)
    for i in range(half_window, len(high) - half_window):
        is_bearish = True
        for j in range(1, half_window + 1):
            if high.iloc[i] <= high.iloc[i - j] or high.iloc[i] <= high.iloc[i + j]:
                is_bearish = False
                break
        if is_bearish:
            bearish_fractals.iloc[i] = 1

    return bullish_fractals, bearish_fractals


def calculate_efficiency_ratio(data: pd.Series, window: int = 10) -> pd.Series:
    """
    Calculate Efficiency Ratio (ER) to measure trending vs. choppy markets.
    ER = abs(price change over period) / sum of absolute price changes

    Args:
        data: Price series (typically close prices)
        window: Lookback period

    Returns:
        Series of efficiency ratios (0-1)
    """
    direction = data.diff(window).abs()
    volatility = data.diff().abs().rolling(window).sum()

    # Avoid division by zero
    efficiency_ratio = (direction / volatility).replace([np.inf, -np.inf], np.nan)
    return efficiency_ratio


def calculate_hurst_exponent(
    data: pd.Series, min_window: int = 10, max_window: int = 100
) -> float:
    """
    Calculate Hurst Exponent to identify mean-reversion vs. trend characteristics.
    H > 0.5: trending, H < 0.5: mean-reverting, H ≈ 0.5: random walk

    Args:
        data: Price series
        min_window: Minimum window size for calculation
        max_window: Maximum window size for calculation

    Returns:
        Hurst exponent value
    """
    if not ADVANCED_STATS:
        return 0.5  # Default to random walk if package not available

    try:
        # Convert to numpy array for hurst calculation
        prices_array = data.dropna().values  # Drop NaNs before calculation
        if len(prices_array) < min_window:  # Check length after dropping NaNs
            logger.warning(
                f"Not enough data ({len(prices_array)}) for Hurst Exponent calculation (min_window={min_window})"
            )
            return 0.5
        H, _, _ = compute_Hc(prices_array, kind="price", simplified=True)
        return H
    except Exception as e:
        logger.warning(f"Error calculating Hurst exponent: {e}")
        return 0.5


def calculate_volatility_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate various volatility metrics. Requires 'close', 'high', 'low'.

    Args:
        df: DataFrame containing OHLC data

    Returns:
        Dictionary of volatility features
    """
    features = {}
    if "close" not in df.columns:
        logger.warning("Missing 'close' column for volatility features.")
        return features

    returns = df["close"].pct_change()  # Calculate returns here

    # Standard rolling volatilities
    for window in [5, 10, 20, 50]:
        features[f"volatility_{window}"] = returns.rolling(window).std()

    # Normalized volatility (current vs historical)
    vol_20 = returns.rolling(20).std()
    vol_50 = returns.rolling(50).std()
    features["volatility_ratio_20_50"] = (vol_20 / vol_50).replace(
        [np.inf, -np.inf], np.nan
    )

    # Volatility of volatility (meta-volatility)
    if "volatility_20" in features:
        features["vol_of_vol_20"] = features["volatility_20"].rolling(20).std()

    # Realized volatility calculation using Parkinson's formula (high-low range based)
    if "high" in df.columns and "low" in df.columns:
        high = df["high"]
        low = df["low"]
        log_hl_sq = (np.log(high / low) ** 2).replace([np.inf, -np.inf], np.nan)
        parkinson_vol = np.sqrt(
            (1.0 / (4.0 * np.log(2.0))) * log_hl_sq.rolling(20).mean()
        )
        # Annualize (optional, depends on use case)
        # features['parkinson_vol_20'] = parkinson_vol * np.sqrt(252)
        features["parkinson_vol_20"] = parkinson_vol
    else:
        logger.warning("Missing 'high' or 'low' for Parkinson volatility.")

    # GARCH volatility forecast if advanced stats available
    valid_returns = returns.dropna()
    if ADVANCED_STATS and len(valid_returns) > 100:
        try:
            # Fit a GARCH(1,1) model
            model = arch_model(
                valid_returns * 100, vol="Garch", p=1, q=1
            )  # Scale returns
            model_fit = model.fit(disp="off", show_warning=False)
            # Get the conditional volatility forecast
            cond_vol = model_fit.conditional_volatility / 100  # Rescale
            features["garch_vol"] = pd.Series(cond_vol, index=valid_returns.index)

        except Exception as e:
            # GARCH often fails, log as info unless debugging
            logger.info(f"Could not calculate GARCH volatility: {e}")

    return features


def calculate_market_regime_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate features that help identify market regimes. Requires 'close', optionally 'volume'.

    Args:
        df: DataFrame containing OHLCV data

    Returns:
        Dictionary of regime indicator features
    """
    features = {}
    if "close" not in df.columns:
        logger.warning("Missing 'close' column for market regime features.")
        return features

    prices = df["close"]
    returns = prices.pct_change()
    volumes = df.get("volume")

    # Check for trend vs range using efficiency ratio
    features["efficiency_ratio_10"] = calculate_efficiency_ratio(prices, 10)
    features["efficiency_ratio_20"] = calculate_efficiency_ratio(prices, 20)

    # ADX calculation is now available via calculate_technical_indicators if configured.
    # This function might still calculate other regime features, but ADX itself should be added via the main function.
    logger.debug(
        "ADX should be calculated via calculate_technical_indicators if needed for regime features."
    )
    # features["adx_14"] = pd.Series(np.nan, index=prices.index) # Remove placeholder if ADX is expected from main function

    # Simple trend detection (close > moving averages)
    sma50 = prices.rolling(50).mean()
    sma200 = prices.rolling(200).mean()
    features["close_gt_sma50"] = (prices > sma50).astype(int)
    features["close_gt_sma200"] = (prices > sma200).astype(int)

    # Simple trend direction (moving averages relative ordering)
    sma20 = prices.rolling(20).mean()
    features["sma20_gt_sma50"] = (sma20 > sma50).astype(int)

    # Volatility regime
    historical_vol = returns.rolling(100).std()
    current_vol = returns.rolling(20).std()
    features["volatility_regime"] = (current_vol / historical_vol).replace(
        [np.inf, -np.inf], np.nan
    )

    # Volume analysis if volume data provided
    if volumes is not None:
        # Volume moving averages
        vol_sma = volumes.rolling(20).mean()
        features["volume_ratio"] = (volumes / vol_sma).replace(
            [np.inf, -np.inf], np.nan
        )

        # Price-volume relationship
        features["price_volume_corr"] = returns.rolling(20).corr(volumes.pct_change())

    # Stationarity test (ADF test)
    if ADVANCED_STATS:
        try:
            # Setup a rolling ADF test window
            window = 100
            # Handle numpy array in lambda: remove nans, check length, then call adfuller
            adf_pvalues = prices.rolling(window).apply(
                lambda x: (
                    adfuller(x[~np.isnan(x)])[1]
                    if len(x[~np.isnan(x)]) >= window
                    else np.nan
                ),
                raw=True,
            )

            features["adf_pvalue"] = adf_pvalues
            # Stationary market (p < 0.05) might be mean-reverting
            features["is_stationary"] = (features["adf_pvalue"] < 0.05).astype(int)
        except Exception as e:
            logger.warning(f"Error calculating stationarity features: {e}")

    return features


def calculate_fractal_features(
    df: pd.DataFrame, window: int = 5
) -> Dict[str, pd.Series]:
    """
    Calculate Williams' Fractals features. Requires 'high', 'low'.

    Args:
        df: DataFrame containing OHLC data
        window: Window size for fractal calculation (passed to helper)

    Returns:
        Dictionary of fractal features
    """
    features = {}
    if not all(col in df.columns for col in ["high", "low"]):
        logger.warning("Missing 'high' or 'low' column for fractal features.")
        return features

    try:
        bullish_fractals, bearish_fractals = calculate_fractals(
            df["high"], df["low"], window=window
        )
        features[f"bullish_fractal_{window}"] = bullish_fractals
        features[f"bearish_fractal_{window}"] = bearish_fractals
        logger.debug(f"Calculated fractal features with window {window}.")
    except Exception as e:
        logger.error(f"Error calculating fractal features: {e}", exc_info=True)

    return features


def calculate_support_resistance_features(
    df: pd.DataFrame, window: int = 50
) -> Dict[str, pd.Series]:
    """
    Calculate features related to support and resistance levels. Requires OHLC.

    Args:
        df: DataFrame with OHLC price data
        window: Lookback window for identifying levels

    Returns:
        Dictionary of support/resistance features
    """
    features = {}

    # Check required columns
    if not all(col in df.columns for col in ["high", "low", "close"]):
        logger.warning(
            "Cannot calculate support/resistance features: missing OHLC data"
        )
        return features

    # Calculate fractal points for potential S/R (using window=5 for standard fractals)
    bullish_fractals, bearish_fractals = calculate_fractals(
        df["high"], df["low"], window=5
    )  # Standard fractal window
    features["bullish_fractal"] = bullish_fractals
    features["bearish_fractal"] = bearish_fractals

    # Identify recent high and low points using the specified window
    high_roll_max = df["high"].rolling(window).max()
    low_roll_min = df["low"].rolling(window).min()

    # Calculate distance from recent extremes
    features["dist_from_recent_high"] = (df["close"] - high_roll_max) / high_roll_max
    features["dist_from_recent_low"] = (df["close"] - low_roll_min) / low_roll_min

    # Within N% of support/resistance
    threshold = 0.01  # 1% threshold
    features["near_resistance"] = (
        features["dist_from_recent_high"].abs() < threshold
    ).astype(int)
    features["near_support"] = (
        features["dist_from_recent_low"].abs() < threshold
    ).astype(int)

    return features


def calculate_pattern_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate features related to candlestick patterns and price formations. Requires OHLC.

    Args:
        df: DataFrame with OHLC price data

    Returns:
        Dictionary of pattern-related features
    """
    features = {}

    # Check required columns
    if not all(col in df.columns for col in ["open", "high", "low", "close"]):
        logger.warning("Cannot calculate pattern features: missing OHLC data")
        return features

    # Candlestick properties
    features["body_size"] = abs(df["close"] - df["open"]) / df["open"]
    features["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df[
        "open"
    ]
    features["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df[
        "open"
    ]

    # Candlestick patterns using pandas_ta removed.
    logger.debug("Candlestick pattern calculation skipped (pandas-ta removed).")
    # features["pattern_doji"] = pd.Series(np.nan, index=df.index) # Example placeholder
    # features["pattern_engulfing"] = pd.Series(np.nan, index=df.index)
    # features["pattern_hammer"] = pd.Series(np.nan, index=df.index)

    return features


def calculate_momentum_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate momentum and oscillator indicators. Requires 'close', optionally HLV.

    Args:
        df: DataFrame with OHLC price data

    Returns:
        Dictionary of momentum-related features
    """
    features = {}

    # Check required columns
    if "close" not in df.columns:
        logger.warning("Missing 'close' column for momentum features.")
        return features
    price_col = "close"

    # Momentum indicators (RSI, MACD) are now calculated in calculate_technical_indicators if configured.
    # Stoch is still skipped. MFI is now available via calculate_technical_indicators.
    logger.debug(
        "Stochastic (Stoch) skipped (pandas-ta removed). MFI should be calculated via calculate_technical_indicators."
    )
    # Placeholders for skipped indicators if needed elsewhere, though they should ideally be added via calculate_technical_indicators
    # features["stoch_k"] = pd.Series(np.nan, index=df.index)
    # features["stoch_d"] = pd.Series(np.nan, index=df.index)
    # features["mfi_14"] = pd.Series(np.nan, index=df.index) # Remove placeholder if MFI is expected from main function

    # Rate of Change (doesn't require pandas-ta)
    for window in [5, 10, 20]:
        features[f"roc_{window}"] = df[price_col].pct_change(window)

    return features


def calculate_vwap_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate VWAP and related features. Requires HLCV.

    Args:
        df: DataFrame with OHLC and volume price data

    Returns:
        Dictionary of VWAP-related features
    """
    features = {}

    # Check required columns
    if not all(col in df.columns for col in ["high", "low", "close", "volume"]):
        logger.warning("Cannot calculate VWAP features: missing required data (HLCV)")
        return features

    # Calculate typical price
    typical_price = (df["high"] + df["low"] + df["close"]) / 3

    # Calculate VWAP using pandas-ta if available (handles daily reset better potentially)
    # VWAP calculation using pandas-ta removed. Rolling VWAP calculation remains.
    # A dedicated VWAP helper could be added to calculate_technical_indicators if needed. VWAP is still skipped there.
    logger.debug(
        "VWAP calculation skipped in calculate_technical_indicators. Using rolling VWAP in calculate_vwap_features."
    )

    # Fallback or additional rolling VWAP calculation
    for window in [20, 50]:  # Shorter windows for rolling VWAP
        vol_sum = df["volume"].rolling(window).sum()
        vwap_roll = (
            (typical_price * df["volume"]).rolling(window).sum() / vol_sum
        ).replace([np.inf, -np.inf], np.nan)

        features[f"vwap_roll_{window}"] = vwap_roll
        # Distance from rolling VWAP
        features[f"dist_from_vwap_roll_{window}"] = (
            df["close"] - vwap_roll
        ) / vwap_roll

    return features
