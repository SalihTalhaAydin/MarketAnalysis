"""
Technical indicators for market data analysis.
"""

import logging
from typing import Any, Dict, List, Tuple  # Add Any here

import numpy as np
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

# Import technical analysis library (pandas-ta)
# Assume it's installed via requirements.txt
try:
    import pandas_ta as ta

    PANDAS_TA_AVAILABLE = True
    logger.info("pandas-ta library loaded successfully.")
except ImportError:
    logger.warning(
        "pandas-ta not found. Please install it (`pip install pandas-ta`) to use technical indicators."
    )
    ta = None  # Define ta as None if import fails
    PANDAS_TA_AVAILABLE = False
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


# Remove this line
# ... (other imports remain the same)


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
    if not PANDAS_TA_AVAILABLE:
        logger.error("pandas-ta not available for calculating indicators")
        return df

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
    open_col = col_mapping["open"]
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
                # Check if the function exists in pandas-ta
                if not hasattr(ta, ta_indicator_name):
                    logger.warning(
                        f"Indicator function '{ta_indicator_name}' (mapped from '{config_indicator_name}') not found in pandas-ta. Skipping."
                    )
                    continue

                indicator_func = getattr(ta, ta_indicator_name)

                # Prepare arguments for the pandas-ta function
                # Most functions accept high, low, close, volume, open as keyword args
                kwargs = {
                    "high": result_df[high_col],
                    "low": result_df[low_col],
                    "close": result_df[close_col],
                    "volume": result_df[volume_col],
                    "open": result_df[open_col],
                    **params,  # Add specific indicator parameters
                }

                # Call the indicator function using keyword arguments
                # pandas-ta functions are generally good at ignoring unused kwargs
                indicator_result = indicator_func(**kwargs)

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
                    # Use pandas-ta's default column names, prefixed with our base name
                    for col in indicator_result.columns:
                        # Clean up potential overlaps in naming (e.g., bbands result BBL_5_2.0)
                        clean_col_suffix = col.replace(
                            f"{ta_indicator_name.upper()}_", ""
                        ).replace(f"{ta_indicator_name.lower()}_", "")
                        result_df[f"{base_col_name}_{clean_col_suffix}"] = (
                            indicator_result[col]
                        )
                elif isinstance(indicator_result, pd.Series):
                    # Handle single-column results (like SMA, RSI, ATR)
                    result_df[base_col_name] = indicator_result
                else:
                    logger.warning(
                        f"Indicator '{ta_indicator_name}' returned unexpected type: {type(indicator_result)}. Skipping."
                    )
                    continue

                # --- Special Handling for ATRr_10 ---
                if config_indicator_name.lower() == "atr":
                    atr_length = params.get(
                        "length", 14
                    )  # Default ATR length in ta is 14
                    atr_col_name = (
                        f"atr_{'length'}{atr_length}"  # Match the generated name
                    )
                    if atr_col_name in result_df.columns:
                        # Calculate ATRr_10 specifically if length is 10
                        if atr_length == 10:
                            atr_col = result_df[atr_col_name]
                            close_prices = result_df[close_col]
                            # Avoid division by zero
                            result_df["ATRr_10"] = (
                                atr_col / close_prices.replace(0, np.nan)
                            ).fillna(0)
                            logger.debug("Calculated ATRr_10 column.")
                        # Also calculate general ATRr if requested (any length)
                        atr_col = result_df[atr_col_name]
                        close_prices = result_df[close_col]
                        result_df[f"ATRr_{atr_length}"] = (
                            atr_col / close_prices.replace(0, np.nan)
                        ).fillna(0)

                    else:
                        logger.warning(
                            f"Could not find calculated ATR column '{atr_col_name}' to compute ATRr."
                        )

                calculated_count += 1

            except Exception as e_ind:
                logger.error(
                    f"Error calculating indicator '{config_indicator_name}' (mapped to '{ta_indicator_name}') with params {params}: {e_ind}",
                    exc_info=True,  # Log traceback for debugging
                )
                # Continue to the next indicator

        # --- Explicitly calculate ATRr_10 for triple barrier ---
        # Ensure required columns are available before attempting calculation
        if all(col in result_df.columns for col in [high_col, low_col, close_col]):
            try:
                atr_10 = ta.atr(
                    result_df[high_col],
                    result_df[low_col],
                    result_df[close_col],
                    length=10,
                )
                close_prices = result_df[close_col]
                # Avoid division by zero and handle potential NaNs from ATR calculation
                result_df["ATRr_10"] = (
                    atr_10 / close_prices.replace(0, np.nan)
                ).fillna(0)
                logger.info("Successfully calculated explicit ATRr_10 column.")
            except Exception as e_atr10:
                logger.error(
                    f"Error explicitly calculating ATRr_10: {e_atr10}", exc_info=True
                )
        else:
            logger.warning(
                "Could not calculate explicit ATRr_10 due to missing HLC columns."
            )

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

    # ADX for trend strength if pandas_ta available and HLC present
    if PANDAS_TA_AVAILABLE and all(c in df.columns for c in ["high", "low", "close"]):
        try:
            adx = ta.adx(df["high"], df["low"], df["close"])
            if isinstance(adx, pd.DataFrame) and len(adx.columns) >= 1:
                features["adx_14"] = adx.iloc[:, 0]
        except Exception as e:
            logger.warning(f"Could not calculate ADX: {e}")
    else:
        features["adx_14"] = pd.Series(np.nan, index=prices.index)

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

    # Candlestick patterns using pandas_ta
    if PANDAS_TA_AVAILABLE:
        try:
            # Use the ta.cdl_pattern function which aggregates multiple patterns
            # Or calculate specific ones if needed
            # Example: Calculate Doji
            doji = ta.cdl_doji(df["open"], df["high"], df["low"], df["close"])
            if doji is not None:
                features["pattern_doji"] = doji

            # Example: Calculate Engulfing
            engulfing = ta.cdl_engulfing(df["open"], df["high"], df["low"], df["close"])
            if engulfing is not None:
                features["pattern_engulfing"] = engulfing

            # Example: Calculate Hammer
            hammer = ta.cdl_hammer(df["open"], df["high"], df["low"], df["close"])
            if hammer is not None:
                features["pattern_hammer"] = hammer

        except Exception as e:
            logger.warning(f"Error calculating pandas-ta patterns: {e}")

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

    # RSI for different timeframes
    if PANDAS_TA_AVAILABLE:
        for window in [7, 14, 21]:
            try:
                features[f"rsi_{window}"] = ta.rsi(df[price_col], length=window)
            except Exception as e:
                logger.warning(f"Error calculating RSI-{window}: {e}")

        # Stochastic oscillator
        if all(col in df.columns for col in ["high", "low", price_col]):
            try:
                stoch = ta.stoch(df["high"], df["low"], df[price_col], k=14, d=3)
                if isinstance(stoch, pd.DataFrame) and len(stoch.columns) >= 2:
                    features["stoch_k"] = stoch.iloc[:, 0]
                    features["stoch_d"] = stoch.iloc[:, 1]
            except Exception as e:
                logger.warning(f"Error calculating Stochastic: {e}")

        # MACD
        try:
            macd = ta.macd(df[price_col])
            if isinstance(macd, pd.DataFrame) and len(macd.columns) >= 3:
                features["macd"] = macd.iloc[:, 0]
                features["macd_signal"] = macd.iloc[:, 1]
                features["macd_histogram"] = macd.iloc[:, 2]
        except Exception as e:
            logger.warning(f"Error calculating MACD: {e}")

        # Money Flow Index if volume available
        if "volume" in df.columns and all(
            col in df.columns for col in ["high", "low", price_col]
        ):
            try:
                features["mfi_14"] = ta.mfi(
                    df["high"], df["low"], df[price_col], df["volume"], length=14
                )
            except Exception as e:
                logger.warning(f"Error calculating MFI: {e}")

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
    if PANDAS_TA_AVAILABLE:
        try:
            # Assuming daily data, anchor='D' might work. Needs testing.
            # Or calculate manually if ta.vwap doesn't fit use case.
            vwap_ta = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
            if vwap_ta is not None:
                features["vwap_ta"] = vwap_ta
                # Avoid division by zero or invalid values
                vwap_safe = vwap_ta.replace(0, np.nan).reindex_like(
                    df["close"]
                )  # Align index
                features["dist_from_vwap_ta"] = (df["close"] - vwap_safe) / vwap_safe

        except Exception as e:
            logger.warning(
                f"Error calculating VWAP with pandas-ta: {e}. Falling back to rolling."
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
