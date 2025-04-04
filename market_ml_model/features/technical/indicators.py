"""
Technical indicators for market data analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple

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


def calculate_technical_indicators(
    df: pd.DataFrame, indicators: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate technical indicators for the given dataframe using pandas-ta.

    Args:
        df: DataFrame with OHLCV data
        indicators: List of indicators to calculate (None = all basic indicators)

    Returns:
        DataFrame with added technical indicators
    """
    if not PANDAS_TA_AVAILABLE:
        logger.error("pandas-ta not available for calculating indicators")
        return df

    # Default indicators if none specified
    if indicators is None:
        indicators = [
            "sma",
            "ema",
            "rsi",
            "macd",
            "bbands",
            "atr",
            "adx",
            "stoch",
            "volume_indicators",
        ]

    # Make a copy of the dataframe
    result_df = df.copy()

    # Check for required columns
    required_cols = ["open", "high", "low", "close"]
    missing_cols = [col for col in required_cols if col not in result_df.columns]

    if missing_cols:
        logger.error(f"Missing required columns for basic indicators: {missing_cols}")
        # Allow proceeding if some indicators don't need all cols (e.g., volume)
        # return df # Original behavior: stop if any OHLC missing

    try:
        # Calculate indicators
        for indicator in indicators:
            try:  # Add inner try-except for individual indicators
                if indicator == "sma":
                    # Simple Moving Averages
                    for period in [20, 50, 200]:
                        result_df[f"sma_{period}"] = ta.sma(
                            result_df["close"], length=period
                        )

                elif indicator == "ema":
                    # Exponential Moving Averages
                    for period in [9, 20, 50]:
                        result_df[f"ema_{period}"] = ta.ema(
                            result_df["close"], length=period
                        )

                elif indicator == "rsi":
                    # Relative Strength Index
                    result_df["rsi_14"] = ta.rsi(result_df["close"], length=14)

                elif indicator == "macd":
                    # MACD
                    macd = ta.macd(result_df["close"])
                    if isinstance(macd, pd.DataFrame) and len(macd.columns) >= 3:
                        result_df["macd"] = macd.iloc[:, 0]
                        result_df["macd_signal"] = macd.iloc[:, 1]
                        result_df["macd_hist"] = macd.iloc[:, 2]

                elif indicator == "bbands":
                    # Bollinger Bands
                    bbands = ta.bbands(result_df["close"], length=20)
                    if isinstance(bbands, pd.DataFrame) and len(bbands.columns) >= 3:
                        result_df["bb_upper"] = bbands.iloc[:, 0]
                        result_df["bb_middle"] = bbands.iloc[:, 1]
                        result_df["bb_lower"] = bbands.iloc[:, 2]
                        result_df["bb_width"] = (
                            bbands.iloc[:, 0] - bbands.iloc[:, 2]
                        ) / bbands.iloc[:, 1]

                elif indicator == "atr":
                    # Average True Range
                    if all(c in result_df.columns for c in ["high", "low", "close"]):
                        result_df["atr_14"] = ta.atr(
                            result_df["high"],
                            result_df["low"],
                            result_df["close"],
                            length=14,
                        )
                        # ATR as a percentage of price (ATRr)
                        result_df["ATRr_10"] = (
                            ta.atr(
                                result_df["high"],
                                result_df["low"],
                                result_df["close"],
                                length=10,
                            )
                            / result_df["close"]
                        )
                    else:
                        logger.warning("Missing HLC for ATR calculation.")

                elif indicator == "adx":
                    # Average Directional Index
                    if all(c in result_df.columns for c in ["high", "low", "close"]):
                        adx = ta.adx(
                            result_df["high"], result_df["low"], result_df["close"]
                        )
                        if isinstance(adx, pd.DataFrame) and len(adx.columns) >= 3:
                            result_df["adx"] = adx.iloc[:, 0]
                            result_df["dmp"] = adx.iloc[
                                :, 1
                            ]  # Plus Directional Movement
                            result_df["dmn"] = adx.iloc[
                                :, 2
                            ]  # Minus Directional Movement
                    else:
                        logger.warning("Missing HLC for ADX calculation.")

                elif indicator == "stoch":
                    # Stochastic Oscillator
                    if all(c in result_df.columns for c in ["high", "low", "close"]):
                        stoch = ta.stoch(
                            result_df["high"], result_df["low"], result_df["close"]
                        )
                        if isinstance(stoch, pd.DataFrame) and len(stoch.columns) >= 2:
                            result_df["stoch_k"] = stoch.iloc[:, 0]
                            result_df["stoch_d"] = stoch.iloc[:, 1]
                    else:
                        logger.warning("Missing HLC for Stochastic calculation.")

                elif indicator == "volume_indicators" and "volume" in result_df.columns:
                    # Volume-based indicators
                    result_df["volume_sma20"] = ta.sma(result_df["volume"], length=20)
                    result_df["volume_ratio"] = (
                        result_df["volume"] / result_df["volume_sma20"]
                    )
                    result_df["obv"] = ta.obv(result_df["close"], result_df["volume"])

                    # Money Flow Index
                    if all(c in result_df.columns for c in ["high", "low", "close"]):
                        result_df["mfi_14"] = ta.mfi(
                            result_df["high"],
                            result_df["low"],
                            result_df["close"],
                            result_df["volume"],
                            length=14,
                        )
                    else:
                        logger.warning("Missing HLC for MFI calculation.")

                else:
                    if (
                        indicator != "volume_indicators"
                    ):  # Avoid warning if volume just missing
                        logger.warning(f"Unsupported or skipped indicator: {indicator}")

            except Exception as e_ind:
                logger.error(f"Error calculating indicator '{indicator}': {e_ind}")

        logger.info(f"Calculated {len(indicators)} indicator groups using pandas-ta")
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
    if f"volatility_20" in features:
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
                vwap_safe = vwap_ta.replace(0, np.nan)
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
