import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Try importing technical analysis libraries with fallbacks
try:
    import pandas_ta as ta
except ImportError:
    logger.warning("pandas-ta not installed. Installing required package...")
    import subprocess
    import sys
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "pandas-ta"])
    import pandas_ta as ta

try:
    from statsmodels.tsa.stattools import adfuller
    from hurst import compute_Hc
    from arch import arch_model
    ADVANCED_STATS = True
except ImportError:
    logger.warning(
        "Advanced statistical packages not installed. Some features will be disabled.")
    ADVANCED_STATS = False


# --- Advanced Feature Engineering Functions ---

def calculate_fractals(high: pd.Series, low: pd.Series, window: int = 5) -> Tuple[pd.Series, pd.Series]:
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

    # Need at least window*2+1 data points
    if len(high) < window*2+1:
        return bullish_fractals, bearish_fractals

    # Bullish fractals (low point with higher lows on both sides)
    for i in range(window, len(low) - window):
        if all(low.iloc[i] < low.iloc[i-j] for j in range(1, window+1)) and \
           all(low.iloc[i] < low.iloc[i+j] for j in range(1, window+1)):
            bullish_fractals.iloc[i] = 1

    # Bearish fractals (high point with lower highs on both sides)
    for i in range(window, len(high) - window):
        if all(high.iloc[i] > high.iloc[i-j] for j in range(1, window+1)) and \
           all(high.iloc[i] > high.iloc[i+j] for j in range(1, window+1)):
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

    efficiency_ratio = direction / volatility
    return efficiency_ratio


def calculate_hurst_exponent(data: pd.Series, min_window: int = 10, max_window: int = 100) -> float:
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
        prices_array = data.values
        H, _, _ = compute_Hc(prices_array, kind='price', simplified=True)
        return H
    except Exception as e:
        logger.warning(f"Error calculating Hurst exponent: {e}")
        return 0.5


def calculate_volatility_features(returns: pd.Series) -> Dict[str, pd.Series]:
    """
    Calculate various volatility metrics.

    Args:
        returns: Series of return values

    Returns:
        Dictionary of volatility features
    """
    features = {}

    # Standard rolling volatilities
    for window in [5, 10, 20, 50]:
        features[f'volatility_{window}'] = returns.rolling(window).std()

    # Normalized volatility (current vs historical)
    features['volatility_ratio_20_50'] = (
        returns.rolling(20).std() / returns.rolling(50).std()
    )

    # Volatility of volatility (meta-volatility)
    features['vol_of_vol_20'] = features['volatility_20'].rolling(20).std()

    # Realized volatility calculation using Parkinson's formula (high-low range based)
    if 'high' in returns.index.names and 'low' in returns.index.names:
        high = returns.xs('high', level=1)
        low = returns.xs('low', level=1)
        features['parkinson_vol_20'] = (
            (1.0 / (4.0 * np.log(2.0))) *
            (np.log(high / low) ** 2).rolling(20).mean() * np.sqrt(252)
        )

    # GARCH volatility forecast if advanced stats available
    if ADVANCED_STATS and len(returns.dropna()) > 100:
        try:
            # Fit a GARCH(1,1) model
            model = arch_model(returns.dropna(), vol='Garch', p=1, q=1)
            model_fit = model.fit(disp='off')
            # Get the conditional volatility forecast
            features['garch_vol'] = pd.Series(
                model_fit.conditional_volatility,
                index=returns.dropna().index
            )
        except Exception as e:
            logger.warning(f"Error calculating GARCH volatility: {e}")

    return features


def calculate_market_regime_features(
    prices: pd.Series,
    returns: pd.Series,
    volumes: Optional[pd.Series] = None
) -> Dict[str, pd.Series]:
    """
    Calculate features that help identify market regimes.

    Args:
        prices: Series of price data (typically close prices)
        returns: Series of return values
        volumes: Optional series of volume data

    Returns:
        Dictionary of regime indicator features
    """
    features = {}

    # Check for trend vs range using efficiency ratio
    features['efficiency_ratio_10'] = calculate_efficiency_ratio(prices, 10)
    features['efficiency_ratio_20'] = calculate_efficiency_ratio(prices, 20)

    # ADX for trend strength if pandas_ta available
    if hasattr(ta, 'adx'):
        # ADX requires high, low, close data which we might not have here
        # This is a placeholder for when we actually have this data
        adx = pd.Series(np.nan, index=prices.index)
        features['adx_14'] = adx

    # Simple trend detection (close > moving averages)
    features['close_gt_sma50'] = (
        prices > prices.rolling(50).mean()).astype(int)
    features['close_gt_sma200'] = (
        prices > prices.rolling(200).mean()).astype(int)

    # Simple trend direction (moving averages relative ordering)
    features['sma20_gt_sma50'] = (
        prices.rolling(20).mean() > prices.rolling(50).mean()
    ).astype(int)

    # Volatility regime
    historical_vol = returns.rolling(100).std()
    current_vol = returns.rolling(20).std()
    features['volatility_regime'] = current_vol / historical_vol

    # Volume analysis if volume data provided
    if volumes is not None:
        # Volume moving averages
        vol_sma = volumes.rolling(20).mean()
        features['volume_ratio'] = volumes / vol_sma

        # Price-volume relationship
        features['price_volume_corr'] = (
            returns.rolling(20).corr(volumes.pct_change())
        )

    # Stationarity test (ADF test)
    if ADVANCED_STATS:
        try:
            # Setup a rolling ADF test window
            window = 100
            pvalues = []
            for i in range(window, len(prices)):
                result = adfuller(prices.iloc[i-window:i])
                pvalues.append(result[1])  # p-value

            features['adf_pvalue'] = pd.Series(
                pvalues, index=prices.index[window:])
            # Stationary market (p < 0.05) might be mean-reverting
            features['is_stationary'] = (
                features['adf_pvalue'] < 0.05).astype(int)
        except Exception as e:
            logger.warning(f"Error calculating stationarity features: {e}")

    return features


def calculate_support_resistance_features(
    df: pd.DataFrame,
    window: int = 50
) -> Dict[str, pd.Series]:
    """
    Calculate features related to support and resistance levels.

    Args:
        df: DataFrame with OHLC price data
        window: Lookback window for identifying levels

    Returns:
        Dictionary of support/resistance features
    """
    features = {}

    # Check required columns
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        logger.warning(
            "Cannot calculate support/resistance features: missing OHLC data")
        return features

    # Calculate fractal points for potential S/R
    bullish_fractals, bearish_fractals = calculate_fractals(
        df['high'], df['low'])
    features['bullish_fractal'] = bullish_fractals
    features['bearish_fractal'] = bearish_fractals

    # Identify recent high and low points
    high_roll_max = df['high'].rolling(window).max()
    low_roll_min = df['low'].rolling(window).min()

    # Calculate distance from recent extremes
    features['dist_from_recent_high'] = (
        df['close'] - high_roll_max) / high_roll_max
    features['dist_from_recent_low'] = (
        df['close'] - low_roll_min) / low_roll_min

    # Within N% of support/resistance
    threshold = 0.01  # 1% threshold
    features['near_resistance'] = (
        features['dist_from_recent_high'].abs() < threshold
    ).astype(int)
    features['near_support'] = (
        features['dist_from_recent_low'].abs() < threshold
    ).astype(int)

    return features


def calculate_pattern_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate features related to candlestick patterns and price formations.

    Args:
        df: DataFrame with OHLC price data

    Returns:
        Dictionary of pattern-related features
    """
    features = {}

    # Check required columns
    if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        logger.warning("Cannot calculate pattern features: missing OHLC data")
        return features

    # Candlestick properties
    features['body_size'] = abs(df['close'] - df['open']) / df['open']
    features['upper_shadow'] = (
        df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
    features['lower_shadow'] = (
        df[['open', 'close']].min(axis=1) - df['low']) / df['open']

    # Candlestick patterns using pandas_ta
    if hasattr(ta, 'cdl_pattern'):
        for pattern in ['doji', 'marubozu', 'hammer', 'engulfing']:
            try:
                pattern_fn = getattr(ta, f'cdl_{pattern}', None)
                if pattern_fn:
                    features[f'pattern_{pattern}'] = pattern_fn(
                        df['open'], df['high'], df['low'], df['close']
                    )
            except Exception as e:
                logger.warning(f"Error calculating pattern {pattern}: {e}")

    return features


def calculate_momentum_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate momentum and oscillator indicators.

    Args:
        df: DataFrame with OHLC price data

    Returns:
        Dictionary of momentum-related features
    """
    features = {}

    # Check required columns
    price_col = 'close' if 'close' in df.columns else df.columns[0]

    # RSI for different timeframes
    for window in [7, 14, 21]:
        try:
            features[f'rsi_{window}'] = ta.rsi(df[price_col], length=window)
        except Exception as e:
            logger.warning(f"Error calculating RSI-{window}: {e}")

    # Stochastic oscillator
    if all(col in df.columns for col in ['high', 'low', price_col]):
        try:
            stoch = ta.stoch(df['high'], df['low'], df[price_col], k=14, d=3)
            if isinstance(stoch, pd.DataFrame):
                features['stoch_k'] = stoch.iloc[:, 0]
                features['stoch_d'] = stoch.iloc[:, 1]
        except Exception as e:
            logger.warning(f"Error calculating Stochastic: {e}")

    # MACD
    try:
        macd = ta.macd(df[price_col])
        if isinstance(macd, pd.DataFrame):
            features['macd'] = macd.iloc[:, 0]
            features['macd_signal'] = macd.iloc[:, 1]
            features['macd_histogram'] = macd.iloc[:, 2]
    except Exception as e:
        logger.warning(f"Error calculating MACD: {e}")

    # Rate of Change
    for window in [5, 10, 20]:
        features[f'roc_{window}'] = df[price_col].pct_change(window)

    # Money Flow Index if volume available
    if 'volume' in df.columns and all(col in df.columns for col in ['high', 'low', price_col]):
        try:
            features['mfi_14'] = ta.mfi(
                df['high'], df['low'], df[price_col], df['volume'], length=14
            )
        except Exception as e:
            logger.warning(f"Error calculating MFI: {e}")

    return features


def calculate_vwap_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate VWAP and related features.

    Args:
        df: DataFrame with OHLC and volume price data

    Returns:
        Dictionary of VWAP-related features
    """
    features = {}

    # Check required columns
    if not all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
        logger.warning("Cannot calculate VWAP features: missing required data")
        return features

    # Calculate typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3

    # Calculate VWAP
    # Note: In real implementation, VWAP should reset daily
    # This is a simplified version
    for window in [20, 50, 200]:
        vol_sum = df['volume'].rolling(window).sum()
        vwap = (typical_price * df['volume']).rolling(window).sum() / vol_sum
        features[f'vwap_{window}'] = vwap

        # Distance from VWAP
        features[f'dist_from_vwap_{window}'] = (df['close'] - vwap) / vwap

    return features


# --- Triple Barrier Implementation ---

def get_triple_barrier_labels(
    prices: pd.Series,
    highs: pd.Series,
    lows: pd.Series,
    atr: pd.Series,
    atr_multiplier_tp: float,
    atr_multiplier_sl: float,
    max_holding_period: int,
    min_return_threshold: float = 0.0
) -> pd.Series:
    """
    Enhanced Triple Barrier Method for labeling.

    Labels:
        1: Take Profit hit first
       -1: Stop Loss hit first
        0: Time barrier hit (neither TP nor SL within max_holding_period)

    Args:
        prices: Series of closing prices (used for entry reference)
        highs: Series of high prices (used to check TP)
        lows: Series of low prices (used to check SL)
        atr: Series of ATR values at entry time
        atr_multiplier_tp: Multiplier for Take Profit distance from entry
        atr_multiplier_sl: Multiplier for Stop Loss distance from entry
        max_holding_period: Maximum number of bars to hold the hypothetical trade
        min_return_threshold: Minimum return threshold to consider for time barrier (0=neutral)

    Returns:
        pd.Series containing the outcome labels (1, 0, -1) or NaN if calculation
        is not possible (e.g., near the end of the series)
    """
    n = len(prices)

    # Input validation
    if not all(len(s) == n for s in [highs, lows, atr]):
        raise ValueError("All input series must have the same length")

    # Initialize labels with NaN
    labels = pd.Series(np.nan, index=prices.index)

    # Calculate target levels for all points upfront
    tp_levels = prices + atr * atr_multiplier_tp
    sl_levels = prices - atr * atr_multiplier_sl

    # Iterate through each potential entry point
    for i in range(n - 1):  # Stop before the last row
        entry_idx = prices.index[i]
        entry_price = prices.iloc[i]
        tp = tp_levels.iloc[i]
        sl = sl_levels.iloc[i]

        # Determine the lookahead window (up to max_holding_period or end of data)
        lookahead_end_idx = min(i + 1 + max_holding_period, n)

        # Check if we have enough lookahead data
        if i + 1 >= n or lookahead_end_idx <= i + 1:
            continue

        window_highs = highs.iloc[i+1:lookahead_end_idx]
        window_lows = lows.iloc[i+1:lookahead_end_idx]
        window_closes = prices.iloc[i+1:lookahead_end_idx]

        # Find the first time TP or SL is hit within the window
        tp_hit_indices = window_highs[window_highs >= tp].index
        sl_hit_indices = window_lows[window_lows <= sl].index

        tp_hit_time = tp_hit_indices[0] if not tp_hit_indices.empty else None
        sl_hit_time = sl_hit_indices[0] if not sl_hit_indices.empty else None

        # Default outcome (time barrier)
        outcome = 0

        # Check for TP or SL hits
        if tp_hit_time is not None and sl_hit_time is not None:
            # Both hit, use the earliest
            outcome = 1 if prices.index.get_loc(
                tp_hit_time) <= prices.index.get_loc(sl_hit_time) else -1
        elif tp_hit_time is not None:
            outcome = 1  # Only TP hit
        elif sl_hit_time is not None:
            outcome = -1  # Only SL hit
        else:
            # Time barrier - check final return if threshold specified
            final_price = window_closes.iloc[-1]
            final_return = (final_price - entry_price) / entry_price

            if min_return_threshold > 0:
                if final_return >= min_return_threshold:
                    outcome = 1  # Positive return at time barrier
                elif final_return <= -min_return_threshold:
                    outcome = -1  # Negative return at time barrier
                else:
                    outcome = 0  # Neutral return at time barrier

        # Assign the calculated outcome to the entry point's index
        labels.loc[entry_idx] = outcome

    return labels


# --- Main Feature Engineering Function ---

def engineer_features(
    df: pd.DataFrame,
    atr_multiplier_tp: float = 2.0,
    atr_multiplier_sl: float = 1.0,
    max_holding_period: int = 10,
    target_type: str = 'triple_barrier',
    additional_features: Optional[List[str]] = None,
    feature_config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Enhanced feature engineering with multiple feature types and configurable generation.

    Args:
        df: Preprocessed DataFrame with OHLCV data
        atr_multiplier_tp: Multiplier for Take Profit distance based on ATR
        atr_multiplier_sl: Multiplier for Stop Loss distance based on ATR
        max_holding_period: Maximum bars for the time barrier
        target_type: Type of target to generate ('triple_barrier', 'directional', etc.)
        additional_features: List of additional feature categories to generate
        feature_config: Dictionary with feature configuration parameters

    Returns:
        DataFrame with added features and target labels
    """
    logger.info("Starting enhanced feature engineering")

    if df is None or df.empty:
        logger.error("Input DataFrame is empty/None for feature engineering")
        return None

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning(
            "Index must be DatetimeIndex for time features, attempting conversion")
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logger.error(f"Failed to convert index to DatetimeIndex: {e}")
            return None

    # Work on a copy
    processed_df = df.copy()

    # Default feature configuration
    if feature_config is None:
        feature_config = {
            'use_ta_lib': True,
            'use_advanced_features': ADVANCED_STATS,
            'volatility_windows': [5, 10, 20, 50],
            'momentum_windows': [7, 14, 21],
            'trend_windows': [20, 50, 200],
        }

    # Default additional features
    if additional_features is None:
        additional_features = [
            'basic_ta', 'volatility', 'trend', 'momentum', 'support_resistance'
        ]

    # --- Feature Engineering Steps ---

    # 1. Calculate basic returns
    if 'close' in processed_df.columns:
        processed_df['return'] = processed_df['close'].pct_change()
        processed_df['log_return'] = np.log(
            processed_df['close'] / processed_df['close'].shift(1))
    else:
        logger.warning("'close' column not found, skipping return calculation")

    # 2. Calculate lagged features
    if 'close' in processed_df.columns:
        processed_df['close_lag1'] = processed_df['close'].shift(1)

    # 3. Time-based features
    processed_df['day_of_week'] = processed_df.index.dayofweek
    processed_df['hour_of_day'] = processed_df.index.hour if hasattr(
        processed_df.index, 'hour') else 0

    # 4. Basic Technical Indicators
    if 'basic_ta' in additional_features and feature_config.get('use_ta_lib', True):
        logger.info("Calculating basic technical indicators")
        try:
            # Only run if we have proper OHLC data
            if all(col in processed_df.columns for col in ['open', 'high', 'low', 'close']):
                # SMAs for various periods
                for window in feature_config.get('trend_windows', [20, 50, 200]):
                    processed_df[f'sma_{window}'] = ta.sma(
                        processed_df['close'], length=window)

                # EMAs for various periods
                for window in [5, 10, 20]:
                    processed_df[f'ema_{window}'] = ta.ema(
                        processed_df['close'], length=window)

                # Bollinger Bands
                bbands = ta.bbands(processed_df['close'], length=20, std=2)
                if isinstance(bbands, pd.DataFrame):
                    processed_df['bb_lower'] = bbands.iloc[:, 0]
                    processed_df['bb_middle'] = bbands.iloc[:, 1]
                    processed_df['bb_upper'] = bbands.iloc[:, 2]
                    processed_df['bb_width'] = (
                        (processed_df['bb_upper'] - processed_df['bb_lower']) /
                        processed_df['bb_middle']
                    )

                # ATR for volatility (ensure it's calculated for triple barrier)
                if 'ATRr_10' not in processed_df.columns:
                    atr = ta.atr(
                        processed_df['high'],
                        processed_df['low'],
                        processed_df['close'],
                        length=10
                    )
                    processed_df['ATRr_10'] = atr

                # RSI
                processed_df['rsi_14'] = ta.rsi(
                    processed_df['close'], length=14)

                # MACD
                macd = ta.macd(processed_df['close'])
                if isinstance(macd, pd.DataFrame) and macd.shape[1] >= 3:
                    processed_df['macd'] = macd.iloc[:, 0]
                    processed_df['macd_signal'] = macd.iloc[:, 1]
                    processed_df['macd_histogram'] = macd.iloc[:, 2]

                # Stochastic Oscillator
                stoch = ta.stoch(
                    processed_df['high'],
                    processed_df['low'],
                    processed_df['close']
                )
                if isinstance(stoch, pd.DataFrame) and stoch.shape[1] >= 2:
                    processed_df['stoch_k'] = stoch.iloc[:, 0]
                    processed_df['stoch_d'] = stoch.iloc[:, 1]

                logger.info(
                    "Basic technical indicators calculated successfully")
            else:
                logger.warning(
                    "Missing required OHLC columns for technical indicators")

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")

    # 5. Volume-based indicators
    if 'volume_indicators' in additional_features and 'volume' in processed_df.columns:
        logger.info("Calculating volume-based indicators")
        try:
            # Volume SMA
            processed_df['volume_sma20'] = ta.sma(
                processed_df['volume'], length=20)
            processed_df['volume_ratio'] = processed_df['volume'] / \
                processed_df['volume_sma20']

            # On-Balance Volume
            processed_df['obv'] = ta.obv(
                processed_df['close'], processed_df['volume'])

            # Money Flow Index
            if all(col in processed_df.columns for col in ['high', 'low', 'close']):
                processed_df['mfi_14'] = ta.mfi(
                    processed_df['high'],
                    processed_df['low'],
                    processed_df['close'],
                    processed_df['volume'],
                    length=14
                )

            # Volume-weighted price
            if all(col in processed_df.columns for col in ['high', 'low', 'close']):
                typical_price = (
                    processed_df['high'] + processed_df['low'] +
                    processed_df['close']
                ) / 3
                processed_df['vwap_1d'] = (
                    (typical_price * processed_df['volume']).rolling(window=390).sum() /
                    processed_df['volume'].rolling(window=390).sum()
                )

            logger.info("Volume indicators calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")

    # 6. Advanced volatility features
    if 'volatility' in additional_features and 'return' in processed_df.columns:
        logger.info("Calculating advanced volatility features")
        try:
            volatility_features = calculate_volatility_features(
                processed_df['return'])
            processed_df = pd.concat(
                [processed_df, pd.DataFrame(volatility_features)], axis=1)
            logger.info("Volatility features calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating volatility features: {e}")

    # 7. Market regime features
    if 'market_regime' in additional_features and 'close' in processed_df.columns:
        logger.info("Calculating market regime features")
        try:
            volumes = processed_df['volume'] if 'volume' in processed_df.columns else None
            regime_features = calculate_market_regime_features(
                processed_df['close'],
                processed_df['return'] if 'return' in processed_df.columns else processed_df['close'].pct_change(
                ),
                volumes
            )
            processed_df = pd.concat(
                [processed_df, pd.DataFrame(regime_features)], axis=1)
            logger.info("Market regime features calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating market regime features: {e}")

    # 8. Support/resistance features
    if 'support_resistance' in additional_features:
        logger.info("Calculating support/resistance features")
        try:
            if all(col in processed_df.columns for col in ['high', 'low', 'close']):
                sr_features = calculate_support_resistance_features(
                    processed_df)
                processed_df = pd.concat(
                    [processed_df, pd.DataFrame(sr_features)], axis=1)
                logger.info(
                    "Support/resistance features calculated successfully")
            else:
                logger.warning(
                    "Missing required columns for support/resistance features")
        except Exception as e:
            logger.error(f"Error calculating support/resistance features: {e}")

    # 9. Pattern recognition features
    if 'patterns' in additional_features:
        logger.info("Calculating pattern recognition features")
        try:
            if all(col in processed_df.columns for col in ['open', 'high', 'low', 'close']):
                pattern_features = calculate_pattern_features(processed_df)
                processed_df = pd.concat(
                    [processed_df, pd.DataFrame(pattern_features)], axis=1)
                logger.info("Pattern features calculated successfully")
            else:
                logger.warning("Missing required columns for pattern features")
        except Exception as e:
            logger.error(f"Error calculating pattern features: {e}")

    # 10. Momentum features
    if 'momentum' in additional_features:
        logger.info("Calculating momentum features")
        try:
            momentum_features = calculate_momentum_features(processed_df)
            processed_df = pd.concat(
                [processed_df, pd.DataFrame(momentum_features)], axis=1)
            logger.info("Momentum features calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating momentum features: {e}")

    # --- Define Target Variable ---

    # Triple Barrier Method
    if target_type == 'triple_barrier':
        logger.info(f"Calculating Triple Barrier Labels (TP: {atr_multiplier_tp}*ATR, "
                    f"SL: {atr_multiplier_sl}*ATR, Max Hold: {max_holding_period} bars)")
        required_cols = ['close', 'high', 'low', 'ATRr_10']

        if not all(col in processed_df.columns for col in required_cols):
            missing = [
                c for c in required_cols if c not in processed_df.columns]
            logger.error(
                f"Missing required columns for Triple Barrier: {missing}")
            return None

        # Drop NaNs from features before calculating labels to ensure alignment
        feature_cols = processed_df.columns.difference(
            ['triple_barrier_label'])
        processed_df.dropna(subset=feature_cols, inplace=True)

        if processed_df.empty:
            logger.error("DataFrame empty after dropping NaNs from features")
            return None

        # Calculate labels on the feature-cleaned data
        processed_df['triple_barrier_label'] = get_triple_barrier_labels(
            prices=processed_df['close'],
            highs=processed_df['high'],
            lows=processed_df['low'],
            atr=processed_df['ATRr_10'],
            atr_multiplier_tp=atr_multiplier_tp,
            atr_multiplier_sl=atr_multiplier_sl,
            max_holding_period=max_holding_period,
            min_return_threshold=0.001  # Small threshold to filter noise
        )

        # Drop rows where the label could not be calculated
        initial_rows = len(processed_df)
        processed_df.dropna(subset=['triple_barrier_label'], inplace=True)
        rows_dropped = initial_rows - len(processed_df)

        if rows_dropped > 0:
            logger.info(
                f"Dropped {rows_dropped} rows due to NaNs from Triple Barrier calculation")

        # Convert label to integer type after dropping NaNs
        if 'triple_barrier_label' in processed_df.columns:
            processed_df['triple_barrier_label'] = processed_df['triple_barrier_label'].astype(
                int)

    # Simple directional prediction (for comparison)
    elif target_type == 'directional':
        logger.info("Calculating simple directional target variable")
        if 'close' not in processed_df.columns:
            logger.error("Missing 'close' column for directional target")
            return None

        # Future N-bar return
        processed_df['future_return'] = processed_df['close'].shift(
            -max_holding_period) / processed_df['close'] - 1

        # Convert to classification target based on threshold
        threshold = 0.001  # Minimum move to consider significant
        processed_df['directional_label'] = 0  # Initialize as neutral
        processed_df.loc[processed_df['future_return']
                         > threshold, 'directional_label'] = 1
        processed_df.loc[processed_df['future_return']
                         < -threshold, 'directional_label'] = -1

        # Drop rows where target is NaN
        processed_df.dropna(subset=['directional_label'], inplace=True)
        processed_df['directional_label'] = processed_df['directional_label'].astype(
            int)

    # --- Clean up and finalize ---

    # Sanitize column names
    original_cols = processed_df.columns
    processed_df.columns = (
        processed_df.columns.str.replace(r'[\s\(\)\%\.]+', '_', regex=True)
        .str.replace(r'_+', '_', regex=True)  # Consolidate underscores
        .str.strip('_')  # Remove leading/trailing underscores
    )

    # Final check for NaNs in the processed DataFrame
    nan_columns = processed_df.columns[processed_df.isna().any()].tolist()
    if nan_columns:
        logger.warning(
            f"NaN values present in columns after processing: {nan_columns}")
        # Replace NaNs with appropriate values based on column type
        for col in nan_columns:
            if np.issubdtype(processed_df[col].dtype, np.number):
                # For numerical columns, use median or 0
                if processed_df[col].median() != 0:
                    processed_df[col].fillna(
                        processed_df[col].median(), inplace=True)
                else:
                    processed_df[col].fillna(0, inplace=True)
            else:
                # For categorical/other columns, use most frequent or 0
                processed_df[col].fillna(0, inplace=True)

    logger.info(
        f"Feature engineering complete. DataFrame shape: {processed_df.shape}")
    return processed_df
