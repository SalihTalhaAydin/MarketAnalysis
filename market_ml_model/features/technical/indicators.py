"""
Technical indicators for market data analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

# Setup logging
logger = logging.getLogger(__name__)

# Try importing technical analysis libraries with fallbacks
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    logger.warning("pandas-ta not installed. Installing required package...")
    import subprocess
    import sys
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pandas-ta"])
        import pandas_ta as ta
        PANDAS_TA_AVAILABLE = True
    except Exception as e:
        logger.error(f"Failed to install pandas-ta: {e}")
        PANDAS_TA_AVAILABLE = False


def calculate_technical_indicators(
    df: pd.DataFrame,
    indicators: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate technical indicators for the given dataframe.

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
            'sma', 'ema', 'rsi', 'macd', 'bbands', 'atr', 'adx', 'stoch'
        ]

    # Make a copy of the dataframe
    result_df = df.copy()

    # Check for required columns
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in result_df.columns]

    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return df

    try:
        # Calculate indicators
        for indicator in indicators:
            if indicator == 'sma':
                # Simple Moving Averages
                for period in [20, 50, 200]:
                    result_df[f'sma_{period}'] = ta.sma(result_df['close'], length=period)

            elif indicator == 'ema':
                # Exponential Moving Averages
                for period in [9, 20, 50]:
                    result_df[f'ema_{period}'] = ta.ema(result_df['close'], length=period)

            elif indicator == 'rsi':
                # Relative Strength Index
                result_df['rsi_14'] = ta.rsi(result_df['close'], length=14)

            elif indicator == 'macd':
                # MACD
                macd = ta.macd(result_df['close'])
                if isinstance(macd, pd.DataFrame) and len(macd.columns) >= 3:
                    result_df['macd'] = macd.iloc[:, 0]
                    result_df['macd_signal'] = macd.iloc[:, 1]
                    result_df['macd_hist'] = macd.iloc[:, 2]

            elif indicator == 'bbands':
                # Bollinger Bands
                bbands = ta.bbands(result_df['close'], length=20)
                if isinstance(bbands, pd.DataFrame) and len(bbands.columns) >= 3:
                    result_df['bb_upper'] = bbands.iloc[:, 0]
                    result_df['bb_middle'] = bbands.iloc[:, 1]
                    result_df['bb_lower'] = bbands.iloc[:, 2]
                    result_df['bb_width'] = ((bbands.iloc[:, 0] - bbands.iloc[:, 2]) / 
                                            bbands.iloc[:, 1])

            elif indicator == 'atr':
                # Average True Range
                result_df['atr_14'] = ta.atr(
                    result_df['high'], result_df['low'], result_df['close'], length=14
                )
                # ATR as a percentage of price (ATRr)
                result_df['ATRr_10'] = ta.atr(
                    result_df['high'], result_df['low'], result_df['close'], length=10
                ) / result_df['close']

            elif indicator == 'adx':
                # Average Directional Index
                adx = ta.adx(result_df['high'], result_df['low'], result_df['close'])
                if isinstance(adx, pd.DataFrame) and len(adx.columns) >= 3:
                    result_df['adx'] = adx.iloc[:, 0]
                    result_df['dmp'] = adx.iloc[:, 1]  # Plus Directional Movement
                    result_df['dmn'] = adx.iloc[:, 2]  # Minus Directional Movement

            elif indicator == 'stoch':
                # Stochastic Oscillator
                stoch = ta.stoch(result_df['high'], result_df['low'], result_df['close'])
                if isinstance(stoch, pd.DataFrame) and len(stoch.columns) >= 2:
                    result_df['stoch_k'] = stoch.iloc[:, 0]
                    result_df['stoch_d'] = stoch.iloc[:, 1]

            elif indicator == 'volume_indicators' and 'volume' in result_df.columns:
                # Volume-based indicators
                result_df['volume_sma20'] = ta.sma(result_df['volume'], length=20)
                result_df['volume_ratio'] = result_df['volume'] / result_df['volume_sma20']
                result_df['obv'] = ta.obv(result_df['close'], result_df['volume'])

                # Money Flow Index
                result_df['mfi_14'] = ta.mfi(
                    result_df['high'], result_df['low'], result_df['close'], 
                    result_df['volume'], length=14
                )

            else:
                logger.warning(f"Unsupported indicator: {indicator}")

        logger.info(f"Added {len(indicators)} indicator groups")
        return result_df

    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return df