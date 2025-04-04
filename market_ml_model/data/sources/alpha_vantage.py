"""
Alpha Vantage data source implementation.
"""

import pandas as pd
import logging
import time
from typing import Optional

# Setup logging
logger = logging.getLogger(__name__)

# Import alpha_vantage with error handling
try:
    from alpha_vantage.timeseries import TimeSeries
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    logger.warning(
        "alpha_vantage not installed. Alpha Vantage data source unavailable.")
    ALPHA_VANTAGE_AVAILABLE = False


def load_from_alpha_vantage(
    ticker: str,
    api_key: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    retry_count: int = 3
) -> Optional[pd.DataFrame]:
    """
    Load historical market data from Alpha Vantage.

    Args:
        ticker: Ticker symbol
        api_key: Alpha Vantage API key
        start_date: Start date for data (YYYY-MM-DD)
        end_date: End date for data (YYYY-MM-DD)
        interval: Data interval ('1d', '60min', '30min', '15min', '5min', '1min')
        retry_count: Number of retries for failed requests

    Returns:
        DataFrame with OHLCV data, or None if download fails
    """
    if not ALPHA_VANTAGE_AVAILABLE:
        logger.error(
            "alpha_vantage not installed. Cannot load from Alpha Vantage.")
        return None

    if not api_key:
        logger.error("Alpha Vantage API key not provided")
        return None

    logger.info(
        f"Loading {interval} data for {ticker} from {start_date} to {end_date} from Alpha Vantage")

    # Map interval to Alpha Vantage format
    av_interval_map = {
        '1d': 'daily',
        '1h': '60min',
        '60min': '60min',
        '30min': '30min',
        '15min': '15min',
        '5min': '5min',
        '1min': '1min'
    }

    av_interval = av_interval_map.get(interval)
    if not av_interval:
        logger.error(f"Unsupported interval for Alpha Vantage: {interval}")
        return None

    # Try multiple times in case of network issues
    for attempt in range(retry_count):
        try:
            # Initialize Alpha Vantage API
            ts = TimeSeries(key=api_key, output_format='pandas')

            # Get data based on interval
            if av_interval == 'daily':
                data, _ = ts.get_daily_adjusted(
                    symbol=ticker, outputsize='full')
            else:
                data, _ = ts.get_intraday(
                    symbol=ticker, interval=av_interval, outputsize='full')

            if data.empty:
                logger.warning(
                    f"Attempt {attempt+1}/{retry_count}: No data for {ticker} from Alpha Vantage")
                time.sleep(1)  # Wait before retry
                continue

            # Rename columns to standard format
            data.columns = [col.split(' ')[1].lower() for col in data.columns]

            # Filter by date range
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)

            # Alpha Vantage data is in reverse chronological order
            data = data.sort_index()

            # Filter date range
            data = data[(data.index >= start_date_dt)
                        & (data.index <= end_date_dt)]

            # Ensure we have expected columns
            expected_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in expected_columns):
                logger.warning(
                    f"Missing expected columns in Alpha Vantage data. Got: {data.columns.tolist()}")

            return data

        except Exception as e:
            logger.warning(
                f"Attempt {attempt+1}/{retry_count}: Error downloading data from Alpha Vantage: {e}")
            time.sleep(1)  # Wait before retry

    logger.error(
        f"Failed to download data for {ticker} from Alpha Vantage after {retry_count} attempts")
    return None