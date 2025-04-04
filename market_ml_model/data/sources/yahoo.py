"""
Yahoo Finance data source implementation.
"""

import logging
import time
from typing import Optional

import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

# Import yfinance with error handling
try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    logger.warning("yfinance not installed. Yahoo Finance data source unavailable.")
    YFINANCE_AVAILABLE = False


def load_from_yahoo(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    adjust_prices: bool = True,
    retry_count: int = 3,
) -> Optional[pd.DataFrame]:
    """
    Load historical market data from Yahoo Finance.

    Args:
        ticker: Ticker symbol
        start_date: Start date for data (YYYY-MM-DD)
        end_date: End date for data (YYYY-MM-DD)
        interval: Data interval (e.g., '1m', '5m', '1h', '1d')
        adjust_prices: Adjust prices for splits/dividends
        retry_count: Number of retries for failed requests

    Returns:
        DataFrame with OHLCV data, or None if download fails
    """
    if not YFINANCE_AVAILABLE:
        logger.error("yfinance not installed. Cannot load from Yahoo Finance.")
        return None

    logger.info(
        f"Loading {interval} data for {ticker} from {start_date} to {end_date} from Yahoo Finance"
    )

    # Try multiple times in case of network issues
    for attempt in range(retry_count):
        try:
            # Use yfinance to download historical market data
            data = yf.download(
                tickers=ticker,  # Ticker symbol(s)
                start=start_date,  # Start date for data retrieval
                end=end_date,  # End date for data retrieval
                interval=interval,  # Data interval
                auto_adjust=adjust_prices,  # Adjust for splits/dividends
                progress=False,  # Suppress download progress bar
            )

            if data.empty:
                logger.warning(
                    f"Attempt {attempt+1}/{retry_count}: No data for {ticker} from Yahoo Finance"
                )
                time.sleep(1)  # Wait before retry
                continue

            # Convert column names to lowercase, handling potential MultiIndex from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                # Keep only the first level (e.g., 'Open', 'Close') and convert to lowercase
                data.columns = data.columns.get_level_values(0).str.lower()
            else:
                # Simple Index case
                data.columns = data.columns.str.lower()

            # Rename 'adj close' to 'close' if auto_adjust=True was used and 'close' doesn't exist
            if "adj close" in data.columns and "close" not in data.columns:
                data = data.rename(columns={"adj close": "close"})
            elif (
                adjust_prices
                and "close" in data.columns
                and "adj close" in data.columns
            ):
                # If auto_adjust=True but somehow both exist, prefer 'adj close' and rename it
                data = data.drop(columns=["close"])
                data = data.rename(columns={"adj close": "close"})
            elif not adjust_prices and "adj close" in data.columns:
                # If auto_adjust=False, remove 'adj close' if it exists
                data = data.drop(columns=["adj close"])

            # Ensure we have expected columns
            expected_columns = ["open", "high", "low", "close", "volume"]
            if not all(col in data.columns for col in expected_columns):
                logger.warning(
                    f"Missing expected columns in Yahoo data. Got: {data.columns.tolist()}"
                )

            return data

        except Exception as e:
            logger.warning(
                f"Attempt {attempt+1}/{retry_count}: Error downloading data for {ticker}: {e}"
            )
            time.sleep(1)  # Wait before retry

    logger.error(f"Failed to download data for {ticker} after {retry_count} attempts")
    return None
