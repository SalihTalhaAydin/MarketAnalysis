"""
Crypto data source implementation using ccxt.
"""

import logging
import time
from typing import Optional

import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

# Import ccxt with error handling
try:
    import ccxt

    CCXT_AVAILABLE = True
except ImportError:
    logger.warning("ccxt not installed. Crypto exchange data source unavailable.")
    CCXT_AVAILABLE = False


def load_from_crypto(
    symbol: str,
    exchange: str = "binance",
    start_date: str = "",
    end_date: str = "",
    interval: str = "1d",
    retry_count: int = 3,
) -> Optional[pd.DataFrame]:
    """
    Load crypto market data using ccxt.

    Args:
        symbol: Crypto symbol pair (e.g., 'BTC/USDT')
        exchange: Exchange name (e.g., 'binance', 'kraken')
        start_date: Start date for data (YYYY-MM-DD)
        end_date: End date for data (YYYY-MM-DD)
        interval: Data interval in ccxt format ('1d', '1h', '5m', etc.)
        retry_count: Number of retries for failed requests

    Returns:
        DataFrame with OHLCV data, or None if download fails
    """
    if not CCXT_AVAILABLE:
        logger.error("ccxt not installed. Cannot load crypto data.")
        return None

    logger.info(
        f"Loading {interval} data for {symbol} from {start_date} to {end_date} from {exchange}"
    )

    # Map interval to ccxt timeframe format
    ccxt_interval_map = {
        "1d": "1d",
        "1h": "1h",
        "4h": "4h",
        "15m": "15m",
        "5m": "5m",
        "1m": "1m",
    }

    ccxt_interval = ccxt_interval_map.get(interval, interval)

    # Convert dates to timestamps
    start_timestamp = (
        int(pd.to_datetime(start_date).timestamp() * 1000) if start_date else None
    )
    end_timestamp = (
        int(pd.to_datetime(end_date).timestamp() * 1000) if end_date else None
    )

    # Try multiple times in case of network issues
    for attempt in range(retry_count):
        try:
            # Initialize exchange
            exchange_class = getattr(ccxt, exchange)
            exchange_instance = exchange_class(
                {
                    "enableRateLimit": True,
                }
            )

            # Check if exchange supports fetchOHLCV
            if not exchange_instance.has["fetchOHLCV"]:
                logger.error(f"Exchange {exchange} does not support OHLCV data")
                return None

            # Fetch data
            ohlcv = []
            since = start_timestamp

            # Loop to get all data (ccxt has pagination limits)
            while True:
                batch = exchange_instance.fetch_ohlcv(symbol, ccxt_interval, since)

                if not batch:
                    break

                ohlcv.extend(batch)

                # Update since for next batch
                since = batch[-1][0] + 1

                # Stop if we reached end date
                if end_timestamp and since > end_timestamp:
                    break

                # Rate limiting
                time.sleep(exchange_instance.rateLimit / 1000)

            if not ohlcv:
                logger.warning(
                    f"Attempt {attempt+1}/{retry_count}: No data for {symbol} from {exchange}"
                )
                time.sleep(1)  # Wait before retry
                continue

            # Convert to DataFrame
            data = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

            # Convert timestamp to datetime
            data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
            data.set_index("timestamp", inplace=True)

            # Filter by end date
            if end_timestamp:
                data = data[data.index.timestamp * 1000 <= end_timestamp]

            # Sort by date
            data = data.sort_index()

            return data

        except Exception as e:
            logger.warning(
                f"Attempt {attempt+1}/{retry_count}: Error downloading crypto data: {e}"
            )
            time.sleep(1)  # Wait before retry

    logger.error(
        f"Failed to download data for {symbol} from {exchange} after {retry_count} attempts"
    )
    return None
