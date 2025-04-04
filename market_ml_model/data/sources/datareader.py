"""
pandas-datareader data source implementation.
"""

import logging
import time
from typing import Optional

import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

# Import pandas_datareader with error handling
try:
    import pandas_datareader as pdr

    DATAREADER_AVAILABLE = True
except ImportError:
    logger.warning("pandas_datareader not installed. Some data sources unavailable.")
    DATAREADER_AVAILABLE = False


def load_from_datareader(
    ticker: str,
    data_source: str,
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None,
    retry_count: int = 3,
) -> Optional[pd.DataFrame]:
    """
    Load market data using pandas_datareader.

    Args:
        ticker: Ticker symbol
        data_source: Data source name for pandas_datareader
        start_date: Start date for data (YYYY-MM-DD)
        end_date: End date for data (YYYY-MM-DD)
        api_key: Optional API key for some data sources
        retry_count: Number of retries for failed requests

    Returns:
        DataFrame with market data, or None if download fails
    """
    if not DATAREADER_AVAILABLE:
        logger.error("pandas_datareader not installed. Cannot load from data reader.")
        return None

    logger.info(
        f"Loading data for {ticker} from {start_date} to {end_date} from {data_source}"
    )

    # Convert dates to datetime
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)

    # Try multiple times in case of network issues
    for attempt in range(retry_count):
        try:
            # Get data with pandas_datareader
            if api_key:
                data = pdr.get_data_datareader(
                    ticker, data_source, start_date_dt, end_date_dt, api_key=api_key
                )
            else:
                data = pdr.get_data_datareader(
                    ticker, data_source, start_date_dt, end_date_dt
                )

            if data.empty:
                logger.warning(
                    f"Attempt {attempt+1}/{retry_count}: No data for {ticker} from {data_source}"
                )
                time.sleep(1)  # Wait before retry
                continue

            # Normalize column names
            data.columns = data.columns.str.lower()

            # Some sources might have different column name conventions
            column_mapping = {
                "adj close": "adj_close",
                "adj. close": "adj_close",
                "last": "close",
                "value": "close",
            }

            data.columns = [column_mapping.get(col, col) for col in data.columns]

            # Sort by date
            data = data.sort_index()

            return data

        except Exception as e:
            logger.warning(
                f"Attempt {attempt+1}/{retry_count}: Error downloading data from {data_source}: {e}"
            )
            time.sleep(1)  # Wait before retry

    logger.error(
        f"Failed to download data for {ticker} from {data_source} after {retry_count} attempts"
    )
    return None
