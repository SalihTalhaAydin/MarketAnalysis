import pandas as pd
import numpy as np
import logging
import time
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import json

# Setup logging
logger = logging.getLogger(__name__)

# Import data providers with error handling
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    logger.warning(
        "yfinance not installed. Yahoo Finance data source unavailable.")
    YFINANCE_AVAILABLE = False

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    logger.warning(
        "ccxt not installed. Crypto exchange data source unavailable.")
    CCXT_AVAILABLE = False

try:
    import pandas_datareader as pdr
    DATAREADER_AVAILABLE = True
except ImportError:
    logger.warning(
        "pandas_datareader not installed. Some data sources unavailable.")
    DATAREADER_AVAILABLE = False

try:
    from alpha_vantage.timeseries import TimeSeries
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    logger.warning(
        "alpha_vantage not installed. Alpha Vantage data source unavailable.")
    ALPHA_VANTAGE_AVAILABLE = False


# --- Data Source Enum ---
class DataSource:
    """Enumeration of supported data sources."""
    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    CSV = "csv"
    CRYPTO = "crypto"
    DATAREADER = "datareader"


# --- Configuration ---
class DataLoaderConfig:
    """Configuration for data loader."""

    def __init__(
        self,
        data_source: str = DataSource.YAHOO,
        api_key: Optional[str] = None,
        rate_limit_pause: float = 1.0,
        cache_dir: Optional[str] = None,
        retry_count: int = 3,
        use_cache: bool = True,
        cache_expiry_days: int = 1,
        default_start_date: str = "2010-01-01",
        default_end_date: Optional[str] = None,
        default_interval: str = "1d",
        crypto_exchange: str = "binance",
        adjust_prices: bool = True,
    ):
        """
        Initialize data loader configuration.

        Args:
            data_source: Data source identifier
            api_key: API key for paid data sources
            rate_limit_pause: Seconds to pause between API calls
            cache_dir: Directory to cache data
            retry_count: Number of retries for failed data requests
            use_cache: Whether to use cached data when available
            cache_expiry_days: Days before cached data expires
            default_start_date: Default start date for data
            default_end_date: Default end date for data (None = today)
            default_interval: Default data interval
            crypto_exchange: CCXT exchange name for crypto data
            adjust_prices: Adjust prices for splits/dividends
        """
        self.data_source = data_source
        self.api_key = api_key
        self.rate_limit_pause = rate_limit_pause
        self.retry_count = retry_count
        self.use_cache = use_cache
        self.cache_expiry_days = cache_expiry_days
        self.default_start_date = default_start_date
        self.default_end_date = default_end_date or datetime.now().strftime("%Y-%m-%d")
        self.default_interval = default_interval
        self.crypto_exchange = crypto_exchange
        self.adjust_prices = adjust_prices

        # Setup cache directory
        if cache_dir:
            self.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
        else:
            self.cache_dir = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataLoaderConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> 'DataLoaderConfig':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data_source': self.data_source,
            'api_key': self.api_key,
            'rate_limit_pause': self.rate_limit_pause,
            'cache_dir': self.cache_dir,
            'retry_count': self.retry_count,
            'use_cache': self.use_cache,
            'cache_expiry_days': self.cache_expiry_days,
            'default_start_date': self.default_start_date,
            'default_end_date': self.default_end_date,
            'default_interval': self.default_interval,
            'crypto_exchange': self.crypto_exchange,
            'adjust_prices': self.adjust_prices,
        }

    def to_json(self, json_path: str) -> None:
        """Save configuration to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)


# --- Data Cache ---
class DataCache:
    """Cache for market data."""

    def __init__(self, cache_dir: str, expiry_days: int = 1):
        """
        Initialize data cache.

        Args:
            cache_dir: Directory to store cached data
            expiry_days: Days before cached data expires
        """
        self.cache_dir = cache_dir
        self.expiry_days = expiry_days
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_path(self, ticker: str, start_date: str, end_date: str, interval: str) -> str:
        """
        Generate cache file path.

        Args:
            ticker: Ticker symbol
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval

        Returns:
            Path to cache file
        """
        # Sanitize ticker for filename (replace characters that might be invalid in filenames)
        safe_ticker = ticker.replace(
            '/', '_').replace('\\', '_').replace(':', '_')

        # Create cache filename
        filename = f"{safe_ticker}_{start_date}_{end_date}_{interval}.parquet"
        return os.path.join(self.cache_dir, filename)

    def is_cache_valid(self, cache_path: str) -> bool:
        """
        Check if cache is valid (exists and not expired).

        Args:
            cache_path: Path to cache file

        Returns:
            True if cache is valid, False otherwise
        """
        if not os.path.exists(cache_path):
            return False

        # Check if cache is expired
        if self.expiry_days > 0:
            file_time = os.path.getmtime(cache_path)
            file_age = (time.time() - file_time) / \
                (24 * 3600)  # Convert to days

            return file_age < self.expiry_days

        return True

    def get_cached_data(self, cache_path: str) -> Optional[pd.DataFrame]:
        """
        Get data from cache.

        Args:
            cache_path: Path to cache file

        Returns:
            DataFrame with cached data, or None if not found
        """
        if not self.is_cache_valid(cache_path):
            return None

        try:
            # Use parquet for efficient storage
            return pd.read_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Failed to read from cache: {e}")
            return None

    def save_to_cache(self, data: pd.DataFrame, cache_path: str) -> bool:
        """
        Save data to cache.

        Args:
            data: DataFrame to cache
            cache_path: Path to cache file

        Returns:
            True if successful, False otherwise
        """
        try:
            data.to_parquet(cache_path, index=True)
            return True
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
            return False


# --- Data Loaders ---

def load_from_yahoo(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    adjust_prices: bool = True,
    retry_count: int = 3
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
        f"Loading {interval} data for {ticker} from {start_date} to {end_date} from Yahoo Finance")

    # Try multiple times in case of network issues
    for attempt in range(retry_count):
        try:
            # Use yfinance to download historical market data
            data = yf.download(
                tickers=ticker,       # Ticker symbol(s)
                start=start_date,     # Start date for data retrieval
                end=end_date,         # End date for data retrieval
                interval=interval,    # Data interval
                auto_adjust=adjust_prices,  # Adjust for splits/dividends
                progress=False        # Suppress download progress bar
            )

            if data.empty:
                logger.warning(
                    f"Attempt {attempt+1}/{retry_count}: No data for {ticker} from Yahoo Finance")
                time.sleep(1)  # Wait before retry
                continue

            # Convert column names to lowercase
            data.columns = data.columns.str.lower()

            # Ensure we have expected columns
            expected_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in expected_columns):
                logger.warning(
                    f"Missing expected columns in Yahoo data. Got: {data.columns.tolist()}")

            return data

        except Exception as e:
            logger.warning(
                f"Attempt {attempt+1}/{retry_count}: Error downloading data for {ticker}: {e}")
            time.sleep(1)  # Wait before retry

    logger.error(
        f"Failed to download data for {ticker} after {retry_count} attempts")
    return None


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


def load_from_csv(
    file_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    date_column: str = 'date',
    date_format: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Load market data from CSV file.

    Args:
        file_path: Path to CSV file
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        date_column: Name of date column
        date_format: Optional date format for parsing

    Returns:
        DataFrame with market data, or None if loading fails
    """
    logger.info(f"Loading data from CSV file: {file_path}")

    try:
        # Read CSV
        data = pd.read_csv(file_path)

        # Check if empty
        if data.empty:
            logger.error(f"CSV file {file_path} is empty")
            return None

        # Convert date column to datetime
        try:
            data[date_column] = pd.to_datetime(
                data[date_column], format=date_format)
        except Exception as e:
            logger.error(f"Failed to parse date column: {e}")
            return None

        # Set date column as index
        data.set_index(date_column, inplace=True)

        # Ensure column names are lowercase
        data.columns = data.columns.str.lower()

        # Filter by date range if provided
        if start_date:
            start_date_dt = pd.to_datetime(start_date)
            data = data[data.index >= start_date_dt]

        if end_date:
            end_date_dt = pd.to_datetime(end_date)
            data = data[data.index <= end_date_dt]

        # Sort by date
        data = data.sort_index()

        # Ensure we have expected columns
        expected_columns = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in expected_columns):
            logger.warning(
                f"Missing expected columns in CSV data. Got: {data.columns.tolist()}")

        return data

    except Exception as e:
        logger.error(f"Failed to load data from CSV file {file_path}: {e}")
        return None


def load_from_crypto(
    symbol: str,
    exchange: str = "binance",
    start_date: str = "",
    end_date: str = "",
    interval: str = "1d",
    retry_count: int = 3
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
        f"Loading {interval} data for {symbol} from {start_date} to {end_date} from {exchange}")

    # Map interval to ccxt timeframe format
    ccxt_interval_map = {
        '1d': '1d',
        '1h': '1h',
        '4h': '4h',
        '15m': '15m',
        '5m': '5m',
        '1m': '1m'
    }

    ccxt_interval = ccxt_interval_map.get(interval, interval)

    # Convert dates to timestamps
    start_timestamp = int(pd.to_datetime(
        start_date).timestamp() * 1000) if start_date else None
    end_timestamp = int(pd.to_datetime(end_date).timestamp()
                        * 1000) if end_date else None

    # Try multiple times in case of network issues
    for attempt in range(retry_count):
        try:
            # Initialize exchange
            exchange_class = getattr(ccxt, exchange)
            exchange_instance = exchange_class({
                'enableRateLimit': True,
            })

            # Check if exchange supports fetchOHLCV
            if not exchange_instance.has['fetchOHLCV']:
                logger.error(
                    f"Exchange {exchange} does not support OHLCV data")
                return None

            # Fetch data
            ohlcv = []
            since = start_timestamp

            # Loop to get all data (ccxt has pagination limits)
            while True:
                batch = exchange_instance.fetch_ohlcv(
                    symbol, ccxt_interval, since)

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
                    f"Attempt {attempt+1}/{retry_count}: No data for {symbol} from {exchange}")
                time.sleep(1)  # Wait before retry
                continue

            # Convert to DataFrame
            data = pd.DataFrame(
                ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Convert timestamp to datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)

            # Filter by end date
            if end_timestamp:
                data = data[data.index.timestamp * 1000 <= end_timestamp]

            # Sort by date
            data = data.sort_index()

            return data

        except Exception as e:
            logger.warning(
                f"Attempt {attempt+1}/{retry_count}: Error downloading crypto data: {e}")
            time.sleep(1)  # Wait before retry

    logger.error(
        f"Failed to download data for {symbol} from {exchange} after {retry_count} attempts")
    return None


def load_from_datareader(
    ticker: str,
    data_source: str,
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None,
    retry_count: int = 3
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
        logger.error(
            "pandas_datareader not installed. Cannot load from data reader.")
        return None

    logger.info(
        f"Loading data for {ticker} from {start_date} to {end_date} from {data_source}")

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
                    f"Attempt {attempt+1}/{retry_count}: No data for {ticker} from {data_source}")
                time.sleep(1)  # Wait before retry
                continue

            # Normalize column names
            data.columns = data.columns.str.lower()

            # Some sources might have different column name conventions
            column_mapping = {
                'adj close': 'adj_close',
                'adj. close': 'adj_close',
                'last': 'close',
                'value': 'close'
            }

            data.columns = [column_mapping.get(
                col, col) for col in data.columns]

            # Sort by date
            data = data.sort_index()

            return data

        except Exception as e:
            logger.warning(
                f"Attempt {attempt+1}/{retry_count}: Error downloading data from {data_source}: {e}")
            time.sleep(1)  # Wait before retry

    logger.error(
        f"Failed to download data for {ticker} from {data_source} after {retry_count} attempts")
    return None


# --- Main DataLoader class ---

class DataLoader:
    """
    Advanced data loader with multiple data sources and caching.
    """

    def __init__(self, config: Optional[DataLoaderConfig] = None):
        """
        Initialize data loader.

        Args:
            config: Configuration for data loader
        """
        self.config = config or DataLoaderConfig()

        # Initialize cache if enabled
        if self.config.use_cache and self.config.cache_dir:
            self.cache = DataCache(
                cache_dir=self.config.cache_dir,
                expiry_days=self.config.cache_expiry_days
            )
        else:
            self.cache = None

        # Last API call timestamp for rate limiting
        self.last_api_call = 0

    def load_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: Optional[str] = None,
        data_source: Optional[str] = None,
        force_reload: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Load market data based on configuration.

        Args:
            ticker: Ticker symbol
            start_date: Start date for data (YYYY-MM-DD), defaults to config
            end_date: End date for data (YYYY-MM-DD), defaults to config
            interval: Data interval, defaults to config
            data_source: Data source, defaults to config
            force_reload: Force reload from source (ignore cache)

        Returns:
            DataFrame with market data, or None if loading fails
        """
        # Use defaults from config if not provided
        start_date = start_date or self.config.default_start_date
        end_date = end_date or self.config.default_end_date
        interval = interval or self.config.default_interval
        data_source = data_source or self.config.data_source

        logger.info(
            f"Loading {ticker} data from {start_date} to {end_date} (interval: {interval})")

        # Check cache first if enabled
        if self.cache and not force_reload:
            cache_path = self.cache.get_cache_path(
                ticker, start_date, end_date, interval)
            cached_data = self.cache.get_cached_data(cache_path)

            if cached_data is not None:
                logger.info(f"Retrieved {ticker} data from cache")
                return cached_data

        # Apply rate limiting
        self._apply_rate_limit()

        # Load from appropriate source
        data = None

        if data_source == DataSource.YAHOO:
            data = load_from_yahoo(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                adjust_prices=self.config.adjust_prices,
                retry_count=self.config.retry_count
            )

        elif data_source == DataSource.ALPHA_VANTAGE:
            data = load_from_alpha_vantage(
                ticker=ticker,
                api_key=self.config.api_key,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                retry_count=self.config.retry_count
            )

        elif data_source == DataSource.CSV:
            # Treat ticker as file path for CSV
            data = load_from_csv(
                file_path=ticker,
                start_date=start_date,
                end_date=end_date
            )

        elif data_source == DataSource.CRYPTO:
            data = load_from_crypto(
                symbol=ticker,
                exchange=self.config.crypto_exchange,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                retry_count=self.config.retry_count
            )

        elif data_source == DataSource.DATAREADER:
            # For datareader, interpret interval as data source name
            data = load_from_datareader(
                ticker=ticker,
                data_source=interval,
                start_date=start_date,
                end_date=end_date,
                api_key=self.config.api_key,
                retry_count=self.config.retry_count
            )

        else:
            logger.error(f"Unsupported data source: {data_source}")
            return None

        # Save to cache if successful
        if data is not None and self.cache:
            cache_path = self.cache.get_cache_path(
                ticker, start_date, end_date, interval)
            self.cache.save_to_cache(data, cache_path)

        return data

    def load_multiple(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: Optional[str] = None,
        data_source: Optional[str] = None,
        column_name: str = 'close',
        force_reload: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Load data for multiple tickers and combine into a single DataFrame.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for data (YYYY-MM-DD), defaults to config
            end_date: End date for data (YYYY-MM-DD), defaults to config
            interval: Data interval, defaults to config
            data_source: Data source, defaults to config
            column_name: Column to extract (e.g., 'close', 'adj_close')
            force_reload: Force reload from source (ignore cache)

        Returns:
            DataFrame with combined data, or None if loading fails
        """
        result = {}

        for ticker in tickers:
            # Load data for this ticker
            data = self.load_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                data_source=data_source,
                force_reload=force_reload
            )

            if data is None:
                logger.warning(f"Failed to load data for {ticker}")
                continue

            # Extract the requested column
            if column_name in data.columns:
                result[ticker] = data[column_name]
            else:
                logger.warning(
                    f"Column '{column_name}' not found in data for {ticker}")

        if not result:
            logger.error("Failed to load data for any ticker")
            return None

        # Combine into a single DataFrame
        combined = pd.DataFrame(result)

        return combined

    def _apply_rate_limit(self) -> None:
        """Apply rate limiting between API calls."""
        if self.config.rate_limit_pause > 0:
            elapsed = time.time() - self.last_api_call

            if elapsed < self.config.rate_limit_pause:
                time.sleep(self.config.rate_limit_pause - elapsed)

        self.last_api_call = time.time()


# --- Data Transformation Functions ---

def resample_data(
    data: pd.DataFrame,
    new_interval: str,
    ohlc_columns: List[str] = ['open', 'high', 'low', 'close'],
    volume_column: Optional[str] = 'volume'
) -> pd.DataFrame:
    """
    Resample OHLCV data to a different interval.

    Args:
        data: DataFrame with OHLCV data
        new_interval: New interval for resampling ('1H', '4H', '1D', etc.)
        ohlc_columns: Column names for OHLC data
        volume_column: Column name for volume data (None if not present)

    Returns:
        Resampled DataFrame
    """
    # Ensure data has DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        logger.warning(
            "Data index is not DatetimeIndex. Attempting conversion.")
        data.index = pd.to_datetime(data.index)

    # Create empty result DataFrame
    result = pd.DataFrame(index=data.index)

    # Define resampling logic for each column type
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }

    # Make sure we have the required columns
    available_ohlc = [col for col in ohlc_columns if col in data.columns]

    if not available_ohlc:
        logger.error("No OHLC columns found in data")
        return data

    # Create a subset of just the OHLC data
    ohlc_data = data[available_ohlc]

    # Map our column names to the expected open/high/low/close names
    column_mapping = {}
    for idx, col in enumerate(available_ohlc):
        if idx < len(ohlc_columns):
            column_mapping[col] = ohlc_columns[idx]

    # Rename columns to standard OHLC names
    ohlc_data = ohlc_data.rename(columns=column_mapping)

    # Make sure we have standard OHLC names
    for col in ['open', 'high', 'low', 'close']:
        if col not in ohlc_data.columns:
            logger.warning(
                f"Missing {col} column for resampling. Using fallbacks.")

            # Use fallbacks
            if col == 'open' and 'close' in ohlc_data.columns:
                ohlc_data['open'] = ohlc_data['close']
            elif col == 'high' and 'close' in ohlc_data.columns:
                ohlc_data['high'] = ohlc_data['close']
            elif col == 'low' and 'close' in ohlc_data.columns:
                ohlc_data['low'] = ohlc_data['close']
            elif col == 'close' and 'open' in ohlc_data.columns:
                ohlc_data['close'] = ohlc_data['open']

    # Resample OHLC data
    resampled = ohlc_data.resample(new_interval).agg(ohlc_dict)

    # Add volume if present
    if volume_column and volume_column in data.columns:
        volume_resampled = data[volume_column].resample(new_interval).sum()
        resampled[volume_column] = volume_resampled

    return resampled


def align_data(
    data_frames: List[pd.DataFrame],
    join_method: str = 'inner'
) -> List[pd.DataFrame]:
    """
    Align multiple DataFrames to the same date index.

    Args:
        data_frames: List of DataFrames to align
        join_method: Join method ('inner', 'outer', 'left', 'right')

    Returns:
        List of aligned DataFrames
    """
    if not data_frames:
        return []

    # Ensure all DataFrames have DatetimeIndex
    for i, df in enumerate(data_frames):
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(
                f"DataFrame {i} index is not DatetimeIndex. Attempting conversion.")
            data_frames[i] = df.copy()
            data_frames[i].index = pd.to_datetime(df.index)

    # Get all unique dates
    if join_method == 'inner':
        # Inner join - keep only dates present in all DataFrames
        common_dates = data_frames[0].index
        for df in data_frames[1:]:
            common_dates = common_dates.intersection(df.index)

    elif join_method == 'outer':
        # Outer join - keep all dates
        common_dates = data_frames[0].index
        for df in data_frames[1:]:
            common_dates = common_dates.union(df.index)

    elif join_method == 'left':
        # Left join - keep dates from first DataFrame
        common_dates = data_frames[0].index

    elif join_method == 'right':
        # Right join - keep dates from last DataFrame
        common_dates = data_frames[-1].index

    else:
        logger.error(f"Unsupported join method: {join_method}")
        return data_frames

    # Reindex all DataFrames to common dates
    aligned_dfs = []
    for df in data_frames:
        aligned_dfs.append(df.reindex(common_dates))

    return aligned_dfs


def detect_outliers(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'zscore',
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detect outliers in data.

    Args:
        data: DataFrame with data
        columns: List of columns to check for outliers (None = all numeric)
        method: Outlier detection method ('zscore', 'iqr', 'percentile')
        threshold: Threshold for outlier detection

    Returns:
        DataFrame with boolean mask (True = outlier)
    """
    # Use all numeric columns if none specified
    if columns is None:
        columns = data.select_dtypes(include=np.number).columns.tolist()
    else:
        # Filter to only include columns that exist and are numeric
        columns = [col for col in columns if col in data.columns and
                   np.issubdtype(data[col].dtype, np.number)]

    if not columns:
        logger.warning("No numeric columns found for outlier detection")
        return pd.DataFrame(False, index=data.index, columns=data.columns)

    # Initialize result with False for all cells
    outliers = pd.DataFrame(False, index=data.index, columns=data.columns)

    # Detect outliers for each column
    for col in columns:
        if method == 'zscore':
            # Z-score method
            mean = data[col].mean()
            std = data[col].std()

            if std == 0:
                continue  # Skip if standard deviation is zero

            z_scores = (data[col] - mean) / std
            outliers[col] = abs(z_scores) > threshold

        elif method == 'iqr':
            # Interquartile range method
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue  # Skip if IQR is zero

            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            outliers[col] = (data[col] < lower_bound) | (
                data[col] > upper_bound)

        elif method == 'percentile':
            # Percentile method
            lower_bound = data[col].quantile(threshold / 100)
            upper_bound = data[col].quantile(1 - threshold / 100)

            outliers[col] = (data[col] < lower_bound) | (
                data[col] > upper_bound)

        else:
            logger.error(f"Unsupported outlier detection method: {method}")
            return outliers

    return outliers


def handle_outliers(
    data: pd.DataFrame,
    outliers: pd.DataFrame,
    method: str = 'winsorize',
    **kwargs
) -> pd.DataFrame:
    """
    Handle outliers in data.

    Args:
        data: DataFrame with data
        outliers: Boolean mask of outliers (True = outlier)
        method: Method to handle outliers ('winsorize', 'clip', 'remove', 'fillna')
        **kwargs: Additional arguments for specific methods

    Returns:
        DataFrame with handled outliers
    """
    result = data.copy()

    if method == 'winsorize':
        # Winsorize outliers (replace with threshold values)
        for col in outliers.columns:
            if outliers[col].any():
                # Get column data
                col_data = result[col]

                # Skip non-numeric columns
                if not np.issubdtype(col_data.dtype, np.number):
                    continue

                # Get outlier indices
                outlier_indices = outliers[col]

                # Get valid (non-outlier) data
                valid_data = col_data[~outlier_indices]

                if valid_data.empty:
                    continue

                # Get threshold values (5th and 95th percentiles of valid data)
                lower_bound = kwargs.get('lower_percentile', 5)
                upper_bound = kwargs.get('upper_percentile', 95)

                lower_threshold = valid_data.quantile(lower_bound / 100)
                upper_threshold = valid_data.quantile(upper_bound / 100)

                # Replace outliers with threshold values
                result.loc[col_data < lower_threshold, col] = lower_threshold
                result.loc[col_data > upper_threshold, col] = upper_threshold

    elif method == 'clip':
        # Clip values to threshold
        for col in outliers.columns:
            if outliers[col].any():
                # Get column data
                col_data = result[col]

                # Skip non-numeric columns
                if not np.issubdtype(col_data.dtype, np.number):
                    continue

                # Get valid (non-outlier) data
                valid_data = col_data[~outliers[col]]

                if valid_data.empty:
                    continue

                # Get threshold values
                min_val = kwargs.get('min_val', valid_data.min())
                max_val = kwargs.get('max_val', valid_data.max())

                # Clip values
                result[col] = result[col].clip(min_val, max_val)

    elif method == 'remove':
        # Remove rows with outliers
        mask = outliers.any(axis=1)
        result = result[~mask]

    elif method == 'fillna':
        # Replace outliers with NaN, then fill using specified method
        fill_method = kwargs.get('fill_method', 'ffill')

        for col in outliers.columns:
            if outliers[col].any():
                # Replace outliers with NaN
                result.loc[outliers[col], col] = np.nan

                # Fill NaN values
                if fill_method == 'ffill':
                    result[col] = result[col].ffill()
                elif fill_method == 'bfill':
                    result[col] = result[col].bfill()
                elif fill_method == 'interpolate':
                    result[col] = result[col].interpolate(method='linear')
                elif fill_method == 'mean':
                    result[col] = result[col].fillna(result[col].mean())
                elif fill_method == 'median':
                    result[col] = result[col].fillna(result[col].median())
                else:
                    logger.warning(f"Unsupported fill method: {fill_method}")

    else:
        logger.error(f"Unsupported outlier handling method: {method}")

    return result


# --- Main Data Loading Function ---

def load_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    data_source: str = DataSource.YAHOO,
    api_key: Optional[str] = None,
    adjust_prices: bool = True,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    handle_missing: bool = True,
    detect_outliers_method: Optional[str] = None,
    verbose: bool = True
) -> Optional[pd.DataFrame]:
    """
    Main function to load market data.

    Args:
        ticker: Ticker symbol
        start_date: Start date for data (YYYY-MM-DD)
        end_date: End date for data (YYYY-MM-DD)
        interval: Data interval (e.g., '1d', '1h', '5m')
        data_source: Data source (yahoo, alpha_vantage, csv, crypto, datareader)
        api_key: API key for paid data sources
        adjust_prices: Adjust prices for splits/dividends
        cache_dir: Directory to cache data
        use_cache: Whether to use cached data
        handle_missing: Whether to handle missing data
        detect_outliers_method: Method to detect outliers (None = disabled)
        verbose: Whether to log detailed information

    Returns:
        DataFrame with market data, or None if loading fails
    """
    # Configure logging level
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Create configuration
    config = DataLoaderConfig(
        data_source=data_source,
        api_key=api_key,
        cache_dir=cache_dir,
        use_cache=use_cache,
        default_start_date=start_date,
        default_end_date=end_date,
        default_interval=interval,
        adjust_prices=adjust_prices
    )

    # Create data loader
    loader = DataLoader(config)

    # Load data
    data = loader.load_data(ticker=ticker)

    if data is None or data.empty:
        logger.error(f"Failed to load data for {ticker}")
        return None

    # Handle missing data if requested
    if handle_missing and data is not None:
        # Check for missing values
        missing = data.isnull().sum().sum()

        if missing > 0:
            logger.info(f"Handling {missing} missing values")

            # Forward fill missing values
            data = data.ffill()

            # Back fill any remaining missing values
            data = data.bfill()

    # Detect and handle outliers if requested
    if detect_outliers_method and data is not None:
        logger.info(
            f"Detecting outliers using {detect_outliers_method} method")

        # Detect outliers
        outliers = detect_outliers(
            data=data,
            method=detect_outliers_method
        )

        # Check if any outliers found
        if outliers.any().any():
            logger.info(f"Handling {outliers.sum().sum()} outliers")

            # Handle outliers
            data = handle_outliers(
                data=data,
                outliers=outliers,
                method='fillna',
                fill_method='interpolate'
            )

    return data
