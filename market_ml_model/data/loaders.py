"""
Core data loading functionality.
"""

import pandas as pd
import numpy as np
import logging
import time
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import json

# Import data source modules
from .sources.yahoo import load_from_yahoo
from .sources.alpha_vantage import load_from_alpha_vantage
from .sources.csv_loader import load_from_csv
from .sources.crypto import load_from_crypto
from .sources.datareader import load_from_datareader
from .sources.data_source import DataSource

# Import cache module
# from .cache.cache import DataCache # Commented out due to missing module/file

# Import transformations
from .transformations import resample_data, align_data, detect_outliers, handle_outliers

# Setup logging
logger = logging.getLogger(__name__)


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


class DataLoader:
    """Advanced data loader with multiple data sources and caching."""

    def __init__(self, config: Optional[DataLoaderConfig] = None):
        """
        Initialize data loader.

        Args:
            config: Configuration for data loader
        """
        self.config = config or DataLoaderConfig()

        # Initialize cache if enabled
        # if self.config.use_cache and self.config.cache_dir:
        #     self.cache = DataCache(
        #         cache_dir=self.config.cache_dir,
        #         expiry_days=self.config.cache_expiry_days
        #     )
        # else:
        self.cache = None # Keep self.cache defined as None for now

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
        # if self.cache and not force_reload:
        #     cache_path = self.cache.get_cache_path(
        #         ticker, start_date, end_date, interval)
        #     cached_data = self.cache.get_cached_data(cache_path)
        #
        #     if cached_data is not None:
        #         logger.info(f"Retrieved {ticker} data from cache")
        #         return cached_data

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
        # if data is not None and self.cache:
        #     cache_path = self.cache.get_cache_path(
        #         ticker, start_date, end_date, interval)
        #     self.cache.save_to_cache(data, cache_path)

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