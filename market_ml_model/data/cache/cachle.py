"""
Data caching functionality to avoid repeated API calls.
"""

import os
import time
import pandas as pd
import logging
from typing import Optional

# Setup logging
logger = logging.getLogger(__name__)


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