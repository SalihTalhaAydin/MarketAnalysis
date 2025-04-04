"""
Data caching functionality.
"""
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataCache:
    """Handles caching of dataframes to disk."""

    def __init__(self, cache_dir: str, expiry_days: int = 1):
        """
        Initialize the data cache.

        Args:
            cache_dir: Directory to store cache files.
            expiry_days: Number of days before cache expires.
        """
        self.cache_dir = cache_dir
        self.expiry_days = expiry_days
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"DataCache initialized. Cache directory: {self.cache_dir}, Expiry: {self.expiry_days} days")

    def get_cache_path(
        self, ticker: str, start_date: str, end_date: str, interval: str
    ) -> str:
        """
        Generate a unique cache file path based on request parameters.

        Args:
            ticker: Ticker symbol.
            start_date: Start date string.
            end_date: End date string.
            interval: Data interval string.

        Returns:
            The full path for the cache file.
        """
        filename = f"{ticker}_{start_date}_{end_date}_{interval}.pkl"
        return os.path.join(self.cache_dir, filename)

    def get_cached_data(self, cache_path: str) -> Optional[pd.DataFrame]:
        """
        Retrieve data from cache if it exists and hasn't expired.

        Args:
            cache_path: Path to the cache file.

        Returns:
            Cached DataFrame or None if not found or expired.
        """
        if not os.path.exists(cache_path):
            logger.debug(f"Cache miss (file not found): {cache_path}")
            return None

        # Check cache expiry
        try:
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - file_mod_time > timedelta(days=self.expiry_days):
                logger.info(f"Cache expired: {cache_path}. Removing.")
                os.remove(cache_path)
                return None
        except OSError as e:
            logger.warning(f"Error checking cache file modification time for {cache_path}: {e}")
            return None # Treat as expired if we can't check time

        # Load data from cache
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            logger.debug(f"Cache hit: {cache_path}")
            return data
        except (pickle.UnpicklingError, EOFError, OSError) as e:
            logger.error(f"Error loading data from cache file {cache_path}: {e}. Removing corrupt file.")
            try:
                os.remove(cache_path)
            except OSError as remove_err:
                logger.error(f"Failed to remove corrupt cache file {cache_path}: {remove_err}")
            return None

    def save_to_cache(self, data: pd.DataFrame, cache_path: str) -> None:
        """
        Save DataFrame to a cache file using pickle.

        Args:
            data: DataFrame to cache.
            cache_path: Path to save the cache file.
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            logger.warning(f"Attempted to cache invalid or empty data for {cache_path}. Skipping.")
            return

        try:
            # Ensure the directory exists (might be redundant but safe)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Data saved to cache: {cache_path}")
        except (pickle.PicklingError, OSError) as e:
            logger.error(f"Error saving data to cache file {cache_path}: {e}")