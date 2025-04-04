"""
CSV data source implementation.
"""

import pandas as pd
import logging
from typing import Optional

# Setup logging
logger = logging.getLogger(__name__)


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