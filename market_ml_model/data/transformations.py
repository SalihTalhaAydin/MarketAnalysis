"""
Data transformation utilities for preprocessing market data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

# Setup logging
logger = logging.getLogger(__name__)


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