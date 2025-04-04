"""
Data transformation utilities for preprocessing market data.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)


def resample_data(
    data: pd.DataFrame,
    new_interval: str,
    ohlc_columns: List[str] = ["open", "high", "low", "close"],
    volume_column: Optional[str] = "volume",
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
        logger.warning("Data index is not DatetimeIndex. Attempting conversion.")
        # Try converting, but operate on a copy to avoid SettingWithCopyWarning
        data = data.copy()
        data.index = pd.to_datetime(data.index)

    # --- Refactored Logic ---
    # Create a new DataFrame to hold standardized OHLC data
    ohlc_std = pd.DataFrame(index=data.index)

    # Map provided ohlc_columns to standard names ('open', 'high', 'low', 'close')
    # Use the first 4 provided column names if they exist in the input data
    col_map = {}
    standard_names = ["open", "high", "low", "close"]
    for i, std_name in enumerate(standard_names):
        if i < len(ohlc_columns) and ohlc_columns[i] in data.columns:
            col_map[std_name] = ohlc_columns[i]
        else:
            col_map[std_name] = None  # Mark as missing initially

    # Populate the standardized DataFrame
    for std_name, original_col in col_map.items():
        if original_col:
            ohlc_std[std_name] = data[original_col]

    # Apply fallbacks for missing standard columns
    fallback_source = None
    if col_map["close"] is not None:  # Prefer close if available
        fallback_source = col_map["close"]
    elif col_map["open"] is not None:  # Else use open
        fallback_source = col_map["open"]

    if fallback_source:
        for std_name, original_col in col_map.items():
            if original_col is None:  # If the standard column was missing
                logger.warning(
                    f"Missing standard column '{std_name}'. Using fallback '{fallback_source}'."
                )
                ohlc_std[std_name] = data[
                    fallback_source
                ]  # Use original data for fallback
    else:
        # If neither open nor close exists, we can't really do OHLC resampling
        logger.error(
            "Cannot perform OHLC resampling: requires at least 'open' or 'close' column."
        )
        # Return original data or empty? Let's return empty for consistency.
        return pd.DataFrame()  # Correct indentation

    # Define the standard aggregation rules
    agg_rules = {"open": "first", "high": "max", "low": "min", "close": "last"}

    # Filter rules to only include columns present in our standardized df
    dynamic_agg_rules = {
        col: rule for col, rule in agg_rules.items() if col in ohlc_std.columns
    }

    if not dynamic_agg_rules:
        logger.error("No valid columns for OHLC resampling after fallbacks.")
        resampled = pd.DataFrame()
    else:
        # Resample the standardized data
        resampled = ohlc_std.resample(new_interval).agg(dynamic_agg_rules)

    # Add volume if present and requested
    if volume_column and volume_column in data.columns:
        volume_resampled = data[volume_column].resample(new_interval).sum()
        # Ensure resampled is not empty before trying to assign volume
        if not resampled.empty:
            resampled[volume_column] = volume_resampled

    return resampled


def align_data(
    data_frames: List[pd.DataFrame], join_method: str = "inner"
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
                f"DataFrame {i} index is not DatetimeIndex. Attempting conversion."
            )
            data_frames[i] = df.copy()
            data_frames[i].index = pd.to_datetime(df.index)

    # Get all unique dates
    if join_method == "inner":
        # Inner join - keep only dates present in all DataFrames
        common_dates = data_frames[0].index
        for df in data_frames[1:]:
            common_dates = common_dates.intersection(df.index)

    elif join_method == "outer":
        # Outer join - keep all dates
        common_dates = data_frames[0].index
        for df in data_frames[1:]:
            common_dates = common_dates.union(df.index)

    elif join_method == "left":
        # Left join - keep dates from first DataFrame
        common_dates = data_frames[0].index

    elif join_method == "right":
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
    method: str = "zscore",
    threshold: float = 3.0,
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
        columns = [
            col
            for col in columns
            if col in data.columns and np.issubdtype(data[col].dtype, np.number)
        ]

    if not columns:
        logger.warning("No numeric columns found for outlier detection")
        return pd.DataFrame(False, index=data.index, columns=data.columns)

    # Initialize result with False for all cells
    outliers = pd.DataFrame(False, index=data.index, columns=data.columns)

    # Detect outliers for each column
    for col in columns:
        if method == "zscore":
            # Z-score method
            mean = data[col].mean()
            std = data[col].std()

            if std == 0:
                continue  # Skip if standard deviation is zero

            z_scores = (data[col] - mean) / std
            outliers[col] = abs(z_scores) > threshold

        elif method == "iqr":
            # Interquartile range method
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue  # Skip if IQR is zero

            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            outliers[col] = (data[col] < lower_bound) | (data[col] > upper_bound)

        elif method == "percentile":
            # Percentile method
            lower_bound = data[col].quantile(threshold / 100)
            upper_bound = data[col].quantile(1 - threshold / 100)

            outliers[col] = (data[col] < lower_bound) | (data[col] > upper_bound)

        else:
            logger.error(f"Unsupported outlier detection method: {method}")
            return outliers

    return outliers


def handle_outliers(
    data: pd.DataFrame, outliers: pd.DataFrame, method: str = "winsorize", **kwargs
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

    if method == "winsorize":
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
                lower_bound = kwargs.get("lower_percentile", 5)
                upper_bound = kwargs.get("upper_percentile", 95)

                lower_threshold = valid_data.quantile(lower_bound / 100)
                upper_threshold = valid_data.quantile(upper_bound / 100)

                # Ensure column can handle float thresholds if necessary
                if pd.api.types.is_integer_dtype(result[col].dtype) and (
                    isinstance(lower_threshold, float) or isinstance(upper_threshold, float)
                ):
                    result[col] = result[col].astype(float)

                # Replace outliers with threshold values
                result.loc[col_data < lower_threshold, col] = lower_threshold
                result.loc[col_data > upper_threshold, col] = upper_threshold

    elif method == "clip":
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
                min_val = kwargs.get("min_val", valid_data.min())
                max_val = kwargs.get("max_val", valid_data.max())

                # Clip values
                result[col] = result[col].clip(min_val, max_val)

    elif method == "remove":
        # Remove rows with outliers
        mask = outliers.any(axis=1)
        result = result[~mask]

    elif method == "fillna":
        # Replace outliers with NaN, then fill using specified method
        fill_method = kwargs.get("fill_method", "ffill")

        for col in outliers.columns:
            if outliers[col].any():
                # Replace outliers with NaN
                result.loc[outliers[col], col] = np.nan

                # Fill NaN values
                if fill_method == "ffill":
                    result[col] = result[col].ffill()
                elif fill_method == "bfill":
                    result[col] = result[col].bfill()
                elif fill_method == "interpolate":
                    result[col] = result[col].interpolate(method="linear")
                elif fill_method == "mean":
                    result[col] = result[col].fillna(result[col].mean())
                elif fill_method == "median":
                    result[col] = result[col].fillna(result[col].median())
                else:
                    logger.warning(f"Unsupported fill method: {fill_method}")

    else:
        logger.error(f"Unsupported outlier handling method: {method}")

    return result


def preprocess_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """
    Performs initial data cleaning and standardization.

    Args:
        df: Raw OHLCV DataFrame (potentially MultiIndex from yfinance).
        ticker: The ticker symbol for which data is being processed.

    Returns:
        DataFrame with standardized columns and basic cleaning,
        or None if fails.
    """
    logger.info("--- Initial Data Preprocessing ---")  # Use logger
    if df is None or df.empty:
        logger.error("Input DataFrame is empty/None for preprocessing.")  # Use logger
        return None

    processed_df = df.copy()

    # --- Standardize Column Names (lowercase) ---
    # Handle potential MultiIndex from yfinance if multiple tickers were loaded
    if isinstance(processed_df.columns, pd.MultiIndex):
        logger.info(f"Extracting data for '{ticker}' from MultiIndex...")  # Use logger
        # Check if ticker is at level 1 (standard yfinance format)
        if ticker in processed_df.columns.get_level_values(1):
            try:
                idx = pd.IndexSlice
                processed_df = processed_df.loc[:, idx[:, ticker]].copy()
                processed_df.columns = processed_df.columns.droplevel(1)
                processed_df.columns = processed_df.columns.str.lower()
                logger.info(
                    f"Extracted {ticker} data from column level 1."
                )  # Use logger
            except Exception as e:
                logger.error(
                    f"Error extracting ticker {ticker} from level 1: {e}"
                )  # Use logger
                return None
        # Check if ticker is at level 0 (current test fixture format)
        elif ticker in processed_df.columns.get_level_values(0):
            try:
                processed_df = processed_df.xs(ticker, level=0, axis=1).copy()
                processed_df.columns = processed_df.columns.str.lower()
                logger.info(
                    f"Extracted {ticker} data from column level 0."
                )  # Use logger
            except KeyError:
                logger.error(f"Ticker {ticker} in level 0 but xs failed.")  # Use logger
                return None
            except Exception as e:
                logger.error(
                    f"Error extracting ticker {ticker} from level 0: {e}"
                )  # Use logger
                return None
        else:
            logger.error(
                f"Ticker {ticker} not found in MultiIndex "  # Use logger
                f"columns (levels 0 or 1). Available: {processed_df.columns}"
            )
            return None
    else:
        # Assume single index columns, convert to lowercase
        processed_df.columns = processed_df.columns.str.lower()

    # --- Basic Preprocessing Steps ---
    # 1. Handle missing values (example: forward fill)
    initial_nan_count = processed_df.isnull().sum().sum()
    processed_df.ffill(inplace=True)
    # Handle any remaining NaNs at the beginning (e.g., drop rows)
    processed_df.dropna(inplace=True)
    final_nan_count = processed_df.isnull().sum().sum()
    if initial_nan_count > 0:
        logger.info(
            f"Handled NaNs (Initial: {initial_nan_count}, "  # Use logger
            f"Final: {final_nan_count})"
        )

    # 2. Ensure correct data types (e.g., numeric for OHLCV)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")
        else:
            logger.warning(
                f"Column '{col}' not found for type conversion."
            )  # Use logger
    processed_df.dropna(inplace=True)  # Drop rows if coercion failed

    logger.info(
        f"Initial preprocessing complete. Shape: {processed_df.shape}"
    )  # Use logger
    return processed_df
