"""
Main feature engineering orchestration module.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .labeling.triple_barrier import get_triple_barrier_labels

# Import components
from .technical.indicators import calculate_technical_indicators

# Setup logging
logger = logging.getLogger(__name__)


def engineer_features(
    df: pd.DataFrame,
    atr_multiplier_tp: float = 2.0,
    atr_multiplier_sl: float = 1.0,
    max_holding_period: int = 10,
    target_type: str = "triple_barrier",
    additional_features: Optional[List[str]] = None,
    feature_config: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Enhanced feature engineering with multiple feature types and configurable generation.

    Args:
        df: Preprocessed DataFrame with OHLCV data
        atr_multiplier_tp: Multiplier for Take Profit distance based on ATR
        atr_multiplier_sl: Multiplier for Stop Loss distance based on ATR
        max_holding_period: Maximum bars for the time barrier
        target_type: Type of target to generate ('triple_barrier', 'directional', etc.)
        additional_features: List of additional feature categories to generate
        feature_config: Dictionary with feature configuration parameters

    Returns:
        DataFrame with added features and target labels
    """
    logger.info("Starting enhanced feature engineering")

    if df is None or df.empty:
        logger.error("Input DataFrame is empty/None for feature engineering")
        return None

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning(
            "Index must be DatetimeIndex for time features, attempting conversion"
        )
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logger.error(f"Failed to convert index to DatetimeIndex: {e}")
            return None

    # Work on a copy
    processed_df = df.copy()

    # Default feature configuration
    if feature_config is None:
        feature_config = {
            "use_ta_lib": True,
            "use_advanced_features": True,
            "volatility_windows": [5, 10, 20, 50],
            "momentum_windows": [7, 14, 21],
            "trend_windows": [20, 50, 200],
        }

    # Default additional features
    if additional_features is None:
        additional_features = [
            "basic_ta",
            "volatility",
            "trend",
            "momentum",
            "support_resistance",
        ]

    # --- Feature Engineering Steps ---

    # 1. Calculate basic returns
    if "close" in processed_df.columns:
        processed_df["return"] = processed_df["close"].pct_change()
        processed_df["log_return"] = np.log(
            processed_df["close"] / processed_df["close"].shift(1)
        )
    else:
        logger.warning("'close' column not found, skipping return calculation")

    # 2. Calculate lagged features
    if "close" in processed_df.columns:
        processed_df["close_lag1"] = processed_df["close"].shift(1)

    # 3. Time-based features
    processed_df["day_of_week"] = processed_df.index.dayofweek
    processed_df["hour_of_day"] = (
        processed_df.index.hour if hasattr(processed_df.index, "hour") else 0
    )

    # 4. Technical Indicators
    if "basic_ta" in additional_features:
        logger.info("Calculating technical indicators")
        processed_df = calculate_technical_indicators(
            processed_df,
            indicators=["sma", "ema", "rsi", "macd", "bbands", "atr", "adx", "stoch"],
        )

    # 5. Volume-based indicators
    if "volume_indicators" in additional_features and "volume" in processed_df.columns:
        logger.info("Calculating volume indicators")
        processed_df = calculate_technical_indicators(
            processed_df, indicators=["volume_indicators"]
        )

    # --- Define Target Variable ---

    # Triple Barrier Method
    if target_type == "triple_barrier":
        logger.info(
            f"Calculating Triple Barrier Labels (TP: {atr_multiplier_tp}*ATR, "
            f"SL: {atr_multiplier_sl}*ATR, Max Hold: {max_holding_period} bars)"
        )
        required_cols = ["close", "high", "low", "ATRr_10"]

        if not all(col in processed_df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in processed_df.columns]
            logger.error(f"Missing required columns for Triple Barrier: {missing}")
            return None

        # Drop NaNs from features before calculating labels to ensure alignment
        feature_cols = processed_df.columns.difference(["triple_barrier_label"])
        processed_df.dropna(subset=feature_cols, inplace=True)

        if processed_df.empty:
            logger.error("DataFrame empty after dropping NaNs from features")
            return None

        # Calculate labels on the feature-cleaned data
        processed_df["triple_barrier_label"] = get_triple_barrier_labels(
            prices=processed_df["close"],
            highs=processed_df["high"],
            lows=processed_df["low"],
            atr=processed_df["ATRr_10"],
            atr_multiplier_tp=atr_multiplier_tp,
            atr_multiplier_sl=atr_multiplier_sl,
            max_holding_period=max_holding_period,
            min_return_threshold=0.001,  # Small threshold to filter noise
        )

        # Drop rows where the label could not be calculated
        initial_rows = len(processed_df)
        processed_df.dropna(subset=["triple_barrier_label"], inplace=True)
        rows_dropped = initial_rows - len(processed_df)

        if rows_dropped > 0:
            logger.info(
                f"Dropped {rows_dropped} rows due to NaNs from Triple Barrier calculation"
            )

        # Convert label to integer type after dropping NaNs
        if "triple_barrier_label" in processed_df.columns:
            processed_df["triple_barrier_label"] = processed_df[
                "triple_barrier_label"
            ].astype(int)

    # Simple directional prediction (for comparison)
    elif target_type == "directional":
        logger.info("Calculating simple directional target variable")
        if "close" not in processed_df.columns:
            logger.error("Missing 'close' column for directional target")
            return None

        # Future N-bar return
        processed_df["future_return"] = (
            processed_df["close"].shift(-max_holding_period) / processed_df["close"] - 1
        )

        # Convert to classification target based on threshold
        threshold = 0.001  # Minimum move to consider significant
        processed_df["directional_label"] = 0  # Initialize as neutral
        processed_df.loc[
            processed_df["future_return"] > threshold, "directional_label"
        ] = 1
        processed_df.loc[
            processed_df["future_return"] < -threshold, "directional_label"
        ] = -1

        # Drop rows where target is NaN
        processed_df.dropna(subset=["directional_label"], inplace=True)
        processed_df["directional_label"] = processed_df["directional_label"].astype(
            int
        )

    # --- Clean up and finalize ---

    # Sanitize column names
    original_cols = processed_df.columns
    processed_df.columns = (
        processed_df.columns.str.replace(r"[\s\(\)\%\.]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)  # Consolidate underscores
        .str.strip("_")  # Remove leading/trailing underscores
    )

    # NOTE: Final NaN handling moved to the end of the function
    logger.info(f"Feature engineering complete. DataFrame shape: {processed_df.shape}")

    # --- Final NaN Handling (Moved Here) ---
    # Final check for NaNs in the processed DataFrame just before returning
    nan_columns = processed_df.columns[processed_df.isna().any()].tolist()
    if nan_columns:
        logger.warning(
            f"NaN values present in columns after final processing: {nan_columns}. Filling..."
        )
        # Replace NaNs with appropriate values based on column type
        for col in nan_columns:
            if np.issubdtype(processed_df[col].dtype, np.number):
                # Check if the column is entirely NaN first
                if processed_df[col].isnull().all():
                    fill_value = 0
                else:
                    # For numerical columns, calculate median excluding NaNs
                    col_median = processed_df[col].median()
                    # If median is NaN (shouldn't happen now due to the check above) or 0, fill with 0, otherwise fill with median
                    fill_value = (
                        col_median if pd.notna(col_median) and col_median != 0 else 0
                    )
                processed_df[col] = processed_df[col].fillna(fill_value)
                logger.debug(f"Filled NaNs in numeric column '{col}' with {fill_value}")
            else:
                # For categorical/other columns, fill with 0 (or mode if appropriate)
                processed_df[col] = processed_df[col].fillna(
                    0
                )  # Using 0 as a simple default
                logger.debug(f"Filled NaNs in non-numeric column '{col}' with 0")

    return processed_df
