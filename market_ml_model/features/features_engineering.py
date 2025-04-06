"""
Main feature engineering orchestration module.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from .labeling.triple_barrier import get_triple_barrier_labels

# Import components
from .technical.indicators import (
    calculate_fractal_features,
    calculate_market_regime_features,
    calculate_momentum_features,
    calculate_technical_indicators,
    calculate_volatility_features,
)

# Setup logging
logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame, feature_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Centralized and deterministic feature engineering based on configuration.

    Args:
        df: DataFrame with OHLCV data (must include 'open', 'high', 'low', 'close', 'volume').
            Index must be DatetimeIndex.
        feature_config: Dictionary defining features and target to generate.
            Keys include:
            * ``technical_indicators`` (list): Configurations for technical indicators.
              Example: ``[{'indicator': 'sma', 'length': 50}, ...]``
            * ``target_config`` (dict): Configuration for the target variable, containing:
                * ``type`` (str): 'triple_barrier' or 'directional'.
                * ``atr_multiplier_tp`` (float): For triple_barrier.
                * ``atr_multiplier_sl`` (float): For triple_barrier.
                * ``max_holding_period`` (int): Max bars for target calculation.
                * ``min_return_threshold`` (float, optional): For triple_barrier.
                * ``threshold`` (float): For directional.

    Returns:
        DataFrame with added features and target labels, ready for model input.
        NaN values in features are handled using forward fill followed by zero fill
        before target calculation.
        Returns None if input is invalid or processing fails.
    """
    logger.info("Starting centralized feature engineering")

    if df is None or df.empty:
        logger.error("Input DataFrame is empty/None for feature engineering.")
        return None
    if feature_config is None:
        logger.error("`feature_config` is required for feature engineering.")
        return None
    if not isinstance(feature_config, dict):
        logger.error("`feature_config` must be a dictionary.")
        return None

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning(
            "Input DataFrame index is not a DatetimeIndex. Feature engineering requires a DatetimeIndex."
        )
        # Attempt conversion or return None if strict type checking is preferred
        try:
            df.index = pd.to_datetime(df.index)
            logger.info("Successfully converted index to DatetimeIndex.")
        except Exception as e:
            logger.error(
                f"Failed to convert index to DatetimeIndex: {e}. Aborting feature engineering."
            )
            return None

    # Work on a copy
    processed_df = df.copy()

    # --- Configuration Extraction ---
    # Expecting a list of dicts for technical indicators, matching the YAML structure
    technical_indicator_configs = feature_config.get("technical_indicators", [])
    target_config = feature_config.get("target_config", {})
    target_type = target_config.get(
        "type", "triple_barrier"
    )  # Default if not specified

    logger.debug(f"Using technical indicator configs: {technical_indicator_configs}")
    logger.debug(f"Using target config: {target_config}")

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

    # 4. Technical Indicators based on indicator_configs
    # The `calculate_technical_indicators` function needs to be adapted
    # or called appropriately based on the structure of indicator_configs.
    # Assuming calculate_technical_indicators can handle a list of indicator types for now.
    # A more robust approach might involve iterating through indicator_configs
    # and calling specific calculation functions or a modified calculate_technical_indicators.

    # Pass the list of configurations directly to the updated function
    if technical_indicator_configs:
        logger.info(
            f"Calculating technical indicators based on config: {len(technical_indicator_configs)} indicators specified."
        )
        processed_df = calculate_technical_indicators(
            processed_df,
            indicator_configs=technical_indicator_configs,  # Pass the list of dicts
        )
    else:
        logger.warning(
            "No indicators specified in feature_config['technical_indicators']. Skipping technical indicators."
        )

    # 5. Additional Feature Categories (Volatility, Regime, Momentum, Fractals)
    feature_sets_to_calculate = {
        "volatility": (
            feature_config.get("volatility_features", False),
            calculate_volatility_features,
        ),
        "regime": (
            feature_config.get("regime_features", False),
            calculate_market_regime_features,
        ),
        "momentum": (
            feature_config.get("momentum_features", False),
            calculate_momentum_features,
        ),
        "fractal": (
            feature_config.get("fractal_features", False),
            calculate_fractal_features,
        ),
    }

    # Use a temporary copy for calculations within this block to avoid modifying
    # the df mid-loop in ways that affect subsequent calculations if they rely on
    # the state *before* adding features from a previous set in this loop.
    temp_df_for_calcs = processed_df.copy()

    for name, (enabled, func) in feature_sets_to_calculate.items():
        if enabled:
            logger.info(f"Calculating {name} features...")
            try:
                # Pass the temporary df copy to the calculation function
                # Allow fractal window config, default 5
                if name == "fractal":
                    fractal_window = feature_config.get("fractal_window", 5)
                    feature_results = func(
                        temp_df_for_calcs, window=fractal_window
                    )  # Pass window
                else:
                    feature_results = func(temp_df_for_calcs)

                if feature_results:
                    added_count = 0
                    for col_name, series in feature_results.items():
                        if col_name in processed_df.columns:
                            logger.warning(
                                f"Column '{col_name}' from {name} features already exists. Overwriting."
                            )
                        # Add the result back to the main processed_df
                        processed_df[col_name] = series.reindex(
                            processed_df.index
                        )  # Ensure index alignment
                        added_count += 1
                    logger.info(f"Successfully added {added_count} {name} features.")
                else:
                    logger.warning(
                        f"{name.capitalize()} feature calculation returned empty results."
                    )
            except Exception as e:
                logger.error(f"Error calculating {name} features: {e}", exc_info=True)

    # 6. Feature Differencing (Optional)
    differencing_config = feature_config.get("differencing", {})
    if differencing_config.get("apply", False):
        period = differencing_config.get("period", 1)
        features_to_diff = differencing_config.get("features", [])
        if features_to_diff:
            logger.info(
                f"Applying differencing (period={period}) to features: {features_to_diff}"
            )
            diff_count = 0
            for feature_name in features_to_diff:
                if feature_name in processed_df.columns:
                    try:
                        # Overwrite original column with differenced version
                        processed_df[feature_name] = processed_df[feature_name].diff(
                            periods=period
                        )
                        diff_count += 1
                    except Exception as e:
                        logger.error(
                            f"Error applying differencing to '{feature_name}': {e}"
                        )
                else:
                    logger.warning(
                        f"Feature '{feature_name}' specified for differencing not found."
                    )
            logger.info(f"Applied differencing to {diff_count} features.")
        else:
            logger.warning(
                "Differencing enabled but no features specified in 'differencing.features'."
            )
    # --- NaN Handling for Features (Before Target Calculation) ---
    # Apply after all feature calculations but before target definition.
    # This handles NaNs in generated features but allows specific handling for target inputs.
    logger.info(
        f"Applying forward fill and zero fill for NaNs in FEATURES before target calculation. Shape: {processed_df.shape}"
    )
    initial_feature_nan_count = processed_df.isna().sum().sum()
    if initial_feature_nan_count > 0:
        # Apply forward fill first
        processed_df.fillna(method="ffill", inplace=True)
        # Apply zero fill for any remaining NaNs (usually at the beginning)
        processed_df.fillna(0, inplace=True)
        final_feature_nan_count = processed_df.isna().sum().sum()
        logger.info(
            f"Feature NaNs before fill: {initial_feature_nan_count}, Feature NaNs after fill: {final_feature_nan_count}"
        )
    else:
        logger.info("No NaNs found in features before target calculation.")

    # --- Define Target Variable ---
    # --- Define Target Variable ---

    # Triple Barrier Method
    if target_type == "triple_barrier":
        # Extract parameters from target_config
        atr_multiplier_tp = target_config.get("atr_multiplier_tp", 2.0)
        atr_multiplier_sl = target_config.get("atr_multiplier_sl", 1.0)
        max_holding_period = target_config.get("max_holding_period", 10)
        min_return_threshold = target_config.get("min_return_threshold", 0.001)
        # Dynamically determine ATR column name based on config or default
        # Check if 'technical_indicators' config specifies an ATR period to use
        atr_period_from_indicators = None
        for ind_conf in technical_indicator_configs:
            if ind_conf.get("indicator", "").lower() == "atr":
                atr_period_from_indicators = ind_conf.get(
                    "length", 14
                )  # Default pandas-ta ATR length is 14
                break
        # Prioritize target_config setting, then indicator config, then default 10
        atr_period = target_config.get(
            "atr_period",
            (
                atr_period_from_indicators
                if atr_period_from_indicators is not None
                else 10
            ),
        )
        atr_col_name = f"ATRr_{atr_period}"  # Construct the column name dynamically

        logger.info(
            f"Calculating Triple Barrier Labels (TP: {atr_multiplier_tp}*ATR, "
            f"SL: {atr_multiplier_sl}*ATR, Max Hold: {max_holding_period} bars, ATR Col: {atr_col_name})"
        )
        required_cols = ["close", "high", "low", atr_col_name]

        if not all(col in processed_df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in processed_df.columns]
            logger.warning(
                f"Missing required columns for Triple Barrier: {missing}. Skipping Triple Barrier labeling."
            )
            # Skip labeling if required columns (likely from pandas-ta) are missing
        else:
            # --- Handle NaNs in the specific ATR column before labeling ---
            initial_rows = len(processed_df)
            processed_df.dropna(subset=[atr_col_name], inplace=True)
            rows_after_drop = len(processed_df)
            if initial_rows > rows_after_drop:
                logger.info(
                    f"Dropped {initial_rows - rows_after_drop} rows due to NaNs in '{atr_col_name}' before triple barrier calculation."
                )

            if processed_df.empty:
                logger.warning(
                    f"DataFrame is empty after dropping NaNs in '{atr_col_name}'. Skipping Triple Barrier labeling."
                )
            else:
                # Calculate labels using the cleaned data
                processed_df["triple_barrier_label"] = get_triple_barrier_labels(
                    prices=processed_df["close"],
                    highs=processed_df["high"],
                    lows=processed_df["low"],
                    atr=processed_df[
                        atr_col_name
                    ],  # Use dynamically determined and cleaned ATR column
                    atr_multiplier_tp=atr_multiplier_tp,
                    atr_multiplier_sl=atr_multiplier_sl,
                    max_holding_period=max_holding_period,
                    min_return_threshold=min_return_threshold,
                )

            # Handle NaNs and convert type (fill with 0 for neutral/undetermined)
            if "triple_barrier_label" in processed_df.columns:
                processed_df["triple_barrier_label"] = pd.to_numeric(
                    processed_df["triple_barrier_label"], errors="coerce"
                )
                processed_df["triple_barrier_label"] = processed_df[
                    "triple_barrier_label"
                ].replace([np.inf, -np.inf], np.nan)
                processed_df["triple_barrier_label"] = processed_df[
                    "triple_barrier_label"
                ].fillna(0)
                processed_df["triple_barrier_label"] = processed_df[
                    "triple_barrier_label"
                ].astype(int)
            else:
                logger.warning("Column 'triple_barrier_label' not created.")

        # Removed the duplicated block and the aggressive dropna for labels.
        # NaNs are now filled with 0 above.

    # Simple directional prediction (for comparison)
    elif target_type == "directional":
        # Extract parameters from target_config
        max_holding_period = target_config.get("max_holding_period", 10)
        threshold = target_config.get("threshold", 0.001)  # Significance threshold

        logger.info(
            f"Calculating directional target (Max Hold: {max_holding_period}, Threshold: {threshold})"
        )
        if "close" not in processed_df.columns:
            logger.error("Missing 'close' column for directional target")
            return None

        # Future N-bar return
        processed_df["future_return"] = (
            processed_df["close"].shift(-max_holding_period) / processed_df["close"] - 1
        )

        # Convert to classification target based on threshold
        # threshold = 0.001  # Now extracted from config
        processed_df["directional_label"] = 0  # Initialize as neutral
        processed_df.loc[
            processed_df["future_return"] > threshold, "directional_label"
        ] = 1
        processed_df.loc[
            processed_df["future_return"] < -threshold, "directional_label"
        ] = -1

        # Drop rows where target is NaN
        # Handle NaNs and convert type (fill with 0 for neutral/undetermined)
        if "directional_label" in processed_df.columns:
            processed_df["directional_label"] = processed_df[
                "directional_label"
            ].fillna(0)
            processed_df["directional_label"] = processed_df[
                "directional_label"
            ].astype(int)
        else:
            logger.warning("Column 'directional_label' not created.")

        # Drop rows where future_return is NaN (which caused directional_label NaN before fillna)
        processed_df.dropna(subset=["future_return"], inplace=True)
        processed_df.drop(
            columns=["future_return"], inplace=True
        )  # Remove intermediate column

    # --- Clean up and finalize ---

    # Sanitize column names
    processed_df.columns = (
        processed_df.columns.str.replace(r"[\s\(\)\%\.]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)  # Consolidate underscores
        .str.strip("_")  # Remove leading/trailing underscores
    )

    # NOTE: Final NaN handling moved to the end of the function
    logger.info(
        f"Feature and target engineering complete. DataFrame shape before returning: {processed_df.shape}"
    )

    # --- Final NaN Handling (Removed - Now handled earlier) ---
    return processed_df
