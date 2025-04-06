"""
Triple barrier method for labeling financial data.
"""

import logging

import numpy as np
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)


def get_triple_barrier_labels(
    prices: pd.Series,
    highs: pd.Series,
    lows: pd.Series,
    atr: pd.Series,
    atr_multiplier_tp: float,
    atr_multiplier_sl: float,
    max_holding_period: int,
    min_return_threshold: float = 0.0,
) -> pd.Series:
    """
    Enhanced Triple Barrier Method for labeling.

    Labels:
        1: Take Profit hit first
        -1: Stop Loss hit first
        0: Time barrier hit (neither TP nor SL within max_holding_period)

    Args:
        prices: Series of closing prices (used for entry reference)
        highs: Series of high prices (used to check TP)
        lows: Series of low prices (used to check SL)
        atr: Series of ATR values at entry time
        atr_multiplier_tp: Multiplier for Take Profit distance from entry
        atr_multiplier_sl: Multiplier for Stop Loss distance from entry
        max_holding_period: Maximum number of bars to hold the hypothetical trade
        min_return_threshold: Minimum return threshold to consider for time barrier (0=neutral)

    Returns:
        pd.Series containing the outcome labels (1, 0, -1) or NaN if calculation
        is not possible (e.g., near the end of the series)
    """
    n = len(prices)

    # Input validation
    if not all(len(s) == n for s in [highs, lows, atr]):
        raise ValueError("All input series must have the same length")

    # Initialize labels with NaN
    labels = pd.Series(np.nan, index=prices.index)

    # Calculate target levels for all points upfront
    tp_levels = prices + atr * atr_multiplier_tp
    sl_levels = prices - atr * atr_multiplier_sl

    # Iterate through each potential entry point
    for i in range(n - 1):  # Stop before the last row
        entry_idx = prices.index[i]
        entry_price = prices.iloc[i]
        tp = tp_levels.iloc[i]
        sl = sl_levels.iloc[i]

        # Determine the lookahead window (up to max_holding_period or end of data)
        lookahead_end_idx = min(i + 1 + max_holding_period, n)

        # Check if we have enough lookahead data
        if i + 1 >= n or lookahead_end_idx <= i + 1:
            continue

        window_highs = highs.iloc[i + 1 : lookahead_end_idx]
        window_lows = lows.iloc[i + 1 : lookahead_end_idx]
        window_closes = prices.iloc[i + 1 : lookahead_end_idx]

        # Find the first time TP or SL is hit within the window
        tp_hit_indices = window_highs[window_highs >= tp].index
        sl_hit_indices = window_lows[window_lows <= sl].index

        tp_hit_time = tp_hit_indices[0] if not tp_hit_indices.empty else None
        sl_hit_time = sl_hit_indices[0] if not sl_hit_indices.empty else None

        # Default outcome (time barrier)
        outcome = 0

        # Check for TP or SL hits
        if tp_hit_time is not None and sl_hit_time is not None:
            # Both hit, use the earliest
            outcome = (
                1
                if prices.index.get_loc(tp_hit_time)
                <= prices.index.get_loc(sl_hit_time)
                else -1
            )
        elif tp_hit_time is not None:
            outcome = 1  # Only TP hit
        elif sl_hit_time is not None:
            outcome = -1  # Only SL hit
        else:
            # Time barrier - check final return if threshold specified
            final_price = window_closes.iloc[-1]
            final_return = (final_price - entry_price) / entry_price

            if min_return_threshold > 0:
                if final_return >= min_return_threshold:
                    outcome = 1  # Positive return at time barrier
                elif final_return <= -min_return_threshold:
                    outcome = -1  # Negative return at time barrier
                else:
                    outcome = 0  # Neutral return at time barrier

        # Assign the calculated outcome to the entry point's index
        labels.loc[entry_idx] = outcome

    return labels
