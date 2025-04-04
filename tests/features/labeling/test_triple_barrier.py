import numpy as np
import pandas as pd
import pytest

# Import the function to test
from market_ml_model.features.labeling.triple_barrier import get_triple_barrier_labels

# --- Fixtures ---


@pytest.fixture
def sample_data():
    """Provides sample price, high, low, and atr series."""
    dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
    prices = pd.Series(np.linspace(100, 110, 20), index=dates)
    highs = prices + 0.5
    lows = prices - 0.5
    atr = pd.Series(np.ones(20) * 0.5, index=dates)  # Constant ATR for simplicity
    return prices, highs, lows, atr


# --- Tests ---


def test_tp_hit_first(sample_data):
    """Test scenario where Take Profit is hit before Stop Loss or Time Barrier."""
    prices, highs, lows, atr = sample_data
    # Modify highs to ensure TP is hit quickly for the first entry
    highs_mod = highs.copy()
    highs_mod.iloc[2] = (
        105  # Entry price at index 0 is ~100, ATR=0.5, TP mult=2 -> TP ~101. TP hit at index 2
    )

    labels = get_triple_barrier_labels(
        prices=prices,
        highs=highs_mod,
        lows=lows,
        atr=atr,
        atr_multiplier_tp=2.0,  # TP = entry + 0.5 * 2.0 = entry + 1.0
        atr_multiplier_sl=2.0,  # SL = entry - 0.5 * 2.0 = entry - 1.0
        max_holding_period=5,
    )
    # Entry at index 0 (~100), TP ~101. High at index 2 is 105. Label should be 1.
    assert labels.iloc[0] == 1
    # Check the last label is NaN, and the second last is not (for max_holding_period=5)
    assert pd.isna(labels.iloc[-1])
    assert pd.notna(labels.iloc[-2])  # Should have calculated a label


def test_sl_hit_first(sample_data):
    """Test scenario where Stop Loss is hit before Take Profit or Time Barrier."""
    prices, highs, lows, atr = sample_data
    # Modify lows AND highs to ensure SL is hit first
    lows_mod = lows.copy()
    highs_mod = highs.copy()
    lows_mod.iloc[3] = 98  # SL ~99 is hit at index 3
    highs_mod.iloc[1:4] = (
        100.5  # Keep highs below TP (~101) up to and including index 3
    )

    labels = get_triple_barrier_labels(
        prices=prices,
        highs=highs_mod,  # Use modified highs
        lows=lows_mod,
        atr=atr,
        atr_multiplier_tp=2.0,  # TP ~ entry + 1.0
        atr_multiplier_sl=2.0,  # SL ~ entry - 1.0
        max_holding_period=5,
    )
    # Entry at index 0 (~100), SL ~99. Low at index 3 is 98. Highs kept below TP. Label should be -1.
    assert labels.iloc[0] == -1
    # Check the last label is NaN, and the second last is not
    assert pd.isna(labels.iloc[-1])
    assert pd.notna(labels.iloc[-2])


def test_time_barrier_hit_neutral(sample_data):
    """Test scenario where Time Barrier is hit, final return is neutral."""
    prices, highs, lows, atr = sample_data
    # Ensure prices stay within TP/SL bounds for the first entry's holding period
    prices_mod = prices.copy()
    prices_mod.iloc[1:7] = (
        100.1  # Keep price flat within bounds for entry 0 (TP~101, SL~99)
    )

    labels = get_triple_barrier_labels(
        prices=prices_mod,
        highs=prices_mod + 0.1,  # Keep highs/lows tight
        lows=prices_mod - 0.1,
        atr=atr,
        atr_multiplier_tp=2.0,  # TP ~ entry + 1.0
        atr_multiplier_sl=2.0,  # SL ~ entry - 1.0
        max_holding_period=5,
        min_return_threshold=0.005,  # Threshold = 0.5%
    )
    # Entry at index 0 (~100). Price at index 6 (end of period) is 100.1.
    # Return = (100.1 - 100) / 100 = 0.001, which is < threshold. Label should be 0.
    assert labels.iloc[0] == 0
    # Check the last label is NaN, and the second last is not
    assert pd.isna(labels.iloc[-1])
    assert pd.notna(labels.iloc[-2])


def test_time_barrier_hit_positive_return(sample_data):
    """Test scenario where Time Barrier is hit, final return meets positive threshold."""
    prices, highs, lows, atr = sample_data
    prices_mod = prices.copy()
    # Make price rise, but not enough to hit TP within 5 bars
    prices_mod.iloc[1:7] = np.linspace(100.1, 100.8, 6)  # Ends at 100.8

    labels = get_triple_barrier_labels(
        prices=prices_mod,
        highs=prices_mod + 0.1,
        lows=prices_mod - 0.1,
        atr=atr,
        atr_multiplier_tp=2.0,  # TP ~ 101
        atr_multiplier_sl=2.0,  # SL ~ 99
        max_holding_period=5,
        min_return_threshold=0.005,  # Threshold = 0.5%
    )
    # Entry at index 0 (~100). Price at index 6 is 100.8.
    # Return = (100.8 - 100) / 100 = 0.008, which is >= threshold. Label should be 1.
    assert labels.iloc[0] == 1
    # Check the last label is NaN, and the second last is not
    assert pd.isna(labels.iloc[-1])
    assert pd.notna(labels.iloc[-2])


def test_time_barrier_hit_negative_return(sample_data):
    """Test scenario where Time Barrier is hit, final return meets negative threshold."""
    prices, highs, lows, atr = sample_data
    prices_mod = prices.copy()
    # Make price fall, but not enough to hit SL within 5 bars
    prices_mod.iloc[1:7] = np.linspace(99.9, 99.2, 6)  # Ends at 99.2

    labels = get_triple_barrier_labels(
        prices=prices_mod,
        highs=prices_mod + 0.1,
        lows=prices_mod - 0.1,
        atr=atr,
        atr_multiplier_tp=2.0,  # TP ~ 101
        atr_multiplier_sl=2.0,  # SL ~ 99
        max_holding_period=5,
        min_return_threshold=0.005,  # Threshold = 0.5%
    )
    # Entry at index 0 (~100). Price at index 6 is 99.2.
    # Return = (99.2 - 100) / 100 = -0.008, which is <= -threshold. Label should be -1.
    assert labels.iloc[0] == -1
    # Check the last label is NaN, and the second last is not
    assert pd.isna(labels.iloc[-1])
    assert pd.notna(labels.iloc[-2])


def test_tp_sl_same_bar(sample_data):
    """Test scenario where TP and SL are hit on the same bar (TP should win)."""
    prices, highs, lows, atr = sample_data
    highs_mod = highs.copy()
    lows_mod = lows.copy()
    # Make TP and SL hit on bar 3 for entry 0
    highs_mod.iloc[3] = 105  # TP ~ 101
    lows_mod.iloc[3] = 95  # SL ~ 99

    labels = get_triple_barrier_labels(
        prices=prices,
        highs=highs_mod,
        lows=lows_mod,
        atr=atr,
        atr_multiplier_tp=2.0,
        atr_multiplier_sl=2.0,
        max_holding_period=5,
    )
    # TP hit time index (3) <= SL hit time index (3). Label should be 1.
    assert labels.iloc[0] == 1


def test_input_validation_length_mismatch(sample_data):
    """Test that ValueError is raised for input series of different lengths."""
    prices, highs, lows, atr = sample_data
    with pytest.raises(ValueError, match="All input series must have the same length"):
        get_triple_barrier_labels(
            prices=prices.iloc[:-1],  # Shorter prices
            highs=highs,
            lows=lows,
            atr=atr,
            atr_multiplier_tp=2.0,
            atr_multiplier_sl=2.0,
            max_holding_period=5,
        )


def test_no_lookahead_data(sample_data):
    """Test behavior when max_holding_period extends beyond available data."""
    prices, highs, lows, atr = sample_data
    labels = get_triple_barrier_labels(
        prices=prices,
        highs=highs,
        lows=lows,
        atr=atr,
        atr_multiplier_tp=2.0,
        atr_multiplier_sl=2.0,
        max_holding_period=50,  # Longer than data length
    )
    # Labels near the end should be NaN because lookahead is insufficient
    assert pd.isna(labels.iloc[-1])  # Use pd.isna()
    # Check a point where lookahead exists but is shorter than max_holding_period
    # Entry at index 15 (price ~107.9). Max lookahead is 4 bars (indices 16-19).
    # TP ~ 108.9, SL ~ 106.9. Assume no hit within 4 bars.
    # Final price at index 19 is ~110. Return > 0. Label should be 1 (if threshold=0).
    # Check a point where lookahead exists but is shorter than max_holding_period
    # Entry at index 18 (n-2). Lookahead window is index 19.
    # Price[18]~109.47, Price[19]~110. TP~110.47, SL~108.47. High[19]~110.5, Low[19]~109.5
    # TP is hit at index 19.
    assert labels.iloc[18] == 1  # TP hit on last bar
