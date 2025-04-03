# import pytest # Not used directly
import pandas as pd
import numpy as np

# Adjust the import path based on your project structure
from market_ml_model.src.backtesting import backtest_strategy

# Fixture (sample_featured_data) is automatically discovered from conftest.py


# --- Test Functions for Backtesting ---


def test_backtest_strategy_basic(sample_featured_data):
    """Test basic backtesting logic with alternating predictions."""
    data_with_predictions = sample_featured_data.copy()
    # Simple alternating prediction for testing
    data_with_predictions['prediction'] = [1, 0, 1, 0, 1, 0]

    performance = backtest_strategy(data_with_predictions)

    assert isinstance(performance, dict)
    assert "total_strategy_return_pct" in performance
    assert "total_market_return_pct" in performance
    assert "num_trades" in performance

    # Basic sanity checks (values depend heavily on sample data and signals)
    # Expected trades: Signal goes 0->1 at index 2 and 4 (after shift)
    # Expect 3 trades: First signal=1 (T1), 0->1 (T2), 0->1 (T3)
    assert performance["num_trades"] == 3
    # Check if returns are calculated (might be positive or negative)
    assert isinstance(performance["total_strategy_return_pct"], float)
    assert isinstance(performance["total_market_return_pct"], float)

    # Rough check on returns based on sample data and alternating signal
    # Market return: (107 / 105.5) - 1 (approx 1.4%)
    # Strategy trades:
    #   Enter at 106 (idx 2 signal), Exit at 106.5 (idx 3 signal 0) -> Gain
    #   Enter at 107.5 (idx 4 signal), Exit at 107 (idx 5 signal 0) -> Loss
    # Overall strategy return should be calculated.
    # Example: Check if strategy return is different from market return
    assert not np.isclose(performance["total_strategy_return_pct"],
                          performance["total_market_return_pct"])


def test_backtest_strategy_all_buy(sample_featured_data):
    """Test backtesting logic when prediction is always 1 (buy/hold)."""
    data_with_predictions = sample_featured_data.copy()
    data_with_predictions['prediction'] = 1  # Always predict buy

    performance = backtest_strategy(data_with_predictions)

    assert isinstance(performance, dict)
    # Strategy return should match market return (buy and hold)
    # Use np.isclose for float comparison
    assert np.isclose(performance["total_strategy_return_pct"],
                      performance["total_market_return_pct"])
    # Should be one trade (entry at the start)
    assert performance["num_trades"] == 1


def test_backtest_strategy_all_sell(sample_featured_data):
    """Test backtesting logic when prediction is always 0 (sell/stay out)."""
    data_with_predictions = sample_featured_data.copy()
    data_with_predictions['prediction'] = 0  # Always predict sell

    performance = backtest_strategy(data_with_predictions)

    assert isinstance(performance, dict)
    # Strategy return should be close to 0 (no positions taken)
    assert np.isclose(performance["total_strategy_return_pct"], 0.0)
    # Market return is still calculated for comparison
    assert "total_market_return_pct" in performance
    # No trades should occur
    assert performance["num_trades"] == 0


def test_backtest_strategy_missing_columns(sample_featured_data):
    """Test backtest with missing required columns."""
    data_missing_close = sample_featured_data.copy()
    data_missing_close['prediction'] = 1
    data_missing_close = data_missing_close.drop(columns=['close'])

    data_missing_pred = sample_featured_data.copy()
    # Missing 'prediction' column

    assert backtest_strategy(data_missing_close) == {}
    assert backtest_strategy(data_missing_pred) == {}


def test_backtest_strategy_empty_input():
    """Test backtest with empty DataFrame."""
    empty_df = pd.DataFrame(columns=['close', 'prediction'])
    assert backtest_strategy(empty_df) == {}

    empty_df_no_cols = pd.DataFrame()
    assert backtest_strategy(empty_df_no_cols) == {}


def test_backtest_strategy_nan_input(sample_featured_data):
    """Test backtest handling NaNs in crucial columns."""
    # NaN in close
    data_nan_close = sample_featured_data.copy()
    data_nan_close['prediction'] = 1
    data_nan_close.loc[data_nan_close.index[1], 'close'] = np.nan
    # backtest drops NaNs in daily_return calculation, should still run
    perf_nan_close = backtest_strategy(data_nan_close)
    assert isinstance(perf_nan_close, dict)
    assert "total_strategy_return_pct" in perf_nan_close  # Check it ran

    # NaN in prediction (after shift, this becomes NaN in signal)
    data_nan_pred = sample_featured_data.copy()
    data_nan_pred['prediction'] = 1.0  # Convert to float for NaN
    data_nan_pred.loc[data_nan_pred.index[1], 'prediction'] = np.nan
    # backtest drops NaNs in signal calculation, should still run
    perf_nan_pred = backtest_strategy(data_nan_pred)
    assert isinstance(perf_nan_pred, dict)
    assert "total_strategy_return_pct" in perf_nan_pred  # Check it ran