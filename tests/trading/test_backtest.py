from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Import the function to test
from market_ml_model.trading.backtest import backtest_strategy

# Import dependencies for mocking specs
from market_ml_model.trading.simulation import TradeManager

# Define path for patching
BACKTEST_PATH = "market_ml_model.trading.backtest"

# --- Fixtures ---


@pytest.fixture
def sample_backtest_data():
    """DataFrame with OHLC, prediction, and ATR."""
    dates = pd.date_range(start="2023-01-01", periods=15, freq="D")
    data = {
        "open": np.linspace(100, 105, 15),
        "high": np.linspace(100.5, 106, 15),
        "low": np.linspace(99.5, 104, 15),
        "close": np.linspace(100.2, 105.5, 15),
        "prediction": [
            0,
            1,
            1,
            0,
            -1,
            -1,
            0,
            0,
            1,
            0,
            -1,
            0,
            1,
            1,
            0,
        ],  # Example signals
        "ATRr_10": np.linspace(0.5, 0.8, 15),  # Example ATR values
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def mock_trade_manager(mocker):
    """Mocks the TradeManager class."""
    mock_instance = MagicMock(spec=TradeManager)
    mock_instance.get_performance_metrics.return_value = {
        "total_return": 0.1,
        "sharpe_ratio": 1.5,
    }  # Dummy metrics
    mock_instance.get_trade_summary.return_value = pd.DataFrame(
        {"entry_time": [], "exit_time": []}
    )  # Empty trades df
    mock_instance.active_trades = {}  # Start with no active trades
    mock_instance.positions = {}  # Add the missing positions attribute
    mock_instance.capital = 10000.0  # Initial capital

    # Patch the class within the backtest module
    mocker.patch(  # Removed unused variable mock_class
        f"{BACKTEST_PATH}.TradeManager", return_value=mock_instance
    )
    return mock_instance  # Return the instance for assertion checks


# --- Tests ---


def test_backtest_strategy_basic_run(sample_backtest_data, mock_trade_manager):
    """Test a basic run of the backtest function."""
    results = backtest_strategy(
        data_with_predictions=sample_backtest_data,
        initial_capital=5000.0,
        risk_per_trade=0.01,
        atr_col="ATRr_10",  # Use the provided ATR column
    )

    # Check TradeManager initialization (via the mock's return value's call)
    # Note: We can't directly assert on the __init__ call args easily with instance mock
    # Instead, check methods called during the loop

    # Check methods were called
    assert mock_trade_manager.update.call_count == len(sample_backtest_data) - 1
    # Check enter_position calls based on signals (non-zero predictions in prev bar)
    # Signals at index 1, 2, 4, 5, 8, 10, 12, 13 -> 8 potential entries
    assert mock_trade_manager.enter_position.call_count == 8
    # Check close_all_positions called at the end if any trades were active
    # (Mock starts with no active trades, so might not be called if all closed by update)
    # mock_trade_manager.close_all_positions.assert_called_once() # This depends on mock state

    # Check performance metrics were retrieved
    mock_trade_manager.get_performance_metrics.assert_called_once()
    assert "total_return_pct" in results
    assert results["total_return_pct"] == pytest.approx(10.0)  # 0.1 * 100


def test_backtest_strategy_entry_logic(sample_backtest_data, mock_trade_manager):
    """Verify arguments passed to enter_position."""
    backtest_strategy(  # Removed unused variable results
        data_with_predictions=sample_backtest_data,
        atr_multiplier_sl=1.5,
        atr_multiplier_tp=3.0,
        atr_col="ATRr_10",
        slippage_pct_per_trade=0.001,
    )

    # Check the first entry call (signal=1 at index 1, entry based on index 0 data)
    first_entry_call = mock_trade_manager.enter_position.call_args_list[0]
    call_kwargs = first_entry_call.kwargs
    # Entry is based on signal at index 1, using data from index 1 (i=2 loop -> i-1=1)
    prev_close = sample_backtest_data["close"].iloc[1]
    prev_atr = sample_backtest_data["ATRr_10"].iloc[1]  # Use ATR from index 1
    expected_sl = prev_close - 1.5 * prev_atr
    expected_tp = prev_close + 3.0 * prev_atr
    # expected_entry = prev_close * (1 + 0.001 * 1)  # Removed unused variable

    assert (
        call_kwargs["timestamp"] == sample_backtest_data.index[1]
    )  # Entry is based on signal at index 1, timestamp is index 1
    assert call_kwargs["direction"] == 1
    # Entry price calculation uses prev_close (from index 1)
    assert call_kwargs["entry_price"] == pytest.approx(prev_close * (1 + 0.001 * 1))
    assert call_kwargs["stop_loss"] == pytest.approx(expected_sl)
    assert call_kwargs["take_profit"] == pytest.approx(expected_tp)
    assert call_kwargs["signal_strength"] >= 0.5  # Default threshold logic


def test_backtest_strategy_no_dynamic_stops(sample_backtest_data, mock_trade_manager):
    """Test backtest with fixed percentage stops."""
    backtest_strategy(  # Removed unused variable results
        data_with_predictions=sample_backtest_data.drop(
            columns=["ATRr_10"]
        ),  # Remove ATR col
        use_dynamic_stops=False,
        atr_multiplier_sl=None,  # Disable ATR stops
        atr_multiplier_tp=None,
    )

    # Check the first entry call
    first_entry_call = mock_trade_manager.enter_position.call_args_list[0]
    call_kwargs = first_entry_call.kwargs
    # Entry is based on signal at index 1, using data from index 1
    prev_close = sample_backtest_data["close"].iloc[1]
    # Default fixed stops are 1% SL, 2% TP
    expected_sl = prev_close * (1 - 0.01)
    expected_tp = prev_close * (1 + 0.02)

    # Check that the calculated fixed percentage stop loss was passed
    # Check that the calculated fixed percentage stop loss was passed
    assert call_kwargs["stop_loss"] == pytest.approx(expected_sl)
    assert call_kwargs["take_profit"] == pytest.approx(expected_tp)


@patch(f"{BACKTEST_PATH}.os.makedirs")
@patch(f"{BACKTEST_PATH}.os.path.dirname")
def test_backtest_strategy_save_trades(
    mock_dirname, mock_makedirs, sample_backtest_data, mock_trade_manager
):
    """Test saving trades log."""
    mock_dirname.return_value = "/fake/output"  # Mock dirname return
    trades_path = "/fake/output/my_trades.csv"
    # Simulate TradeManager returning a non-empty DataFrame
    mock_trade_manager.get_trade_summary.return_value = pd.DataFrame(
        {"symbol": ["TEST"], "pnl": [10]}
    )

    with patch("pandas.DataFrame.to_csv") as mock_to_csv:
        backtest_strategy(  # Removed unused variable results
            data_with_predictions=sample_backtest_data, output_trades_path=trades_path
        )
        mock_trade_manager.get_trade_summary.assert_called_once()
        mock_makedirs.assert_called_once_with(
            "/fake/output"
        )  # Correct call signature (no exist_ok)
        mock_to_csv.assert_called_once_with(trades_path, index=False)


@patch(f"{BACKTEST_PATH}.os.makedirs")
def test_backtest_strategy_save_report(
    mock_makedirs, sample_backtest_data, mock_trade_manager
):
    """Test saving detailed report."""
    output_dir = "/fake/report_dir"
    backtest_strategy(  # Removed unused variable results
        data_with_predictions=sample_backtest_data,
        output_dir=output_dir,
        save_detailed_report=True,
    )
    mock_makedirs.assert_called_once_with(
        output_dir
    )  # Correct call signature (no exist_ok)
    mock_trade_manager.generate_report.assert_called_once()
    # Check benchmark_equity was passed (or None)
    assert "benchmark_equity" in mock_trade_manager.generate_report.call_args.kwargs


def test_backtest_strategy_missing_cols(sample_backtest_data):
    """Test error handling for missing columns."""
    data_missing = sample_backtest_data.drop(columns=["close"])
    with patch(f"{BACKTEST_PATH}.logger") as mock_logger:
        results = backtest_strategy(data_missing)
        assert results == {}
        mock_logger.error.assert_called_once_with(
            "Missing required columns for backtesting: ['close']"
        )


def test_backtest_strategy_empty_data():
    """Test error handling for empty data."""
    # Create an empty DataFrame *with* required columns
    empty_df = pd.DataFrame(columns=["high", "low", "close", "prediction", "ATRr_10"])
    empty_df.index = pd.to_datetime(empty_df.index)  # Ensure datetime index

    with patch(f"{BACKTEST_PATH}.logger") as mock_logger:
        results = backtest_strategy(empty_df)
        assert results == {}
        # Check that the warning for empty data was logged
        mock_logger.warning.assert_called_once_with("No data provided for backtesting.")
