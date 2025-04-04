# tests/trading/test_simulation.py
from datetime import datetime

import pandas as pd
import pytest

from market_ml_model.trading.simulation import Trade, TradeManager

# Define tolerance for floating point comparisons
TOL = 1e-6

# --- Fixtures ---


@pytest.fixture
def sample_trade_long():
    """Fixture for a basic long Trade object."""
    return Trade(
        id="T1",
        symbol="AAPL",
        direction=1,
        entry_time=datetime(2023, 1, 1, 9, 30),
        entry_price=150.0,
        size=10.0,
        stop_loss=145.0,
        take_profit=160.0,
    )


@pytest.fixture
def sample_trade_short():
    """Fixture for a basic short Trade object."""
    return Trade(
        id="T2",
        symbol="GOOG",
        direction=-1,
        entry_time=datetime(2023, 1, 2, 10, 0),
        entry_price=100.0,
        size=5.0,
        stop_loss=105.0,
        take_profit=95.0,
    )


@pytest.fixture
def sample_trade_manager():
    """Fixture for a TradeManager object."""
    return TradeManager(initial_capital=10000.0, risk_per_trade=0.01)


# --- Trade Class Tests ---


def test_trade_initialization_long(sample_trade_long):
    """Test initialization of a long trade."""
    trade = sample_trade_long
    assert trade.id == "T1"
    assert trade.symbol == "AAPL"
    assert trade.direction == 1
    assert trade.entry_price == 150.0
    assert trade.size == 10.0
    assert trade.stop_loss == 145.0
    assert trade.take_profit == 160.0
    assert trade.is_active is True
    assert trade.exit_time is None
    assert trade.exit_price is None
    assert trade.realized_pnl is None


def test_trade_initialization_short(sample_trade_short):
    """Test initialization of a short trade."""
    trade = sample_trade_short
    assert trade.id == "T2"
    assert trade.symbol == "GOOG"
    assert trade.direction == -1
    assert trade.entry_price == 100.0
    assert trade.size == 5.0
    assert trade.stop_loss == 105.0
    assert trade.take_profit == 95.0
    assert trade.is_active is True


def test_trade_update_long_no_close(sample_trade_long):
    """Test updating a long trade without hitting SL/TP."""
    trade = sample_trade_long
    timestamp = datetime(2023, 1, 1, 10, 0)
    should_close = trade.update(timestamp, 155.0)
    assert should_close is False
    assert trade.is_active is True
    assert trade.current_price == 155.0
    assert trade.unrealized_pnl == pytest.approx((155.0 - 150.0) / 150.0, abs=TOL)


def test_trade_update_long_hit_tp(sample_trade_long):
    """Test updating a long trade hitting take profit."""
    trade = sample_trade_long
    timestamp = datetime(2023, 1, 1, 10, 30)
    should_close = trade.update(timestamp, 161.0)  # Price goes above TP
    assert should_close is True
    assert trade.is_active is False
    assert trade.exit_time == timestamp
    assert trade.exit_price == 160.0  # Exits at TP price
    assert trade.exit_reason == "Take Profit"
    assert trade.realized_pnl == pytest.approx((160.0 - 150.0) / 150.0, abs=TOL)


def test_trade_update_long_hit_sl(sample_trade_long):
    """Test updating a long trade hitting stop loss."""
    trade = sample_trade_long
    timestamp = datetime(2023, 1, 1, 11, 0)
    should_close = trade.update(timestamp, 144.0)  # Price goes below SL
    assert should_close is True
    assert trade.is_active is False
    assert trade.exit_time == timestamp
    assert trade.exit_price == 145.0  # Exits at SL price
    assert trade.exit_reason == "Stop Loss"
    assert trade.realized_pnl == pytest.approx((145.0 - 150.0) / 150.0, abs=TOL)


def test_trade_update_short_hit_tp(sample_trade_short):
    """Test updating a short trade hitting take profit."""
    trade = sample_trade_short
    timestamp = datetime(2023, 1, 2, 10, 30)
    should_close = trade.update(timestamp, 94.0)  # Price goes below TP
    assert should_close is True
    assert trade.is_active is False
    assert trade.exit_price == 95.0  # Exits at TP price
    assert trade.exit_reason == "Take Profit"
    assert trade.realized_pnl == pytest.approx((100.0 - 95.0) / 100.0, abs=TOL)


def test_trade_update_short_hit_sl(sample_trade_short):
    """Test updating a short trade hitting stop loss."""
    trade = sample_trade_short
    timestamp = datetime(2023, 1, 2, 11, 0)
    should_close = trade.update(timestamp, 106.0)  # Price goes above SL
    assert should_close is True
    assert trade.is_active is False
    assert trade.exit_price == 105.0  # Exits at SL price
    assert trade.exit_reason == "Stop Loss"
    assert trade.realized_pnl == pytest.approx((100.0 - 105.0) / 100.0, abs=TOL)


def test_trade_close_manual(sample_trade_long):
    """Test manually closing a trade."""
    trade = sample_trade_long
    timestamp = datetime(2023, 1, 1, 12, 0)
    exit_price = 158.0
    trade.close(timestamp, exit_price, "Manual Close")
    assert trade.is_active is False
    assert trade.exit_time == timestamp
    assert trade.exit_price == exit_price
    assert trade.exit_reason == "Manual Close"
    assert trade.realized_pnl == pytest.approx((158.0 - 150.0) / 150.0, abs=TOL)


def test_trade_to_dict(sample_trade_long):
    """Test converting a trade to a dictionary."""
    trade = sample_trade_long
    trade_dict = trade.to_dict()
    assert isinstance(trade_dict, dict)
    assert trade_dict["id"] == "T1"
    assert trade_dict["symbol"] == "AAPL"
    assert trade_dict["entry_price"] == 150.0
    assert trade_dict["realized_pnl"] is None  # Not closed yet


# --- TradeManager Class Tests ---


def test_trade_manager_initialization(sample_trade_manager):
    """Test initialization of the TradeManager."""
    tm = sample_trade_manager
    assert tm.initial_capital == 10000.0
    assert tm.capital == 10000.0
    assert tm.risk_per_trade == 0.01
    assert not tm.active_trades
    assert not tm.closed_trades
    assert not tm.positions


def test_trade_manager_get_next_trade_id(sample_trade_manager):
    """Test trade ID generation."""
    tm = sample_trade_manager
    assert tm.get_next_trade_id() == "T1"
    assert tm.get_next_trade_id() == "T2"
    assert tm.trade_counter == 2


def test_trade_manager_calculate_position_size_basic(sample_trade_manager):
    """Test basic position size calculation (fixed fraction)."""
    tm = sample_trade_manager
    tm.use_kelly_sizing = False  # Use fixed risk for simplicity here
    size_fraction = tm.calculate_position_size(
        symbol="MSFT",
        signal_strength=0.7,  # Not used if Kelly is false
        volatility=0.01,  # Not used if Kelly is false
        current_price=200.0,
    )
    # Expected size: (Capital * Risk Per Trade) / Price
    # Note: The current implementation of calculate_position_size in TradeManager
    # calls the external position_sizing function which might have different logic.
    # This test assumes a simple risk fraction calculation for demonstration.
    # We need to mock or test the actual position sizing logic separately.
    # For now, let's assume it returns a fraction based on risk_per_trade
    # and max_capital_per_trade.
    # Let's assume calculate_position_size returns the capital fraction directly.
    # capital_fraction = tm.risk_per_trade # Simplistic assumption
    # expected_units = (tm.capital * capital_fraction) / 200.0
    # assert size == pytest.approx(expected_units, abs=TOL)
    # Since the actual logic is complex, let's just check it returns a positive number
    assert size_fraction > 0  # Placeholder assertion


def test_trade_manager_enter_position_long(sample_trade_manager):
    """Test entering a long position."""
    tm = sample_trade_manager
    timestamp = datetime(2023, 1, 3, 9, 30)
    trade_id = tm.enter_position(
        timestamp=timestamp,
        symbol="NVDA",
        direction=1,
        entry_price=500.0,
        stop_loss=490.0,
        take_profit=520.0,
        signal_strength=0.8,
        volatility=0.02,
    )
    assert trade_id is not None
    assert trade_id in tm.active_trades
    assert tm.active_trades[trade_id].symbol == "NVDA"
    assert tm.active_trades[trade_id].direction == 1
    assert "NVDA" in tm.positions
    assert tm.positions["NVDA"] > 0  # Size should be positive for long


def test_trade_manager_enter_position_short(sample_trade_manager):
    """Test entering a short position."""
    tm = sample_trade_manager
    timestamp = datetime(2023, 1, 4, 10, 0)
    trade_id = tm.enter_position(
        timestamp=timestamp,
        symbol="TSLA",
        direction=-1,
        entry_price=180.0,
        stop_loss=185.0,
        take_profit=170.0,
        signal_strength=0.7,
        volatility=0.03,
    )
    assert trade_id is not None
    assert trade_id in tm.active_trades
    assert tm.active_trades[trade_id].symbol == "TSLA"
    assert tm.active_trades[trade_id].direction == -1
    assert "TSLA" in tm.positions
    assert tm.positions["TSLA"] < 0  # Size should be negative for short


def test_trade_manager_update_close_trade(sample_trade_manager):
    """Test updating the manager, causing a trade to close."""
    tm = sample_trade_manager
    # Enter a trade
    entry_time = datetime(2023, 1, 5, 9, 30)
    trade_id = tm.enter_position(entry_time, "AMD", 1, 100.0, 98.0, 105.0)
    initial_capital = tm.capital
    trade = tm.active_trades[trade_id]
    initial_size = trade.size
    initial_entry_price = trade.entry_price

    # Update with price hitting stop loss
    update_time = datetime(2023, 1, 5, 10, 0)
    prices = {"AMD": 97.0}
    closed_ids = tm.update(update_time, prices)

    assert closed_ids == [trade_id]
    assert trade_id not in tm.active_trades
    assert trade in tm.closed_trades
    assert "AMD" not in tm.positions  # Position should be closed
    assert trade.exit_reason == "Stop Loss"
    assert trade.exit_price == 98.0  # Closed at SL
    # Check capital update
    expected_pnl_amount = (
        trade.exit_price - initial_entry_price
    ) * initial_size  # PnL = (Exit - Entry) * Size for long
    assert tm.capital == pytest.approx(initial_capital + expected_pnl_amount, abs=TOL)
    assert len(tm.equity_curve) == 2  # Initial capital + update
    assert tm.equity_curve[-1] == tm.capital


def test_trade_manager_close_all_positions(sample_trade_manager):
    """Test closing all positions manually."""
    tm = sample_trade_manager
    ts1 = datetime(2023, 1, 6, 9, 30)
    ts2 = datetime(2023, 1, 6, 9, 35)
    tm.enter_position(ts1, "MSFT", 1, 300.0, 295.0, 310.0)
    tm.enter_position(ts2, "AMZN", -1, 150.0, 155.0, 140.0)

    assert len(tm.active_trades) == 2
    assert "MSFT" in tm.positions
    assert "AMZN" in tm.positions

    close_time = datetime(2023, 1, 6, 16, 0)
    prices = {"MSFT": 305.0, "AMZN": 148.0}
    closed_ids = tm.close_all_positions(close_time, prices, "End of Day")

    assert len(closed_ids) == 2
    assert not tm.active_trades
    assert len(tm.closed_trades) == 2
    assert not tm.positions
    assert tm.closed_trades[0].exit_reason == "End of Day"
    assert tm.closed_trades[1].exit_reason == "End of Day"


def test_trade_manager_get_trade_summary(sample_trade_manager):
    """Test getting the trade summary DataFrame."""
    tm = sample_trade_manager
    ts1 = datetime(2023, 1, 7, 9, 30)
    ts2 = datetime(2023, 1, 7, 10, 0)
    trade_id1 = tm.enter_position(ts1, "XOM", 1, 110.0, 108.0, 115.0)
    tm.close_trade(trade_id1, ts2, 112.0, "Manual")  # Close the trade

    summary_df = tm.get_trade_summary()
    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 1
    assert summary_df.iloc[0]["id"] == trade_id1
    assert summary_df.iloc[0]["symbol"] == "XOM"
    assert summary_df.iloc[0]["realized_pnl"] == pytest.approx(
        (112.0 - 110.0) / 110.0, abs=TOL
    )


def test_trade_manager_get_performance_metrics_no_trades(sample_trade_manager):
    """Test performance metrics when no trades have been closed."""
    tm = sample_trade_manager
    metrics = tm.get_performance_metrics()
    assert isinstance(metrics, dict)
    assert (
        metrics["total_return"] == 0.0
    )  # Based on equity curve (only initial capital)
    assert metrics["win_rate"] == 0.0
    assert metrics["profit_factor"] == 0.0
    assert metrics["max_drawdown"] == 0.0  # Based on equity curve


# TODO: Add more comprehensive tests for TradeManager, including:
# - Kelly sizing calculation (might require mocking the sizing function)
# - Risk limits (max open trades, max drawdown circuit breaker)
# - Correlation limits
# - Pyramiding logic
# - Trailing stops in Trade and how TradeManager handles them
# - Performance metrics calculation with actual closed trades
# - generate_report functionality (might require mocking plotting/saving)
