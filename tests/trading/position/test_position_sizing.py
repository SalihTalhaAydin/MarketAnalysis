from unittest.mock import patch

import pytest

# Import the function to test
from market_ml_model.trading.position.position_sizing import calculate_position_size

# Define path for patching logger
POS_SIZING_PATH = "market_ml_model.trading.position.position_sizing"

# --- Test Data ---
CAPITAL = 10000.0
SIGNAL_STRONG_BUY = 0.9
SIGNAL_WEAK_BUY = 0.6
SIGNAL_NEUTRAL = 0.5
SIGNAL_STRONG_SELL = 0.1  # Note: signal strength is 0-1, direction handled elsewhere
VOLATILITY_NORMAL = 0.01  # Matches target_volatility
VOLATILITY_HIGH = 0.02
VOLATILITY_LOW = 0.005
MAX_RISK = 0.02
MAX_CAPITAL = 0.25
TARGET_VOL = 0.01  # Default target volatility in the function


# --- Helper for Kelly Calculation ---
def calculate_kelly(win_rate, payoff):
    if payoff <= 0:
        return 0
    kelly_f = win_rate - (1 - win_rate) / payoff
    return max(0, kelly_f * 0.5)  # Half Kelly


# --- Tests ---


def test_fixed_fractional_sizing():
    """Test fixed fractional sizing (use_kelly=False)."""
    # Add dummy entry/stop prices required by the new signature
    size = calculate_position_size(
        capital=CAPITAL,
        signal_strength=SIGNAL_STRONG_BUY,  # Ignored
        volatility=VOLATILITY_NORMAL,  # Ignored
        entry_price=100.0,
        stop_loss_price=98.0,  # Represents 2% risk if entry is 100
        max_risk_per_trade=MAX_RISK,
        max_capital_per_trade=MAX_CAPITAL,
    )
    # Expected: MAX_RISK * (TARGET_VOL / VOLATILITY_NORMAL) = 0.02 * (0.01 / 0.01) = 0.02
    # Formula: min(MAX_CAPITAL, (MAX_RISK * entry_price) / abs(entry_price - stop_loss_price))
    # Expected: min(0.25, (0.02 * 100) / 2) = min(0.25, 1.0) = 0.25
    assert size == pytest.approx(MAX_CAPITAL)


def test_fixed_fractional_high_volatility():
    """Test fixed fractional sizing with high volatility."""
    # Add dummy entry/stop prices
    size = calculate_position_size(
        capital=CAPITAL,
        signal_strength=SIGNAL_STRONG_BUY,  # Ignored
        volatility=VOLATILITY_HIGH,  # Ignored
        entry_price=100.0,
        stop_loss_price=98.0,
        max_risk_per_trade=MAX_RISK,
        max_capital_per_trade=MAX_CAPITAL,
    )
    # Expected: MAX_RISK * (TARGET_VOL / VOLATILITY_HIGH) = 0.02 * (0.01 / 0.02) = 0.01
    # Volatility is ignored in the new function. Expected: 0.25
    assert size == pytest.approx(MAX_CAPITAL)


def test_fixed_fractional_low_volatility():
    """Test fixed fractional sizing with low volatility."""
    # Add dummy entry/stop prices
    size = calculate_position_size(
        capital=CAPITAL,
        signal_strength=SIGNAL_STRONG_BUY,  # Ignored
        volatility=VOLATILITY_LOW,  # Ignored
        entry_price=100.0,
        stop_loss_price=98.0,
        max_risk_per_trade=MAX_RISK,
        max_capital_per_trade=MAX_CAPITAL,
    )
    # Expected: MAX_RISK * (TARGET_VOL / VOLATILITY_LOW) = 0.02 * (0.01 / 0.005) = 0.04
    # Volatility scalar capped at 2.0
    # Volatility is ignored. Expected: 0.25
    assert size == pytest.approx(MAX_CAPITAL)


def test_kelly_sizing_estimated():
    """Test Kelly sizing estimated from signal strength."""
    # Signal 0.9 -> Edge = (0.9 - 0.5) * 2 = 0.8
    # Win Rate = 0.5 + 0.8 / 2 = 0.9
    # Payoff = 2.0 (default)
    # Kelly F = 0.9 - (1 - 0.9) / 2.0 = 0.9 - 0.1 / 2.0 = 0.9 - 0.05 = 0.85
    # Half Kelly = 0.85 * 0.5 = 0.425
    # expected_kelly = 0.425 # Removed unused variable
    # Volatility scalar = 1.0
    # Expected size = min(0.425 * 1.0, MAX_CAPITAL) = min(0.425, 0.25) = 0.25
    # Kelly sizing removed, test is now invalid for its original purpose.
    # Adapt to test fixed fractional sizing instead.
    # Add dummy entry/stop prices
    size = calculate_position_size(
        capital=CAPITAL,
        signal_strength=SIGNAL_STRONG_BUY,  # Ignored
        volatility=VOLATILITY_NORMAL,  # Ignored
        entry_price=100.0,
        stop_loss_price=98.0,
        max_risk_per_trade=MAX_RISK,  # Use default risk
        max_capital_per_trade=MAX_CAPITAL,
    )
    assert size == pytest.approx(MAX_CAPITAL)  # Limited by max capital


def test_kelly_sizing_estimated_weak_signal():
    """Test Kelly sizing estimated from a weaker signal."""
    # Signal 0.6 -> Edge = (0.6 - 0.5) * 2 = 0.2
    # Win Rate = 0.5 + 0.2 / 2 = 0.6
    # Payoff = 2.0 (default)
    # Kelly F = 0.6 - (1 - 0.6) / 2.0 = 0.6 - 0.4 / 2.0 = 0.6 - 0.2 = 0.4
    # Half Kelly = 0.4 * 0.5 = 0.2
    # expected_kelly = 0.2 # Removed unused variable
    # Volatility scalar = 1.0
    # Expected size = min(0.2 * 1.0, MAX_CAPITAL) = min(0.2, 0.25) = 0.2
    # Kelly sizing removed, test is now invalid. Adapt to test fixed fractional.
    # Add dummy entry/stop prices
    size = calculate_position_size(
        capital=CAPITAL,
        signal_strength=SIGNAL_WEAK_BUY,  # Ignored
        volatility=VOLATILITY_NORMAL,  # Ignored
        entry_price=100.0,
        stop_loss_price=98.0,
        max_risk_per_trade=MAX_RISK,  # Use default risk
        max_capital_per_trade=MAX_CAPITAL,
    )
    # Kelly removed. Expected: 0.25
    assert size == pytest.approx(MAX_CAPITAL)


def test_kelly_sizing_provided_stats():
    """Test Kelly sizing using provided win rate and payoff."""
    # win_rate = 0.65 # Removed unused variable
    # payoff = 2.5 # Removed unused variable
    # expected_kelly = calculate_kelly( # Removed unused variable
    #     win_rate, payoff
    # )  # 0.65 - (0.35 / 2.5) = 0.65 - 0.14 = 0.51 -> Half = 0.255
    # Volatility scalar = 1.0
    # Expected size = min(0.255 * 1.0, MAX_CAPITAL) = min(0.255, 0.25) = 0.25
    # Kelly sizing removed, test is now invalid. Adapt to test fixed fractional.
    # Add dummy entry/stop prices
    size = calculate_position_size(
        capital=CAPITAL,
        signal_strength=SIGNAL_STRONG_BUY,  # Ignored
        volatility=VOLATILITY_NORMAL,  # Ignored
        entry_price=100.0,
        stop_loss_price=98.0,
        max_risk_per_trade=MAX_RISK,  # Use default risk
        max_capital_per_trade=MAX_CAPITAL,
    )
    assert size == pytest.approx(MAX_CAPITAL)  # Limited by max capital


def test_kelly_sizing_high_volatility():
    """Test Kelly sizing adjusted for high volatility."""
    # win_rate = 0.65 # Removed unused variable
    # payoff = 2.5 # Removed unused variable
    # expected_kelly = calculate_kelly(win_rate, payoff)  # 0.255 # Removed unused variable
    # Volatility scalar = TARGET_VOL / VOLATILITY_HIGH = 0.01 / 0.02 = 0.5
    # Expected size = min(0.255 * 0.5, MAX_CAPITAL) = min(0.1275, 0.25) = 0.1275
    # Kelly sizing removed, test is now invalid. Adapt to test fixed fractional.
    # Add dummy entry/stop prices
    size = calculate_position_size(
        capital=CAPITAL,
        signal_strength=SIGNAL_STRONG_BUY,  # Ignored
        volatility=VOLATILITY_HIGH,  # Ignored
        entry_price=100.0,
        stop_loss_price=98.0,
        max_risk_per_trade=MAX_RISK,  # Use default risk
        max_capital_per_trade=MAX_CAPITAL,
    )
    # Kelly removed. Expected: 0.25
    assert size == pytest.approx(MAX_CAPITAL)


def test_kelly_sizing_low_volatility():
    """Test Kelly sizing adjusted for low volatility (capped)."""
    # win_rate = 0.65 # Removed unused variable
    # payoff = 2.5 # Removed unused variable
    # expected_kelly = calculate_kelly(win_rate, payoff)  # 0.255 # Removed unused variable
    # Volatility scalar = TARGET_VOL / VOLATILITY_LOW = 0.01 / 0.005 = 2.0 (capped at 2.0)
    # Expected size = min(0.255 * 2.0, MAX_CAPITAL) = min(0.51, 0.25) = 0.25
    # Kelly sizing removed, test is now invalid. Adapt to test fixed fractional.
    # Add dummy entry/stop prices
    size = calculate_position_size(
        capital=CAPITAL,
        signal_strength=SIGNAL_STRONG_BUY,  # Ignored
        volatility=VOLATILITY_LOW,  # Ignored
        entry_price=100.0,
        stop_loss_price=98.0,
        max_risk_per_trade=MAX_RISK,  # Use default risk
        max_capital_per_trade=MAX_CAPITAL,
    )
    assert size == pytest.approx(MAX_CAPITAL)  # Limited by max capital


def test_max_capital_constraint():
    """Test that position size is capped by max_capital_per_trade."""
    # Use fixed fractional, but set risk high enough that it would exceed max_capital
    high_risk = 0.5
    # Add dummy entry/stop prices
    # Corrected call for fixed fractional sizing, ensuring no repeated keywords
    size = calculate_position_size(
        capital=CAPITAL,
        signal_strength=SIGNAL_STRONG_BUY,  # Ignored
        volatility=VOLATILITY_NORMAL,  # Ignored
        entry_price=100.0,
        stop_loss_price=98.0,
        max_risk_per_trade=high_risk,  # 0.5
        max_capital_per_trade=MAX_CAPITAL,  # 0.25 - This is the only instance needed
    )
    # Expected: min(0.5 * 1.0, 0.25) = 0.25
    assert size == pytest.approx(MAX_CAPITAL)


def test_zero_capital():
    """Test behavior with zero capital."""
    # Add dummy entry/stop prices
    size = calculate_position_size(
        capital=0,
        signal_strength=0.8,
        volatility=0.01,
        entry_price=100.0,
        stop_loss_price=98.0,
    )
    assert size == 0.0


def test_zero_volatility():
    """Test behavior with zero volatility."""
    # Add dummy entry/stop prices
    size = calculate_position_size(
        capital=CAPITAL,
        signal_strength=0.8,
        volatility=0.0,
        entry_price=100.0,
        stop_loss_price=98.0,
    )
    assert size == 0.0  # Function should return 0 for zero volatility


def test_negative_kelly():
    """Test Kelly when formula results in negative fraction (should be clamped to 0)."""
    # Low win rate, low payoff
    # win_rate = 0.4 # Removed unused variable
    # payoff = 1.1 # Removed unused variable
    # Kelly F = 0.4 - (0.6 / 1.1) = 0.4 - 0.545... = negative
    # Kelly sizing removed, test is now invalid. Adapt to test fixed fractional.
    # Add dummy entry/stop prices
    size = calculate_position_size(
        capital=CAPITAL,
        signal_strength=0.5,  # Ignored
        volatility=VOLATILITY_NORMAL,  # Ignored
        entry_price=100.0,
        stop_loss_price=98.0,
        max_risk_per_trade=MAX_RISK,  # Use default risk
    )
    # Kelly removed. Expected: 0.25
    assert size == pytest.approx(MAX_CAPITAL)


@patch(f"{POS_SIZING_PATH}.logger")
def test_kelly_zero_payoff(mock_logger):
    """Test Kelly fallback when payoff ratio is zero."""
    # Kelly sizing removed, test is now invalid. Adapt to test fixed fractional.
    # Add dummy entry/stop prices
    # This call was already updated in the previous step, ensure it's correct.
    # It uses fixed fractional sizing.
    size = calculate_position_size(
        capital=CAPITAL,
        signal_strength=0.8,  # Ignored
        volatility=VOLATILITY_NORMAL,  # Ignored
        entry_price=100.0,
        stop_loss_price=98.0,
        max_risk_per_trade=MAX_RISK,
    )
    # Kelly removed, warning is no longer logged.
    # Assert the calculated fixed fractional size.
    # Expected: min(0.25, (0.02 * 100) / 2) = 0.25
    assert size == pytest.approx(MAX_CAPITAL)
