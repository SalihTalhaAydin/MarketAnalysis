"""
Position sizing algorithms.
"""

import logging
from typing import Optional

# Setup logging
logger = logging.getLogger(__name__)


def calculate_position_size(
    capital: float,
    signal_strength: float,
    volatility: float,
    current_price: Optional[
        float
    ] = None,  # Kept for signature consistency, but not used
    max_risk_per_trade: float = 0.02,
    max_capital_per_trade: float = 0.25,
    # Added entry_price and stop_loss_price for correct calculation
    entry_price: Optional[float] = None,
    stop_loss_price: Optional[float] = None,
) -> float:
    """
    Calculate position size fraction based on fixed fractional risk.

    This function calculates the fraction of capital to *allocate* to a trade
    such that the potential loss (if the stop-loss is hit) equals the desired
    risk fraction (`max_risk_per_trade`) of the total capital.

    Formula:
        Risk Amount = Capital * max_risk_per_trade
        Risk per Unit = abs(entry_price - stop_loss_price)
        Units = Risk Amount / Risk per Unit
        Capital Allocated = Units * entry_price
        Capital Fraction = Capital Allocated / Capital
        Capital Fraction = (max_risk_per_trade * entry_price) / abs(entry_price - stop_loss_price)

    The result is capped by `max_capital_per_trade`.

    Args:
        capital: Current total account equity.
        signal_strength: Signal strength (currently unused in fixed fractional).
        volatility: Market volatility (currently unused in fixed fractional).
        current_price: Current price (unused, entry_price is used).
        max_risk_per_trade: Maximum risk per trade (fraction of capital, e.g., 0.02 for 2%).
        max_capital_per_trade: Maximum capital allocation per trade (fraction, e.g., 0.25 for 25%).
        entry_price: The price at which the trade will be entered.
        stop_loss_price: The price at which the trade will be stopped out.

    Returns:
        Position size as a fraction of capital to allocate. Returns 0.0 if inputs are invalid.
    """
    print(
        f"DEBUG_SIZING - Inputs: capital={capital}, signal_strength={signal_strength}, "
        f"volatility={volatility}, current_price={current_price}, "
        f"max_risk_per_trade={max_risk_per_trade}, max_capital_per_trade={max_capital_per_trade}, "
        f"entry_price={entry_price}, stop_loss_price={stop_loss_price}"
    )
    # Validate inputs
    if capital <= 0:
        logger.warning("Capital is zero or negative, cannot size position.")
        return 0.0
    if signal_strength < 0 or signal_strength > 1:
        logger.warning(
            f"Signal strength {signal_strength} out of bounds [0, 1]. Clamping."
        )
        signal_strength = max(0, min(1, signal_strength))
    if volatility <= 1e-6:  # Use small epsilon instead of 0
        logger.warning("Volatility is zero or negative, cannot size position.")
        return 0.0

    # --- Fixed Fractional Calculation ---
    if entry_price is None or stop_loss_price is None:
        logger.warning(
            "Entry price or stop loss price not provided. Cannot calculate fixed fractional size."
        )
        return 0.0
    if entry_price <= 0 or stop_loss_price <= 0:
        logger.warning(
            f"Entry price ({entry_price}) or stop loss price ({stop_loss_price}) is non-positive."
        )
        return 0.0

    risk_per_unit = abs(entry_price - stop_loss_price)
    print(f"DEBUG_SIZING - Calculated risk_per_unit: {risk_per_unit}")
    if risk_per_unit < 1e-9:  # Avoid division by zero if entry == stop
        logger.warning(
            f"Risk per unit is zero (entry={entry_price}, stop={stop_loss_price}). Cannot size position."
        )
        return 0.0

    # Calculate the ideal fraction of capital to allocate based on risk
    # Formula: Capital Fraction = (max_risk_per_trade * entry_price) / abs(entry_price - stop_loss_price)
    position_size_fraction = (max_risk_per_trade * entry_price) / risk_per_unit
    print(
        f"DEBUG_SIZING - Initial position_size_fraction (before cap): {position_size_fraction}"
    )

    # Cap the allocation by max_capital_per_trade
    position_size_fraction = min(position_size_fraction, max_capital_per_trade)
    print(
        f"DEBUG_SIZING - position_size_fraction after cap ({max_capital_per_trade}): {position_size_fraction}"
    )

    # Ensure final size is non-negative
    position_size_fraction = max(0, position_size_fraction)

    logger.info(
        f"Fixed Fractional Calculation: "
        f"Capital={capital:.2f}, MaxRisk={max_risk_per_trade:.4f}, "
        f"Entry={entry_price:.4f}, Stop={stop_loss_price:.4f}, "
        f"Risk/Unit={risk_per_unit:.4f} -> Alloc Fraction={position_size_fraction:.6f} "
        f"(capped by {max_capital_per_trade:.4f})"
    )

    # Optionally convert fraction to units if price is given
    # if current_price is not None and current_price > 0:
    #     capital_to_allocate = capital * position_size_fraction
    #     units = capital_to_allocate / current_price
    #     return units # Return units instead of fraction
    # else:
    #     return position_size_fraction # Return fraction

    print(
        f"DEBUG_SIZING - Final position_size_fraction returned: {position_size_fraction}"
    )
    return position_size_fraction  # Return fraction by default
