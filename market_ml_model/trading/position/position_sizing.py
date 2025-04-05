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
    current_price: Optional[float] = None,  # Added for potential unit calculation
    max_risk_per_trade: float = 0.02,
    max_capital_per_trade: float = 0.25,
) -> float:
    """
    Calculate position size as a fraction of capital using various methods.

    Args:
        capital: Current capital
        signal_strength: Signal strength (0-1, where 0.5 is neutral)
        volatility: Current market volatility (e.g., ATR percentage)
        current_price: Current price (optional, used if returning units)
        max_risk_per_trade: Maximum risk per trade (fraction of capital)
        max_capital_per_trade: Maximum capital allocation per trade (fraction)

    Returns:
        Position size (fraction of capital)
    """
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

    # Default to fixed fractional sizing
    base_fraction = max_risk_per_trade
    logger.info(f"Initial base fraction set to max_risk_per_trade: {base_fraction:.4f}")

    # Volatility adjustment (reduce position size in volatile markets)
    # Normalize volatility (e.g., assume target daily volatility is 1%)
    target_volatility = 0.01
    volatility_scalar = target_volatility / max(
        1e-6, volatility
    )  # Avoid division by zero

    # Limit adjustment range (e.g., 25%-200% of base size)
    volatility_scalar = min(2.0, max(0.25, volatility_scalar))
    logger.info(
        f"Volatility scalar: {volatility_scalar:.2f} (based on vol={volatility:.4f})"
    )

    # Calculate final position size fraction based on the minimum constraint
    position_size_fraction = min(
        base_fraction * volatility_scalar,  # Volatility-adjusted base fraction
        max_capital_per_trade,  # Max capital allocation constraint
    )

    # Ensure final size is non-negative
    position_size_fraction = max(0, position_size_fraction)

    logger.info(f"Calculated position size fraction: {position_size_fraction:.4f}")

    # Optionally convert fraction to units if price is given
    # if current_price is not None and current_price > 0:
    #     capital_to_allocate = capital * position_size_fraction
    #     units = capital_to_allocate / current_price
    #     return units # Return units instead of fraction
    # else:
    #     return position_size_fraction # Return fraction

    return position_size_fraction  # Return fraction by default
