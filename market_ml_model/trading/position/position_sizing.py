"""
Position sizing algorithms.
"""

import logging
from typing import Optional

import numpy as np

# Setup logging
logger = logging.getLogger(__name__)


def calculate_position_size(
    capital: float,
    signal_strength: float,
    volatility: float,
    current_price: Optional[float] = None,  # Added for potential unit calculation
    max_risk_per_trade: float = 0.02,
    max_capital_per_trade: float = 0.25,
    use_kelly: bool = True,
    win_rate: Optional[float] = None,
    payoff_ratio: Optional[float] = None,
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
        use_kelly: Whether to use Kelly criterion
        win_rate: Historical win rate (if available)
        payoff_ratio: Historical payoff ratio (if available)

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

    # Kelly criterion calculation
    kelly_pct = max_risk_per_trade  # Default to fixed fractional if Kelly fails
    if use_kelly:
        # If historical win rate and payoff available, use those
        if win_rate is not None and payoff_ratio is not None:
            if payoff_ratio <= 0:
                logger.warning("Payoff ratio must be positive for Kelly criterion.")
                kelly_win_rate = 0.5  # Fallback to neutral
                kelly_payoff = 1.0
            else:
                kelly_win_rate = max(0, min(1, win_rate))
                kelly_payoff = payoff_ratio
        else:
            # Estimate from signal strength if historical data unavailable
            # Calculate edge from signal strength (0.5 = no edge, 1.0 = certain)
            edge = max(0, (signal_strength - 0.5) * 2)  # Rescale to 0-1 edge
            kelly_win_rate = 0.5 + edge / 2  # Win probability based on edge

            # Default payoff ratio (e.g., based on typical TP/SL ratio)
            kelly_payoff = 2.0
            logger.info(
                f"Estimating Kelly params: Win Rate={kelly_win_rate:.2f}, Payoff={kelly_payoff:.1f}"
            )

        # Kelly formula: f* = (p * b - (1 - p)) / b = p - (1-p)/b
        # Ensure payoff is positive
        if kelly_payoff > 1e-6:
            kelly_f = kelly_win_rate - (1 - kelly_win_rate) / kelly_payoff
            # Use half Kelly for safety, ensure non-negative
            kelly_pct = max(0, kelly_f * 0.5)
            logger.info(f"Kelly fraction (half): {kelly_pct:.4f}")
        else:
            logger.warning(
                "Payoff ratio is zero or negative, cannot use Kelly. Defaulting to max risk."
            )
            kelly_pct = max_risk_per_trade

    else:
        # Fixed fractional approach
        kelly_pct = max_risk_per_trade
        logger.info(f"Using fixed fractional sizing: {kelly_pct:.4f}")

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
        kelly_pct * volatility_scalar,  # Volatility-adjusted Kelly/FixedFractional
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
