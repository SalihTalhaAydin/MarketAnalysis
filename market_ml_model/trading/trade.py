# market_ml_model/trading/trade.py
import logging
from typing import Any, Dict, List, Optional

import pandas as pd  # Added import

logger = logging.getLogger(__name__)


class Trade:
    """
    Class to represent a single trade with advanced tracking.
    """

    def __init__(
        self,
        id: str,
        symbol: str,
        direction: int,  # 1 for long, -1 for short
        entry_time: Any,
        entry_price: float,
        size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: bool = False,
        trailing_stop_distance: float = 0.0,
        trailing_stop_activation: float = 0.0,
        tags: Optional[List[str]] = None,
    ):
        """
        Initialize a trade object.

        Args:
            id: Unique trade identifier
            symbol: Asset symbol
            direction: Trade direction (1=long, -1=short)
            entry_time: Entry timestamp
            entry_price: Entry price
            size: Position size
            stop_loss: Stop loss price
            take_profit: Take profit price
            trailing_stop: Whether to use trailing stop
            trailing_stop_distance: Distance for trailing stop
            trailing_stop_activation: Activation threshold for trailing stop
            tags: Optional tags for analysis
        """
        # Trade identification
        self.id = id
        self.symbol = symbol
        self.direction = direction
        self.tags = tags or []

        # Entry details
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.size = size

        # Exit targets
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop
        self.trailing_stop_distance = trailing_stop_distance
        self.trailing_stop_activation = trailing_stop_activation

        # Exit details (to be set later)
        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None

        # State tracking
        self.is_active = True
        self.max_favorable_excursion = 0.0  # Best profit seen during trade
        self.max_adverse_excursion = 0.0  # Worst drawdown seen during trade
        self.current_price = entry_price
        self.trailing_stop_price = None
        self.trailing_stop_activated = False

        # Performance tracking
        self.unrealized_pnl = 0.0
        self.realized_pnl = None
        self.duration = None

        # Initialize trailing stop if used
        if trailing_stop and direction == 1:  # Long position
            self.trailing_stop_price = entry_price * (1 - trailing_stop_distance)
        elif trailing_stop and direction == -1:  # Short position
            self.trailing_stop_price = entry_price * (1 + trailing_stop_distance)

    def update(self, timestamp: Any, current_price: float) -> bool:
        """
        Update trade status with current price information.

        Args:
            timestamp: Current timestamp
            current_price: Current market price

        Returns:
            bool: True if trade should be closed, False otherwise
        """
        if not self.is_active:
            return False

        self.current_price = current_price

        # Calculate unrealized P&L
        if self.direction == 1:  # Long position
            self.unrealized_pnl = (current_price - self.entry_price) / self.entry_price

            # Update max favorable/adverse excursion
            self.max_favorable_excursion = max(
                self.max_favorable_excursion, self.unrealized_pnl
            )
            self.max_adverse_excursion = min(
                self.max_adverse_excursion, self.unrealized_pnl
            )

            # Check trailing stop activation
            if (
                self.trailing_stop
                and not self.trailing_stop_activated
                and self.unrealized_pnl >= self.trailing_stop_activation
            ):
                self.trailing_stop_activated = True
                self.trailing_stop_price = current_price * (
                    1 - self.trailing_stop_distance
                )
                logger.info(
                    f"Trailing stop activated for trade {self.id} at {self.trailing_stop_price}"
                )

            # Update trailing stop if activated
            if self.trailing_stop and self.trailing_stop_activated:
                new_stop = current_price * (1 - self.trailing_stop_distance)
                if (
                    self.trailing_stop_price is None
                    or new_stop > self.trailing_stop_price
                ):  # Check if None before comparing
                    self.trailing_stop_price = new_stop

            # Check stop loss
            if self.stop_loss is not None and current_price <= self.stop_loss:
                self.close(timestamp, self.stop_loss, "Stop Loss")
                return True

            # Check trailing stop
            if (
                self.trailing_stop
                and self.trailing_stop_activated
                and self.trailing_stop_price is not None  # Check if None
                and current_price <= self.trailing_stop_price
            ):
                self.close(timestamp, self.trailing_stop_price, "Trailing Stop")
                return True

            # Check take profit
            if self.take_profit is not None and current_price >= self.take_profit:
                self.close(timestamp, self.take_profit, "Take Profit")
                return True

        elif self.direction == -1:  # Short position
            self.unrealized_pnl = (self.entry_price - current_price) / self.entry_price

            # Update max favorable/adverse excursion
            self.max_favorable_excursion = max(
                self.max_favorable_excursion, self.unrealized_pnl
            )
            self.max_adverse_excursion = min(
                self.max_adverse_excursion, self.unrealized_pnl
            )

            # Check trailing stop activation
            if (
                self.trailing_stop
                and not self.trailing_stop_activated
                and self.unrealized_pnl >= self.trailing_stop_activation
            ):
                self.trailing_stop_activated = True
                self.trailing_stop_price = current_price * (
                    1 + self.trailing_stop_distance
                )
                logger.info(
                    f"Trailing stop activated for trade {self.id} at {self.trailing_stop_price}"
                )

            # Update trailing stop if activated
            if self.trailing_stop and self.trailing_stop_activated:
                new_stop = current_price * (1 + self.trailing_stop_distance)
                if (
                    self.trailing_stop_price is None
                    or new_stop < self.trailing_stop_price
                ):  # Check if None before comparing
                    self.trailing_stop_price = new_stop

            # Check stop loss
            if self.stop_loss is not None and current_price >= self.stop_loss:
                self.close(timestamp, self.stop_loss, "Stop Loss")
                return True

            # Check trailing stop
            if (
                self.trailing_stop
                and self.trailing_stop_activated
                and self.trailing_stop_price is not None  # Check if None
                and current_price >= self.trailing_stop_price
            ):
                self.close(timestamp, self.trailing_stop_price, "Trailing Stop")
                return True

            # Check take profit
            if self.take_profit is not None and current_price <= self.take_profit:
                self.close(timestamp, self.take_profit, "Take Profit")
                return True

        return False

    def close(self, timestamp: Any, price: float, reason: str) -> None:
        """
        Close the trade.

        Args:
            timestamp: Exit timestamp
            price: Exit price
            reason: Reason for exit
        """
        if not self.is_active:
            return

        self.is_active = False
        self.exit_time = timestamp
        self.exit_price = price
        self.exit_reason = reason

        # Calculate realized P&L
        if self.entry_price == 0:  # Avoid division by zero
            self.realized_pnl = 0.0
        elif self.direction == 1:  # Long position
            self.realized_pnl = (self.exit_price - self.entry_price) / self.entry_price
        else:  # Short position
            self.realized_pnl = (self.entry_price - self.exit_price) / self.entry_price

        # Calculate duration
        try:
            # Attempt conversion if timestamps are not already datetime objects
            entry_dt = pd.to_datetime(self.entry_time)
            exit_dt = pd.to_datetime(self.exit_time)
            self.duration = exit_dt - entry_dt
        except (TypeError, ValueError):
            # Handle cases where conversion fails or types are incompatible
            self.duration = None
            logger.debug(
                f"Could not calculate duration for trade {self.id} due to timestamp types."
            )

        logger.info(
            f"Trade {self.id} closed: {self.symbol} {self.direction} at {self.exit_price}. "
            f"P&L: {self.realized_pnl:.2%}, Reason: {self.exit_reason}"
        )

    def to_dict(self) -> Dict:
        """
        Convert trade to dictionary representation.

        Returns:
            Dictionary with trade information
        """
        return {
            "id": self.id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "size": self.size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "exit_time": self.exit_time,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "realized_pnl": self.realized_pnl,
            "max_favorable_excursion": self.max_favorable_excursion,
            "max_adverse_excursion": self.max_adverse_excursion,
            "duration": (
                str(self.duration) if self.duration is not None else None
            ),  # Convert duration to string
            "tags": ",".join(self.tags) if self.tags else "",
        }
