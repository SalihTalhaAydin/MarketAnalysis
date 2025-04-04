"""
Trade simulation system for backtesting strategies.
"""

import logging
import os
from datetime import timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

# Import utils
from ..utils.metrics import calculate_returns_metrics

# Import from position sizing
from .position.position_sizing import calculate_position_size

# Setup logging
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
                if new_stop > self.trailing_stop_price:
                    self.trailing_stop_price = new_stop

            # Check stop loss
            if self.stop_loss is not None and current_price <= self.stop_loss:
                self.close(timestamp, self.stop_loss, "Stop Loss")
                return True

            # Check trailing stop
            if (
                self.trailing_stop
                and self.trailing_stop_activated
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
                if new_stop < self.trailing_stop_price:
                    self.trailing_stop_price = new_stop

            # Check stop loss
            if self.stop_loss is not None and current_price >= self.stop_loss:
                self.close(timestamp, self.stop_loss, "Stop Loss")
                return True

            # Check trailing stop
            if (
                self.trailing_stop
                and self.trailing_stop_activated
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
        if self.direction == 1:  # Long position
            self.realized_pnl = (self.exit_price - self.entry_price) / self.entry_price
        else:  # Short position
            self.realized_pnl = (self.entry_price - self.exit_price) / self.entry_price

        # Calculate duration
        if hasattr(self.exit_time, "timestamp") and hasattr(
            self.entry_time, "timestamp"
        ):
            self.duration = self.exit_time - self.entry_time
        else:
            self.duration = None

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
            "duration": self.duration,
            "tags": ",".join(self.tags) if self.tags else "",
        }


class TradeManager:
    """
    Advanced trade management system with risk controls.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.02,
        max_capital_per_trade: float = 0.25,
        use_kelly_sizing: bool = True,
        max_open_trades: int = 5,
        max_correlated_trades: int = 2,
        max_drawdown_pct: float = 0.15,
        trailing_stop_enabled: bool = True,
        trailing_stop_activation: float = 0.01,
        trailing_stop_distance: float = 0.005,
        pyramiding_allowed: bool = False,
    ):
        """
        Initialize the trade manager.

        Args:
            initial_capital: Starting capital
            risk_per_trade: Maximum risk per trade (fraction of capital)
            max_capital_per_trade: Maximum capital allocation per trade
            use_kelly_sizing: Whether to use Kelly criterion for position sizing
            max_open_trades: Maximum number of open trades
            max_correlated_trades: Maximum correlated trades allowed
            max_drawdown_pct: Maximum acceptable drawdown percentage
            trailing_stop_enabled: Whether to use trailing stops
            trailing_stop_activation: Return required to activate trailing stop
            trailing_stop_distance: Distance for trailing stop (fraction of price)
            pyramiding_allowed: Whether to allow adding to existing positions
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_capital_per_trade = max_capital_per_trade
        self.use_kelly_sizing = use_kelly_sizing
        self.max_open_trades = max_open_trades
        self.max_correlated_trades = max_correlated_trades
        self.max_drawdown_pct = max_drawdown_pct
        self.trailing_stop_enabled = trailing_stop_enabled
        self.trailing_stop_activation = trailing_stop_activation
        self.trailing_stop_distance = trailing_stop_distance
        self.pyramiding_allowed = pyramiding_allowed

        # State tracking
        self.active_trades = {}  # id -> Trade
        self.closed_trades = []  # List of closed Trade objects
        self.positions = {}  # symbol -> size
        self.equity_curve = [initial_capital]
        self.equity_timestamps = []
        self._first_update_done = False  # Flag to handle initial timestamp
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0

        # Performance tracking
        self.running_win_rate = 0.5  # Start with neutral assumption
        self.running_payoff_ratio = 2.0  # Start with TP/SL ratio assumption

        # Trade correlation tracking (symbol -> list of correlated symbols)
        self.correlated_assets = {}

        # Trade ID counter
        self.trade_counter = 0

    def get_next_trade_id(self) -> str:
        """
        Generate the next trade ID.

        Returns:
            Unique trade ID
        """
        self.trade_counter += 1
        return f"T{self.trade_counter}"

    def calculate_position_size(
        self,
        symbol: str,
        signal_strength: float,
        volatility: float,
        current_price: float,
    ) -> float:
        """
        Calculate appropriate position size.

        Args:
            symbol: Asset symbol
            signal_strength: Strength of the signal (0-1)
            volatility: Asset volatility
            current_price: Current asset price

        Returns:
            Number of units to trade
        """
        # Check circuit breakers
        if len(self.active_trades) >= self.max_open_trades:
            logger.info(
                f"Maximum number of open trades ({self.max_open_trades}) reached."
            )
            return 0.0

        if self.current_drawdown >= self.max_drawdown_pct:
            logger.info(
                f"Maximum drawdown ({self.max_drawdown_pct:.2%}) reached. No new positions."
            )
            return 0.0

        # Check if position already exists and pyramiding is disabled
        if symbol in self.positions and not self.pyramiding_allowed:
            logger.info(
                f"Position already exists for {symbol} and pyramiding is disabled."
            )
            return 0.0

        # Check correlation limits
        if symbol in self.correlated_assets:
            # Count current active correlated trades
            correlated_symbols = self.correlated_assets[symbol]
            active_correlated = sum(
                1 for s in correlated_symbols if s in self.positions
            )

            if active_correlated >= self.max_correlated_trades:
                logger.info(
                    f"Maximum correlated trades ({self.max_correlated_trades}) reached for {symbol}."
                )
                return 0.0

        # Calculate position size as capital fraction
        capital_fraction = calculate_position_size(
            capital=self.capital,
            signal_strength=signal_strength,
            volatility=volatility,
            max_risk_per_trade=self.risk_per_trade,
            max_capital_per_trade=self.max_capital_per_trade,
            use_kelly=self.use_kelly_sizing,
            win_rate=self.running_win_rate,
            payoff_ratio=self.running_payoff_ratio,
        )

        # Calculate number of units
        capital_amount = self.capital * capital_fraction
        units = capital_amount / current_price

        return units

    def enter_position(
        self,
        timestamp: Any,
        symbol: str,
        direction: int,  # 1 for long, -1 for short
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        signal_strength: float = 0.6,
        volatility: float = 0.01,
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Enter a new position or add to existing position.

        Args:
            timestamp: Entry timestamp
            symbol: Asset symbol
            direction: Trade direction (1=long, -1=short)
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            signal_strength: Signal strength (0-1)
            volatility: Asset volatility
            tags: Optional tags for analysis

        Returns:
            Trade ID if successful, None otherwise
        """
        if direction not in [1, -1]:
            logger.warning(
                f"Invalid direction: {direction}. Must be 1 (long) or -1 (short)."
            )
            return None

        # Calculate position size
        size = self.calculate_position_size(
            symbol, signal_strength, volatility, entry_price
        )

        if size <= 0:
            logger.info(f"Position size calculation returned {size}. No trade entered.")
            return None

        # Create trade ID
        trade_id = self.get_next_trade_id()

        # Create trade object
        trade = Trade(
            id=trade_id,
            symbol=symbol,
            direction=direction,
            entry_time=timestamp,
            entry_price=entry_price,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=self.trailing_stop_enabled,
            trailing_stop_distance=self.trailing_stop_distance,
            trailing_stop_activation=self.trailing_stop_activation,
            tags=tags,
        )

        # Add to active trades
        self.active_trades[trade_id] = trade

        # Update positions
        if symbol in self.positions:
            # Existing position - add size
            self.positions[symbol] += size * direction
        else:
            # New position
            self.positions[symbol] = size * direction

        logger.info(
            f"Entered {direction} position in {symbol} at {entry_price}. Size: {size}, Trade ID: {trade_id}"
        )
        return trade_id

    def update(self, timestamp: Any, prices: Dict[str, float]) -> List[str]:
        """
        Update all active trades with current prices.

        Args:
            timestamp: Current timestamp
            prices: Dictionary of current prices (symbol -> price)

        Returns:
            List of closed trade IDs
        """
        closed_trades = []

        # Update each active trade
        for trade_id, trade in list(self.active_trades.items()):
            symbol = trade.symbol

            if symbol in prices:
                current_price = prices[symbol]

                # Update the trade
                if trade.update(timestamp, current_price):
                    # Trade was closed
                    closed_trades.append(trade_id)

                    # Update position
                    if symbol in self.positions:
                        # Reduce position
                        self.positions[symbol] -= trade.size * trade.direction

                        # Remove if zero
                        if abs(self.positions[symbol]) < 1e-6:
                            del self.positions[symbol]

                    # Add to closed trades
                    self.closed_trades.append(trade)

                    # Remove from active trades
                    del self.active_trades[trade_id]

                    # Update capital
                    trade_pnl = trade.realized_pnl * trade.size * trade.entry_price
                    self.capital += trade_pnl

                    # Update running metrics for Kelly
                    self._update_performance_metrics()
            else:
                logger.warning(
                    f"No price data for {symbol}. Cannot update trade {trade_id}."
                )

        # Update equity curve and timestamps
        if not self._first_update_done:
            # On the first update, just record the timestamp for the initial capital
            self.equity_timestamps.append(timestamp)
            self._first_update_done = True
            # self.equity_curve already contains initial capital, so don't append here
        else:
            # On subsequent updates, record timestamp and updated capital
            self.equity_timestamps.append(timestamp)
            self.equity_curve.append(self.capital)

        # Update drawdown
        peak_capital = max(self.equity_curve)
        self.current_drawdown = (peak_capital - self.capital) / peak_capital
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        return closed_trades

    def close_trade(
        self, trade_id: str, timestamp: Any, price: float, reason: str
    ) -> bool:
        """
        Manually close a trade.

        Args:
            trade_id: Trade ID to close
            timestamp: Exit timestamp
            price: Exit price
            reason: Reason for exit

        Returns:
            True if trade was closed, False otherwise
        """
        if trade_id not in self.active_trades:
            logger.warning(f"Trade {trade_id} not found in active trades.")
            return False

        trade = self.active_trades[trade_id]
        symbol = trade.symbol

        # Close the trade
        trade.close(timestamp, price, reason)

        # Update position
        if symbol in self.positions:
            # Reduce position
            self.positions[symbol] -= trade.size * trade.direction

            # Remove if zero
            if abs(self.positions[symbol]) < 1e-6:
                del self.positions[symbol]

        # Add to closed trades
        self.closed_trades.append(trade)

        # Remove from active trades
        del self.active_trades[trade_id]

        # Update capital
        trade_pnl = trade.realized_pnl * trade.size * trade.entry_price
        self.capital += trade_pnl

        # Update running metrics for Kelly
        self._update_performance_metrics()

        # Update equity curve
        self.equity_timestamps.append(timestamp)
        self.equity_curve.append(self.capital)

        return True

    def close_all_positions(
        self, timestamp: Any, prices: Dict[str, float], reason: str
    ) -> List[str]:
        """
        Close all active positions.

        Args:
            timestamp: Exit timestamp
            prices: Dictionary of current prices (symbol -> price)
            reason: Reason for exit

        Returns:
            List of closed trade IDs
        """
        closed_trades = []

        for trade_id, trade in list(self.active_trades.items()):
            symbol = trade.symbol

            if symbol in prices:
                # Close the trade
                self.close_trade(trade_id, timestamp, prices[symbol], reason)
                closed_trades.append(trade_id)
            else:
                logger.warning(
                    f"No price data for {symbol}. Cannot close trade {trade_id}."
                )

        return closed_trades

    def _update_performance_metrics(self) -> None:
        """
        Update running performance metrics for Kelly calculation.
        """
        if not self.closed_trades:
            return

        # Calculate win rate (last 50 trades or all if fewer)
        recent_trades = self.closed_trades[-50:]
        wins = sum(1 for trade in recent_trades if trade.realized_pnl > 0)
        self.running_win_rate = wins / len(recent_trades)

        # Calculate payoff ratio
        winning_trades = [trade for trade in recent_trades if trade.realized_pnl > 0]
        losing_trades = [trade for trade in recent_trades if trade.realized_pnl < 0]

        if winning_trades and losing_trades:
            avg_win = sum(trade.realized_pnl for trade in winning_trades) / len(
                winning_trades
            )
            avg_loss = sum(abs(trade.realized_pnl) for trade in losing_trades) / len(
                losing_trades
            )

            if avg_loss > 0:
                self.running_payoff_ratio = avg_win / avg_loss

    def add_correlated_assets(self, symbol: str, correlated_symbols: List[str]) -> None:
        """
        Add correlation information for risk management.

        Args:
            symbol: Asset symbol
            correlated_symbols: List of correlated asset symbols
        """
        self.correlated_assets[symbol] = correlated_symbols

        # Ensure symmetric relationships
        for corr_symbol in correlated_symbols:
            if corr_symbol not in self.correlated_assets:
                self.correlated_assets[corr_symbol] = [symbol]
            elif symbol not in self.correlated_assets[corr_symbol]:
                self.correlated_assets[corr_symbol].append(symbol)

    def get_trade_summary(self) -> pd.DataFrame:
        """
        Get summary of all closed trades.

        Returns:
            DataFrame with trade information
        """
        if not self.closed_trades:
            return pd.DataFrame()

        # Convert trades to dictionaries
        trade_dicts = [trade.to_dict() for trade in self.closed_trades]

        # Create DataFrame
        return pd.DataFrame(trade_dicts)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        if not self.closed_trades:
            return {
                "total_return": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
            }

        # Calculate returns from equity curve
        equity_series = pd.Series(self.equity_curve, index=self.equity_timestamps)
        returns = equity_series.pct_change().fillna(0)

        # Calculate metrics
        metrics = calculate_returns_metrics(returns)

        # Add trade-specific metrics
        trades_df = self.get_trade_summary()

        metrics["total_trades"] = len(trades_df)
        metrics["winning_trades"] = sum(trades_df["realized_pnl"] > 0)
        metrics["losing_trades"] = sum(trades_df["realized_pnl"] < 0)
        metrics["breakeven_trades"] = sum(trades_df["realized_pnl"] == 0)

        metrics["win_rate"] = (
            metrics["winning_trades"] / metrics["total_trades"]
            if metrics["total_trades"] > 0
            else 0
        )

        # Average trade metrics
        metrics["avg_trade_return"] = (
            trades_df["realized_pnl"].mean() if not trades_df.empty else 0
        )
        metrics["avg_winner"] = (
            trades_df.loc[trades_df["realized_pnl"] > 0, "realized_pnl"].mean()
            if any(trades_df["realized_pnl"] > 0)
            else 0
        )
        metrics["avg_loser"] = (
            trades_df.loc[trades_df["realized_pnl"] < 0, "realized_pnl"].mean()
            if any(trades_df["realized_pnl"] < 0)
            else 0
        )

        # Profit factor
        gross_profit = (
            trades_df.loc[trades_df["realized_pnl"] > 0, "realized_pnl"].sum()
            if any(trades_df["realized_pnl"] > 0)
            else 0
        )
        gross_loss = (
            abs(trades_df.loc[trades_df["realized_pnl"] < 0, "realized_pnl"].sum())
            if any(trades_df["realized_pnl"] < 0)
            else 0
        )

        metrics["gross_profit"] = gross_profit
        metrics["gross_loss"] = gross_loss
        metrics["profit_factor"] = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        # Duration statistics
        if "duration" in trades_df.columns and trades_df["duration"].notna().any():
            # Convert to timedeltas if not already
            if not isinstance(trades_df["duration"].iloc[0], timedelta):
                trades_df["duration"] = pd.to_timedelta(trades_df["duration"])

            # Calculate statistics
            metrics["avg_trade_duration"] = trades_df["duration"].mean()
            metrics["min_trade_duration"] = trades_df["duration"].min()
            metrics["max_trade_duration"] = trades_df["duration"].max()

        # Exit reasons
        exit_reasons = trades_df["exit_reason"].value_counts().to_dict()
        for reason, count in exit_reasons.items():
            metrics[f'exit_{reason.lower().replace(" ", "_")}'] = count

        return metrics

    def generate_report(
        self,
        output_dir: Optional[str] = None,
        benchmark_equity: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Args:
            output_dir: Directory to save report files
            benchmark_equity: Optional benchmark equity curve

        Returns:
            Dictionary with report data
        """
        # Create output directory if needed
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get performance metrics
        metrics = self.get_performance_metrics()

        # Convert equity curve to Series
        equity_series = pd.Series(self.equity_curve, index=self.equity_timestamps)
        returns = equity_series.pct_change().fillna(0)

        # Print performance summary
        self._print_performance_summary(metrics)

        # Generate plots if output directory provided
        if output_dir:
            try:
                from ..utils.visualization import (
                    plot_drawdowns,
                    plot_equity_curve,
                    plot_monthly_returns,
                )

                # Equity curve
                plot_equity_curve(
                    equity_curve=equity_series,
                    benchmark_curve=benchmark_equity,
                    title=f"Equity Curve - Return: {(self.capital / self.initial_capital - 1):.2%}",
                    filename=os.path.join(output_dir, "equity_curve.png"),
                )

                # Drawdowns
                plot_drawdowns(
                    returns=returns,
                    top_n=5,
                    filename=os.path.join(output_dir, "drawdowns.png"),
                )

                # Monthly returns heatmap
                plot_monthly_returns(
                    returns=returns,
                    filename=os.path.join(output_dir, "monthly_returns.png"),
                )
            except ImportError:
                logger.warning("Visualization modules not available for plotting")

        # Save trade summary
        if output_dir:
            trades_df = self.get_trade_summary()
            trades_df.to_csv(os.path.join(output_dir, "trades.csv"), index=False)

            # Save performance metrics
            pd.Series(metrics).to_csv(
                os.path.join(output_dir, "performance_metrics.csv")
            )

            # Save equity curve
            equity_series.to_csv(os.path.join(output_dir, "equity_curve.csv"))

        return {
            "metrics": metrics,
            "equity_curve": equity_series,
            "trades": (
                self.get_trade_summary() if self.closed_trades else pd.DataFrame()
            ),
        }

    def _print_performance_summary(self, metrics: Dict[str, Any]) -> None:
        """
        Print a formatted performance summary.

        Args:
            metrics: Dictionary of performance metrics
        """
        print("\n" + "=" * 50)
        print(" " * 15 + "PERFORMANCE SUMMARY")
        print("=" * 50)

        # Returns
        print("\n-- RETURNS --")
        print(f"Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
        print(f"Win/Loss Ratio: {metrics.get('win_loss_ratio', 0):.2f}")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")

        # Risk
        print("\n-- RISK --")
        print(f"Volatility (Ann.): {metrics.get('volatility', 0):.2%}")
        print(f"Downside Volatility: {metrics.get('downside_volatility', 0):.2%}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"VaR (95%): {metrics.get('var_95', 0):.2%}")
        print(f"CVaR (95%): {metrics.get('cvar_95', 0):.2%}")

        # Risk-Adjusted
        print("\n-- RISK-ADJUSTED METRICS --")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        print(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")

        # Trade Statistics
        print("\n-- TRADE STATISTICS --")
        print(f"Winning Days: {metrics.get('win_days', 0)}")
        print(f"Losing Days: {metrics.get('lose_days', 0)}")
        print(f"Avg. Win: {metrics.get('avg_win', 0):.2%}")
        print(f"Avg. Loss: {metrics.get('avg_loss', 0):.2%}")
        print(f"Max Consecutive Wins: {metrics.get('max_consecutive_wins', 0)}")
        print(f"Max Consecutive Losses: {metrics.get('max_consecutive_losses', 0)}")

        # Benchmark Comparison (if available)
        if "alpha" in metrics and "beta" in metrics:
            print("\n-- BENCHMARK COMPARISON --")
            print(f"Alpha (Ann.): {metrics.get('alpha', 0):.2%}")
            print(f"Beta: {metrics.get('beta', 0):.2f}")
            print(f"Information Ratio: {metrics.get('information_ratio', 0):.2f}")

        print("\n" + "=" * 50)
