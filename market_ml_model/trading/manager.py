# market_ml_model/trading/manager.py
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

# Import Trade class from the same directory
from .trade import Trade

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    VISUALIZATION_AVAILABLE = True
except ImportError:
    logging.warning("Matplotlib/Seaborn not installed. Visualization unavailable.")
    VISUALIZATION_AVAILABLE = False
    plt = None
    sns = None

# Import utils (adjust path if necessary)
try:
    from ..utils.metrics import calculate_returns_metrics

    UTILS_AVAILABLE = True
except ImportError:
    logging.warning("Utils module not found. Performance reporting may be limited.")
    UTILS_AVAILABLE = False

    # Define dummy function if needed
    def calculate_returns_metrics(*args, **kwargs):
        return {}


# Import from position sizing (adjust path if necessary)
try:
    from .position.position_sizing import calculate_position_size

    POSITION_SIZING_AVAILABLE = True
except ImportError:
    logging.warning("Position sizing module not found. Using basic sizing.")
    POSITION_SIZING_AVAILABLE = False

    # Define dummy function if needed
    def calculate_position_size(*args, **kwargs):
        return 0.1  # Example fallback


logger = logging.getLogger(__name__)


class TradeManager:
    """
    Advanced trade management system with risk controls.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.02,
        max_capital_per_trade: float = 0.25,
        use_kelly_sizing: bool = True,  # Keep for potential future use in position sizing logic
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
        self.active_trades: Dict[str, Trade] = {}  # id -> Trade
        self.closed_trades: List[Trade] = []  # List of closed Trade objects
        self.positions: Dict[str, float] = {}  # symbol -> size
        # Simplified Equity Curve Tracking: Start with initial capital and timestamp=None
        self.equity_curve: List[Dict[str, Any]] = [
            {"timestamp": None, "equity": initial_capital}
        ]
        # self.equity_timestamps: List[Any] = [] # Removed - Use equity_curve directly
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = initial_capital  # Track peak equity for drawdown calc

        # Performance tracking (for potential future use, e.g., adaptive sizing)
        self.running_win_rate = 0.5  # Start with neutral assumption
        self.running_payoff_ratio = 2.0  # Start with TP/SL ratio assumption

        # Trade correlation tracking (symbol -> list of correlated symbols)
        self.correlated_assets: Dict[str, List[str]] = {}

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

    def _check_risk_limits(self, symbol: str) -> bool:
        """Check if risk limits allow entering a new trade."""
        # Check max open trades
        if len(self.active_trades) >= self.max_open_trades:
            logger.debug(
                f"Risk limit check failed: Max open trades ({self.max_open_trades}) reached."
            )
            return False

        # Check max drawdown
        if self.current_drawdown >= self.max_drawdown_pct:
            logger.debug(
                f"Risk limit check failed: Max drawdown ({self.max_drawdown_pct:.2%}) reached."
            )
            return False

        # Check if position already exists and pyramiding is disabled
        if symbol in self.positions and not self.pyramiding_allowed:
            logger.debug(
                f"Risk limit check failed: Position exists for {symbol} and pyramiding disabled."
            )
            return False

        # Check correlation limits
        if symbol in self.correlated_assets:
            correlated_symbols = self.correlated_assets[symbol]
            # Count active trades in correlated symbols (including the current symbol)
            active_correlated = sum(
                1
                for trade in self.active_trades.values()
                if trade.symbol in correlated_symbols or trade.symbol == symbol
            )
            if active_correlated >= self.max_correlated_trades:
                logger.debug(
                    f"Risk limit check failed: Max correlated trades ({self.max_correlated_trades}) reached for group including {symbol}."
                )
                return False

        return True  # All checks passed

    def calculate_position_size(
        self,
        symbol: str,
        signal_strength: float,  # Keep for potential future use
        volatility: float,  # Keep for potential future use
        entry_price: float,
        stop_loss_price: float,
    ) -> float:
        """
        Calculate appropriate position size based on risk parameters.

        Args:
            symbol: Asset symbol
            signal_strength: Strength of the signal (0-1) - currently unused but kept
            volatility: Asset volatility - currently unused but kept
            entry_price: Proposed entry price
            stop_loss_price: Proposed stop loss price

        Returns:
            Number of units (size) to trade, or 0.0 if trade not allowed.
        """
        # Check risk limits first
        if not self._check_risk_limits(symbol):
            return 0.0

        # Calculate position size using the external function
        # This function encapsulates the logic (e.g., fixed fractional, Kelly)
        if not POSITION_SIZING_AVAILABLE:
            logger.warning(
                "Using basic fixed fractional sizing due to missing position sizing module."
            )
            capital_fraction = self.risk_per_trade  # Simple fallback
        else:
            capital_fraction = calculate_position_size(
                capital=self.capital,
                signal_strength=signal_strength,
                volatility=volatility,
                max_risk_per_trade=self.risk_per_trade,
                max_capital_per_trade=self.max_capital_per_trade,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                # Add win_rate=self.running_win_rate, payoff_ratio=self.running_payoff_ratio if using Kelly
            )

        # Calculate number of units
        capital_to_allocate = self.capital * capital_fraction

        if entry_price <= 1e-9:  # Use a small epsilon instead of zero
            logger.warning(
                f"Entry price ({entry_price}) is zero or negative for {symbol}. Cannot calculate units."
            )
            return 0.0

        units = capital_to_allocate / entry_price

        logger.info(
            f"Position sizing for {symbol}: Capital={self.capital:.2f}, "
            f"Fraction={capital_fraction:.6f}, Price={entry_price:.4f}, "
            f"Stop={stop_loss_price:.4f}, Calculated Units={units:.6f}"
        )

        # Add minimum position size check if needed (e.g., from AssetConfig)
        # min_size = asset_config.min_position_size
        # if units < min_size: units = 0.0

        return units

    def enter_position(
        self,
        timestamp: Any,
        symbol: str,
        direction: int,  # 1 for long, -1 for short
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        signal_strength: float = 0.6,  # Default or from signal
        volatility: float = 0.01,  # Default or from features
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Enter a new position or add to existing position if allowed.

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

        # Calculate position size (includes risk checks)
        size = self.calculate_position_size(
            symbol, signal_strength, volatility, entry_price, stop_loss
        )

        if size <= 0:
            logger.info(
                f"Position size calculation returned {size:.6f} for {symbol}. No trade entered."
            )
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

        # Update positions dictionary (tracks net position size per symbol)
        self.positions[symbol] = self.positions.get(symbol, 0) + (size * direction)

        logger.info(
            f"Entered {'LONG' if direction == 1 else 'SHORT'} position in {symbol} at {entry_price:.4f}. "
            f"Size: {size:.6f}, SL: {stop_loss:.4f}, TP: {take_profit:.4f}, Trade ID: {trade_id}"
        )
        return trade_id

    # Removed duplicated _record_equity method definition
    def _record_equity(self, timestamp: Any):
        """Helper to record equity curve point."""
        last_entry = self.equity_curve[-1]
        # Always check the last entry
        # Check if this is the first actual timestamp being recorded
        is_first_update = (
            len(self.equity_curve) == 1 and self.equity_curve[0]["timestamp"] is None
        )

        if is_first_update:
            # This is the first update after initialization.
            # Add a *new* entry for this timestamp. Keep the initial entry as is.
            self.equity_curve.append({"timestamp": timestamp, "equity": self.capital})
        else:
            # Not the first update, check the last entry
            last_entry = self.equity_curve[-1]
            if timestamp != last_entry["timestamp"]:
                # Append new entry if timestamp is different
                self.equity_curve.append(
                    {"timestamp": timestamp, "equity": self.capital}
                )
            elif timestamp == last_entry["timestamp"]:
                # Update equity for the same timestamp
                last_entry["equity"] = self.capital

    def update(self, timestamp: Any, prices: Dict[str, float]) -> List[str]:
        """
        Update all active trades with current prices and manage exits.

        Args:
            timestamp: Current timestamp
            prices: Dictionary of current prices (symbol -> price)

        Returns:
            List of closed trade IDs during this update cycle.
        """
        closed_trade_ids = []
        # Calculate portfolio value before processing updates (for potential metrics)
        # current_portfolio_value = self.capital
        # for trade in self.active_trades.values():
        #     # ... (calculation logic if needed) ...

        # Update each active trade
        for trade_id, trade in list(
            self.active_trades.items()
        ):  # Use list() for safe iteration
            symbol = trade.symbol
            if symbol in prices:
                current_price = prices[symbol]
                should_close = trade.update(timestamp, current_price)
                if should_close:
                    closed_trade_ids.append(trade_id)
                    self.positions[symbol] -= trade.size * trade.direction
                    if abs(self.positions[symbol]) < 1e-9:
                        del self.positions[symbol]
                    self.closed_trades.append(trade)
                    del self.active_trades[trade_id]
                    trade_pnl_amount = (
                        trade.realized_pnl * trade.size * trade.entry_price
                    )
                    self.capital += trade_pnl_amount
                    self._update_performance_metrics()
            else:
                logger.warning(
                    f"No price data for {symbol} at {timestamp}. Cannot update trade {trade_id}."
                )

        # Record equity AFTER processing all updates for this timestamp
        self._record_equity(timestamp)

        # Update drawdown based on cash capital
        self.peak_equity = max(self.peak_equity, self.capital)
        if self.peak_equity > 0:  # Avoid division by zero
            self.current_drawdown = (self.peak_equity - self.capital) / self.peak_equity
        else:
            self.current_drawdown = 0.0
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        return closed_trade_ids

    def close_trade(
        self, trade_id: str, timestamp: Any, price: float, reason: str
    ) -> bool:
        """
        Manually close a specific active trade.

        Args:
            trade_id: Trade ID to close
            timestamp: Exit timestamp
            price: Exit price
            reason: Reason for exit

        Returns:
            True if trade was closed successfully, False otherwise.
        """
        if trade_id not in self.active_trades:
            logger.warning(f"Trade {trade_id} not found in active trades.")
            return False

        trade = self.active_trades[trade_id]
        symbol = trade.symbol

        # Close the trade object
        trade.close(timestamp, price, reason)

        # Update position dictionary
        self.positions[symbol] -= trade.size * trade.direction
        if abs(self.positions[symbol]) < 1e-9:
            del self.positions[symbol]

        # Move trade from active to closed
        self.closed_trades.append(trade)
        del self.active_trades[trade_id]

        # Update capital
        trade_pnl_amount = trade.realized_pnl * trade.size * trade.entry_price
        self.capital += trade_pnl_amount

        # Update running metrics (optional)
        self._update_performance_metrics()

        # Record equity AFTER processing the close for this timestamp
        self._record_equity(timestamp)

        # Update drawdown
        self.peak_equity = max(self.peak_equity, self.capital)
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - self.capital) / self.peak_equity
        else:
            self.current_drawdown = 0.0
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        return True

    def close_all_positions(
        self, timestamp: Any, prices: Dict[str, float], reason: str = "Strategy End"
    ) -> List[str]:
        """
        Close all currently active positions.

        Args:
            timestamp: Exit timestamp
            prices: Dictionary of current prices (symbol -> price)
            reason: Reason for exit

        Returns:
            List of closed trade IDs.
        """
        closed_trade_ids = []
        active_trade_ids = list(self.active_trades.keys())  # Get IDs before iterating

        for trade_id in active_trade_ids:
            trade = self.active_trades.get(trade_id)  # Check if still exists
            if trade:
                symbol = trade.symbol
                if symbol in prices:
                    if self.close_trade(trade_id, timestamp, prices[symbol], reason):
                        closed_trade_ids.append(trade_id)
                else:
                    logger.warning(
                        f"No price data for {symbol} at {timestamp}. Cannot close trade {trade_id}."
                    )
            else:
                logger.warning(
                    f"Trade {trade_id} disappeared during close_all_positions loop."
                )

        # Equity is recorded within the close_trade calls

        return closed_trade_ids

    def _update_performance_metrics(self) -> None:
        """
        Update running performance metrics (e.g., for adaptive sizing).
        Placeholder implementation.
        """
        if not self.closed_trades:
            return

        # Example: Update win rate based on last N trades
        n_recent = 50
        recent_trades = self.closed_trades[-n_recent:]
        if not recent_trades:
            return

        wins = sum(1 for trade in recent_trades if trade.realized_pnl > 0)
        self.running_win_rate = wins / len(recent_trades)

        # Example: Update payoff ratio
        winning_pnls = [
            trade.realized_pnl for trade in recent_trades if trade.realized_pnl > 0
        ]
        losing_pnls = [
            abs(trade.realized_pnl) for trade in recent_trades if trade.realized_pnl < 0
        ]

        if winning_pnls and losing_pnls:
            avg_win = sum(winning_pnls) / len(winning_pnls)
            avg_loss = sum(losing_pnls) / len(losing_pnls)
            if avg_loss > 0:
                self.running_payoff_ratio = avg_win / avg_loss
        # logger.debug(f"Updated running metrics: Win Rate={self.running_win_rate:.2f}, Payoff={self.running_payoff_ratio:.2f}")

    def add_correlated_assets(self, symbol: str, correlated_symbols: List[str]) -> None:
        """
        Define correlation relationships between assets for risk management.

        Args:
            symbol: Asset symbol
            correlated_symbols: List of asset symbols correlated with the main symbol
        """
        # Ensure the main symbol has an entry
        if symbol not in self.correlated_assets:
            self.correlated_assets[symbol] = []
        # Add unique correlated symbols
        for corr_symbol in correlated_symbols:
            if (
                corr_symbol != symbol
                and corr_symbol not in self.correlated_assets[symbol]
            ):
                self.correlated_assets[symbol].append(corr_symbol)

            # Ensure symmetric relationship
            if corr_symbol not in self.correlated_assets:
                self.correlated_assets[corr_symbol] = []
            if symbol not in self.correlated_assets[corr_symbol]:
                self.correlated_assets[corr_symbol].append(symbol)
        logger.info(
            f"Updated correlations for {symbol}: {self.correlated_assets.get(symbol)}"
        )

    def get_trade_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all closed trades.

        Returns:
            DataFrame with detailed information for each closed trade.
        """
        if not self.closed_trades:
            return pd.DataFrame()

        trade_dicts = [trade.to_dict() for trade in self.closed_trades]
        summary_df = pd.DataFrame(trade_dicts)

        # Convert relevant columns to appropriate types
        for col in ["entry_time", "exit_time"]:
            if col in summary_df.columns:
                summary_df[col] = pd.to_datetime(summary_df[col], errors="coerce")
        for col in [
            "entry_price",
            "exit_price",
            "size",
            "stop_loss",
            "take_profit",
            "realized_pnl",
            "max_favorable_excursion",
            "max_adverse_excursion",
        ]:
            if col in summary_df.columns:
                summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")
        if "duration" in summary_df.columns:
            summary_df["duration"] = pd.to_timedelta(
                summary_df["duration"], errors="coerce"
            )

        return summary_df

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return comprehensive performance metrics based on the equity curve and closed trades.

        Returns:
            Dictionary containing various performance metrics.
        """
        metrics = {}
        # Extract valid timestamps and corresponding equity values directly from equity_curve
        # Filter out the initial entry if its timestamp is still None
        valid_entries = [
            entry for entry in self.equity_curve if entry["timestamp"] is not None
        ]
        valid_timestamps = [entry["timestamp"] for entry in valid_entries]
        equity_values = [entry["equity"] for entry in valid_entries]

        # Need at least two points (initial capital + one update) for metrics
        if len(valid_timestamps) < 1 or len(equity_values) <= 1:
            logger.warning("Not enough equity data to calculate performance metrics.")
            return {"message": "Not enough equity data"}

        try:
            # Ensure index has same length as values
            equity_index = pd.to_datetime(valid_timestamps[: len(equity_values)])
            equity_series = pd.Series(equity_values, index=equity_index)
        except Exception as e:
            logger.error(
                f"Could not convert equity timestamps to datetime or align series: {e}. Using integer index."
            )
            equity_index = pd.RangeIndex(len(equity_values))
            equity_series = pd.Series(equity_values, index=equity_index)

        returns = equity_series.pct_change().fillna(0)

        # Calculate metrics using utility function if available
        if UTILS_AVAILABLE:
            metrics = calculate_returns_metrics(returns)
        else:
            # Basic metrics calculation if utils not available
            metrics["total_return_pct"] = (
                self.capital / self.initial_capital - 1
            ) * 100
            metrics["max_drawdown_pct"] = (
                self.max_drawdown * 100
            )  # Use tracked max drawdown
            # Add more basic calculations if needed

        # Add trade-specific metrics
        trades_df = self.get_trade_summary()
        num_trades = len(trades_df)
        metrics["num_trades"] = num_trades

        if num_trades > 0:
            winning_trades = trades_df[trades_df["realized_pnl"] > 0]
            losing_trades = trades_df[trades_df["realized_pnl"] < 0]
            num_winners = len(winning_trades)
            num_losers = len(losing_trades)

            metrics["num_winning_trades"] = num_winners
            metrics["num_losing_trades"] = num_losers
            metrics["win_rate_pct"] = (
                (num_winners / num_trades) * 100 if num_trades > 0 else 0
            )

            metrics["avg_trade_pnl_pct"] = trades_df["realized_pnl"].mean() * 100
            metrics["avg_win_pnl_pct"] = (
                winning_trades["realized_pnl"].mean() * 100 if num_winners > 0 else 0
            )
            metrics["avg_loss_pnl_pct"] = (
                losing_trades["realized_pnl"].mean() * 100 if num_losers > 0 else 0
            )

            gross_profit = winning_trades["realized_pnl"].sum()
            gross_loss = abs(losing_trades["realized_pnl"].sum())
            metrics["profit_factor"] = (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            )

            if "duration" in trades_df.columns and trades_df["duration"].notna().any():
                metrics["avg_trade_duration_str"] = str(trades_df["duration"].mean())

            exit_reasons = trades_df["exit_reason"].value_counts().to_dict()
            metrics["exit_reasons"] = exit_reasons
        else:
            metrics["win_rate_pct"] = 0
            metrics["profit_factor"] = 0

        return metrics

    def generate_report(
        self,
        output_dir: Optional[str] = None,
        benchmark_equity: Optional[pd.Series] = None,  # Keep benchmark option
    ) -> Dict[str, Any]:
        """
        Generate a performance report including metrics, trades, and optional plots.

        Args:
            output_dir: Directory to save report files (plots, CSVs).
            benchmark_equity: Optional benchmark equity curve for comparison plots.

        Returns:
            Dictionary containing report data (metrics, trades DataFrame, equity Series).
        """
        logger.info("Generating performance report...")
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create report directory {output_dir}: {e}")
                output_dir = None

        # Get performance metrics
        metrics = self.get_performance_metrics()

        # Prepare equity curve Series
        valid_timestamps = [ts for ts in self.equity_timestamps if ts is not None]
        equity_values = (
            self.equity_curve
            if self.equity_timestamps[0] is not None
            else self.equity_curve[1:]
        )
        equity_series = pd.Series()  # Default empty
        if len(valid_timestamps) > 0 and len(equity_values) > 0:
            try:
                equity_index = pd.to_datetime(valid_timestamps[: len(equity_values)])
                equity_series = pd.Series(equity_values, index=equity_index)
                equity_series = equity_series[equity_series.index.notna()]
            except Exception:
                logger.exception("Failed to create equity series for report.")

        # Print performance summary to console
        self._print_performance_summary(metrics)

        # Generate plots if output directory and visualization available
        plot_paths = {}
        if output_dir and VISUALIZATION_AVAILABLE and not equity_series.empty:
            logger.info(f"Generating plots in {output_dir}...")
            try:
                from ..utils.visualization import (  # Import plotting functions locally
                    plot_drawdowns,
                    plot_equity_curve,
                    plot_monthly_returns,
                )

                # Equity curve plot
                equity_plot_path = os.path.join(output_dir, "equity_curve.png")
                plot_equity_curve(
                    equity_curve=equity_series,
                    benchmark_curve=benchmark_equity,
                    title=f"Equity Curve - Return: {metrics.get('total_return_pct', 0):.2f}%",
                    filename=equity_plot_path,
                )
                plot_paths["equity_curve"] = equity_plot_path

                # Drawdowns plot
                returns = equity_series.pct_change().fillna(0)
                drawdown_plot_path = os.path.join(output_dir, "drawdowns.png")
                plot_drawdowns(
                    returns=returns,
                    top_n=5,
                    filename=drawdown_plot_path,
                )
                plot_paths["drawdowns"] = drawdown_plot_path

                # Monthly returns heatmap
                monthly_plot_path = os.path.join(output_dir, "monthly_returns.png")
                plot_monthly_returns(
                    returns=returns,
                    filename=monthly_plot_path,
                )
                plot_paths["monthly_returns"] = monthly_plot_path

            except ImportError:
                logger.warning(
                    "Visualization module not found. Skipping plot generation."
                )
            except Exception as e:
                logger.exception(f"Error generating plots: {e}")

        # Save trade summary and metrics CSVs if output directory provided
        trades_df = self.get_trade_summary()
        csv_paths = {}
        if output_dir:
            try:
                trades_csv_path = os.path.join(output_dir, "trades.csv")
                trades_df.to_csv(trades_csv_path, index=False)
                csv_paths["trades"] = trades_csv_path

                metrics_csv_path = os.path.join(output_dir, "performance_metrics.csv")
                # Convert metrics dict to Series/DataFrame for saving
                pd.Series(metrics, name="Value").to_csv(metrics_csv_path)
                csv_paths["metrics"] = metrics_csv_path

                if not equity_series.empty:
                    equity_csv_path = os.path.join(output_dir, "equity_curve.csv")
                    equity_series.to_csv(equity_csv_path, header=["Equity"])
                    csv_paths["equity_curve"] = equity_csv_path

            except Exception as e:
                logger.exception(f"Error saving report CSV files: {e}")

        return {
            "metrics": metrics,
            "equity_curve": equity_series,
            "trades": trades_df,
            "plot_paths": plot_paths,
            "csv_paths": csv_paths,
        }

    def _print_performance_summary(self, metrics: Dict[str, Any]) -> None:
        """Print a formatted performance summary to the console."""
        print("\n" + "=" * 60)
        print(" " * 20 + "PERFORMANCE SUMMARY")
        print("=" * 60)

        def print_metric(label, key, fmt=".2f"):
            value = metrics.get(key)
            if value is not None:
                try:
                    # Handle percentage formatting specifically
                    if "pct" in key or "rate" in key:
                        print(f"{label:<30}: {value:{fmt}}%")
                    else:
                        print(f"{label:<30}: {value:{fmt}}")
                except (TypeError, ValueError):
                    print(
                        f"{label:<30}: {value}"
                    )  # Print as string if formatting fails
            else:
                print(f"{label:<30}: N/A")

        # Returns
        print("\n-- Returns --")
        print_metric("Total Return (%)", "total_return_pct")
        print_metric("Annualized Return (%)", "annualized_return_pct")

        # Risk
        print("\n-- Risk --")
        print_metric("Volatility (Ann.) (%)", "annualized_volatility_pct")
        print_metric("Max Drawdown (%)", "max_drawdown_pct")
        print_metric("Sharpe Ratio", "sharpe_ratio")
        print_metric("Sortino Ratio", "sortino_ratio")
        print_metric("Calmar Ratio", "calmar_ratio")

        # Trades
        print("\n-- Trades --")
        print_metric("Total Trades", "num_trades", "d")  # Integer format
        print_metric("Win Rate (%)", "win_rate_pct")
        print_metric("Profit Factor", "profit_factor")
        print_metric("Avg Trade P&L (%)", "avg_trade_pnl_pct")
        print_metric("Avg Win P&L (%)", "avg_win_pnl_pct")
        print_metric("Avg Loss P&L (%)", "avg_loss_pnl_pct")
        print_metric(
            "Avg Trade Duration", "avg_trade_duration_str", "s"
        )  # String format

        # Exit Reasons
        if metrics.get("exit_reasons"):
            print("\n-- Exit Reasons --")
            for reason, count in metrics["exit_reasons"].items():
                print(f"{reason:<30}: {count}")

        print("\n" + "=" * 60)
