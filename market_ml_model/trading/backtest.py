"""
Main backtesting functionality for trading strategies.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

# Import trade management
from .simulation import TradeManager
from .position.position_sizing import calculate_position_size
from ..utils.metrics import calculate_returns_metrics

# Setup logging
logger = logging.getLogger(__name__)


def backtest_strategy(
    data_with_predictions: pd.DataFrame,
    transaction_cost_pct: float = 0.001,
    slippage_pct_per_trade: float = 0.0005,
    initial_capital: float = 10000.0,
    risk_per_trade: float = 0.02,
    use_kelly_sizing: bool = True,
    atr_multiplier_sl: Optional[float] = 1.0,
    atr_multiplier_tp: Optional[float] = 2.0,
    atr_col: str = 'ATRr_10',
    trailing_stop_pct: Optional[float] = 0.01,
    trailing_stop_activation_pct: float = 0.01,
    max_open_trades: int = 5,
    max_drawdown_pct: float = 0.25,
    use_dynamic_stops: bool = True,
    signal_threshold: float = 0.6,
    allow_pyramiding: bool = False,
    benchmark_col: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_trades_path: Optional[str] = None,
    save_detailed_report: bool = False,
) -> Dict[str, Any]:
    """
    Advanced vectorized backtest function with comprehensive risk management.

    Args:
        data_with_predictions: DataFrame with OHLC ('open', 'high', 'low',
                              'close'), 'prediction' column (-1, 0, 1),
                              and the specified ATR column.
                              Index should be datetime.
        transaction_cost_pct: Percentage cost per transaction.
        slippage_pct_per_trade: Percentage slippage applied to each trade.
        initial_capital: Starting capital for the backtest.
        risk_per_trade: Maximum risk per trade (fraction of capital).
        use_kelly_sizing: Whether to use Kelly criterion for position sizing.
        atr_multiplier_sl: Multiplier for ATR to set stop loss distance.
        atr_multiplier_tp: Multiplier for ATR to set take profit distance.
        atr_col: Name of the column containing ATR values.
        trailing_stop_pct: Percentage for trailing stop distance (if None, disabled).
        trailing_stop_activation_pct: Percentage return to activate trailing stop.
        max_open_trades: Maximum number of open trades at any time.
        max_drawdown_pct: Maximum acceptable drawdown percentage.
        use_dynamic_stops: Whether to use dynamic stop levels based on ATR.
        signal_threshold: Threshold for signal strength (0.5-1.0).
        allow_pyramiding: Whether to allow adding to existing positions.
        benchmark_col: Column to use as benchmark for comparison (typically 'close').
        output_dir: Directory to save report files.
        output_trades_path: Specific path to save trade log CSV.
        save_detailed_report: Whether to save a detailed performance report.

    Returns:
        Dictionary summarizing the backtest performance.
    """
    logger.info("--- Starting Advanced Vectorized Backtest ---")

    # Validate inputs
    required_cols = ['high', 'low', 'close', 'prediction']

    # Add ATR column to required if dynamic stops are used
    if use_dynamic_stops and (atr_multiplier_sl is not None or atr_multiplier_tp is not None):
        required_cols.append(atr_col)

    missing = [c for c in required_cols if c not in data_with_predictions.columns]
    if missing:
        logger.error(f"Missing required columns for backtesting: {missing}")
        return {}

    if data_with_predictions.empty:
        logger.warning("No data provided for backtesting.")
        return {}

    # Ensure index is DatetimeIndex
    if not isinstance(data_with_predictions.index, pd.DatetimeIndex):
        logger.warning("Index is not DatetimeIndex. Attempting conversion...")
        try:
            data_with_predictions.index = pd.to_datetime(
                data_with_predictions.index)
        except Exception as e:
            logger.error(f"Failed to convert index to DatetimeIndex: {e}")
            logger.warning(
                "Continuing with original index, but timestamp functionality may be limited.")

    # Initialize trade manager
    trade_manager = TradeManager(
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        max_capital_per_trade=0.25,  # Max 25% of capital per trade
        use_kelly_sizing=use_kelly_sizing,
        max_open_trades=max_open_trades,
        max_correlated_trades=2,
        max_drawdown_pct=max_drawdown_pct,
        trailing_stop_enabled=trailing_stop_pct is not None,
        trailing_stop_activation=trailing_stop_activation_pct,
        trailing_stop_distance=trailing_stop_pct if trailing_stop_pct is not None else 0.01,
        pyramiding_allowed=allow_pyramiding
    )

    # Create working copy of data
    df = data_with_predictions.copy()

    # Generate sample asset name if not provided
    symbol = "BACKTEST_ASSET"

    # Track benchmark if requested
    if benchmark_col and benchmark_col in df.columns:
        benchmark_values = [initial_capital]
        benchmark_returns = df[benchmark_col].pct_change().fillna(0)
    else:
        benchmark_values = None
        benchmark_returns = None

    # Get default volatility estimate from ATR if available
    if atr_col in df.columns:
        default_volatility = df[atr_col].mean() / df['close'].mean()
    else:
        default_volatility = df['close'].pct_change().std()

    # Iterate through each bar
    for i in range(1, len(df)):
        current_idx = df.index[i]
        prev_idx = df.index[i-1]

        # Current market data
        current_close = df.iloc[i]['close']
        current_high = df.iloc[i]['high']
        current_low = df.iloc[i]['low']

        # Previous data for decisions
        prev_signal = df.iloc[i-1]['prediction']
        prev_close = df.iloc[i-1]['close']

        # Current volatility
        if atr_col in df.columns:
            current_volatility = df.iloc[i][atr_col] / current_close
        else:
            current_volatility = default_volatility

        # Update existing positions with current prices
        trade_manager.update(current_idx, {symbol: current_close})

        # Process signal for new trades
        if prev_signal != 0:  # Non-zero signal
            # Calculate stop loss and take profit prices
            if use_dynamic_stops and atr_col in df.columns:
                atr_value = df.iloc[i-1][atr_col]

                if prev_signal == 1:  # Long signal
                    stop_loss = prev_close - \
                        (atr_multiplier_sl *
                         atr_value) if atr_multiplier_sl is not None else None
                    take_profit = prev_close + \
                        (atr_multiplier_tp *
                         atr_value) if atr_multiplier_tp is not None else None
                else:  # Short signal
                    stop_loss = prev_close + \
                        (atr_multiplier_sl *
                         atr_value) if atr_multiplier_sl is not None else None
                    take_profit = prev_close - \
                        (atr_multiplier_tp *
                         atr_value) if atr_multiplier_tp is not None else None
            else:
                # Default fixed percentage stops (1% SL, 2% TP)
                # Calculate these regardless of atr_multiplier values when use_dynamic_stops is False
                if prev_signal == 1:  # Long signal
                    stop_loss = prev_close * (1 - 0.01)
                    take_profit = prev_close * (1 + 0.02)
                else:  # Short signal
                    stop_loss = prev_close * (1 + 0.01)
                    take_profit = prev_close * (1 - 0.02)
                # Fixed stops are calculated and should be used unless explicitly disabled
                # by a different mechanism (not currently implemented based on atr_multiplier being None)

            # Calculate entry price with slippage
            entry_price = prev_close * \
                (1 + slippage_pct_per_trade * prev_signal)

            # Convert signal to signal strength (simplistic approach)
            signal_strength = max(
                0.5, min(1.0, abs(prev_signal) * signal_threshold))

            # Only enter if symbol not in positions or pyramiding allowed
            can_enter = (symbol not in trade_manager.positions or
                         (allow_pyramiding and
                          (trade_manager.positions[symbol] > 0) == (prev_signal > 0)))

            if can_enter:
                # Apply transaction cost to entry
                trade_manager.capital -= trade_manager.capital * transaction_cost_pct

                # Enter position
                trade_manager.enter_position(
                    timestamp=prev_idx,
                    symbol=symbol,
                    direction=prev_signal,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    signal_strength=signal_strength,
                    volatility=current_volatility,
                    tags=['backtest']
                )

        # Update benchmark if tracking
        if benchmark_values is not None:
            benchmark_values.append(
                benchmark_values[-1] * (1 + benchmark_returns.iloc[i]))

    # Close any remaining positions at the end
    if trade_manager.active_trades:
        final_prices = {symbol: df.iloc[-1]['close']}
        trade_manager.close_all_positions(
            df.index[-1], final_prices, "End of Backtest")

    # Calculate final metrics
    metrics = trade_manager.get_performance_metrics()

    # Create benchmark equity curve if available
    if benchmark_values is not None:
        benchmark_equity = pd.Series(
            benchmark_values, index=df.index[:len(benchmark_values)])
    else:
        benchmark_equity = None

    # Generate report if requested
    if save_detailed_report or output_dir:
        output_path = output_dir or "./"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        trade_manager.generate_report(
            output_dir=output_path,
            benchmark_equity=benchmark_equity
        )

    # Export trades if requested
    if output_trades_path:
        trades_df = trade_manager.get_trade_summary()
        if not trades_df.empty:
            try:
                # Ensure directory exists
                output_dir = os.path.dirname(output_trades_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                trades_df.to_csv(output_trades_path, index=False)
                logger.info(f"Trades exported to {output_trades_path}")
            except Exception as e:
                logger.error(f"Failed to export trades: {e}")

    # Prepare summary for return
    performance_summary = {
        "total_return_pct": metrics.get("total_return", 0) * 100,
        "annualized_return_pct": metrics.get("annualized_return", 0) * 100,
        "volatility_pct": metrics.get("volatility", 0) * 100,
        "sharpe_ratio": metrics.get("sharpe_ratio", 0),
        "sortino_ratio": metrics.get("sortino_ratio", 0),
        "calmar_ratio": metrics.get("calmar_ratio", 0),
        "max_drawdown_pct": metrics.get("max_drawdown", 0) * 100,
        "win_rate_pct": metrics.get("win_rate", 0) * 100,
        "profit_factor": metrics.get("profit_factor", 0),
        "num_trades": metrics.get("total_trades", 0),
        "num_winning_trades": metrics.get("winning_trades", 0),
        "num_losing_trades": metrics.get("losing_trades", 0),
        "avg_trade_return_pct": metrics.get("avg_trade_return", 0) * 100,
        "avg_winner_pct": metrics.get("avg_winner", 0) * 100,
        "avg_loser_pct": metrics.get("avg_loser", 0) * 100,
        "transaction_costs_pct": transaction_cost_pct * 100,
        "slippage_pct": slippage_pct_per_trade * 100,
        "final_capital": trade_manager.capital,
        "stop_loss_setting": f"{atr_multiplier_sl}*ATR" if atr_multiplier_sl is not None else "None",
        "take_profit_setting": f"{atr_multiplier_tp}*ATR" if atr_multiplier_tp is not None else "None",
        "trailing_stop_setting": f"{trailing_stop_pct:.2%}" if trailing_stop_pct is not None else "None",
    }

    logger.info(f"Backtest completed: Return {performance_summary['total_return_pct']:.2f}%, "
                f"Win Rate {performance_summary['win_rate_pct']:.2f}%, "
                f"Trades {performance_summary['num_trades']}")

    return performance_summary