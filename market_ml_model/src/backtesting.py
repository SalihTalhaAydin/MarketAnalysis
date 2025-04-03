import pandas as pd
import os  # Added for directory creation


def backtest_strategy(
    data_with_predictions: pd.DataFrame,
    transaction_cost_pct: float = 0.001,  # Example: 0.1% cost per trade
    stop_loss_pct: float | None = None,   # Stop-loss % (e.g., 0.02 for 2%)
    take_profit_pct: float | None = None,  # Take-profit % (e.g., 0.04 for 4%)
    output_trades_path: str | None = None,  # Specify trade log output
):
    """
    Performs a basic vectorized backtest for a long/short strategy.

    Assumptions:
    - Enters long (pred=1) or short (pred=-1) based on *previous* bar's signal.
    - Assumes entry at the 'close' price of the signal bar (prev bar's close).
    - Exits if signal flips/goes flat OR if stop-loss/take-profit is hit.
    - SL/TP checked against 'high'/'low' of the *current* bar.
    - If SL/TP hit, exit assumed at SL/TP price (approximated return).
    - Transaction costs apply on entries, exits, and flips.
    - Slippage is NOT modeled.

    Args:
        data_with_predictions: DataFrame with OHLC ('open', 'high', 'low',
                               'close'), and 'prediction' column (-1, 0, 1).
                               Index should be datetime.

    Returns:
        A dictionary summarizing the basic backtest performance,
        or an empty dict if errors occur.
    """
    print("--- Backtesting Strategy (Long/Short Vectorized) ---")
    sl_info = f"{stop_loss_pct*100:.2f}%" if stop_loss_pct else "None"
    tp_info = f"{take_profit_pct*100:.2f}%" if take_profit_pct else "None"
    print(f"    Tx Cost: {transaction_cost_pct*100:.3f}% | SL: {sl_info} | "
          f"TP: {tp_info}")

    required_cols = ['high', 'low', 'close', 'prediction']
    if not all(col in data_with_predictions.columns for col in required_cols):
        print(f"Error: Missing required columns for backtesting: "
              f"{required_cols}")
        return {}
    if data_with_predictions.empty:
        print("Warning: No data provided for backtesting.")
        return {}

    print(f"    Backtesting data shape: {data_with_predictions.shape}")

    # Calculate daily returns of the asset
    df = data_with_predictions.copy()  # Work on a copy
    df['daily_return'] = df['close'].pct_change(fill_method=None)

    # Shift prediction by 1 (trade based on previous bar's signal)
    # Fill NaNs from shift early and ensure integer type.
    # Assume prediction is -1, 0, or 1.
    df['signal'] = df['prediction'].shift(1).fillna(0).astype(int)

    # --- Calculate Returns with Stop-Loss and Take-Profit ---
    df['entry_price'] = df['close'].shift(1)  # Entry assumed at prior close

    # Calculate SL/TP levels only when a position is initiated or held
    if stop_loss_pct:
        df['stop_loss_price'] = df['entry_price'] * \
            (1 - stop_loss_pct * df['signal'].replace(-1, 1))
    else:
        df['stop_loss_price'] = pd.NA
    if take_profit_pct:
        df['take_profit_price'] = df['entry_price'] * \
            (1 + take_profit_pct * df['signal'].replace(-1, 1))
    else:
        df['take_profit_price'] = pd.NA

    # Correct SL/TP calculation for shorts (SL is above entry, TP is below)
    if stop_loss_pct:
        df.loc[df['signal'] == -1, 'stop_loss_price'] = \
            df['entry_price'] * (1 + stop_loss_pct)
    if take_profit_pct:
        df.loc[df['signal'] == -1, 'take_profit_price'] = \
            df['entry_price'] * (1 - take_profit_pct)

    # Calculate raw strategy return based on signal (before SL/TP adjustment)
    df['raw_strategy_return_signal'] = df['signal'] * df['daily_return']

    # Drop initial NaN rows BEFORE calculating hits/exits
    df.dropna(
        subset=['daily_return', 'entry_price', 'raw_strategy_return_signal'],
        inplace=True
    )

    if df.empty:  # Check after dropping NaNs
        print("Warning: No valid data remaining after calculating "
              "returns/signals.")
        return {}

    # --- Now calculate SL/TP hits and exits on the cleaned df ---

    # Determine if SL/TP was hit during the current bar
    # Initialize hit flags to False
    sl_hit_long = pd.Series(False, index=df.index)
    tp_hit_long = pd.Series(False, index=df.index)
    sl_hit_short = pd.Series(False, index=df.index)
    tp_hit_short = pd.Series(False, index=df.index)

    # Only perform comparison on rows where SL/TP is defined and signal matches
    if stop_loss_pct:  # Only calculate if SL is enabled
        long_sl_mask = (df['signal'] == 1) & df['stop_loss_price'].notna()
        if long_sl_mask.any():
            sl_hit_long[long_sl_mask] = (
                df.loc[long_sl_mask, 'low'] <=
                df.loc[long_sl_mask, 'stop_loss_price']
            ).astype(bool)

        short_sl_mask = (df['signal'] == -1) & df['stop_loss_price'].notna()
        if short_sl_mask.any():
            sl_hit_short[short_sl_mask] = (
                df.loc[short_sl_mask, 'high'] >=
                df.loc[short_sl_mask, 'stop_loss_price']
            ).astype(bool)  # Note: >= for short SL

    if take_profit_pct:  # Only calculate if TP is enabled
        long_tp_mask = (df['signal'] == 1) & df['take_profit_price'].notna()
        if long_tp_mask.any():
            tp_hit_long[long_tp_mask] = (
                df.loc[long_tp_mask, 'high'] >=
                df.loc[long_tp_mask, 'take_profit_price']
            ).astype(bool)

        short_tp_mask = (df['signal'] == -1) & df['take_profit_price'].notna()
        if short_tp_mask.any():
            tp_hit_short[short_tp_mask] = (
                df.loc[short_tp_mask, 'low'] <=
                df.loc[short_tp_mask, 'take_profit_price']
            ).astype(bool)  # Note: <= for short TP

    # Combine SL/TP hit flags
    sl_hit = sl_hit_long | sl_hit_short
    tp_hit = tp_hit_long | tp_hit_short

    # Adjust return for SL/TP hits (prioritize SL)
    # Note: Approximates exit at SL/TP level. More accurate needs iteration.
    df['raw_strategy_return'] = df['raw_strategy_return_signal']  # Default
    if stop_loss_pct:
        # Return is -SL% if long SL hit, +SL% if short SL hit
        df.loc[sl_hit, 'raw_strategy_return'] = -stop_loss_pct * df['signal']
    if take_profit_pct:
        # Apply TP only if SL wasn't hit on the same bar
        tp_applicable = tp_hit & ~sl_hit
        df.loc[tp_applicable, 'raw_strategy_return'] = \
            take_profit_pct * df['signal']

    # Identify changes in position state (on cleaned df)
    was_flat = df['signal'].shift(1).fillna(0) == 0
    was_long = df['signal'].shift(1) == 1
    was_short = df['signal'].shift(1) == -1

    is_long = df['signal'] == 1
    is_short = df['signal'] == -1
    is_flat = df['signal'] == 0

    # Identify entry/exit events based on signals (on cleaned df)
    long_entry = is_long & (was_flat | was_short)
    short_entry = is_short & (was_flat | was_long)
    long_exit_signal = (is_flat | is_short) & was_long  # Signal-based exit
    short_exit_signal = (is_flat | is_long) & was_short  # Signal-based exit

    # Identify actual exit events (signal change OR SL/TP hit)
    # Note: SL/TP hits force an exit regardless of the next signal
    actual_long_exit = long_exit_signal | (sl_hit_long | tp_hit_long)
    actual_short_exit = short_exit_signal | (sl_hit_short | tp_hit_short)

    # Any event incurring cost: entry or exit (signal change OR SL/TP hit)
    transaction_event = (long_entry | short_entry |
                         actual_long_exit | actual_short_exit)

    # Apply transaction costs (calculated based on events determined above)
    df['transaction_cost'] = transaction_event * transaction_cost_pct
    df['strategy_return'] = df['raw_strategy_return'] - df['transaction_cost']

    # Note: df.dropna was moved earlier

    # Already dropped NaNs earlier

    if df.empty:  # Check after dropping NaNs
        print("Warning: No valid data remaining after calculating "
              "returns/signals.")
        return {}

    # Calculate cumulative returns
    df['cumulative_market_return'] = (
        (1 + df['daily_return']).cumprod() - 1
    )
    df['cumulative_strategy_return'] = (
        (1 + df['strategy_return']).cumprod() - 1
    )

    # Calculate basic performance metrics
    total_strategy_return = df['cumulative_strategy_return'].iloc[-1]
    total_market_return = df['cumulative_market_return'].iloc[-1]

    # Use the boolean masks calculated *after* dropna for trade logging
    # These masks are already aligned with the final 'df'
    long_entry_final = long_entry
    short_entry_final = short_entry
    # For logging, we need the indices where signal-based exits occurred
    long_exit_signal_final = long_exit_signal
    short_exit_signal_final = short_exit_signal
    # And indices where SL/TP hits occurred
    sl_hit_final = sl_hit
    tp_hit_final = tp_hit

    num_long_trades = long_entry_final.sum()  # Based on signal entry
    num_short_trades = short_entry_final.sum()  # Based on signal entry
    num_trades = num_long_trades + num_short_trades
    num_sl_hits = sl_hit_final.sum()
    num_tp_hits = tp_hit_final.sum()

    # Calculate Win Rate based on periods (simpler vectorized approach)
    # A period is winning if the raw strategy return is positive while
    # holding a position
    winning_periods = df[(df['signal'] != 0) & (df['raw_strategy_return'] > 0)]
    # Count periods holding long OR short
    holding_periods = df[df['signal'] != 0]
    win_rate_periods = (len(winning_periods) / len(holding_periods) * 100
                        if not holding_periods.empty else 0)
    # Note: A more accurate win rate should be calculated from the exported
    # trades list.

    print("Basic backtesting logic complete.")

    # --- Export Trades Logic ---
    if output_trades_path:
        print(f"    Attempting to export trades to: {output_trades_path}")
        # Ensure df has datetime index for clarity, if not already
        if not isinstance(df.index, pd.DatetimeIndex):
            print(
                "Warning: DataFrame index is not DatetimeIndex. "
                "Trade times might lack context."
            )
            # Consider converting:
            # df.index = pd.to_datetime(df.index) if appropriate

        long_entry_indices = df.index[long_entry_final]
        short_entry_indices = df.index[short_entry_final]
        # Exits can be going flat or flipping directly
        # Use signal exits for pairing entries with exits
        long_exit_indices = df.index[long_exit_signal_final]
        short_exit_indices = df.index[short_exit_signal_final]

        trades_list = []
        tz = df.index.tz
        last_exit_idx = (pd.Timestamp.min.tz_localize(tz)
                         if tz else pd.Timestamp.min)

        # Process Long Trades
        for entry_idx in long_entry_indices:
            possible_exits = long_exit_indices[
                (long_exit_indices > entry_idx) &
                (long_exit_indices > last_exit_idx)
            ]
            if not possible_exits.empty:
                exit_idx = possible_exits[0]
                entry_price = df.loc[entry_idx, 'close']
                exit_price = df.loc[exit_idx, 'close']
                pnl = exit_price - entry_price
                pnl_pct = ((exit_price / entry_price - 1)
                           if entry_price != 0 else 0)
                # Check if exit corresponds to SL/TP hit approximation
                exit_reason = "Signal"
                # Check if this exit corresponds to an SL/TP hit
                # Use sl_hit/tp_hit masks directly (already aligned with df)
                if sl_hit[exit_idx]:
                    exit_reason = "StopLoss"
                    # Adjust PnL for approximate SL exit price
                    exit_price = (df.loc[entry_idx, 'entry_price'] *
                                  (1 - stop_loss_pct)
                                  if stop_loss_pct else exit_price)
                elif tp_hit[exit_idx]:
                    exit_reason = "TakeProfit"
                    # Adjust PnL for approximate TP exit price
                    exit_price = (df.loc[entry_idx, 'entry_price'] *
                                  (1 + take_profit_pct)
                                  if take_profit_pct else exit_price)

                pnl = exit_price - entry_price
                pnl_pct = ((exit_price / entry_price - 1)
                           if entry_price != 0 else 0)

                trades_list.append({
                    'Entry Timestamp': entry_idx,
                    'Entry Price': entry_price,
                    'Exit Timestamp': exit_idx,
                    'Exit Price': exit_price,  # Note: Approx SL/TP price
                    'Trade Type': 'Long',
                    'Exit Reason': exit_reason,  # Added
                    'Profit/Loss ($)': pnl,
                    'Profit/Loss (%)': pnl_pct
                })
                # Update last exit time considered
                last_exit_idx = max(last_exit_idx, exit_idx)

        # Process Short Trades
        # Reset for shorts
        tz = df.index.tz
        last_exit_idx = (pd.Timestamp.min.tz_localize(tz)
                         if tz else pd.Timestamp.min)
        for entry_idx in short_entry_indices:
            possible_exits = short_exit_indices[
                (short_exit_indices > entry_idx) &
                (short_exit_indices > last_exit_idx)
            ]
            if not possible_exits.empty:
                exit_idx = possible_exits[0]
                entry_price = df.loc[entry_idx, 'close']
                exit_price = df.loc[exit_idx, 'close']
                pnl = entry_price - exit_price  # PnL for short is Entry - Exit
                # PnL % for short
                pnl_pct = ((entry_price / exit_price - 1)
                           if exit_price != 0 else 0)
                # Check if exit corresponds to SL/TP hit approximation
                exit_reason = "Signal"
                # Check if this exit corresponds to an SL/TP hit
                if sl_hit[exit_idx]:
                    exit_reason = "StopLoss"
                    # Adjust PnL for approximate SL exit price
                    exit_price = (df.loc[entry_idx, 'entry_price'] *
                                  (1 + stop_loss_pct)
                                  if stop_loss_pct else exit_price)
                elif tp_hit[exit_idx]:
                    exit_reason = "TakeProfit"
                    # Adjust PnL for approximate TP exit price
                    exit_price = (df.loc[entry_idx, 'entry_price'] *
                                  (1 - take_profit_pct)
                                  if take_profit_pct else exit_price)

                pnl = entry_price - exit_price  # PnL for short is Entry - Exit
                pnl_pct = ((entry_price / exit_price - 1)
                           if exit_price != 0 else 0)

                trades_list.append({
                    'Entry Timestamp': entry_idx,
                    'Entry Price': entry_price,
                    'Exit Timestamp': exit_idx,
                    'Exit Price': exit_price,  # Note: Approx SL/TP price
                    'Trade Type': 'Short',
                    'Exit Reason': exit_reason,  # Added
                    'Profit/Loss ($)': pnl,
                    'Profit/Loss (%)': pnl_pct
                })
                # Update last exit time considered
                last_exit_idx = max(last_exit_idx, exit_idx)
            # else: Could log open trade at the end if needed

        if trades_list:
            # Sort trades by entry time for clarity before saving
            trades_df = pd.DataFrame(trades_list).sort_values(
                by='Entry Timestamp'
            )
            try:
                # Ensure directory exists
                output_dir = os.path.dirname(output_trades_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    print(f"    Created directory: {output_dir}")

                trades_df.to_csv(output_trades_path, index=False)
                print(
                    f"    Successfully exported {len(trades_df)} trades to "
                    f"{output_trades_path}"
                )
            except Exception as e:
                print(
                    f"Error: Failed to export trades to "
                    f"{output_trades_path} - {e}"
                )
        else:
            print("    No completed trades found to export.")
    # --- End Export Trades Logic ---

    # Calculate win rate from exported trades if available
    win_rate_trades = 0
    if 'trades_df' in locals() and not trades_df.empty:
        winning_trades = trades_df[trades_df['Profit/Loss ($)'] > 0]
        win_rate_trades = (len(winning_trades) / len(trades_df) * 100)

    performance_summary = {
        "total_strategy_return_pct": total_strategy_return * 100,
        "total_market_return_pct": total_market_return * 100,  # Buy & Hold
        "num_trades": int(num_trades),
        "num_long_trades": int(num_long_trades),
        "num_short_trades": int(num_short_trades),
        "win_rate_pct (based on trades)": win_rate_trades,  # More accurate
        "num_stop_loss_hits": int(num_sl_hits),
        "num_take_profit_hits": int(num_tp_hits),
        "stop_loss_setting_pct": (stop_loss_pct * 100
                                  if stop_loss_pct else "None"),
        "take_profit_setting_pct": (take_profit_pct * 100
                                    if take_profit_pct else "None"),
        "transaction_costs_applied_pct": transaction_cost_pct * 100,
        # --- TBD: Add Sharpe, Sortino, Max Drawdown, Profit Factor ---
    }
    return performance_summary