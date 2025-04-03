import pandas as pd


def backtest_strategy(
    data_with_predictions: pd.DataFrame,
    # Note: SL/TP implementation adds complexity, omitted in this basic version
    # stop_loss_pct: float | None = 0.02,
    # take_profit_pct: float | None = 0.04
):
    """
    Performs a basic vectorized backtest for a simple long-only strategy.

    Assumptions:
    - Buys/Holds when prediction is 1, sells/stays out when prediction is 0.
    - Trades occur at the 'close' price of the signal bar.
    - No transaction costs or slippage included.
    - Stop-loss / Take-profit are NOT implemented in this basic version.

    Args:
        data_with_predictions: DataFrame with OHLC data ('close') and
                               'prediction' column (0 or 1).

    Returns:
        A dictionary summarizing the basic backtest performance.
    """
    print("--- Backtesting Strategy (Basic Vectorized) ---")
    required_cols = ['close', 'prediction']
    if not all(col in data_with_predictions.columns for col in required_cols):
        print(f"Error: Missing required columns ({required_cols}) "
              "for backtesting.")
        return {}
    if data_with_predictions.empty:
        print("Warning: No data provided for backtesting.")
        return {}

    print(f"Backtesting data shape: {data_with_predictions.shape}")
    # sl_info = f"{stop_loss_pct*100}%" if stop_loss_pct else "None" # Omitted
    # Note: SL/TP info display omitted for basic backtest
    # print(f"Using SL={sl_info}, TP={tp_info}") # Omitted

    # Calculate daily returns of the asset
    df = data_with_predictions.copy()  # Work on a copy
    df['daily_return'] = df['close'].pct_change(fill_method=None)

    # Calculate strategy returns (hold asset when signal is 1, else 0 return)
    # Shift prediction by 1 (trade based on previous day's signal)
    df['signal'] = df['prediction'].shift(1)
    df['strategy_return'] = df['signal'] * df['daily_return']

    # Drop initial NaN rows from pct_change and shift
    subset_cols = ['daily_return', 'signal', 'strategy_return']
    df.dropna(subset=subset_cols, inplace=True)

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

    # Count trades (entry points: where signal changes from 0 to 1)
    # Ensure signal is integer type first
    df['signal'] = df['signal'].fillna(0).astype(int)
    # Count entries: where signal changes 0->1 OR first valid signal is 1
    entries = ((df['signal'] == 1) & (df['signal'].shift(1) == 0))
    # Check if the very first signal (after NaNs are dropped) is 1
    if not df.empty and df['signal'].iloc[0] == 1:
        # Manually add 1 trade if the first action is a buy
        num_trades = entries.sum() + 1
    else:
        num_trades = entries.sum()

    print("Basic backtesting logic complete.")
    performance_summary = {
        "total_strategy_return_pct": total_strategy_return * 100,
        "total_market_return_pct": total_market_return * 100,  # Buy & Hold
        "num_trades": num_trades
        # --- TBD: Add more metrics (Sharpe, Drawdown, Win Rate) ---
        # Requires more complex calculations (e.g., tracking equity curve)
    }
    return performance_summary