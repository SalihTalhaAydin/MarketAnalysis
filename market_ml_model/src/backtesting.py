import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import warnings
import time

# Setup logging
logger = logging.getLogger(__name__)

# Try to import visualization and statistical libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib/Seaborn not available. Plotting disabled.")
    PLOTTING_AVAILABLE = False

# Try to import statistical packages
try:
    from scipy import stats
    import statsmodels.api as sm
    STATS_AVAILABLE = True
except ImportError:
    logger.warning(
        "Advanced statistical packages not available. Some metrics disabled.")
    STATS_AVAILABLE = False

# Try to import pyfolio for performance analytics
try:
    import pyfolio as pf
    PYFOLIO_AVAILABLE = True
except ImportError:
    logger.warning(
        "Pyfolio not available. Advanced performance analytics disabled.")
    PYFOLIO_AVAILABLE = False


# --- Utility Functions ---

def calculate_returns_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive return-based performance metrics.

    Args:
        returns: Series of strategy returns

    Returns:
        Dictionary of performance metrics
    """
    metrics = {}

    # Basic return metrics
    metrics['total_return'] = (1 + returns).prod() - 1
    metrics['annualized_return'] = (
        1 + metrics['total_return']) ** (252 / len(returns)) - 1

    # Count winning and losing days
    win_days = returns[returns > 0]
    lose_days = returns[returns < 0]

    metrics['win_rate'] = len(win_days) / \
        len(returns) if len(returns) > 0 else 0
    metrics['win_days'] = len(win_days)
    metrics['lose_days'] = len(lose_days)

    # Average returns
    metrics['avg_return'] = returns.mean()
    metrics['avg_win'] = win_days.mean() if len(win_days) > 0 else 0
    metrics['avg_loss'] = lose_days.mean() if len(lose_days) > 0 else 0

    # Return volatility
    metrics['volatility'] = returns.std() * np.sqrt(252)  # Annualized

    # Downside risk measures
    metrics['downside_volatility'] = returns[returns < 0].std(
    ) * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0
    metrics['max_drawdown'] = (
        returns.cumsum() - returns.cumsum().cummax()).min()

    # Risk-adjusted return metrics
    if metrics['volatility'] > 0:
        metrics['sharpe_ratio'] = metrics['annualized_return'] / \
            metrics['volatility']
    else:
        metrics['sharpe_ratio'] = np.nan

    if metrics['downside_volatility'] > 0:
        metrics['sortino_ratio'] = metrics['annualized_return'] / \
            metrics['downside_volatility']
    else:
        metrics['sortino_ratio'] = np.nan

    if abs(metrics['max_drawdown']) > 0:
        metrics['calmar_ratio'] = metrics['annualized_return'] / \
            abs(metrics['max_drawdown'])
    else:
        metrics['calmar_ratio'] = np.nan

    # Value at Risk (VaR)
    metrics['var_95'] = returns.quantile(0.05)
    metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean() if len(
        returns[returns <= metrics['var_95']]) > 0 else 0

    # Profit factor and win/loss ratio
    gross_profits = win_days.sum()
    gross_losses = abs(lose_days.sum())

    if gross_losses > 0:
        metrics['profit_factor'] = gross_profits / gross_losses
    else:
        metrics['profit_factor'] = np.inf if gross_profits > 0 else 0

    if metrics['avg_loss'] != 0:
        metrics['win_loss_ratio'] = abs(
            metrics['avg_win'] / metrics['avg_loss'])
    else:
        metrics['win_loss_ratio'] = np.inf if metrics['avg_win'] > 0 else 0

    # Largest consecutive wins and losses
    # Calculate streak information
    signs = np.sign(returns)
    win_streak = (signs > 0).astype(int)
    lose_streak = (signs < 0).astype(int)

    # Identify where streaks begin/end
    win_streak_ends = np.append(0, np.where(np.diff(win_streak) != 0)[0] + 1)
    lose_streak_ends = np.append(0, np.where(np.diff(lose_streak) != 0)[0] + 1)

    # Calculate streak lengths
    win_streak_lengths = np.diff(np.append(win_streak_ends, len(win_streak)))
    lose_streak_lengths = np.diff(
        np.append(lose_streak_ends, len(lose_streak)))

    # Get only streaks of wins/losses (not zeros)
    win_streak_lengths = win_streak_lengths[win_streak[win_streak_ends] > 0]
    lose_streak_lengths = lose_streak_lengths[lose_streak[lose_streak_ends] > 0]

    metrics['max_consecutive_wins'] = max(
        win_streak_lengths) if len(win_streak_lengths) > 0 else 0
    metrics['max_consecutive_losses'] = max(
        lose_streak_lengths) if len(lose_streak_lengths) > 0 else 0

    # Statistical properties if available
    if STATS_AVAILABLE:
        # Skewness and kurtosis
        metrics['skewness'] = stats.skew(returns)
        metrics['kurtosis'] = stats.kurtosis(returns)

        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(returns.dropna())
        metrics['jb_statistic'] = jb_stat
        metrics['jb_pvalue'] = jb_pvalue

        # Autocorrelation
        if len(returns) > 1:
            metrics['autocorrelation_1d'] = returns.autocorr(1)

    return metrics


def calculate_drawdowns(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Calculate drawdown statistics.

    Args:
        equity_curve: Series of equity values or cumulative returns

    Returns:
        DataFrame with drawdown information
    """
    # Convert to returns if equity curve is provided
    if equity_curve.iloc[0] != 0:  # Equity curve (not returns)
        returns = equity_curve.pct_change().fillna(0)
    else:  # Already returns
        returns = equity_curve

    # Calculate wealth index (cumulative returns)
    if returns.iloc[0] == 0:
        # Start with 1 if first return is 0
        wealth_index = (1 + returns).cumprod()
    else:
        # Otherwise use provided equity curve
        wealth_index = equity_curve

    # Calculate previous peaks
    previous_peaks = wealth_index.cummax()

    # Calculate drawdowns
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    # Find drawdown periods
    is_drawdown = drawdowns < 0

    # Group consecutive drawdown periods
    drawdown_groups = (is_drawdown.astype(int).diff() != 0).cumsum()
    drawdown_periods = []

    for group_id in drawdown_groups[is_drawdown].unique():
        # Get the group
        group = drawdowns[drawdown_groups == group_id]

        if not group.empty:
            start_date = group.index[0]
            end_date = group.index[-1]
            lowest_point_date = group.idxmin()
            max_drawdown = group.min()

            # Find recovery date (if any)
            recovery_indices = wealth_index.index > lowest_point_date
            if any(recovery_indices) and any(wealth_index[recovery_indices] >= previous_peaks[lowest_point_date]):
                recovery_indices = wealth_index[recovery_indices] >= previous_peaks[lowest_point_date]
                recovery_date = wealth_index[recovery_indices].index[0] if any(
                    recovery_indices) else None
            else:
                recovery_date = None

            # Calculate durations
            duration_days = (end_date - start_date).days if hasattr(start_date,
                                                                    'days') else (end_date - start_date)
            recovery_duration = (
                recovery_date - lowest_point_date).days if recovery_date and hasattr(recovery_date, 'days') else np.nan

            drawdown_periods.append({
                'start_date': start_date,
                'end_date': end_date,
                'lowest_point_date': lowest_point_date,
                'recovery_date': recovery_date,
                'max_drawdown': max_drawdown,
                'duration_days': duration_days,
                'recovery_duration_days': recovery_duration
            })

    # Create a DataFrame from drawdown periods
    if drawdown_periods:
        drawdown_df = pd.DataFrame(drawdown_periods)
        return drawdown_df.sort_values('max_drawdown')
    else:
        return pd.DataFrame(columns=[
            'start_date', 'end_date', 'lowest_point_date', 'recovery_date',
            'max_drawdown', 'duration_days', 'recovery_duration_days'
        ])


def calculate_alpha_beta(returns: pd.Series, benchmark_returns: pd.Series) -> Tuple[float, float]:
    """
    Calculate alpha and beta relative to a benchmark.

    Args:
        returns: Series of strategy returns
        benchmark_returns: Series of benchmark returns

    Returns:
        Tuple of (alpha, beta)
    """
    if not STATS_AVAILABLE:
        return 0, 1

    # Align data
    aligned_data = pd.DataFrame({
        'strategy': returns,
        'benchmark': benchmark_returns
    }).dropna()

    if aligned_data.empty or len(aligned_data) < 2:
        return 0, 1

    # Add constant for alpha intercept
    X = sm.add_constant(aligned_data['benchmark'])

    # Fit regression model
    try:
        model = sm.OLS(aligned_data['strategy'], X).fit()
        alpha = model.params['const'] * 252  # Annualized alpha
        beta = model.params['benchmark']
        return alpha, beta
    except Exception as e:
        logger.warning(f"Error calculating alpha/beta: {e}")
        return 0, 1


def calculate_position_size(
    capital: float,
    signal_strength: float,
    volatility: float,
    max_risk_per_trade: float = 0.02,
    max_capital_per_trade: float = 0.25,
    use_kelly: bool = True,
    win_rate: Optional[float] = None,
    payoff_ratio: Optional[float] = None
) -> float:
    """
    Calculate position size using various methods.

    Args:
        capital: Current capital
        signal_strength: Signal strength (0-1)
        volatility: Current market volatility
        max_risk_per_trade: Maximum risk per trade (fraction of capital)
        max_capital_per_trade: Maximum capital allocation per trade
        use_kelly: Whether to use Kelly criterion
        win_rate: Historical win rate (if available)
        payoff_ratio: Historical payoff ratio (if available)

    Returns:
        Position size (fraction of capital)
    """
    # Validate inputs
    if capital <= 0 or signal_strength <= 0 or volatility <= 0:
        return 0.0

    # Kelly criterion calculation
    if use_kelly:
        # If historical win rate and payoff available, use those
        if win_rate is not None and payoff_ratio is not None:
            kelly_win_rate = win_rate
            kelly_payoff = payoff_ratio
        else:
            # Calculate edge from signal strength (0.5 = no edge, 1.0 = certain)
            edge = max(0, (signal_strength - 0.5) * 2)  # Rescale to 0-1

            # Calculate win probability from edge
            kelly_win_rate = 0.5 + edge / 2  # Conservative adjustment

            # Default payoff ratio
            kelly_payoff = 2.0  # From system's TP/SL ratio

        # Kelly formula: f* = (p * b - (1 - p)) / b
        # Where p = win probability, b = payoff ratio
        kelly_f = (kelly_win_rate * kelly_payoff -
                   (1 - kelly_win_rate)) / kelly_payoff

        # Use half Kelly for safety
        kelly_pct = max(0, kelly_f * 0.5)
    else:
        # Default fixed percentage approach
        kelly_pct = max_risk_per_trade

    # Volatility adjustment (reduce position size in volatile markets)
    # Normalize around 1% volatility
    volatility_scalar = 0.01 / max(0.001, volatility)

    # Limit adjustment range (25%-200% of base size)
    volatility_scalar = min(2.0, max(0.25, volatility_scalar))

    # Calculate position size with all constraints
    position_size = min(
        kelly_pct * volatility_scalar,  # Volatility-adjusted Kelly
        max_risk_per_trade,             # Maximum risk per trade
        max_capital_per_trade           # Maximum capital per trade
    )

    return position_size


def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark_curve: Optional[pd.Series] = None,
    title: str = "Equity Curve",
    filename: Optional[str] = None
) -> None:
    """
    Plot equity curve and benchmark if available.

    Args:
        equity_curve: Series of equity values
        benchmark_curve: Optional benchmark equity curve
        title: Plot title
        filename: If provided, save plot to this file
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available. Skipping equity curve plot.")
        return

    plt.figure(figsize=(12, 6))

    # Plot strategy equity curve
    plt.plot(equity_curve.index, equity_curve.values,
             label='Strategy', linewidth=2)

    # Add benchmark if available
    if benchmark_curve is not None:
        plt.plot(benchmark_curve.index, benchmark_curve.values, label='Benchmark',
                 linewidth=1.5, alpha=0.7, linestyle='--')

    # Add drawdown shading
    if len(equity_curve) > 1:
        # Calculate drawdowns
        peaks = equity_curve.cummax()
        drawdowns = (equity_curve - peaks) / peaks

        # Plot drawdowns
        plt.fill_between(
            equity_curve.index,
            0,
            drawdowns.values,
            alpha=0.3,
            color='red',
            label='Drawdowns'
        )

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(alpha=0.3)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Equity curve plot saved to {filename}")
    else:
        plt.tight_layout()
        plt.show()

    plt.close()


def plot_drawdowns(
    returns: pd.Series,
    top_n: int = 5,
    title: str = "Top Drawdowns",
    filename: Optional[str] = None
) -> None:
    """
    Plot the top N drawdowns.

    Args:
        returns: Series of returns
        top_n: Number of top drawdowns to plot
        title: Plot title
        filename: If provided, save plot to this file
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available. Skipping drawdown plot.")
        return

    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()

    # Calculate drawdowns
    drawdown_info = calculate_drawdowns(cum_returns)

    if drawdown_info.empty:
        logger.info("No drawdowns to plot.")
        return

    # Sort by max drawdown (ascending, worst first)
    drawdown_info = drawdown_info.sort_values('max_drawdown')

    # Limit to top N
    n_drawdowns = min(top_n, len(drawdown_info))
    top_drawdowns = drawdown_info.iloc[:n_drawdowns]

    # Plot
    plt.figure(figsize=(12, 6))

    # Plot cumulative returns
    plt.plot(cum_returns.index, cum_returns.values,
             label='Equity Curve', color='blue', alpha=0.7)

    # Highlight drawdown periods
    colors = plt.cm.tab10(np.linspace(0, 1, n_drawdowns))

    for i, (_, row) in enumerate(top_drawdowns.iterrows()):
        start = row['start_date']
        end = row['end_date']
        low_point = row['lowest_point_date']

        # Get the specific period for this drawdown
        period_mask = (cum_returns.index >= start) & (cum_returns.index <= end)
        dd_period = cum_returns[period_mask]

        # Plot the drawdown period
        plt.plot(dd_period.index, dd_period.values, color=colors[i], linewidth=2,
                 label=f"DD {i+1}: {row['max_drawdown']:.2%}")

        # Mark the lowest point
        plt.scatter(low_point, cum_returns[low_point], color=colors[i], s=100, zorder=5,
                    marker='o', edgecolor='black')

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(alpha=0.3)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Drawdown plot saved to {filename}")
    else:
        plt.tight_layout()
        plt.show()

    plt.close()


def plot_monthly_returns(
    returns: pd.Series,
    title: str = "Monthly Returns",
    filename: Optional[str] = None
) -> None:
    """
    Plot monthly returns heatmap.

    Args:
        returns: Series of returns
        title: Plot title
        filename: If provided, save plot to this file
    """
    if not PLOTTING_AVAILABLE:
        logger.warning(
            "Plotting not available. Skipping monthly returns plot.")
        return

    # Ensure index is datetime
    if not isinstance(returns.index, pd.DatetimeIndex):
        logger.warning(
            "Returns index is not DatetimeIndex. Cannot create monthly returns plot.")
        return

    # Calculate monthly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

    # Create a pivot table with years as rows and months as columns
    monthly_pivot = pd.Series(
        index=monthly_returns.index,
        data=monthly_returns.values
    ).groupby([lambda x: x.year, lambda x: x.month]).mean()

    monthly_pivot = monthly_pivot.unstack(level=-1)

    # Plot heatmap
    plt.figure(figsize=(12, 8))

    sns.heatmap(
        monthly_pivot,
        annot=True,
        fmt=".2%",
        cmap="RdYlGn",
        center=0,
        linewidths=1,
        cbar_kws={"shrink": .75}
    )

    plt.title(title)
    plt.xlabel('Month')
    plt.ylabel('Year')

    # Set month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(np.arange(12) + 0.5, month_names)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Monthly returns plot saved to {filename}")
    else:
        plt.tight_layout()
        plt.show()

    plt.close()


def print_performance_summary(metrics: Dict[str, Any]) -> None:
    """
    Print a formatted performance summary.

    Args:
        metrics: Dictionary of performance metrics
    """
    print("\n" + "="*50)
    print(" "*15 + "PERFORMANCE SUMMARY")
    print("="*50)

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
    print(
        f"Max Consecutive Losses: {metrics.get('max_consecutive_losses', 0)}")

    # Benchmark Comparison (if available)
    if 'alpha' in metrics and 'beta' in metrics:
        print("\n-- BENCHMARK COMPARISON --")
        print(f"Alpha (Ann.): {metrics.get('alpha', 0):.2%}")
        print(f"Beta: {metrics.get('beta', 0):.2f}")
        print(f"Information Ratio: {metrics.get('information_ratio', 0):.2f}")

    print("\n" + "="*50)


# --- Trade Management ---

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
        tags: Optional[List[str]] = None
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
        self.max_adverse_excursion = 0.0    # Worst drawdown seen during trade
        self.current_price = entry_price
        self.trailing_stop_price = None
        self.trailing_stop_activated = False

        # Performance tracking
        self.unrealized_pnl = 0.0
        self.realized_pnl = None
        self.duration = None

        # Initialize trailing stop if used
        if trailing_stop and direction == 1:  # Long position
            self.trailing_stop_price = entry_price * \
                (1 - trailing_stop_distance)
        elif trailing_stop and direction == -1:  # Short position
            self.trailing_stop_price = entry_price * \
                (1 + trailing_stop_distance)

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
            self.unrealized_pnl = (
                current_price - self.entry_price) / self.entry_price

            # Update max favorable/adverse excursion
            self.max_favorable_excursion = max(
                self.max_favorable_excursion, self.unrealized_pnl)
            self.max_adverse_excursion = min(
                self.max_adverse_excursion, self.unrealized_pnl)

            # Check trailing stop activation
            if (self.trailing_stop and not self.trailing_stop_activated and
                    self.unrealized_pnl >= self.trailing_stop_activation):
                self.trailing_stop_activated = True
                self.trailing_stop_price = current_price * \
                    (1 - self.trailing_stop_distance)
                logger.info(
                    f"Trailing stop activated for trade {self.id} at {self.trailing_stop_price}")

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
            if (self.trailing_stop and self.trailing_stop_activated and
                    current_price <= self.trailing_stop_price):
                self.close(timestamp, self.trailing_stop_price,
                           "Trailing Stop")
                return True

            # Check take profit
            if self.take_profit is not None and current_price >= self.take_profit:
                self.close(timestamp, self.take_profit, "Take Profit")
                return True

        elif self.direction == -1:  # Short position
            self.unrealized_pnl = (
                self.entry_price - current_price) / self.entry_price

            # Update max favorable/adverse excursion
            self.max_favorable_excursion = max(
                self.max_favorable_excursion, self.unrealized_pnl)
            self.max_adverse_excursion = min(
                self.max_adverse_excursion, self.unrealized_pnl)

            # Check trailing stop activation
            if (self.trailing_stop and not self.trailing_stop_activated and
                    self.unrealized_pnl >= self.trailing_stop_activation):
                self.trailing_stop_activated = True
                self.trailing_stop_price = current_price * \
                    (1 + self.trailing_stop_distance)
                logger.info(
                    f"Trailing stop activated for trade {self.id} at {self.trailing_stop_price}")

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
            if (self.trailing_stop and self.trailing_stop_activated and
                    current_price >= self.trailing_stop_price):
                self.close(timestamp, self.trailing_stop_price,
                           "Trailing Stop")
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
            self.realized_pnl = (
                self.exit_price - self.entry_price) / self.entry_price
        else:  # Short position
            self.realized_pnl = (self.entry_price -
                                 self.exit_price) / self.entry_price

        # Calculate duration
        if hasattr(self.exit_time, 'timestamp') and hasattr(self.entry_time, 'timestamp'):
            self.duration = self.exit_time - self.entry_time
        else:
            self.duration = None

        logger.info(f"Trade {self.id} closed: {self.symbol} {self.direction} at {self.exit_price}. "
                    f"P&L: {self.realized_pnl:.2%}, Reason: {self.exit_reason}")

    def to_dict(self) -> Dict:
        """
        Convert trade to dictionary representation.

        Returns:
            Dictionary with trade information
        """
        return {
            'id': self.id,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_time': self.entry_time,
            'entry_price': self.entry_price,
            'size': self.size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'exit_time': self.exit_time,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'realized_pnl': self.realized_pnl,
            'max_favorable_excursion': self.max_favorable_excursion,
            'max_adverse_excursion': self.max_adverse_excursion,
            'duration': self.duration,
            'tags': ','.join(self.tags) if self.tags else '',
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
        pyramiding_allowed: bool = False
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
        self.positions = {}      # symbol -> size
        self.equity_curve = [initial_capital]
        self.equity_timestamps = []
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
        current_price: float
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
                f"Maximum number of open trades ({self.max_open_trades}) reached.")
            return 0.0

        if self.current_drawdown >= self.max_drawdown_pct:
            logger.info(
                f"Maximum drawdown ({self.max_drawdown_pct:.2%}) reached. No new positions.")
            return 0.0

        # Check if position already exists and pyramiding is disabled
        if symbol in self.positions and not self.pyramiding_allowed:
            logger.info(
                f"Position already exists for {symbol} and pyramiding is disabled.")
            return 0.0

        # Check correlation limits
        if symbol in self.correlated_assets:
            # Count current active correlated trades
            correlated_symbols = self.correlated_assets[symbol]
            active_correlated = sum(
                1 for s in correlated_symbols if s in self.positions)

            if active_correlated >= self.max_correlated_trades:
                logger.info(
                    f"Maximum correlated trades ({self.max_correlated_trades}) reached for {symbol}.")
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
            payoff_ratio=self.running_payoff_ratio
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
        tags: Optional[List[str]] = None
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
                f"Invalid direction: {direction}. Must be 1 (long) or -1 (short).")
            return None

        # Calculate position size
        size = self.calculate_position_size(
            symbol, signal_strength, volatility, entry_price)

        if size <= 0:
            logger.info(
                f"Position size calculation returned {size}. No trade entered.")
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
            tags=tags
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
            f"Entered {direction} position in {symbol} at {entry_price}. Size: {size}, Trade ID: {trade_id}")
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
                    f"No price data for {symbol}. Cannot update trade {trade_id}.")

        # Update equity curve
        self.equity_timestamps.append(timestamp)
        self.equity_curve.append(self.capital)

        # Update drawdown
        peak_capital = max(self.equity_curve)
        self.current_drawdown = (peak_capital - self.capital) / peak_capital
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        return closed_trades

    def close_trade(self, trade_id: str, timestamp: Any, price: float, reason: str) -> bool:
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

    def close_all_positions(self, timestamp: Any, prices: Dict[str, float], reason: str) -> List[str]:
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
                    f"No price data for {symbol}. Cannot close trade {trade_id}.")

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
        winning_trades = [
            trade for trade in recent_trades if trade.realized_pnl > 0]
        losing_trades = [
            trade for trade in recent_trades if trade.realized_pnl < 0]

        if winning_trades and losing_trades:
            avg_win = sum(
                trade.realized_pnl for trade in winning_trades) / len(winning_trades)
            avg_loss = sum(abs(trade.realized_pnl)
                           for trade in losing_trades) / len(losing_trades)

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
                'total_return': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0
            }

        # Calculate returns from equity curve
        equity_series = pd.Series(
            self.equity_curve, index=self.equity_timestamps)
        returns = equity_series.pct_change().fillna(0)

        # Calculate metrics
        metrics = calculate_returns_metrics(returns)

        # Add trade-specific metrics
        trades_df = self.get_trade_summary()

        metrics['total_trades'] = len(trades_df)
        metrics['winning_trades'] = sum(trades_df['realized_pnl'] > 0)
        metrics['losing_trades'] = sum(trades_df['realized_pnl'] < 0)
        metrics['breakeven_trades'] = sum(trades_df['realized_pnl'] == 0)

        metrics['win_rate'] = metrics['winning_trades'] / \
            metrics['total_trades'] if metrics['total_trades'] > 0 else 0

        # Average trade metrics
        metrics['avg_trade_return'] = trades_df['realized_pnl'].mean(
        ) if not trades_df.empty else 0
        metrics['avg_winner'] = trades_df.loc[trades_df['realized_pnl'] > 0,
                                              'realized_pnl'].mean() if any(trades_df['realized_pnl'] > 0) else 0
        metrics['avg_loser'] = trades_df.loc[trades_df['realized_pnl'] < 0,
                                             'realized_pnl'].mean() if any(trades_df['realized_pnl'] < 0) else 0

        # Profit factor
        gross_profit = trades_df.loc[trades_df['realized_pnl'] > 0, 'realized_pnl'].sum(
        ) if any(trades_df['realized_pnl'] > 0) else 0
        gross_loss = abs(trades_df.loc[trades_df['realized_pnl'] < 0, 'realized_pnl'].sum(
        )) if any(trades_df['realized_pnl'] < 0) else 0

        metrics['gross_profit'] = gross_profit
        metrics['gross_loss'] = gross_loss
        metrics['profit_factor'] = gross_profit / \
            gross_loss if gross_loss > 0 else float('inf')

        # Duration statistics
        if 'duration' in trades_df.columns and trades_df['duration'].notna().any():
            # Convert to timedeltas if not already
            if not isinstance(trades_df['duration'].iloc[0], timedelta):
                trades_df['duration'] = pd.to_timedelta(trades_df['duration'])

            # Calculate statistics
            metrics['avg_trade_duration'] = trades_df['duration'].mean()
            metrics['min_trade_duration'] = trades_df['duration'].min()
            metrics['max_trade_duration'] = trades_df['duration'].max()

        # Exit reasons
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
        for reason, count in exit_reasons.items():
            metrics[f'exit_{reason.lower().replace(" ", "_")}'] = count

        return metrics

    def plot_equity_curve(
        self,
        benchmark_equity: Optional[pd.Series] = None,
        filename: Optional[str] = None
    ) -> None:
        """
        Plot the equity curve.

        Args:
            benchmark_equity: Optional benchmark equity curve
            filename: If provided, save plot to this file
        """
        if not PLOTTING_AVAILABLE:
            logger.warning(
                "Plotting not available. Skipping equity curve plot.")
            return

        equity_series = pd.Series(
            self.equity_curve, index=self.equity_timestamps)

        plot_equity_curve(
            equity_curve=equity_series,
            benchmark_curve=benchmark_equity,
            title=f"Equity Curve - Return: {(self.capital / self.initial_capital - 1):.2%}",
            filename=filename
        )

    def generate_report(
        self,
        output_dir: Optional[str] = None,
        benchmark_equity: Optional[pd.Series] = None
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
        equity_series = pd.Series(
            self.equity_curve, index=self.equity_timestamps)
        returns = equity_series.pct_change().fillna(0)

        # Print performance summary
        print_performance_summary(metrics)

        # Generate plots if output directory provided
        if output_dir and PLOTTING_AVAILABLE:
            # Equity curve
            self.plot_equity_curve(
                benchmark_equity=benchmark_equity,
                filename=os.path.join(output_dir, 'equity_curve.png')
            )

            # Drawdowns
            plot_drawdowns(
                returns=returns,
                top_n=5,
                filename=os.path.join(output_dir, 'drawdowns.png')
            )

            # Monthly returns heatmap
            plot_monthly_returns(
                returns=returns,
                filename=os.path.join(output_dir, 'monthly_returns.png')
            )

            # Trade outcomes by exit reason
            try:
                trades_df = self.get_trade_summary()

                if not trades_df.empty:
                    plt.figure(figsize=(10, 6))
                    exit_reasons = trades_df['exit_reason'].value_counts()
                    exit_reasons.plot(kind='bar')
                    plt.title('Trade Exit Reasons')
                    plt.ylabel('Count')
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(
                        output_dir, 'exit_reasons.png'), dpi=300)
                    plt.close()

                    # P&L by exit reason
                    plt.figure(figsize=(10, 6))
                    pnl_by_reason = trades_df.groupby(
                        'exit_reason')['realized_pnl'].mean()

                    # Create color map based on P&L
                    colors = ['g' if x > 0 else 'r' for x in pnl_by_reason]

                    pnl_by_reason.plot(kind='bar', color=colors)
                    plt.title('Average P&L by Exit Reason')
                    plt.ylabel('Average P&L (%)')
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(
                        output_dir, 'pnl_by_exit_reason.png'), dpi=300)
                    plt.close()
            except Exception as e:
                logger.warning(f"Error generating trade statistics plots: {e}")

        # Save trade summary
        if output_dir:
            trades_df = self.get_trade_summary()
            trades_df.to_csv(os.path.join(
                output_dir, 'trades.csv'), index=False)

            # Save performance metrics
            pd.Series(metrics).to_csv(os.path.join(
                output_dir, 'performance_metrics.csv'))

            # Save equity curve
            equity_series.to_csv(os.path.join(output_dir, 'equity_curve.csv'))

        return {
            'metrics': metrics,
            'equity_curve': equity_series,
            'trades': self.get_trade_summary() if self.closed_trades else pd.DataFrame()
        }


# --- Main Backtesting Function ---

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
                # Default fixed percentage stops
                if prev_signal == 1:  # Long signal
                    stop_loss = prev_close * \
                        (1 - 0.01) if atr_multiplier_sl is not None else None
                    take_profit = prev_close * \
                        (1 + 0.02) if atr_multiplier_tp is not None else None
                else:  # Short signal
                    stop_loss = prev_close * \
                        (1 + 0.01) if atr_multiplier_sl is not None else None
                    take_profit = prev_close * \
                        (1 - 0.02) if atr_multiplier_tp is not None else None

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
