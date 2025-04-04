"""
Performance metrics for backtesting and strategy evaluation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple

# Setup logging
logger = logging.getLogger(__name__)

# Try to import statistical packages
try:
    from scipy import stats
    import statsmodels.api as sm
    STATS_AVAILABLE = True
except ImportError:
    logger.warning(
        "Advanced statistical packages (scipy, statsmodels) not available. Some metrics disabled.")
    STATS_AVAILABLE = False


def calculate_returns_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive return-based performance metrics.

    Args:
        returns: Series of strategy returns

    Returns:
        Dictionary of performance metrics
    """
    metrics = {}
    if returns.empty:
        logger.warning("Returns series is empty, cannot calculate metrics.")
        return metrics

    # Basic return metrics
    metrics['total_return'] = (1 + returns).prod() - 1
    metrics['annualized_return'] = (
        1 + metrics['total_return']) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0

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
    downside_returns = returns[returns < 0]
    metrics['downside_volatility'] = downside_returns.std(
    ) * np.sqrt(252) if len(downside_returns) > 0 else 0

    # Calculate cumulative returns for drawdown
    cumulative_returns = (1 + returns).cumprod()
    previous_peaks = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - previous_peaks) / previous_peaks
    metrics['max_drawdown'] = drawdowns.min() if not drawdowns.empty else 0


    # Risk-adjusted return metrics
    if metrics['volatility'] > 1e-6: # Avoid division by zero
        metrics['sharpe_ratio'] = metrics['annualized_return'] / \
            metrics['volatility']
    else:
        metrics['sharpe_ratio'] = np.nan

    if metrics['downside_volatility'] > 1e-6: # Avoid division by zero
        metrics['sortino_ratio'] = metrics['annualized_return'] / \
            metrics['downside_volatility']
    else:
        metrics['sortino_ratio'] = np.nan

    if abs(metrics['max_drawdown']) > 1e-6: # Avoid division by zero
        metrics['calmar_ratio'] = metrics['annualized_return'] / \
            abs(metrics['max_drawdown'])
    else:
        metrics['calmar_ratio'] = np.nan

    # Value at Risk (VaR)
    metrics['var_95'] = returns.quantile(0.05)
    cvar_returns = returns[returns <= metrics['var_95']]
    metrics['cvar_95'] = cvar_returns.mean() if len(cvar_returns) > 0 else 0

    # Profit factor and win/loss ratio
    gross_profits = win_days.sum()
    gross_losses = abs(lose_days.sum())

    if gross_losses > 1e-6: # Avoid division by zero
        metrics['profit_factor'] = gross_profits / gross_losses
    else:
        metrics['profit_factor'] = np.inf if gross_profits > 0 else 0

    if abs(metrics['avg_loss']) > 1e-6: # Avoid division by zero
        metrics['win_loss_ratio'] = abs(
            metrics['avg_win'] / metrics['avg_loss'])
    else:
        metrics['win_loss_ratio'] = np.inf if metrics['avg_win'] > 0 else 0

    # Largest consecutive wins and losses
    signs = np.sign(returns.fillna(0)) # Fill NaNs before sign
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
    win_streak_lengths = win_streak_lengths[win_streak[win_streak_ends] > 0] if len(win_streak_ends) > 0 else np.array([])
    lose_streak_lengths = lose_streak_lengths[lose_streak[lose_streak_ends] > 0] if len(lose_streak_ends) > 0 else np.array([])


    metrics['max_consecutive_wins'] = max(
        win_streak_lengths) if len(win_streak_lengths) > 0 else 0
    metrics['max_consecutive_losses'] = max(
        lose_streak_lengths) if len(lose_streak_lengths) > 0 else 0

    # Statistical properties if available
    if STATS_AVAILABLE:
        valid_returns = returns.dropna()
        if len(valid_returns) > 1:
            # Skewness and kurtosis
            metrics['skewness'] = stats.skew(valid_returns)
            metrics['kurtosis'] = stats.kurtosis(valid_returns) # Fisher's kurtosis (normal=0)

            # Jarque-Bera test for normality
            try:
                jb_stat, jb_pvalue = stats.jarque_bera(valid_returns)
                metrics['jb_statistic'] = jb_stat
                metrics['jb_pvalue'] = jb_pvalue
            except Exception as e:
                 logger.warning(f"Could not calculate Jarque-Bera: {e}")
                 metrics['jb_statistic'] = np.nan
                 metrics['jb_pvalue'] = np.nan


            # Autocorrelation
            metrics['autocorrelation_1d'] = valid_returns.autocorr(1)
        else:
             logger.warning("Not enough valid returns for statistical tests.")


    return metrics


def calculate_drawdowns(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Calculate drawdown statistics.

    Args:
        equity_curve: Series of equity values or cumulative returns

    Returns:
        DataFrame with drawdown information
    """
    if equity_curve.empty:
        logger.warning("Equity curve is empty, cannot calculate drawdowns.")
        return pd.DataFrame(columns=[
            'start_date', 'end_date', 'lowest_point_date', 'recovery_date',
            'max_drawdown', 'duration_days', 'recovery_duration_days'
        ])

    # Ensure index is datetime if possible
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        try:
            equity_curve.index = pd.to_datetime(equity_curve.index)
        except Exception:
             logger.warning("Could not convert index to DatetimeIndex for drawdown calculation.")


    # Calculate wealth index (cumulative returns starting from 1)
    wealth_index = equity_curve / equity_curve.iloc[0] if equity_curve.iloc[0] != 0 else (1 + equity_curve.pct_change().fillna(0)).cumprod()


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
            end_date = group.index[-1] # End of the drawdown period in data
            lowest_point_date = group.idxmin()
            max_drawdown = group.min()
            peak_value_at_low = previous_peaks[lowest_point_date]

            # Find recovery date (first time equity >= peak value at the low point)
            recovery_mask = (wealth_index.index > lowest_point_date) & (wealth_index >= peak_value_at_low)
            recovery_date = wealth_index[recovery_mask].index[0] if recovery_mask.any() else None


            # Calculate durations
            duration_days = np.nan
            recovery_duration_days = np.nan
            if isinstance(equity_curve.index, pd.DatetimeIndex):
                 duration_days = (end_date - start_date).days
                 if recovery_date:
                     recovery_duration_days = (recovery_date - lowest_point_date).days


            drawdown_periods.append({
                'start_date': start_date,
                'end_date': end_date, # Note: This is end of observed drawdown, not necessarily recovery
                'lowest_point_date': lowest_point_date,
                'recovery_date': recovery_date,
                'max_drawdown': max_drawdown,
                'duration_days': duration_days,
                'recovery_duration_days': recovery_duration_days
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
    Calculate alpha and beta relative to a benchmark using OLS regression.

    Args:
        returns: Series of strategy returns (daily or other frequency)
        benchmark_returns: Series of benchmark returns (same frequency)

    Returns:
        Tuple of (annualized_alpha, beta)
    """
    if not STATS_AVAILABLE:
        logger.warning("statsmodels not available, cannot calculate alpha/beta.")
        return np.nan, np.nan

    # Align data and drop NaNs
    aligned_data = pd.DataFrame({
        'strategy': returns,
        'benchmark': benchmark_returns
    }).dropna()

    if aligned_data.empty or len(aligned_data) < 2:
        logger.warning("Not enough overlapping data to calculate alpha/beta.")
        return np.nan, np.nan

    # Add constant for alpha intercept
    X = sm.add_constant(aligned_data['benchmark'])
    y = aligned_data['strategy']

    # Fit regression model
    try:
        model = sm.OLS(y, X).fit()
        # Alpha is the intercept (annualized assuming daily returns)
        # Beta is the coefficient of the benchmark returns
        alpha = model.params['const'] * 252
        beta = model.params['benchmark']
        return alpha, beta
    except Exception as e:
        logger.warning(f"Error calculating alpha/beta via OLS: {e}")
        return np.nan, np.nan