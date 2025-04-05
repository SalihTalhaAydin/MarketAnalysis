"""
Visualization utilities for backtesting and analysis.
"""

import logging
from typing import Optional, Tuple
from matplotlib.axes import Axes  # Import Axes

import numpy as np
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib/Seaborn not available. Plotting disabled.")
    PLOTTING_AVAILABLE = False


# Need calculate_drawdowns for plot_drawdowns
# (Ideally, this would be imported from utils.metrics, but keep it here for now
#  to avoid circular dependency if metrics also imports visualization)
def calculate_drawdowns(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Calculate drawdown statistics. (Copied from metrics for dependency)

    Args:
        equity_curve: Series of equity values or cumulative returns

    Returns:
        DataFrame with drawdown information
    """
    if equity_curve.empty:
        logger.warning("Equity curve is empty, cannot calculate drawdowns.")
        return pd.DataFrame(
            columns=[
                "start_date",
                "end_date",
                "lowest_point_date",
                "recovery_date",
                "max_drawdown",
                "duration_days",
                "recovery_duration_days",
            ]
        )

    # Ensure index is datetime if possible
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        try:
            equity_curve.index = pd.to_datetime(equity_curve.index)
        except Exception:
            logger.warning(
                "Could not convert index to DatetimeIndex for drawdown calculation."
            )

    # Calculate wealth index (cumulative returns starting from 1)
    wealth_index = (
        equity_curve / equity_curve.iloc[0]
        if equity_curve.iloc[0] != 0
        else (1 + equity_curve.pct_change().fillna(0)).cumprod()
    )

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
            end_date = group.index[-1]  # End of the drawdown period in data
            lowest_point_date = group.idxmin()
            max_drawdown = group.min()
            peak_value_at_low = previous_peaks[lowest_point_date]

            # Find recovery date (first time equity >= peak value at the low point)
            recovery_mask = (wealth_index.index > lowest_point_date) & (
                wealth_index >= peak_value_at_low
            )
            recovery_date = (
                wealth_index[recovery_mask].index[0] if recovery_mask.any() else None
            )

            # Calculate durations
            duration_days = np.nan
            recovery_duration_days = np.nan
            if isinstance(equity_curve.index, pd.DatetimeIndex):
                duration_days = (end_date - start_date).days
                if recovery_date:
                    recovery_duration_days = (recovery_date - lowest_point_date).days

            drawdown_periods.append(
                {
                    "start_date": start_date,
                    "end_date": end_date,  # Note: This is end of observed drawdown, not necessarily recovery
                    "lowest_point_date": lowest_point_date,
                    "recovery_date": recovery_date,
                    "max_drawdown": max_drawdown,
                    "duration_days": duration_days,
                    "recovery_duration_days": recovery_duration_days,
                }
            )

    # Create a DataFrame from drawdown periods
    if drawdown_periods:
        drawdown_df = pd.DataFrame(drawdown_periods)
        return drawdown_df.sort_values("max_drawdown")
    else:
        return pd.DataFrame(
            columns=[
                "start_date",
                "end_date",
                "lowest_point_date",
                "recovery_date",
                "max_drawdown",
                "duration_days",
                "recovery_duration_days",
            ]
        )


def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark_curve: Optional[pd.Series] = None,
    title: str = "Equity Curve",
    figsize: Tuple[int, int] = (12, 6),
    filename: Optional[str] = None,
    ax: Optional[Axes] = None,  # Added ax parameter
) -> None:
    """
    Plot equity curve and benchmark if available, optionally on a given Axes.

    Args:
        equity_curve: Series of equity values
        benchmark_curve: Optional benchmark equity curve
        title: Plot title
        figsize: Figure size (used only if ax is None)
        filename: If provided, save plot to this file (used only if ax is None)
        ax: Optional Matplotlib Axes object to plot on. If None, a new figure/axes is created.
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available. Skipping equity curve plot.")
        return

    # Determine if we need to create a figure and axes
    create_figure = ax is None
    if create_figure:
        fig, ax = plt.subplots(figsize=figsize)  # Create fig and ax if ax is None
    else:
        fig = ax.get_figure()  # Get the figure from the provided ax

    # Plot strategy equity curve using ax
    ax.plot(equity_curve.index, equity_curve.values, label="Strategy", linewidth=2)

    # Add benchmark if available
    if benchmark_curve is not None:
        # Align benchmark to strategy index if necessary
        # Align benchmark to strategy index if necessary
        aligned_benchmark = benchmark_curve.reindex(equity_curve.index).ffill()
        ax.plot(  # Use ax.plot
            aligned_benchmark.index,
            aligned_benchmark.values,
            label="Benchmark",
            linewidth=1.5,
            alpha=0.7,
            linestyle="--",
        )

    # Add drawdown shading
    if len(equity_curve) > 1:
        # Calculate drawdowns relative to the start value
        wealth_index = (
            equity_curve / equity_curve.iloc[0]
            if equity_curve.iloc[0] != 0
            else (1 + equity_curve.pct_change().fillna(0)).cumprod()
        )
        peaks = wealth_index.cummax()
        drawdowns = (wealth_index - peaks) / peaks

        # Plot drawdowns below the equity curve
        # Plot drawdowns below the equity curve using ax
        ax.fill_between(  # Use ax.fill_between
            equity_curve.index,
            equity_curve.values,  # Start fill from equity curve
            equity_curve.values
            * (1 + drawdowns.values),  # End fill based on drawdown %
            where=drawdowns < 0,  # Only fill where there is a drawdown
            alpha=0.3,
            color="red",
            label="Drawdown",
        )

    # Set plot attributes using ax
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(alpha=0.3)

    # Only manage figure layout/saving/showing/closing if we created it
    if create_figure:
        try:
            fig.tight_layout()  # Use fig.tight_layout()
            if filename:
                fig.savefig(filename, dpi=300, bbox_inches="tight")  # Use fig.savefig()
                logger.info(f"Equity curve plot saved to {filename}")
            else:
                plt.show()  # plt.show() is still okay here
        finally:
            # Ensure figure is closed even if saving/showing fails
            plt.close(fig)  # Close the specific figure


def plot_drawdowns(
    returns: pd.Series,
    top_n: int = 5,
    title: str = "Top Drawdowns",
    figsize: Tuple[int, int] = (12, 6),  # Added figsize default
    filename: Optional[str] = None,
) -> None:
    """
    Plot the top N drawdowns highlighted on the equity curve.

    Args:
        returns: Series of returns
        top_n: Number of top drawdowns to plot
        title: Plot title
        figsize: Figure size
        filename: If provided, save plot to this file
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available. Skipping drawdown plot.")
        return

    # Calculate cumulative returns (wealth index starting at 1)
    cum_returns = (1 + returns).cumprod()

    # Calculate drawdowns
    drawdown_info = calculate_drawdowns(cum_returns)  # Use the helper function

    if drawdown_info.empty:
        logger.info("No drawdowns to plot.")
        return

    # Sort by max drawdown (ascending, worst first)
    drawdown_info = drawdown_info.sort_values("max_drawdown")

    # Limit to top N
    n_drawdowns = min(top_n, len(drawdown_info))
    top_drawdowns = drawdown_info.iloc[:n_drawdowns]

    # Plot
    plt.figure(figsize=figsize)

    # Plot cumulative returns
    plt.plot(
        cum_returns.index,
        cum_returns.values,
        label="Equity Curve",
        color="blue",
        alpha=0.7,
    )

    # Highlight drawdown periods
    colors = plt.cm.viridis(np.linspace(0, 1, n_drawdowns))  # Use a different colormap

    for i, (_, row) in enumerate(top_drawdowns.iterrows()):
        start = row["start_date"]
        # Use recovery date if available, otherwise end date of observed drawdown
        end = (
            row["recovery_date"] if pd.notna(row["recovery_date"]) else row["end_date"]
        )
        low_point = row["lowest_point_date"]

        # Get the specific period for this drawdown (from start to end/recovery)
        period_mask = (cum_returns.index >= start) & (cum_returns.index <= end)
        dd_period = cum_returns[period_mask]

        # Plot the drawdown period
        plt.plot(
            dd_period.index,
            dd_period.values,
            color=colors[i],
            linewidth=2,
            label=f"DD {i + 1}: {row['max_drawdown']:.2%}",
        )

        # Mark the lowest point
        if low_point in cum_returns.index:  # Ensure low_point exists in index
            plt.scatter(
                low_point,
                cum_returns[low_point],
                color=colors[i],
                s=100,
                zorder=5,
                marker="o",
                edgecolor="black",
            )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity (Log Scale)")  # Use log scale for better visibility
    plt.yscale("log")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()  # Use tight_layout before saving/showing

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Drawdown plot saved to {filename}")
    else:
        plt.show()

    plt.close()


def plot_monthly_returns(
    returns: pd.Series,
    title: str = "Monthly Returns Heatmap",  # More descriptive title
    figsize: Tuple[int, int] = (12, 8),  # Added figsize default
    filename: Optional[str] = None,
) -> None:
    """
    Plot monthly returns heatmap.

    Args:
        returns: Series of returns (index must be DatetimeIndex)
        title: Plot title
        figsize: Figure size
        filename: If provided, save plot to this file
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available. Skipping monthly returns plot.")
        return

    # Ensure index is datetime
    if not isinstance(returns.index, pd.DatetimeIndex):
        logger.warning(
            "Returns index is not DatetimeIndex. Cannot create monthly returns plot."
        )
        return

    # Calculate monthly returns
    monthly_returns = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)

    # Create a pivot table with years as rows and months as columns
    # Handle potential multi-index from groupby if needed
    monthly_pivot = (
        monthly_returns.groupby(
            [monthly_returns.index.year, monthly_returns.index.month]
        )
        .mean()
        .unstack(level=-1)
    )

    # Plot heatmap
    plt.figure(figsize=figsize)

    sns.heatmap(
        monthly_pivot,
        annot=True,
        fmt=".1%",  # Use percentage format
        cmap="RdYlGn",  # Red-Yellow-Green colormap
        center=0,
        linewidths=0.5,  # Reduced linewidth
        linecolor="lightgrey",  # Added linecolor
        cbar_kws={"shrink": 0.8},  # Adjusted color bar size
    )

    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel("Year")

    # Set month names
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    # Ensure ticks match the columns present in the pivot table
    plt.xticks(
        np.arange(len(monthly_pivot.columns)) + 0.5,
        month_names[: len(monthly_pivot.columns)],
    )
    plt.yticks(rotation=0)  # Keep year labels horizontal
    plt.tight_layout()  # Use tight_layout before saving/showing

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Monthly returns plot saved to {filename}")
    else:
        plt.show()

    plt.close()
