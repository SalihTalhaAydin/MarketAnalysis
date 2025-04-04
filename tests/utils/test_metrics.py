# tests/utils/test_metrics.py
import pandas as pd
import numpy as np
import pytest
from market_ml_model.utils.metrics import calculate_returns_metrics

# Define tolerance for floating point comparisons
TOL = 1e-6

@pytest.fixture
def sample_returns_positive():
    """Fixture for a simple positive returns series."""
    dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
    return pd.Series([0.01, 0.02, 0.005, 0.015, 0.01], index=dates, name="PositiveReturns")

@pytest.fixture
def sample_returns_mixed():
    """Fixture for a mixed positive/negative returns series."""
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    return pd.Series([0.02, -0.01, 0.03, -0.02, 0.01, 0.005, -0.005, 0.015, -0.01, 0.02], index=dates, name="MixedReturns")

@pytest.fixture
def sample_returns_empty():
    """Fixture for an empty returns series."""
    return pd.Series([], dtype=float, name="EmptyReturns")

@pytest.fixture
def sample_returns_constant():
    """Fixture for a constant returns series (zero volatility)."""
    dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
    return pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=dates, name="ConstantReturns")


def test_calculate_returns_metrics_positive(sample_returns_positive):
    """Test metrics calculation with only positive returns."""
    metrics = calculate_returns_metrics(sample_returns_positive)

    assert isinstance(metrics, dict)
    assert metrics['total_return'] == pytest.approx((1.01 * 1.02 * 1.005 * 1.015 * 1.01) - 1, abs=TOL)
    assert metrics['win_rate'] == pytest.approx(1.0, abs=TOL)
    assert metrics['win_days'] == 5
    assert metrics['lose_days'] == 0
    assert metrics['avg_win'] == pytest.approx(sample_returns_positive.mean(), abs=TOL)
    assert metrics['avg_loss'] == 0.0
    assert metrics['max_drawdown'] == pytest.approx(0.0, abs=TOL) # No drawdowns
    assert metrics['profit_factor'] == np.inf # No losses
    assert metrics['max_consecutive_wins'] == 5
    assert metrics['max_consecutive_losses'] == 0
    assert metrics['sharpe_ratio'] > 0
    assert np.isnan(metrics['sortino_ratio']) # Sortino is NaN when downside deviation is zero
    assert np.isnan(metrics['calmar_ratio']) # Calmar is NaN when max drawdown is zero

def test_calculate_returns_metrics_mixed(sample_returns_mixed):
    """Test metrics calculation with mixed returns."""
    metrics = calculate_returns_metrics(sample_returns_mixed)
    expected_total_return = (1 + sample_returns_mixed).prod() - 1

    assert isinstance(metrics, dict)
    assert metrics['total_return'] == pytest.approx(expected_total_return, abs=TOL)
    assert metrics['win_rate'] == pytest.approx(6 / 10, abs=TOL)
    assert metrics['win_days'] == 6
    assert metrics['lose_days'] == 4
    assert metrics['avg_win'] == pytest.approx(sample_returns_mixed[sample_returns_mixed > 0].mean(), abs=TOL)
    assert metrics['avg_loss'] == pytest.approx(sample_returns_mixed[sample_returns_mixed < 0].mean(), abs=TOL)
    assert metrics['max_drawdown'] < 0
    assert metrics['profit_factor'] > 0
    assert metrics['max_consecutive_wins'] == 2 # [0.01, 0.005] and [0.015, -0.01, 0.02] -> last two
    assert metrics['max_consecutive_losses'] == 1 # Individual losses
    assert 'sharpe_ratio' in metrics
    assert 'sortino_ratio' in metrics
    assert 'calmar_ratio' in metrics

def test_calculate_returns_metrics_empty(sample_returns_empty):
    """Test metrics calculation with an empty series."""
    metrics = calculate_returns_metrics(sample_returns_empty)
    assert isinstance(metrics, dict)
    assert not metrics # Should return an empty dictionary

def test_calculate_returns_metrics_constant(sample_returns_constant):
    """Test metrics calculation with constant zero returns."""
    metrics = calculate_returns_metrics(sample_returns_constant)

    assert isinstance(metrics, dict)
    assert metrics['total_return'] == pytest.approx(0.0, abs=TOL)
    assert metrics['annualized_return'] == pytest.approx(0.0, abs=TOL)
    assert metrics['win_rate'] == 0.0
    assert metrics['win_days'] == 0
    assert metrics['lose_days'] == 0
    assert metrics['avg_win'] == 0.0
    assert metrics['avg_loss'] == 0.0
    assert metrics['volatility'] == pytest.approx(0.0, abs=TOL)
    assert metrics['downside_volatility'] == 0.0
    assert metrics['max_drawdown'] == pytest.approx(0.0, abs=TOL)
    assert np.isnan(metrics['sharpe_ratio']) # Division by zero volatility
    assert np.isnan(metrics['sortino_ratio']) # Division by zero downside volatility
    assert np.isnan(metrics['calmar_ratio']) # Division by zero max drawdown
    assert metrics['profit_factor'] == 0.0 # Zero profit, zero loss
    assert metrics['win_loss_ratio'] == 0.0 # Zero avg win, zero avg loss
    assert metrics['max_consecutive_wins'] == 0
    assert metrics['max_consecutive_losses'] == 0

# TODO: Add tests for calculate_drawdowns
# TODO: Add tests for calculate_alpha_beta