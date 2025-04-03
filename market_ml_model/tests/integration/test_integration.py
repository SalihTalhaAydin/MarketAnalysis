import pytest
import io
from contextlib import redirect_stdout

# Adjust the import path based on your project structure
from market_ml_model.src.main_strategy import run_trading_strategy_analysis
# Import fixtures if needed directly (though mocker is used here)
# from ..conftest import sample_raw_ohlcv_data # Example if needed

# Note: This test relies on mocking `load_data` and potentially `pandas_ta`.
# It tests the overall flow of the `run_trading_strategy_analysis` function.


@pytest.mark.slow  # Mark as slow as it runs the full pipeline (mocked)
def test_run_trading_strategy_analysis_integration(
    mocker, sample_raw_ohlcv_data
):
    """
    Test the full workflow integration using mocked data loading.
    """
    # Mock load_data to return our sample data fixture
    # Make sure the path to patch is correct relative to where it's *used*
    # Patch load_data where it's used in main_strategy
    mocker.patch(
        'market_ml_model.src.main_strategy.load_data',
        return_value=sample_raw_ohlcv_data  # Use fixture
    )

    # Mock pandas_ta if necessary (e.g., in CI environment without it)
    # This prevents errors if pandas_ta isn't installed where tests run
    # Mock pandas_ta if necessary (e.g., in CI environment without it)
    # This prevents errors if pandas_ta isn't installed where tests run
    try:
        import pandas_ta as ta  # Check if it exists
        _ = ta  # Assign to unused var to satisfy linter
    except ImportError:
        # If pandas_ta is not installed, mock its usage within main_strategy
        # Patch the 'ta' variable *within* the main_strategy module
        mocker.patch('market_ml_model.src.main_strategy.ta', None)
        print("\nMocking pandas_ta as None for integration test.")

    # Capture print output to check logs
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            # Run the main analysis function with sample parameters
            run_trading_strategy_analysis(
                symbol="TEST_INTEGRATION",  # Use a distinct symbol
                timeframe="1d",
                start_date="2023-01-01",  # Match fixture dates
                end_date="2023-01-06"    # Match fixture dates
            )
            # If it runs without errors, it's a basic pass
            run_successful = True
        except Exception as e:
            run_successful = False
            pytest.fail(
                f"Integration test failed: {e}"
            )

    assert run_successful  # Ensure the function completed

    # Add more specific assertions based on expected print output
    output = f.getvalue()
    # print(f"\nCaptured Output:\n{output}\n") # Uncomment for debugging output
    assert "=== Running Analysis for TEST_INTEGRATION (1d) ===" in output
    assert "--- Preprocessing and Engineering Features ---" in output
    assert "--- Training Classification Model ---" in output
    assert "--- Making Predictions ---" in output
    assert "--- Backtesting Strategy (Basic Vectorized) ---" in output
    assert "--- Performance Summary (Basic Backtest) ---" in output
    # Check for a specific metric output (adjust based on sample data results)
    assert "total_strategy_return_pct" in output