# import pytest  # Not needed directly if no pytest functions used
import pandas as pd
# import numpy as np # Not used in this file

# Adjust the import path based on your project structure
# Assuming tests/ is at the same level as src/
# Import the new preprocessing function
from market_ml_model.src.preprocessing import preprocess_data

# Fixtures (e.g., sample_raw_ohlcv_data) are automatically discovered
# from ../conftest.py by pytest.


# --- Test Functions for Preprocessing ---


def test_preprocess_data_basic(sample_raw_ohlcv_data):
    """Test basic preprocessing on a standard DataFrame."""
    ticker = "TEST"
    processed_df = preprocess_data(sample_raw_ohlcv_data.copy(), ticker)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)
    assert not processed_df.isnull().any().any()  # Check for NaNs

    # Check basic preprocessing results
    assert 'open' in processed_df.columns  # Check columns are lowercase
    assert 'high' in processed_df.columns
    assert 'low' in processed_df.columns
    assert 'close' in processed_df.columns
    assert 'volume' in processed_df.columns

    # Check data types (assuming fixture has numeric-like data)
    assert pd.api.types.is_numeric_dtype(processed_df['close'])
    assert pd.api.types.is_numeric_dtype(processed_df['volume'])


def test_preprocess_data_multiindex(
    sample_multiindex_raw_data
):
    """Test preprocessing handling MultiIndex input (level 0)."""
    ticker = "AAPL"  # Ticker exists in the fixture
    processed_df = preprocess_data(sample_multiindex_raw_data.copy(), ticker)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)
    assert not isinstance(processed_df.columns, pd.MultiIndex)  # Flattened
    assert not processed_df.isnull().any().any()

    # Check basic columns are present and lowercase
    assert 'close' in processed_df.columns
    assert 'volume' in processed_df.columns

# --- Add this test to test_preprocessing.py ---


def test_preprocess_data_yfinance_multiindex(
    sample_yfinance_multiindex_data
):
    """Test preprocessing handling yfinance-style MultiIndex (level 1)."""
    ticker = "MSFT"  # Ticker exists in the fixture
    processed_df = preprocess_data(
        sample_yfinance_multiindex_data.copy(), ticker
    )

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)
    assert not isinstance(processed_df.columns, pd.MultiIndex)  # Flattened
    assert not processed_df.isnull().any().any()

    # Check basic columns are present and lowercase
    assert 'close' in processed_df.columns
    assert 'volume' in processed_df.columns


def test_preprocess_data_multiindex_ticker_not_found(
    sample_multiindex_raw_data
):
    """Test preprocessing with MultiIndex and non-existent ticker."""
    ticker = "GOOG"  # Ticker does NOT exist in the fixture
    processed_df = preprocess_data(sample_multiindex_raw_data.copy(), ticker)
    assert processed_df is None  # Should return None if ticker not found


def test_preprocess_data_empty_input():
    """Test handling of empty DataFrame input."""
    empty_df = pd.DataFrame()
    processed_df = preprocess_data(empty_df, "TEST")
    assert processed_df is None


def test_preprocess_data_missing_ohlc(sample_raw_ohlcv_data):
    """Test handling if essential OHLC columns are missing."""
    df_missing_close = sample_raw_ohlcv_data.copy().drop(columns=['Close'])
    # The function tries to lowercase columns, then checks for 'close'
    processed_df_no_close = preprocess_data(df_missing_close, "TEST")

    # Preprocessing should still run but might warn about missing 'close'
    # The function doesn't fail if 'close' is missing for type conversion,
    # but subsequent steps (like feature engineering) would.
    # Here, we just check if it ran without erroring out immediately.
    # A more robust test could capture warnings.
    assert processed_df_no_close is not None
    # Ensure 'close' is still missing after processing
    assert 'close' not in processed_df_no_close.columns

    # Test missing volume (less critical for some features)
    df_missing_volume = sample_raw_ohlcv_data.copy().drop(columns=['Volume'])
    processed_df_no_vol = preprocess_data(df_missing_volume, "TEST")
    assert processed_df_no_vol is not None  # Should likely still process
    assert 'volume' not in processed_df_no_vol.columns
    # Check if basic columns (except volume) are present
    assert 'close' in processed_df_no_vol.columns