# import pytest  # Not needed directly if no pytest functions used
import pandas as pd
# import numpy as np # Not used in this file

# Adjust the import path based on your project structure
# Assuming tests/ is at the same level as src/
# from ...src.main_strategy import preprocess_and_engineer_features
# Or if running pytest from the root directory (MarketAnalysis):
from market_ml_model.src.main_strategy import preprocess_and_engineer_features

# Fixtures (e.g., sample_raw_ohlcv_data) are automatically discovered
# from ../conftest.py by pytest.


# --- Test Functions for Preprocessing ---


def test_preprocess_and_engineer_features_basic(sample_raw_ohlcv_data):
    """Test basic feature engineering on a standard DataFrame."""
    ticker = "TEST"
    processed_df = preprocess_and_engineer_features(
        sample_raw_ohlcv_data.copy(), ticker
    )

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)
    assert not processed_df.isnull().any().any()  # Check for NaNs

    # Check if expected columns are added
    # Only check for indicators if pandas_ta was available
    try:
        import pandas_ta as ta
    except ImportError:
        ta = None

    if ta:
        assert ('sma_20' in processed_df.columns
                or 'SMA_20' in processed_df.columns)
        assert ('rsi_14' in processed_df.columns
                or 'RSI_14' in processed_df.columns)
        # Add checks for other indicators if needed (e.g., bbands, atr)
        # Example check for bbands middle
        assert 'BBM_20_2.0' in processed_df.columns
        assert 'ATRr_14' in processed_df.columns  # Example check for ATR
    else:
        # If ta not installed, ensure these columns *aren't* present
        assert ('sma_20' not in processed_df.columns
                and 'SMA_20' not in processed_df.columns)
        assert ('rsi_14' not in processed_df.columns
                and 'RSI_14' not in processed_df.columns)
        assert 'BBM_20_2.0' not in processed_df.columns
        assert 'ATRr_14' not in processed_df.columns

    assert 'return' in processed_df.columns
    assert 'close_lag1' in processed_df.columns
    assert 'target_direction' in processed_df.columns

    # Check target calculation logic
    # Ensure the column contains only 0s and 1s
    assert processed_df['target_direction'].isin([0, 1]).all()
    # Note: Specific value checks removed due to larger/randomized fixture


def test_preprocess_and_engineer_features_multiindex(
    sample_multiindex_raw_data
):
    """Test feature engineering handling MultiIndex input."""
    ticker = "AAPL"  # Ticker exists in the fixture
    processed_df = preprocess_and_engineer_features(
        sample_multiindex_raw_data.copy(), ticker
    )

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)
    assert not isinstance(processed_df.columns, pd.MultiIndex)  # Flattened
    assert not processed_df.isnull().any().any()

    # Check for indicators conditionally based on pandas_ta availability
    try:
        import pandas_ta as ta
    except ImportError:
        ta = None

    if ta:
        assert ('sma_20' in processed_df.columns
                or 'SMA_20' in processed_df.columns)
    else:
        assert ('sma_20' not in processed_df.columns
                and 'SMA_20' not in processed_df.columns)

    assert 'close' in processed_df.columns  # Ensure basic columns are present
    assert 'return' in processed_df.columns
    assert 'target_direction' in processed_df.columns


def test_preprocess_and_engineer_features_multiindex_ticker_not_found(
    sample_multiindex_raw_data
):
    """Test feature engineering with MultiIndex and non-existent ticker."""
    ticker = "GOOG"  # Ticker does NOT exist in the fixture
    processed_df = preprocess_and_engineer_features(
        sample_multiindex_raw_data.copy(), ticker
    )
    assert processed_df is None  # Should return None if ticker not found


def test_preprocess_and_engineer_features_empty_input():
    """Test handling of empty DataFrame input."""
    empty_df = pd.DataFrame()
    processed_df = preprocess_and_engineer_features(empty_df, "TEST")
    assert processed_df is None


def test_preprocess_and_engineer_features_missing_ohlc(sample_raw_ohlcv_data):
    """Test handling if essential OHLC columns are missing."""
    df_missing_close = sample_raw_ohlcv_data.copy().drop(columns=['Close'])
    # The function tries to lowercase columns, so check for 'close'
    processed_df_no_close = preprocess_and_engineer_features(
        df_missing_close, "TEST"
    )

    # Behavior might depend on pandas_ta. If it fails gracefully,
    # features requiring 'close' should be absent.
    if processed_df_no_close is not None:
        assert 'return' not in processed_df_no_close.columns
        assert 'target_direction' not in processed_df_no_close.columns
        # Indicators might still be calculated if they don't strictly
        # need 'close' (e.g., some use high/low/open), but many common ones do.
        try:
            import pandas_ta as ta
        except ImportError:
            ta = None
        if ta:
            # SMA, RSI, BBands typically need 'close'
            assert 'SMA_20' not in processed_df_no_close.columns
            assert 'RSI_14' not in processed_df_no_close.columns
            assert 'BBM_20_2.0' not in processed_df_no_close.columns
            # ATR uses High, Low, Close - might fail or produce NaNs
            # assert 'ATRr_14' not in processed_df_no_close.columns # Check

    # Test missing volume (less critical for some features)
    df_missing_volume = sample_raw_ohlcv_data.copy().drop(columns=['Volume'])
    processed_df_no_vol = preprocess_and_engineer_features(
        df_missing_volume, "TEST"
    )
    assert processed_df_no_vol is not None  # Should likely still process
    assert 'volume' not in processed_df_no_vol.columns
    # Check if features NOT requiring volume are still present
    assert 'return' in processed_df_no_vol.columns
    assert 'target_direction' in processed_df_no_vol.columns
    try:
        import pandas_ta as ta
    except ImportError:
        ta = None
    if ta:
        assert 'SMA_20' in processed_df_no_vol.columns  # SMA needs close