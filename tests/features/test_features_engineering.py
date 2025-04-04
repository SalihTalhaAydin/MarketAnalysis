from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

# Import the function to test
from market_ml_model.features.features_engineering import engineer_features

# Define the path for patching
FEATURES_PATH = "market_ml_model.features.features_engineering"

# --- Fixtures ---


@pytest.fixture
def sample_ohlcv_df():
    """Provides a basic OHLCV DataFrame."""
    dates = pd.date_range(start="2023-01-01 09:00", periods=20, freq="H")
    data = {
        "open": np.linspace(100, 105, 20) + np.random.randn(20) * 0.1,
        "high": np.linspace(100.5, 105.5, 20) + np.random.randn(20) * 0.1,
        "low": np.linspace(99.5, 104.5, 20) + np.random.randn(20) * 0.1,
        "close": np.linspace(100.2, 105.2, 20) + np.random.randn(20) * 0.1,
        "volume": np.linspace(1000, 1500, 20).astype(int),
    }
    df = pd.DataFrame(data, index=dates)
    # Ensure high >= low, high >= close, low <= close etc.
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    return df


@pytest.fixture
def mock_indicator_df(sample_ohlcv_df):
    """DataFrame returned by mocked calculate_technical_indicators."""
    df = sample_ohlcv_df.copy()
    df["SMA_10"] = np.linspace(101, 106, 20)
    df["RSI_14"] = np.linspace(40, 60, 20)
    df["ATRr_10"] = np.linspace(0.5, 0.8, 20)  # Needed for triple barrier
    return df


@pytest.fixture
def mock_labels(
    sample_ohlcv_df,
):  # Depend on sample_ohlcv_df to get the correct index length
    """Series returned by mocked get_triple_barrier_labels."""
    # The index should match the index of the dataframe *after* feature NaNs are dropped
    # This typically means dropping the first row due to pct_change/shift
    expected_index_for_labeling = sample_ohlcv_df.index[1:]  # Index of length 19
    return pd.Series(
        np.random.choice([-1, 0, 1], size=len(expected_index_for_labeling)),
        index=expected_index_for_labeling,
    )


# --- Mocks ---


@pytest.fixture
def mock_calc_indicators(mocker, mock_indicator_df):
    """Mock for calculate_technical_indicators that adds columns."""

    def side_effect(df, *args, **kwargs):
        # Simulate adding indicator columns to the input df
        output_df = df.copy()
        for col in mock_indicator_df.columns:
            if col not in output_df.columns and col in [
                "SMA_10",
                "RSI_14",
                "ATRr_10",
            ]:  # Add only indicator cols
                # Align index before assigning
                output_df[col] = mock_indicator_df[col].reindex(output_df.index)
        return output_df

    return mocker.patch(
        f"{FEATURES_PATH}.calculate_technical_indicators", side_effect=side_effect
    )


@pytest.fixture
def mock_get_labels(mocker, mock_labels):
    """Mock for get_triple_barrier_labels."""
    return mocker.patch(
        f"{FEATURES_PATH}.get_triple_barrier_labels", return_value=mock_labels
    )


# Use the specific mocks in tests instead of a combined autouse fixture
# This makes dependencies clearer.


# --- Tests ---


# Inject specific mocks needed for each test
def test_engineer_features_basic(
    sample_ohlcv_df, mock_calc_indicators, mock_get_labels
):
    """Test basic feature generation (returns, lags, time)."""
    # This test doesn't use the mocks, but they are injected by pytest
    # Disable TA and labels for this test
    result_df = engineer_features(
        sample_ohlcv_df, additional_features=[], target_type=None  # No TA  # No labels
    )

    assert "return" in result_df.columns
    assert "log_return" in result_df.columns
    assert "close_lag1" in result_df.columns
    assert "day_of_week" in result_df.columns
    assert "hour_of_day" in result_df.columns

    # Check mocks were NOT called
    mock_calc_indicators.assert_not_called()
    mock_get_labels.assert_not_called()

    # Initial NaNs in return/lag columns are expected to be filled by the end
    assert not result_df["return"].isnull().any()
    assert not result_df["log_return"].isnull().any()
    assert not result_df["close_lag1"].isnull().any()
    # Check the first row values are filled (not necessarily 0)
    assert pd.notna(result_df["return"].iloc[0])
    assert pd.notna(result_df["log_return"].iloc[0])
    assert pd.notna(result_df["close_lag1"].iloc[0])

    # Check time features
    assert result_df["day_of_week"].iloc[0] == 6  # 2023-01-01 was a Sunday
    assert result_df["hour_of_day"].iloc[0] == 9


def test_engineer_features_with_ta(
    sample_ohlcv_df, mock_calc_indicators, mock_get_labels, mock_indicator_df
):
    """Test feature generation including technical indicators."""
    # Test uses mock_calc_indicators
    result_df = engineer_features(
        sample_ohlcv_df,
        additional_features=["basic_ta"],  # Request TA
        target_type=None,  # No labels
    )

    # Check TA mock was called
    mock_calc_indicators.assert_called_once()
    # Check that the result contains columns from the mocked indicator df
    # Check that the result contains columns from the mocked indicator df
    # Note: The final NaN handling might fill NaNs introduced by indicators
    assert "SMA_10" in result_df.columns
    assert "RSI_14" in result_df.columns
    assert "ATRr_10" in result_df.columns
    # Check basic features are still there
    assert "return" in result_df.columns  # Should exist as mock adds cols now
    assert "day_of_week" in result_df.columns
    # Check label mock was not called
    mock_get_labels.assert_not_called()


def test_engineer_features_triple_barrier(
    sample_ohlcv_df,
    mock_calc_indicators,
    mock_get_labels,
    mock_indicator_df,
    mock_labels,
):
    """Test feature generation with triple barrier labels."""
    # Test uses both mocks
    # The mock_calc_indicators fixture now uses a side_effect to add columns

    result_df = engineer_features(
        sample_ohlcv_df,
        additional_features=["basic_ta"],  # Need ATR from TA
        target_type="triple_barrier",
        atr_multiplier_tp=1.5,
        atr_multiplier_sl=0.8,
        max_holding_period=5,
    )

    # Check mocks were called
    mock_calc_indicators.assert_called_once()
    mock_get_labels.assert_called_once()

    # Check arguments passed to get_triple_barrier_labels
    call_args, call_kwargs = mock_get_labels.call_args
    # Verify the index of the data passed to the mock matches the adjusted mock_labels index
    pd.testing.assert_index_equal(call_kwargs["prices"].index, mock_labels.index)
    pd.testing.assert_index_equal(call_kwargs["highs"].index, mock_labels.index)
    pd.testing.assert_index_equal(call_kwargs["lows"].index, mock_labels.index)
    pd.testing.assert_index_equal(call_kwargs["atr"].index, mock_labels.index)
    # Check other parameters
    # Verify the index of the data passed to the mock starts after the initial NaN row
    assert call_kwargs["prices"].index[0] == sample_ohlcv_df.index[1]
    # Check other parameters
    assert call_kwargs["atr_multiplier_tp"] == 1.5
    assert call_kwargs["atr_multiplier_sl"] == 0.8
    assert call_kwargs["max_holding_period"] == 5

    # Check label column exists and has correct type
    assert "triple_barrier_label" in result_df.columns
    assert pd.api.types.is_integer_dtype(result_df["triple_barrier_label"])

    # Check that the final index matches the mock_labels index
    # (The internal dropna based on the label column should align them)
    pd.testing.assert_index_equal(result_df.index, mock_labels.index)


def test_engineer_features_directional_target(
    sample_ohlcv_df, mock_calc_indicators, mock_get_labels
):
    """Test feature generation with directional target labels."""
    # Test doesn't use mocks directly but they are injected
    holding_period = 3
    result_df = engineer_features(
        sample_ohlcv_df,
        additional_features=[],  # No TA needed
        target_type="directional",
        max_holding_period=holding_period,
    )

    assert "directional_label" in result_df.columns
    assert pd.api.types.is_integer_dtype(result_df["directional_label"])
    mock_get_labels.assert_not_called()  # Triple barrier mock not called

    # Check calculation for a specific point
    # Example: index 5, compare close at index 5 vs index 5+3=8
    close_5 = sample_ohlcv_df["close"].iloc[5]
    close_8 = sample_ohlcv_df["close"].iloc[8]
    expected_label = 0
    if close_8 / close_5 - 1 > 0.001:
        expected_label = 1
    elif close_8 / close_5 - 1 < -0.001:
        expected_label = -1
    assert result_df["directional_label"].iloc[5] == expected_label

    # Check last few rows are NOT dropped anymore because final NaN handling fills them
    assert len(result_df) == len(sample_ohlcv_df)
    # Check that the last few directional labels are 0 (due to NaN future_return being filled)
    assert (result_df["directional_label"].iloc[-holding_period:] == 0).all()


def test_engineer_features_column_sanitization(
    sample_ohlcv_df, mock_calc_indicators, mock_get_labels
):
    """Test that column names are sanitized."""
    # Test uses mock_calc_indicators side effect implicitly
    # Add a column with problematic characters
    df_dirty = sample_ohlcv_df.copy()
    df_dirty["Bad Name (%.%)"] = 1

    # Adjust the mock side effect for this test to add the dirty column name
    def side_effect_dirty(df, *args, **kwargs):
        output_df = df.copy()
        output_df["Bad Name (%.%)"] = 1
        # Add other indicators just to simulate the normal return structure
        output_df["SMA_10"] = 1
        output_df["RSI_14"] = 1
        output_df["ATRr_10"] = 1
        return output_df

    mock_calc_indicators.side_effect = side_effect_dirty

    result_df = engineer_features(
        df_dirty, additional_features=["basic_ta"], target_type=None
    )

    assert "Bad_Name" in result_df.columns
    assert "Bad Name (%.%)" not in result_df.columns


@patch(f"{FEATURES_PATH}.logger")
def test_engineer_features_missing_required_cols_barrier(
    mock_logger, sample_ohlcv_df, mock_calc_indicators, mock_get_labels
):
    """Test error handling when required columns for triple barrier are missing."""
    # Test uses mock_calc_indicators and mock_get_labels
    # Simulate indicator calculation NOT returning ATR
    mock_indicator_df_no_atr = sample_ohlcv_df.copy()
    mock_indicator_df_no_atr["SMA_10"] = 1

    # Adjust side effect to NOT add ATR
    def side_effect_no_atr(df, *args, **kwargs):
        output_df = df.copy()
        output_df["SMA_10"] = 1
        return output_df

    mock_calc_indicators.side_effect = side_effect_no_atr

    result_df = engineer_features(
        sample_ohlcv_df, additional_features=["basic_ta"], target_type="triple_barrier"
    )

    assert result_df is None
    mock_logger.error.assert_any_call(
        "Missing required columns for Triple Barrier: ['ATRr_10']"
    )
    mock_get_labels.assert_not_called()


@patch(f"{FEATURES_PATH}.logger")
def test_engineer_features_missing_close_directional(
    mock_logger, sample_ohlcv_df, mock_calc_indicators, mock_get_labels
):
    """Test error handling when 'close' is missing for directional target."""
    # Test doesn't use mocks directly
    df_no_close = sample_ohlcv_df.drop(columns=["close"])

    result_df = engineer_features(
        df_no_close, additional_features=[], target_type="directional"
    )

    assert result_df is None
    mock_logger.error.assert_called_once_with(
        "Missing 'close' column for directional target"
    )


def test_engineer_features_final_nan_handling(
    sample_ohlcv_df, mock_calc_indicators, mock_get_labels
):
    """Test that final NaNs are handled."""
    # Test uses mock_calc_indicators
    # Simulate indicator calculation returning NaNs
    mock_indicator_df_with_nan = sample_ohlcv_df.copy()
    mock_indicator_df_with_nan["SMA_10"] = np.nan  # All NaN column
    mock_indicator_df_with_nan["RSI_14"] = [np.nan] * 5 + list(
        range(15)
    )  # Leading NaNs

    # Adjust side effect to return NaNs
    def side_effect_nan(df, *args, **kwargs):
        output_df = df.copy()
        output_df["SMA_10"] = np.nan  # All NaN column
        output_df["RSI_14"] = [np.nan] * 5 + list(range(15))  # Leading NaNs
        output_df["ATRr_10"] = 1  # Add ATRr_10 as it might be needed implicitly
        return output_df

    mock_calc_indicators.side_effect = side_effect_nan

    result_df = engineer_features(
        sample_ohlcv_df,
        additional_features=["basic_ta"],
        target_type=None,  # No labels to simplify NaN checking
    )

    # Check that the final DataFrame has no NaNs
    assert not result_df.isnull().any().any()
    # Check that SMA_10 (all NaN originally) is filled with 0
    assert (result_df["SMA_10"] == 0).all()
    # Check that RSI_14 leading NaNs are filled (likely with median of non-NaN part)
    assert not result_df["RSI_14"].iloc[:5].isnull().any()


def test_engineer_features_empty_input(mock_calc_indicators, mock_get_labels):
    """Test handling of empty input DataFrame."""
    result_df = engineer_features(pd.DataFrame())
    assert result_df is None
