import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

# Import functions to test
from market_ml_model.data.transformations import (
    resample_data,
    align_data,
    detect_outliers,
    handle_outliers,
    preprocess_data
)

# Define the path for patching the logger
TRANSFORMATIONS_PATH = 'market_ml_model.data.transformations'

# --- Fixtures ---

@pytest.fixture
def sample_daily_df():
    """DataFrame with daily frequency."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq='D')
    data = {
        'open': np.arange(100, 110),
        'high': np.arange(101, 111) + 0.5,
        'low': np.arange(99, 109) - 0.5,
        'close': np.arange(100, 110) + 0.2,
        'volume': np.arange(1000, 1100, 10)
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def sample_df_with_nan():
    """DataFrame with some NaN values."""
    dates = pd.date_range(start="2023-01-01", periods=5, freq='D')
    data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50],
        'C': [100, 200, 300, 400, np.nan]
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def sample_df_for_outliers():
    """DataFrame with potential outliers."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq='D')
    data = {
        'value1': [10, 11, 12, 10, 11, 13, 9, 100, 11, 12], # 100 is outlier
        'value2': [20, 21, 19, 22, 20, -50, 21, 20, 18, 22], # -50 is outlier
        'constant': [5] * 10,
        'string': ['a'] * 10
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def sample_multiindex_df():
    """DataFrame with MultiIndex columns (like yfinance)."""
    dates = pd.date_range(start="2023-01-01", periods=3, freq='D')
    tickers = ['AAPL', 'MSFT']
    metrics = ['Open', 'High', 'Low', 'Close', 'Volume']
    columns = pd.MultiIndex.from_product([metrics, tickers])
    data = np.arange(3 * len(columns)).reshape(3, len(columns))
    df = pd.DataFrame(data, index=dates, columns=columns)
    # Add some variation
    df[('Close', 'AAPL')] = [150, 151, 150.5]
    df[('Volume', 'MSFT')] = [10000, 11000, 10500]
    return df


# --- Tests for resample_data ---

def test_resample_data_daily_to_weekly(sample_daily_df):
    """Test resampling daily data to weekly."""
    resampled_df = resample_data(sample_daily_df, 'W') # W for weekly

    assert isinstance(resampled_df.index, pd.DatetimeIndex)
    assert len(resampled_df) == 3 # 2023-01-01 is week 1, 02-08 is week 2, 09-10 is week 3

    # Check first week (Jan 1 only, as resampling defaults to week ending Sunday)
    week1 = resampled_df.iloc[0]
    assert week1['open'] == 100
    assert week1['high'] == 101.5
    assert week1['low'] == 98.5
    assert week1['close'] == 100.2
    assert week1['volume'] == 1000

    # Check second week (Jan 2 to Jan 8)
    week2 = resampled_df.iloc[1]
    assert week2['open'] == 101 # open of Jan 2
    assert week2['high'] == 108.5 # high of Jan 8
    assert week2['low'] == 99.5 # low of Jan 2
    assert week2['close'] == 107.2 # close of Jan 8
    assert week2['volume'] == (1010 + 1020 + 1030 + 1040 + 1050 + 1060 + 1070) # Sum for Jan 2-8

    # Check third week (Jan 9 to Jan 10)
    week3 = resampled_df.iloc[2]
    assert week3['open'] == 108 # open of Jan 9
    assert week3['high'] == 110.5 # high of Jan 10
    assert week3['low'] == 106.5 # low of Jan 9
    assert week3['close'] == 109.2 # close of Jan 10
    assert week3['volume'] == (1080 + 1090) # Sum for Jan 9-10

def test_resample_data_no_volume(sample_daily_df):
    """Test resampling without a volume column."""
    df_no_volume = sample_daily_df.drop(columns=['volume'])
    resampled_df = resample_data(df_no_volume, 'W', volume_column=None)
    assert 'volume' not in resampled_df.columns
    assert len(resampled_df) == 3

def test_resample_data_missing_ohlc(sample_daily_df):
    """Test resampling with missing OHLC columns (uses fallbacks)."""
    df_missing = sample_daily_df[['open', 'close', 'volume']].copy()
    resampled_df = resample_data(df_missing, 'W')

    # Check results for the 3 weeks (High/Low fallbacks applied)
    # Week 1 (Jan 1)
    week1 = resampled_df.iloc[0]
    assert week1['open'] == 100
    assert week1['high'] == 100.2 # close value
    assert week1['low'] == 100.2 # close value
    assert week1['close'] == 100.2
    assert week1['volume'] == 1000

    # Week 2 (Jan 2 - Jan 8)
    week2 = resampled_df.iloc[1]
    assert week2['open'] == 101
    assert week2['high'] == 107.2 # Max close in week 2
    assert week2['low'] == 101.2 # Min close in week 2
    assert week2['close'] == 107.2
    assert week2['volume'] == 7280

    # Week 3 (Jan 9 - Jan 10)
    week3 = resampled_df.iloc[2]
    assert week3['open'] == 108
    assert week3['high'] == 109.2 # Max close in week 3
    assert week3['low'] == 108.2 # Min close in week 3
    assert week3['close'] == 109.2
    assert week3['volume'] == 2170

def test_resample_data_non_datetime_index(sample_daily_df):
    """Test resampling with a non-DatetimeIndex."""
    df_non_dt = sample_daily_df.reset_index() # Index becomes RangeIndex
    df_non_dt.index = df_non_dt['index'].astype(str) # Make index strings
    df_non_dt = df_non_dt.drop(columns=['index'])

    resampled_df = resample_data(df_non_dt, 'W')
    assert isinstance(resampled_df.index, pd.DatetimeIndex)
    assert len(resampled_df) == 3

# --- Tests for align_data ---

@pytest.fixture
def dataframes_to_align():
    """Provides a list of DataFrames with slightly different indices."""
    dates1 = pd.date_range('2023-01-01', periods=5, freq='D')
    df1 = pd.DataFrame({'A': range(5)}, index=dates1)

    dates2 = pd.date_range('2023-01-03', periods=5, freq='D')
    df2 = pd.DataFrame({'B': range(10, 15)}, index=dates2)

    dates3 = pd.date_range('2023-01-02', periods=4, freq='D')
    df3 = pd.DataFrame({'C': range(20, 24)}, index=dates3)
    return [df1, df2, df3]

def test_align_data_inner(dataframes_to_align):
    """Test align_data with inner join."""
    aligned = align_data(dataframes_to_align, join_method='inner')
    common_index = pd.to_datetime(['2023-01-03', '2023-01-04', '2023-01-05'])
    for df in aligned:
        pd.testing.assert_index_equal(df.index, common_index)
    # Check values at common dates
    assert aligned[0].loc['2023-01-03', 'A'] == 2
    assert aligned[1].loc['2023-01-03', 'B'] == 10
    assert aligned[2].loc['2023-01-03', 'C'] == 21

def test_align_data_outer(dataframes_to_align):
    """Test align_data with outer join."""
    aligned = align_data(dataframes_to_align, join_method='outer')
    full_index = pd.to_datetime([
        '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
        '2023-01-05', '2023-01-06', '2023-01-07'
    ])
    for df in aligned:
        pd.testing.assert_index_equal(df.index, full_index)
    # Check values and NaNs
    assert aligned[0].loc['2023-01-01', 'A'] == 0
    assert pd.isna(aligned[1].loc['2023-01-01', 'B'])
    assert pd.isna(aligned[2].loc['2023-01-01', 'C'])
    assert aligned[1].loc['2023-01-07', 'B'] == 14
    assert pd.isna(aligned[0].loc['2023-01-07', 'A'])

def test_align_data_left(dataframes_to_align):
    """Test align_data with left join."""
    aligned = align_data(dataframes_to_align, join_method='left')
    left_index = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    for df in aligned:
        pd.testing.assert_index_equal(df.index, left_index)
    # Check values and NaNs
    assert aligned[0].loc['2023-01-01', 'A'] == 0
    assert pd.isna(aligned[1].loc['2023-01-01', 'B'])
    assert aligned[2].loc['2023-01-02', 'C'] == 20
    assert pd.isna(aligned[1].loc['2023-01-02', 'B'])

def test_align_data_right(dataframes_to_align):
    """Test align_data with right join (uses last df index)."""
    aligned = align_data(dataframes_to_align, join_method='right')
    right_index = pd.to_datetime(['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    for df in aligned:
        pd.testing.assert_index_equal(df.index, right_index)
     # Check values and NaNs
    assert aligned[2].loc['2023-01-02', 'C'] == 20
    assert aligned[0].loc['2023-01-02', 'A'] == 1
    assert pd.isna(aligned[1].loc['2023-01-02', 'B'])

def test_align_data_empty_list():
    """Test align_data with an empty list."""
    assert align_data([]) == []

# --- Tests for detect_outliers ---

def test_detect_outliers_zscore(sample_df_for_outliers):
    """Test outlier detection using z-score."""
    outliers = detect_outliers(sample_df_for_outliers, method='zscore', threshold=2.0)
    assert outliers.loc['2023-01-08', 'value1'] == True # 100 is outlier
    assert outliers.loc['2023-01-06', 'value2'] == True # -50 is outlier
    assert not outliers['constant'].any() # Constant column has no outliers
    assert not outliers['string'].any() # String column ignored
    assert outliers.sum().sum() == 2 # Total outliers detected

def test_detect_outliers_iqr(sample_df_for_outliers):
    """Test outlier detection using IQR."""
    outliers = detect_outliers(sample_df_for_outliers, method='iqr', threshold=1.5)
    assert outliers.loc['2023-01-08', 'value1'] == True # 100 is outlier
    assert outliers.loc['2023-01-06', 'value2'] == True # -50 is outlier
    assert not outliers['constant'].any()
    assert not outliers['string'].any()
    assert outliers.sum().sum() == 2

def test_detect_outliers_percentile(sample_df_for_outliers):
    """Test outlier detection using percentile."""
    # Threshold 10 means keep 10th to 90th percentile
    outliers = detect_outliers(sample_df_for_outliers, method='percentile', threshold=10)
    assert outliers.loc['2023-01-08', 'value1'] == True # 100 is outlier (highest)
    assert outliers.loc['2023-01-07', 'value1'] == True # 9 is outlier (lowest)
    assert outliers.loc['2023-01-06', 'value2'] == True # -50 is outlier (lowest)
    # Depending on exact percentile calculation, highest value2 might also be outlier
    assert not outliers['constant'].any()
    assert not outliers['string'].any()
    assert outliers['value1'].sum() == 2
    assert outliers['value2'].sum() >= 1 # Allow for potential upper outlier too

def test_detect_outliers_specific_columns(sample_df_for_outliers):
    """Test outlier detection on specific columns."""
    outliers = detect_outliers(sample_df_for_outliers, columns=['value1'], method='zscore', threshold=2.0)
    assert outliers.loc['2023-01-08', 'value1'] == True
    assert not outliers['value2'].any() # value2 should not have been checked
    assert not outliers['constant'].any()
    assert not outliers['string'].any()
    assert outliers.sum().sum() == 1

def test_detect_outliers_zero_std_iqr(sample_df_for_outliers):
    """Test outlier detection handles zero std dev / IQR."""
    outliers_z = detect_outliers(sample_df_for_outliers[['constant']], method='zscore')
    outliers_iqr = detect_outliers(sample_df_for_outliers[['constant']], method='iqr')
    assert not outliers_z['constant'].any()
    assert not outliers_iqr['constant'].any()

@patch(f'{TRANSFORMATIONS_PATH}.logger')
def test_detect_outliers_unsupported_method(mock_logger, sample_df_for_outliers):
    """Test detect_outliers with unsupported method."""
    outliers = detect_outliers(sample_df_for_outliers, method='unsupported')
    mock_logger.error.assert_called_with("Unsupported outlier detection method: unsupported")
    # Should return all False
    assert not outliers.any().any()

# --- Tests for handle_outliers ---

@pytest.fixture
def outlier_mask(sample_df_for_outliers):
    """Provides a sample outlier mask."""
    mask = pd.DataFrame(False, index=sample_df_for_outliers.index, columns=sample_df_for_outliers.columns)
    mask.loc['2023-01-08', 'value1'] = True
    mask.loc['2023-01-06', 'value2'] = True
    return mask

def test_handle_outliers_winsorize(sample_df_for_outliers, outlier_mask):
    """Test handling outliers using winsorize."""
    handled_df = handle_outliers(sample_df_for_outliers, outlier_mask, method='winsorize', lower_percentile=5, upper_percentile=95)
    # Calculate expected bounds based on non-outlier data
    valid_v1 = sample_df_for_outliers.loc[~outlier_mask['value1'], 'value1']
    lower_v1 = valid_v1.quantile(0.05)
    upper_v1 = valid_v1.quantile(0.95)
    valid_v2 = sample_df_for_outliers.loc[~outlier_mask['value2'], 'value2']
    lower_v2 = valid_v2.quantile(0.05)
    upper_v2 = valid_v2.quantile(0.95)

    # Check if outliers were replaced by bounds
    assert handled_df.loc['2023-01-08', 'value1'] == upper_v1
    assert handled_df.loc['2023-01-06', 'value2'] == lower_v2
    # Check non-outlier values remain unchanged
    assert handled_df.loc['2023-01-01', 'value1'] == sample_df_for_outliers.loc['2023-01-01', 'value1']
    assert handled_df.loc['2023-01-01', 'value2'] == sample_df_for_outliers.loc['2023-01-01', 'value2']
    assert (handled_df['string'] == sample_df_for_outliers['string']).all() # String column untouched

def test_handle_outliers_clip(sample_df_for_outliers, outlier_mask):
    """Test handling outliers using clip."""
    handled_df = handle_outliers(sample_df_for_outliers, outlier_mask, method='clip')
    # Calculate expected bounds based on non-outlier data min/max
    valid_v1 = sample_df_for_outliers.loc[~outlier_mask['value1'], 'value1']
    min_v1, max_v1 = valid_v1.min(), valid_v1.max()
    valid_v2 = sample_df_for_outliers.loc[~outlier_mask['value2'], 'value2']
    min_v2, max_v2 = valid_v2.min(), valid_v2.max()

    assert handled_df.loc['2023-01-08', 'value1'] == max_v1
    assert handled_df.loc['2023-01-06', 'value2'] == min_v2
    assert handled_df.loc['2023-01-01', 'value1'] == sample_df_for_outliers.loc['2023-01-01', 'value1']
    assert (handled_df['string'] == sample_df_for_outliers['string']).all()

def test_handle_outliers_remove(sample_df_for_outliers, outlier_mask):
    """Test handling outliers using remove."""
    handled_df = handle_outliers(sample_df_for_outliers, outlier_mask, method='remove')
    # Rows with outliers should be removed
    assert '2023-01-08' not in handled_df.index
    assert '2023-01-06' not in handled_df.index
    assert len(handled_df) == len(sample_df_for_outliers) - 2

def test_handle_outliers_fillna_interpolate(sample_df_for_outliers, outlier_mask):
    """Test handling outliers using fillna with interpolate."""
    df_copy = sample_df_for_outliers.copy()
    handled_df = handle_outliers(df_copy, outlier_mask, method='fillna', fill_method='interpolate')

    # Check if the outlier values were replaced by interpolated values
    # Value1 outlier at index 7 ('2023-01-08') between 9 (idx 6) and 11 (idx 8) -> should be 10
    assert handled_df.loc['2023-01-08', 'value1'] == 10.0
    # Value2 outlier at index 5 ('2023-01-06') between 20 (idx 4) and 21 (idx 6) -> should be 20.5
    assert handled_df.loc['2023-01-06', 'value2'] == 20.5
    assert (handled_df['string'] == sample_df_for_outliers['string']).all()

def test_handle_outliers_fillna_ffill(sample_df_for_outliers, outlier_mask):
    """Test handling outliers using fillna with ffill."""
    df_copy = sample_df_for_outliers.copy()
    handled_df = handle_outliers(df_copy, outlier_mask, method='fillna', fill_method='ffill')
    # Value1 outlier at index 7 should take value from index 6 (9)
    assert handled_df.loc['2023-01-08', 'value1'] == 9.0
    # Value2 outlier at index 5 should take value from index 4 (20)
    assert handled_df.loc['2023-01-06', 'value2'] == 20.0

@patch(f'{TRANSFORMATIONS_PATH}.logger')
def test_handle_outliers_unsupported_method(mock_logger, sample_df_for_outliers, outlier_mask):
    """Test handle_outliers with unsupported method."""
    handled_df = handle_outliers(sample_df_for_outliers, outlier_mask, method='unsupported')
    mock_logger.error.assert_called_with("Unsupported outlier handling method: unsupported")
    # Should return original df copy
    pd.testing.assert_frame_equal(handled_df, sample_df_for_outliers)

# --- Tests for preprocess_data ---

def test_preprocess_data_standard_df(sample_daily_df):
    """Test preprocessing a standard DataFrame."""
    df_copy = sample_daily_df.copy()
    # Introduce some potential issues
    df_copy.columns = ['Open', 'High', 'Low', 'Close', 'Volume'] # Uppercase
    df_copy.loc['2023-01-03', 'High'] = np.nan
    df_copy.loc['2023-01-05', 'Volume'] = 'bad_data'

    processed_df = preprocess_data(df_copy, ticker="ANY") # Ticker doesn't matter for single index

    assert list(processed_df.columns) == ['open', 'high', 'low', 'close', 'volume'] # Lowercase
    assert not processed_df.isnull().any().any() # NaNs handled
    # Row with 'bad_data' in Volume should be dropped after coercion and dropna
    assert '2023-01-05' not in processed_df.index
    # Check NaN fill (ffill)
    assert processed_df.loc['2023-01-03', 'high'] == processed_df.loc['2023-01-02', 'high']
    # Check dtypes
    for col in ['open', 'high', 'low', 'close', 'volume']:
        assert pd.api.types.is_numeric_dtype(processed_df[col])

def test_preprocess_data_multiindex_level1(sample_multiindex_df):
    """Test preprocessing MultiIndex DataFrame (ticker level 1)."""
    processed_df = preprocess_data(sample_multiindex_df, ticker='AAPL')

    assert isinstance(processed_df.columns, pd.Index) # No longer MultiIndex
    assert list(processed_df.columns) == ['open', 'high', 'low', 'close', 'volume']
    assert processed_df.loc['2023-01-01', 'close'] == 150 # Check AAPL value

def test_preprocess_data_multiindex_level0(sample_multiindex_df):
    """Test preprocessing MultiIndex DataFrame (ticker level 0)."""
    # Create a new DataFrame with ticker at level 0
    dates = pd.date_range(start="2023-01-01", periods=3, freq='D')
    columns_level0 = pd.MultiIndex.from_product([['AAPL', 'MSFT'], ['Open', 'Close']])
    data_level0 = np.arange(3 * len(columns_level0)).reshape(3, len(columns_level0))
    df_level0 = pd.DataFrame(data_level0, index=dates, columns=columns_level0)
    # Add specific values
    df_level0[('AAPL', 'Close')] = [160, 161, 162]
    df_level0[('MSFT', 'Close')] = [260, 261, 262]

    processed_df = preprocess_data(df_level0, ticker='AAPL')

    assert isinstance(processed_df.columns, pd.Index)
    assert list(processed_df.columns) == ['open', 'close']
    assert processed_df.loc['2023-01-01', 'close'] == 160

@patch(f'{TRANSFORMATIONS_PATH}.logger')
def test_preprocess_data_multiindex_ticker_not_found(mock_logger, sample_multiindex_df):
    """Test preprocessing MultiIndex with ticker not found."""
    processed_df = preprocess_data(sample_multiindex_df, ticker='GOOG')
    assert processed_df is None
    mock_logger.error.assert_called_once()
    assert "GOOG not found" in mock_logger.error.call_args[0][0]

def test_preprocess_data_empty_input():
    """Test preprocessing with empty or None input."""
    assert preprocess_data(None, "ANY") is None
    assert preprocess_data(pd.DataFrame(), "ANY") is None

@patch(f'{TRANSFORMATIONS_PATH}.logger')
def test_preprocess_data_column_warning(mock_logger, sample_daily_df):
    """Test warning when expected columns like 'volume' are missing."""
    df_missing_vol = sample_daily_df.drop(columns=['volume'])
    processed_df = preprocess_data(df_missing_vol, "ANY")
    mock_logger.warning.assert_any_call("Column 'volume' not found for type conversion.")
    assert 'volume' not in processed_df.columns