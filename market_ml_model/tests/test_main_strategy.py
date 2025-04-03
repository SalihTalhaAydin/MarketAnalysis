import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

# Adjust the import path based on your project structure
# If tests/ is at the same level as src/, this might work:
# from ..src.main_strategy import (
# Or if running pytest from the root directory (MarketAnalysis):
from market_ml_model.src.main_strategy import (
    preprocess_and_engineer_features,
    train_classification_model,
    predict_with_model,
    backtest_strategy,
    # run_trading_strategy_analysis # Commented out: unused for now
)
# Assuming data_loader is also needed for context or fixtures
# from market_ml_model.src.data_loader import load_data  # Unused for now

# --- Fixtures for Test Data ---


@pytest.fixture
def sample_raw_ohlcv_data() -> pd.DataFrame:
    """Creates a sample raw OHLCV DataFrame similar to yfinance output."""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03',
                            '2023-01-04', '2023-01-05', '2023-01-06'])
    data = {
        'Open': [100, 101, 102, 103, 104, 105],
        'High': [102, 103, 103, 105, 106, 106],
        'Low': [99, 100, 101, 102, 103, 104],
        'Close': [101, 102, 101.5, 104, 105.5, 104.5],
        'Volume': [1000, 1100, 1050, 1200, 1150, 1250],
        'Adj Close': [101, 102, 101.5, 104, 105.5, 104.5]  # Often included
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df


@pytest.fixture
def sample_multiindex_raw_data() -> pd.DataFrame:
    """Creates a sample raw MultiIndex DataFrame."""
    tickers = ['AAPL', 'MSFT']
    metrics = ['Open', 'High', 'Low', 'Close', 'Volume']
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    index = pd.MultiIndex.from_product([dates, tickers],
                                       names=['Date', 'Ticker'])
    data = np.random.rand(len(index), len(metrics)) * 100
    df = pd.DataFrame(data, index=index, columns=metrics)
    # Unstack to get the yfinance-like format
    df_unstacked = df.unstack(level='Ticker')
    # Rename columns slightly to match yfinance (e.g., ('Close', 'AAPL'))
    df_unstacked.columns = pd.MultiIndex.from_tuples(
        [(metric, ticker) for metric in metrics for ticker in tickers]
    )
    # Reorder columns to group by metric
    df_unstacked = df_unstacked.swaplevel(0, 1, axis=1).sort_index(axis=1)
    return df_unstacked


@pytest.fixture
def sample_featured_data() -> pd.DataFrame:
    """Creates a sample DataFrame with engineered features and target."""
    dates = pd.to_datetime(['2023-01-05', '2023-01-06', '2023-01-07',
                            '2023-01-08', '2023-01-09', '2023-01-10'])
    data = {
        'open': [104, 105, 104.5, 106, 107, 106.5],
        'high': [106, 106, 106.5, 107, 108, 107.5],
        'low': [103, 104, 104, 105, 106, 106],
        'close': [105.5, 104.5, 106, 106.5, 107.5, 107],
        'volume': [1150, 1250, 1100, 1300, 1200, 1350],
        'SMA_20': [103, 103.5, 104, 104.5, 105, 105.5],
        'RSI_14': [60, 55, 65, 70, 75, 72],
        'return': [0.01, -0.01, 0.015, 0.005, 0.01, -0.005],
        'close_lag1': [104, 105.5, 104.5, 106, 106.5, 107.5],
        'target_direction': [0, 1, 1, 1, 0, 0]  # Example target
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df


@pytest.fixture
def sample_trained_model(sample_featured_data) -> RandomForestClassifier:
    """Creates and fits a simple RandomForestClassifier for testing predict."""
    features = sample_featured_data[['SMA_20', 'RSI_14', 'return',
                                     'close_lag1']]
    target = sample_featured_data['target_direction']
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(features, target)
    return model


# --- Test Functions ---

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
    else:
        # If ta not installed, ensure these columns *aren't* present
        assert ('sma_20' not in processed_df.columns
                and 'SMA_20' not in processed_df.columns)
        assert ('rsi_14' not in processed_df.columns
                and 'RSI_14' not in processed_df.columns)

    assert 'return' in processed_df.columns
    assert 'close_lag1' in processed_df.columns
    assert 'target_direction' in processed_df.columns

    # Check target calculation logic (simple case)
    # Example: 2nd last target should be 0 (last close < 2nd last close)
    assert processed_df['target_direction'].iloc[-2] == 0


def test_preprocess_and_engineer_features_multiindex(
    sample_multiindex_raw_data
):
    """Test feature engineering handling MultiIndex input."""
    ticker = "AAPL"
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


def test_preprocess_and_engineer_features_empty_input():
    """Test handling of empty DataFrame input."""
    empty_df = pd.DataFrame()
    processed_df = preprocess_and_engineer_features(empty_df, "TEST")
    assert processed_df is None


def test_preprocess_and_engineer_features_missing_ohlc(sample_raw_ohlcv_data):
    """Test handling if essential OHLC columns are missing."""
    df_missing = sample_raw_ohlcv_data.copy().drop(columns=['Close'])
    # This might raise an error or return None depending on pandas_ta/logic
    # For now, let's assume it might proceed but lack features
    processed_df = preprocess_and_engineer_features(df_missing, "TEST")
    # Assert based on expected behavior (e.g., None, or df missing features)
    # If pandas_ta fails gracefully:
    if processed_df is not None:
        assert 'sma_20' not in processed_df.columns  # Example check
    # Or if it should fail completely:
    # assert processed_df is None # Uncomment if this is expected


def test_train_classification_model(sample_featured_data):
    """Test model training returns a fitted model."""
    features = sample_featured_data[['SMA_20', 'RSI_14', 'return',
                                     'close_lag1']]
    target = sample_featured_data['target_direction']

    model = train_classification_model(features, target)

    assert model is not None
    assert isinstance(model, RandomForestClassifier)
    # Check if model is fitted by trying to predict
    try:
        model.predict(features.iloc[[0]])
    except NotFittedError:
        pytest.fail("Model should be fitted after train_classification_model")
    except Exception as e:
        pytest.fail(f"Model prediction failed unexpectedly: {e}")


def test_train_classification_model_empty_input():
    """Test training with empty features or target."""
    empty_features = pd.DataFrame()
    empty_target = pd.Series(dtype=int)
    some_features = pd.DataFrame({'a': [1, 2]})
    some_target = pd.Series([0, 1])

    assert train_classification_model(empty_features, some_target) is None
    assert train_classification_model(some_features, empty_target) is None


def test_train_classification_model_mismatch_length():
    """Test training with mismatched feature and target lengths."""
    features = pd.DataFrame({'a': [1, 2, 3]})
    target = pd.Series([0, 1])
    assert train_classification_model(features, target) is None


def test_predict_with_model(sample_trained_model, sample_featured_data):
    """Test prediction generation."""
    features_live = sample_featured_data[['SMA_20', 'RSI_14', 'return',
                                          'close_lag1']]
    predictions = predict_with_model(sample_trained_model, features_live)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(features_live)
    # Check if all predictions are either 0 or 1
    assert all(p in [0, 1] for p in predictions)


def test_predict_with_model_not_fitted():
    """Test prediction with a model that hasn't been fitted."""
    model = RandomForestClassifier()  # Not fitted
    features_live = pd.DataFrame({'a': [1, 2]})
    predictions = predict_with_model(model, features_live)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 0  # Should return empty array on failure


def test_predict_with_model_invalid_model():
    """Test prediction with an invalid model object."""
    model = "not a model"
    features_live = pd.DataFrame({'a': [1, 2]})
    predictions = predict_with_model(model, features_live)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 0


def test_predict_with_model_empty_features():
    """Test prediction with empty features."""
    model = RandomForestClassifier(n_estimators=1).fit(
        pd.DataFrame({'a': [1, 2]}), pd.Series([0, 1])
    )  # Fitted
    features_live = pd.DataFrame()
    predictions = predict_with_model(model, features_live)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 0


def test_backtest_strategy_basic(sample_featured_data):
    """Test basic backtesting logic."""
    # Add dummy predictions
    data_with_predictions = sample_featured_data.copy()
    # Simple alternating prediction for testing
    data_with_predictions['prediction'] = [1, 0, 1, 0, 1, 0]

    performance = backtest_strategy(data_with_predictions)

    assert isinstance(performance, dict)
    assert "total_strategy_return_pct" in performance
    assert "total_market_return_pct" in performance
    assert "num_trades" in performance

    # Basic sanity checks (values depend heavily on sample data and signals)
    assert performance["num_trades"] > 0  # Trades based on alternating signal
    # Check if strategy return is calculated (might be positive or negative)
    assert isinstance(performance["total_strategy_return_pct"], float)


def test_backtest_strategy_missing_columns(sample_featured_data):
    """Test backtest with missing required columns."""
    data_missing_close = sample_featured_data.copy()
    data_missing_close['prediction'] = 1
    data_missing_close = data_missing_close.drop(columns=['close'])

    data_missing_pred = sample_featured_data.copy()
    # Missing 'prediction' column

    assert backtest_strategy(data_missing_close) == {}
    assert backtest_strategy(data_missing_pred) == {}


def test_backtest_strategy_empty_input():
    """Test backtest with empty DataFrame."""
    empty_df = pd.DataFrame(columns=['close', 'prediction'])
    assert backtest_strategy(empty_df) == {}

# --- Integration Test (Optional, can be slow) ---

# @pytest.mark.slow # Mark as slow if it takes time
# def test_run_trading_strategy_analysis_integration(mocker):
#     """
#     Test the full workflow integration.
#     Requires mocking data loading and potentially pandas_ta if not installed.
#     """
#     # Mock load_data to return our sample data
#     mocker.patch('market_ml_model.src.main_strategy.load_data',
#                  return_value=sample_raw_ohlcv_data())

#     # Mock pandas_ta if necessary (e.g., in CI environment)
#     # try:
#     #     import pandas_ta
#     # except ImportError:
#     #      mocker.patch('market_ml_model.src.main_strategy.ta', None)

#     # Capture print output to check logs (optional)
#     # import io
#     # from contextlib import redirect_stdout
#     # f = io.StringIO()
#     # with redirect_stdout(f):
#     try:
#         # Run the main analysis function
#         run_trading_strategy_analysis(
#             symbol="TEST",
#             timeframe="1d",
#             start_date="2023-01-01",
#             end_date="2023-01-06"
#         )
#         # If it runs without errors, it's a basic pass
#         assert True
#     except Exception as e:
#         pytest.fail(f"run_trading_strategy_analysis failed: {e}")

    # Add more specific assertions if needed, e.g., check print output
    # output = f.getvalue()
    # assert "Analysis Complete" in output
    # assert "Performance Summary" in output