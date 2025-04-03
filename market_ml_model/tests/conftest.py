import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- Fixtures for Test Data ---


@pytest.fixture
def sample_raw_ohlcv_data() -> pd.DataFrame:
    """Creates a larger sample raw OHLCV DataFrame (30 days)."""
    num_days = 30
    start_date = '2023-01-01'
    dates = pd.date_range(start=start_date, periods=num_days, freq='D')
    np.random.seed(42)  # for reproducibility

    # Create Open prices
    open_prices = 100 + np.cumsum(np.random.randn(num_days) * 0.5)

    # Create High, Low, Close based on Open using vectorized operations
    high_add = np.random.rand(num_days) * 2
    low_sub = np.random.rand(num_days) * 2
    close_noise = np.random.randn(num_days) * 0.75

    high_prices = open_prices + high_add
    low_prices = open_prices - low_sub
    close_prices = open_prices + close_noise

    # Ensure High >= Low, High >= Open/Close, Low <= Open/Close
    high_prices = np.maximum.reduce([high_prices, open_prices, close_prices])
    low_prices = np.minimum.reduce([low_prices, open_prices, close_prices])

    # Create Volume
    volume = 1000 + np.random.randint(0, 500, size=num_days)

    data = {
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume,
        'Adj Close': close_prices  # Simple assignment for testing
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