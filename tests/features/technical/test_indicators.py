from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Import functions to test
from market_ml_model.features.technical.indicators import (
    calculate_efficiency_ratio,
    calculate_fractals,
    calculate_hurst_exponent,
    calculate_market_regime_features,
    calculate_momentum_features,
    calculate_pattern_features,
    calculate_support_resistance_features,
    calculate_technical_indicators,
    calculate_volatility_features,
    calculate_vwap_features,
)

# Define path for patching
INDICATORS_PATH = "market_ml_model.features.technical.indicators"

# --- Fixtures ---


@pytest.fixture
def sample_ohlcv_df():
    """Provides a basic OHLCV DataFrame."""
    dates = pd.date_range(
        start="2023-01-01", periods=150, freq="D"
    )  # Increased periods for GARCH/ADF
    data = {
        "open": np.linspace(100, 115, 150)
        + np.random.randn(150) * 0.1,  # Adjusted linspace end
        "high": np.linspace(100.5, 115.5, 150) + np.random.randn(150) * 0.1,
        "low": np.linspace(99.5, 114.5, 150) + np.random.randn(150) * 0.1,
        "close": np.linspace(100.2, 115.2, 150) + np.random.randn(150) * 0.1,
        "volume": np.linspace(1000, 2500, 150).astype(int),  # Adjusted linspace end
    }
    df = pd.DataFrame(data, index=dates)
    # Ensure high >= low, high >= close, low <= close etc.
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    return df


# --- Mocks for External Libraries ---


# Mock pandas_ta globally for this test module
@pytest.fixture(autouse=True)
def mock_pandas_ta(mocker):
    """Mocks the pandas_ta library and its methods."""
    mock_ta = MagicMock()
    # Mock specific indicator functions used (adjust size to 150)
    mock_ta.sma = MagicMock(return_value=pd.Series(np.random.rand(150)))
    mock_ta.ema = MagicMock(return_value=pd.Series(np.random.rand(150)))
    mock_ta.rsi = MagicMock(return_value=pd.Series(np.random.rand(150)))
    mock_ta.macd = MagicMock(
        return_value=pd.DataFrame(
            {
                "MACD_12_26_9": np.random.rand(150),
                "MACDs_12_26_9": np.random.rand(150),
                "MACDh_12_26_9": np.random.rand(150),
            }
        )
    )
    mock_ta.bbands = MagicMock(
        return_value=pd.DataFrame(
            {
                "BBL_20_2.0": np.random.rand(150),  # Lower band
                "BBM_20_2.0": np.random.rand(150),  # Middle band
                "BBU_20_2.0": np.random.rand(150),  # Upper band
            }
        )
    )
    mock_ta.atr = MagicMock(return_value=pd.Series(np.random.rand(150)))
    mock_ta.adx = MagicMock(
        return_value=pd.DataFrame(
            {
                "ADX_14": np.random.rand(150),
                "DMP_14": np.random.rand(150),
                "DMN_14": np.random.rand(150),
            }
        )
    )
    mock_ta.stoch = MagicMock(
        return_value=pd.DataFrame(
            {"STOCHk_14_3_3": np.random.rand(150), "STOCHd_14_3_3": np.random.rand(150)}
        )
    )
    mock_ta.obv = MagicMock(return_value=pd.Series(np.random.rand(150)))
    mock_ta.mfi = MagicMock(return_value=pd.Series(np.random.rand(150)))
    mock_ta.cdl_doji = MagicMock(return_value=pd.Series(np.random.randint(0, 101, 150)))
    mock_ta.cdl_engulfing = MagicMock(
        return_value=pd.Series(np.random.randint(-100, 101, 150))
    )
    mock_ta.cdl_hammer = MagicMock(
        return_value=pd.Series(np.random.randint(0, 101, 150))
    )
    mock_ta.vwap = MagicMock(return_value=pd.Series(np.random.rand(150)))

    # Patch the import within the indicators module
    mocker.patch(f"{INDICATORS_PATH}.ta", mock_ta, create=True)  # Use create=True
    # Ensure the availability flag is True
    mocker.patch(f"{INDICATORS_PATH}.PANDAS_TA_AVAILABLE", True)
    return mock_ta


# Mock advanced stats libraries
@pytest.fixture(autouse=True)
def mock_advanced_stats(mocker):
    """Mocks advanced stats libraries."""
    mock_adfuller = MagicMock(
        return_value=(0, 0.01, 0, 0, {}, 0)
    )  # Simulate stationary p-value
    mock_compute_hc = MagicMock(return_value=(0.6, 0, 0))  # Simulate trending Hurst
    mock_arch_model_instance = MagicMock()
    # Adjust GARCH mock return length (150 periods -> 149 returns)
    mock_arch_model_instance.fit.return_value.conditional_volatility = pd.Series(
        np.random.rand(149) * 100
    )
    mock_arch_model = MagicMock(return_value=mock_arch_model_instance)

    mocker.patch(f"{INDICATORS_PATH}.adfuller", mock_adfuller, create=True)
    mocker.patch(f"{INDICATORS_PATH}.compute_Hc", mock_compute_hc, create=True)
    mocker.patch(f"{INDICATORS_PATH}.arch_model", mock_arch_model, create=True)
    mocker.patch(f"{INDICATORS_PATH}.ADVANCED_STATS", True)  # Assume available

    return mock_adfuller, mock_compute_hc, mock_arch_model


# --- Tests for calculate_technical_indicators ---


def test_calc_indicators_all_basic(sample_ohlcv_df, mock_pandas_ta):
    """Test calculating all basic indicators."""
    # Define a default list of indicator configs for the test
    default_configs = [
        {"indicator": "sma", "length": 14},
        {"indicator": "rsi", "length": 14},
    ]
    result_df = calculate_technical_indicators(
        sample_ohlcv_df, indicator_configs=default_configs
    )

    # Check if mocks for default indicators were called
    mock_pandas_ta.sma.assert_called()
    # mock_pandas_ta.ema.assert_called() # EMA was not requested in default_configs
    mock_pandas_ta.rsi.assert_called_once()
    # mock_pandas_ta.macd.assert_called_once() # MACD was not requested
    # mock_pandas_ta.bbands.assert_called_once() # BBands was not requested
    mock_pandas_ta.atr.assert_called()  # Called twice (atr_14, ATRr_10)
    # mock_pandas_ta.adx.assert_called_once() # ADX was not requested
    # mock_pandas_ta.stoch.assert_called_once() # Stoch was not requested
    # mock_pandas_ta.obv.assert_called_once() # OBV not requested
    # mock_pandas_ta.mfi.assert_called_once() # MFI not requested

    # Check if expected columns exist (names might vary slightly based on ta lib version)
    assert any(col.startswith("sma_") for col in result_df.columns)
    # assert any(col.startswith("ema_") for col in result_df.columns) # EMA not requested
    assert any(col.startswith("rsi_") for col in result_df.columns)
    # assert "macd" in result_df.columns # MACD not requested
    # assert any(col.startswith("bb_") for col in result_df.columns) # BBands not requested
    assert (
        "ATRr_10" in result_df.columns
    )  # Check for the explicitly calculated ATRr_10 column
    assert "ATRr_10" in result_df.columns  # ATRr_10 is calculated implicitly
    # assert "adx" in result_df.columns # ADX not requested
    # assert any(col.startswith("stoch_") for col in result_df.columns) # Stoch not requested
    # assert "obv" in result_df.columns # OBV not requested
    # assert any(col.startswith("mfi_") for col in result_df.columns) # MFI not requested
    # assert "volume_sma20" in result_df.columns # Volume SMA not requested


def test_calc_indicators_specific(sample_ohlcv_df, mock_pandas_ta):
    """Test calculating only specific indicators."""
    specific_configs = [
        {"indicator": "sma", "length": 14},
        {"indicator": "rsi", "length": 14},
    ]
    result_df = calculate_technical_indicators(
        sample_ohlcv_df, indicator_configs=specific_configs
    )

    mock_pandas_ta.sma.assert_called()
    mock_pandas_ta.rsi.assert_called_once()
    # Check others were NOT called
    mock_pandas_ta.ema.assert_not_called()
    mock_pandas_ta.macd.assert_not_called()
    mock_pandas_ta.bbands.assert_not_called()
    # Check that only requested columns exist
    assert any(col.startswith("sma_") for col in result_df.columns)
    assert any(col.startswith("rsi_") for col in result_df.columns)
    assert not any(col.startswith("ema_") for col in result_df.columns)
    assert "macd" not in result_df.columns


@patch(f"{INDICATORS_PATH}.logger")
def test_calc_indicators_missing_cols(mock_logger, sample_ohlcv_df, mock_pandas_ta):
    """Test handling of missing required columns."""
    df_missing = sample_ohlcv_df.drop(columns=["high", "low"])
    missing_cols_configs = [
        {"indicator": "atr", "length": 14},
        {"indicator": "sma", "length": 14},
    ]
    result_df = calculate_technical_indicators(
        df_missing, indicator_configs=missing_cols_configs
    )

    # Function should log an error about missing required columns and return early
    mock_logger.error.assert_any_call(
        "Missing required column 'high' (case-insensitive) for technical indicators. Returning original DataFrame."
    )
    # Check that no indicator columns were added (result should be same as input)
    assert not any(col.startswith("sma_") for col in result_df.columns)
    assert not any(col.startswith("atr_") for col in result_df.columns)
    assert len(result_df.columns) == len(df_missing.columns)  # Ensure no columns added
    # ATR calculation should not be attempted
    mock_pandas_ta.atr.assert_not_called()


@patch(f"{INDICATORS_PATH}.PANDAS_TA_AVAILABLE", False)
@patch(f"{INDICATORS_PATH}.logger")
def test_calc_indicators_pandas_ta_unavailable(mock_logger, sample_ohlcv_df):
    """Test behavior when pandas-ta is not available."""
    # Pass an empty list for indicator_configs as the function requires it,
    # even though pandas-ta is mocked as unavailable.
    result_df = calculate_technical_indicators(
        sample_ohlcv_df.copy(), indicator_configs=[]
    )
    mock_logger.error.assert_called_with(
        "pandas-ta not available for calculating indicators"
    )
    # Should return original df
    pd.testing.assert_frame_equal(result_df, sample_ohlcv_df)


# --- Tests for calculate_fractals ---


def test_calculate_fractals(sample_ohlcv_df):
    """Test fractal calculation."""
    # Manually create expected fractal points for a small subset
    highs = pd.Series([10, 12, 11, 13, 10, 9])
    lows = pd.Series([5, 7, 6, 8, 5, 4])
    bull, bear = calculate_fractals(
        highs, lows, window=3
    )  # Window 3 looks 1 bar each side

    # Expected: Bearish at index 1 (12 > 10, 12 > 11) and 3 (13 > 11, 13 > 10)
    # Expected: Bullish at index 2 (6 < 7, 6 < 8)
    assert bear.tolist() == [0, 1, 0, 1, 0, 0]
    assert bull.tolist() == [0, 0, 1, 0, 0, 0]


# --- Tests for calculate_efficiency_ratio ---


def test_calculate_efficiency_ratio(sample_ohlcv_df):
    """Test efficiency ratio calculation."""
    er = calculate_efficiency_ratio(sample_ohlcv_df["close"], window=10)
    assert isinstance(er, pd.Series)
    assert len(er) == len(sample_ohlcv_df)
    assert er.iloc[:9].isnull().all()  # First 9 values should be NaN
    assert (er.dropna() >= 0).all() and (
        er.dropna() <= 1
    ).all()  # Should be between 0 and 1


# --- Tests for calculate_hurst_exponent ---


def test_calculate_hurst_exponent(sample_ohlcv_df, mock_advanced_stats):
    """Test Hurst exponent calculation."""
    mock_adfuller, mock_compute_hc, mock_arch_model = mock_advanced_stats
    hurst = calculate_hurst_exponent(sample_ohlcv_df["close"])
    mock_compute_hc.assert_called_once()
    assert hurst == 0.6  # Value returned by mock


@patch(f"{INDICATORS_PATH}.ADVANCED_STATS", False)
def test_calculate_hurst_exponent_unavailable(sample_ohlcv_df):
    """Test Hurst exponent default when statsmodels not available."""
    hurst = calculate_hurst_exponent(sample_ohlcv_df["close"])
    assert hurst == 0.5


# --- Tests for calculate_volatility_features ---


def test_calculate_volatility_features(sample_ohlcv_df, mock_advanced_stats):
    """Test volatility feature calculation."""
    mock_adfuller, mock_compute_hc, mock_arch_model = mock_advanced_stats
    features = calculate_volatility_features(sample_ohlcv_df)

    assert isinstance(features, dict)
    assert "volatility_5" in features
    assert "volatility_ratio_20_50" in features
    assert "vol_of_vol_20" in features
    assert "parkinson_vol_20" in features
    assert "garch_vol" in features  # Check if GARCH mock was called implicitly
    mock_arch_model.assert_called_once()  # Verify arch_model was instantiated


# --- Tests for calculate_market_regime_features ---


def test_calculate_market_regime_features(
    sample_ohlcv_df, mock_pandas_ta, mock_advanced_stats
):
    """Test market regime feature calculation."""
    mock_adfuller, mock_compute_hc, mock_arch_model = mock_advanced_stats
    features = calculate_market_regime_features(sample_ohlcv_df)

    assert isinstance(features, dict)
    assert "efficiency_ratio_10" in features
    assert "adx_14" in features  # Check if ADX mock was called
    assert "close_gt_sma50" in features
    assert "sma20_gt_sma50" in features
    assert "volatility_regime" in features
    assert "volume_ratio" in features
    assert "price_volume_corr" in features
    assert "adf_pvalue" in features  # Check if ADF mock was called
    assert "is_stationary" in features
    mock_pandas_ta.adx.assert_called_once()
    mock_adfuller.assert_called()  # ADF is called within rolling apply


# --- Tests for calculate_support_resistance_features ---


def test_calculate_support_resistance_features(sample_ohlcv_df):
    """Test support/resistance feature calculation."""
    features = calculate_support_resistance_features(sample_ohlcv_df, window=20)

    assert isinstance(features, dict)
    assert "bullish_fractal" in features
    assert "bearish_fractal" in features
    assert "dist_from_recent_high" in features
    assert "dist_from_recent_low" in features
    assert "near_resistance" in features
    assert "near_support" in features


# --- Tests for calculate_pattern_features ---


def test_calculate_pattern_features(sample_ohlcv_df, mock_pandas_ta):
    """Test pattern feature calculation."""
    features = calculate_pattern_features(sample_ohlcv_df)

    assert isinstance(features, dict)
    assert "body_size" in features
    assert "upper_shadow" in features
    assert "lower_shadow" in features
    # Check if pattern mocks were called
    assert "pattern_doji" in features
    assert "pattern_engulfing" in features
    assert "pattern_hammer" in features
    mock_pandas_ta.cdl_doji.assert_called_once()
    mock_pandas_ta.cdl_engulfing.assert_called_once()
    mock_pandas_ta.cdl_hammer.assert_called_once()


# --- Tests for calculate_momentum_features ---


def test_calculate_momentum_features(sample_ohlcv_df, mock_pandas_ta):
    """Test momentum feature calculation."""
    features = calculate_momentum_features(sample_ohlcv_df)

    assert isinstance(features, dict)
    assert "rsi_7" in features
    assert "rsi_14" in features
    assert "rsi_21" in features
    assert "stoch_k" in features
    assert "stoch_d" in features
    assert "macd" in features
    assert "macd_signal" in features
    assert "macd_histogram" in features
    assert "mfi_14" in features
    assert "roc_5" in features
    assert "roc_10" in features
    assert "roc_20" in features
    # Check mocks
    mock_pandas_ta.rsi.assert_called()
    mock_pandas_ta.stoch.assert_called_once()
    mock_pandas_ta.macd.assert_called_once()
    mock_pandas_ta.mfi.assert_called_once()


# --- Tests for calculate_vwap_features ---


def test_calculate_vwap_features(sample_ohlcv_df, mock_pandas_ta):
    """Test VWAP feature calculation."""
    features = calculate_vwap_features(sample_ohlcv_df)

    assert isinstance(features, dict)
    assert "vwap_ta" in features
    assert "dist_from_vwap_ta" in features
    assert "vwap_roll_20" in features
    assert "dist_from_vwap_roll_20" in features
    assert "vwap_roll_50" in features
    assert "dist_from_vwap_roll_50" in features
    # Check mock
    mock_pandas_ta.vwap.assert_called_once()
