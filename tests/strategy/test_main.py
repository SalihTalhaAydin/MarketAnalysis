import os
from unittest.mock import MagicMock, mock_open, patch

import numpy as np  # Add numpy import
import pandas as pd  # Add pandas import
import pytest

# Import sklearn components for mocking specs

# Import config classes and detector to test
from market_ml_model.strategy.main import (  # Import the main strategy class; Import the class to test
    AssetConfig,
    EnhancedTradingStrategy,
    FeatureConfig,
    MarketRegimeDetector,
    ModelConfig,
    StrategyConfig,
    WalkForwardConfig,
)

# Define path for patching within the strategy.main module
STRATEGY_PATH = "market_ml_model.strategy.main"


# Mock the dependencies imported by strategy.main at the module level
# This avoids errors if the actual modules are not fully installed or have issues
@pytest.fixture(autouse=True)
def mock_strategy_dependencies(mocker):
    mocker.patch(f"{STRATEGY_PATH}.DataLoader", MagicMock())
    mocker.patch(f"{STRATEGY_PATH}.DataLoaderConfig", MagicMock())
    mocker.patch(f"{STRATEGY_PATH}.load_data", MagicMock())
    mocker.patch(f"{STRATEGY_PATH}.engineer_features", MagicMock())
    mocker.patch(f"{STRATEGY_PATH}.train_classification_model", MagicMock())
    # mocker.patch(f"{STRATEGY_PATH}.create_feature_pipeline", MagicMock()) # Removed patch for removed import
    mocker.patch(f"{STRATEGY_PATH}.ModelPredictorBase", MagicMock())
    mocker.patch(f"{STRATEGY_PATH}.PredictionManager", MagicMock())
    mocker.patch(f"{STRATEGY_PATH}.SignalGenerator", MagicMock())
    mocker.patch(f"{STRATEGY_PATH}.predict_with_model", MagicMock())
    mocker.patch(
        f"{STRATEGY_PATH}.load_model", MagicMock(return_value=(MagicMock(), {}))
    )  # Return dummy model/meta
    mocker.patch(f"{STRATEGY_PATH}.generate_model_report", MagicMock())
    mocker.patch(f"{STRATEGY_PATH}.backtest_strategy", MagicMock())
    mocker.patch(f"{STRATEGY_PATH}.TradeManager", MagicMock())
    mocker.patch(f"{STRATEGY_PATH}.calculate_returns_metrics", MagicMock())
    mocker.patch(f"{STRATEGY_PATH}.plot_equity_curve", MagicMock())
    mocker.patch(f"{STRATEGY_PATH}.plot_monthly_returns", MagicMock())
    mocker.patch(f"{STRATEGY_PATH}.plot_drawdowns", MagicMock())
    mocker.patch(f"{STRATEGY_PATH}.KMeans", MagicMock())
    mocker.patch(f"{STRATEGY_PATH}.StandardScaler", MagicMock())
    mocker.patch(f"{STRATEGY_PATH}.SKLEARN_AVAILABLE_FOR_REGIME", True)
    mocker.patch(
        f"{STRATEGY_PATH}.MODULES_AVAILABLE", True
    )  # Assume modules are available for tests


# --- Tests for AssetConfig ---


def test_asset_config_defaults():
    """Test AssetConfig default values."""
    config = AssetConfig(symbol="TEST")
    assert config.symbol == "TEST"
    assert config.timeframe == "1d"
    assert config.data_source == "yahoo"
    assert config.commission_pct == 0.001
    assert config.slippage_pct == 0.0005
    assert config.min_position_size == 0.01
    assert config.max_position_size == 1.0
    assert config.correlation_group is None


def test_asset_config_custom():
    """Test AssetConfig with custom values."""
    params = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "data_source": "crypto",
        "commission_pct": 0.0008,
        "slippage_pct": 0.0002,
        "min_position_size": 0.001,
        "max_position_size": 0.5,
        "correlation_group": "CRYPTO",
    }
    config = AssetConfig(**params)
    for key, value in params.items():
        assert getattr(config, key) == value


def test_asset_config_dict_conversion():
    """Test AssetConfig to_dict and from_dict."""
    params = {"symbol": "AAPL", "timeframe": "5m", "commission_pct": 0.0}
    config1 = AssetConfig(**params)
    config_dict = config1.to_dict()
    # Check if all original params are in the dict (plus defaults)
    for key, value in params.items():
        assert config_dict[key] == value
    assert config_dict["data_source"] == "yahoo"  # Check a default value

    config2 = AssetConfig.from_dict(config_dict)
    assert config1.symbol == config2.symbol
    assert config1.timeframe == config2.timeframe
    assert config1.commission_pct == config2.commission_pct
    assert config1.data_source == config2.data_source  # Check default restored


# --- Tests for FeatureConfig ---


def test_feature_config_defaults():
    """Test FeatureConfig default values."""
    config = FeatureConfig()
    assert config.technical_indicators is True
    assert config.volatility_features is True
    assert config.trend_features is True
    assert config.pattern_features is False
    assert config.price_action_features is True
    assert config.volume_features is True
    assert config.vwap_features is False
    assert config.support_resistance_features is True
    assert config.time_features is True
    assert config.regime_features is True
    assert config.atr_multiplier_tp == 2.0
    assert config.atr_multiplier_sl == 1.0
    assert config.max_holding_period == 10
    assert config.target_type == "triple_barrier"
    assert config.feature_selection_enabled is True
    assert config.feature_selection_method == "importance"
    assert config.pca_enabled is False
    assert config.max_features == 50


def test_feature_config_dict_conversion():
    """Test FeatureConfig to_dict and from_dict."""
    params = {
        "pattern_features": True,
        "max_features": 30,
        "target_type": "directional",
    }
    config1 = FeatureConfig(**params)
    config_dict = config1.to_dict()
    for key, value in params.items():
        assert config_dict[key] == value
    assert config_dict["technical_indicators"] is True  # Check default

    config2 = FeatureConfig.from_dict(config_dict)
    assert config1.pattern_features == config2.pattern_features
    assert config1.max_features == config2.max_features
    assert config1.target_type == config2.target_type
    assert config1.technical_indicators == config2.technical_indicators


# --- Tests for ModelConfig ---


def test_model_config_defaults():
    """Test ModelConfig default values."""
    config = ModelConfig()
    assert config.model_type == "lightgbm"
    assert config.ensemble_models == []
    assert config.optimize_hyperparams is True
    assert config.optimization_method == "random"
    assert config.optimization_trials == 25
    assert config.cv_folds == 5
    assert config.validation_size == 0.2
    assert config.scoring_metric == "f1_weighted"
    assert config.early_stopping_rounds == 50
    assert config.probability_threshold == 0.60
    assert config.signal_neutral_zone == (0.45, 0.55)
    assert config.signal_trend_filter_ma == 50
    assert config.signal_volatility_filter_atr == 14
    assert config.signal_cooling_period == 3
    assert config.risk_per_trade == 0.02
    assert config.use_kelly_sizing is True
    assert config.max_drawdown_pct == 0.20
    assert config.max_open_trades == 5
    assert config.max_correlation_exposure == 2
    assert config.regime_adaptation_enabled is True
    assert config.regime_models == {}


def test_model_config_dict_conversion():
    """Test ModelConfig to_dict and from_dict."""
    params = {
        "model_type": "xgboost",
        "optimize_hyperparams": False,
        "risk_per_trade": 0.01,
    }
    config1 = ModelConfig(**params)
    config_dict = config1.to_dict()
    for key, value in params.items():
        assert config_dict[key] == value
    assert config_dict["cv_folds"] == 5  # Check default

    config2 = ModelConfig.from_dict(config_dict)
    assert config1.model_type == config2.model_type
    assert config1.optimize_hyperparams == config2.optimize_hyperparams
    assert config1.risk_per_trade == config2.risk_per_trade
    assert config1.cv_folds == config2.cv_folds


# --- Tests for WalkForwardConfig ---


def test_walkforward_config_defaults():
    """Test WalkForwardConfig default values."""
    config = WalkForwardConfig()
    assert config.enabled is True
    assert config.initial_train_periods == 1000
    assert config.test_periods == 200
    assert config.step_periods == 200
    assert config.min_train_periods == 800
    assert config.retrain_frequency == 1
    assert config.rolling_window is False
    assert config.preserve_model_history is True
    assert config.early_stopping_drawdown == 0.25
    assert config.performance_tracking_window == 10


def test_walkforward_config_dict_conversion():
    """Test WalkForwardConfig to_dict and from_dict."""
    params = {"enabled": False, "step_periods": 100, "rolling_window": True}
    config1 = WalkForwardConfig(**params)
    config_dict = config1.to_dict()
    for key, value in params.items():
        assert config_dict[key] == value
    assert config_dict["initial_train_periods"] == 1000  # Check default

    config2 = WalkForwardConfig.from_dict(config_dict)
    assert config1.enabled == config2.enabled
    assert config1.step_periods == config2.step_periods
    assert config1.rolling_window == config2.rolling_window
    assert config1.initial_train_periods == config2.initial_train_periods


# --- Tests for StrategyConfig ---


@patch(f"{STRATEGY_PATH}.os.makedirs")  # Mock makedirs during init
def test_strategy_config_defaults(mock_makedirs):
    """Test StrategyConfig default values."""
    config = StrategyConfig()
    assert config.strategy_name == "Enhanced ML Strategy"
    assert isinstance(config.strategy_run_id, str)
    assert config.data_start_date == "2018-01-01"
    assert len(config.assets) == 1
    assert isinstance(config.assets[0], AssetConfig)
    assert config.assets[0].symbol == "SPY"
    assert isinstance(config.feature_config, FeatureConfig)
    assert isinstance(config.model_config, ModelConfig)
    assert isinstance(config.walkforward_config, WalkForwardConfig)
    assert config.output_dir.startswith(
        "strategy_results" + os.path.sep + "enhanced_ml_strategy_"
    )
    assert config.parallel_processing is True
    assert config.random_state == 42
    assert config.debug_mode is False
    mock_makedirs.assert_called_once()  # Check output dir creation attempt


@patch(f"{STRATEGY_PATH}.os.makedirs")
def test_strategy_config_custom(mock_makedirs):
    """Test StrategyConfig with custom values."""
    asset1 = AssetConfig("MSFT")
    asset2 = AssetConfig("TSLA", timeframe="1h")
    feature_cfg = FeatureConfig(max_features=20)
    model_cfg = ModelConfig(model_type="random_forest")
    wf_cfg = WalkForwardConfig(enabled=False)

    config = StrategyConfig(
        strategy_name="Test Strat",
        data_start_date="2020-01-01",
        assets=[asset1, asset2],
        feature_config=feature_cfg,
        model_config=model_cfg,
        walkforward_config=wf_cfg,
        output_dir="custom_output",
        parallel_processing=False,
        random_state=123,
        debug_mode=True,
    )

    assert config.strategy_name == "Test Strat"
    assert config.data_start_date == "2020-01-01"
    assert len(config.assets) == 2
    assert config.assets[0].symbol == "MSFT"
    assert config.assets[1].symbol == "TSLA"
    assert config.feature_config.max_features == 20
    assert config.model_config.model_type == "random_forest"
    assert config.walkforward_config.enabled is False
    assert config.output_dir.startswith("custom_output" + os.path.sep + "test_strat_")
    assert config.parallel_processing is False
    assert config.random_state == 123
    assert config.debug_mode is True
    mock_makedirs.assert_called_once()


@patch(f"{STRATEGY_PATH}.os.makedirs")
def test_strategy_config_dict_conversion(mock_makedirs, tmp_path):
    """Test StrategyConfig to_dict and from_dict."""
    # Use tmp_path for output_dir to avoid actual directory creation issues
    output_base = str(tmp_path / "strategy_output")
    config1 = StrategyConfig(
        assets=[AssetConfig("GOOG")],
        feature_config=FeatureConfig(max_features=15),
        model_config=ModelConfig(optimize_hyperparams=False),
        walkforward_config=WalkForwardConfig(step_periods=50),
        output_dir=output_base,
    )
    # The output_dir in the config object includes the run_id
    run_output_dir = config1.output_dir
    assert run_output_dir.startswith(output_base)

    config_dict = config1.to_dict()
    assert config_dict["strategy_name"] == config1.strategy_name
    assert config_dict["strategy_run_id"] == config1.strategy_run_id
    assert len(config_dict["assets"]) == 1
    assert config_dict["assets"][0]["symbol"] == "GOOG"
    assert config_dict["feature_config"]["max_features"] == 15
    assert config_dict["model_config"]["optimize_hyperparams"] is False
    assert config_dict["walkforward_config"]["step_periods"] == 50
    assert (
        config_dict["output_dir"] == run_output_dir
    )  # Check specific run dir is saved

    # Test loading from dict
    config2 = StrategyConfig.from_dict(config_dict)
    # Note: run_id and output_dir will be regenerated on load
    assert config2.strategy_name == config1.strategy_name
    assert config2.strategy_run_id != config1.strategy_run_id  # Should be different
    assert len(config2.assets) == 1
    assert config2.assets[0].symbol == "GOOG"
    assert config2.feature_config.max_features == 15
    assert config2.model_config.optimize_hyperparams is False
    assert config2.walkforward_config.step_periods == 50
    # The output_dir on the loaded config will point to a *new* run directory
    assert config2.output_dir != run_output_dir
    assert config2.output_dir.startswith(output_base)  # Should still use the base path
    assert config2.output_dir.endswith(config2.strategy_run_id)  # Ends with new run id


@patch(f"{STRATEGY_PATH}.os.makedirs")
@patch(f"{STRATEGY_PATH}.open", new_callable=mock_open)
@patch(f"{STRATEGY_PATH}.json.dump")
def test_strategy_config_save(mock_json_dump, mock_open_func, mock_makedirs, tmp_path):
    """Test saving StrategyConfig to JSON."""
    output_base = str(tmp_path / "save_test")
    config = StrategyConfig(output_dir=output_base)
    expected_path = os.path.join(config.output_dir, "strategy_config.json")

    saved_path = config.save_config()

    assert saved_path == expected_path
    mock_open_func.assert_called_once_with(expected_path, "w")
    mock_json_dump.assert_called_once()
    # Check that the dict passed to dump matches config.to_dict()
    assert mock_json_dump.call_args[0][0] == config.to_dict()


@patch(f"{STRATEGY_PATH}.os.makedirs")
@patch(f"{STRATEGY_PATH}.open", new_callable=mock_open)
@patch(f"{STRATEGY_PATH}.json.load")
def test_strategy_config_load(mock_json_load, mock_open_func, mock_makedirs, tmp_path):
    """Test loading StrategyConfig from JSON."""
    config_dict_to_load = {
        "strategy_name": "Loaded Strat",
        "strategy_run_id": "old_run_id",  # This should be ignored/regenerated
        "description": "Loaded desc",
        "data_start_date": "2021-01-01",
        "data_end_date": "2023-01-01",
        "assets": [
            {
                "symbol": "XYZ",
                "timeframe": "1d",
                "data_source": "yahoo",
                "commission_pct": 0.001,
                "slippage_pct": 0.0005,
                "min_position_size": 0.01,
                "max_position_size": 1.0,
                "correlation_group": None,
            }
        ],
        "feature_config": {
            "technical_indicators": False,
            "max_features": 10,
        },  # Partial feature config
        "model_config": {"model_type": "random_forest"},  # Partial model config
        "walkforward_config": {"enabled": False},  # Partial WF config
        "output_dir": str(
            tmp_path / "loaded_output" / "loaded_strat_old_run_id"
        ),  # Saved output dir
        "parallel_processing": False,
        "random_state": 99,
        "debug_mode": True,
    }
    mock_json_load.return_value = config_dict_to_load
    config_path = "/fake/config.json"

    config = StrategyConfig.load_config(config_path)

    mock_open_func.assert_called_once_with(config_path, "r")
    mock_json_load.assert_called_once()
    assert isinstance(config, StrategyConfig)
    assert config.strategy_name == "Loaded Strat"
    assert config.data_start_date == "2021-01-01"
    assert len(config.assets) == 1
    assert config.assets[0].symbol == "XYZ"
    assert config.feature_config.technical_indicators is False  # Loaded value
    assert config.feature_config.max_features == 10  # Loaded value
    assert config.feature_config.volatility_features is True  # Default value
    assert config.model_config.model_type == "random_forest"  # Loaded value
    assert config.model_config.optimize_hyperparams is True  # Default value
    assert config.walkforward_config.enabled is False  # Loaded value
    assert config.walkforward_config.initial_train_periods == 1000  # Default value
    assert config.parallel_processing is False
    assert config.random_state == 99
    assert config.debug_mode is True
    # Check that output_dir is regenerated based on loaded name and new run_id
    assert config.output_dir != config_dict_to_load["output_dir"]
    assert config.output_dir.startswith(
        str(tmp_path / "loaded_output")
    )  # Base path from loaded config
    assert config.output_dir.endswith(config.strategy_run_id)  # Ends with new run id


# --- Tests for MarketRegimeDetector ---


@pytest.fixture
def sample_regime_features():
    """DataFrame with features suitable for regime detection."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    data = {
        "volatility_ratio_20_50": np.random.rand(100),
        "efficiency_ratio_20": np.random.rand(100),
        "adf_pvalue": np.random.rand(100) * 0.1,  # Mostly low p-values
    }
    return pd.DataFrame(data, index=dates)


@patch(f"{STRATEGY_PATH}.StandardScaler")
@patch(f"{STRATEGY_PATH}.KMeans")
def test_regime_detector_init(MockKMeans, MockStandardScaler):
    """Test MarketRegimeDetector initialization."""
    # Assume sklearn is available for this test
    with patch(f"{STRATEGY_PATH}.SKLEARN_AVAILABLE_FOR_REGIME", True):
        detector = MarketRegimeDetector(n_regimes=4, lookback_window=30)
        assert detector.n_regimes == 4
        assert detector.lookback_window == 30
        assert detector.use_clustering is True
        MockStandardScaler.assert_called_once()
        MockKMeans.assert_called_once_with(n_clusters=4, random_state=42, n_init=10)
        assert detector.is_fitted is False


@patch(f"{STRATEGY_PATH}.StandardScaler")
@patch(f"{STRATEGY_PATH}.KMeans")
def test_regime_detector_detect_clustering(
    MockKMeans, MockStandardScaler, sample_regime_features
):
    """Test regime detection using clustering."""
    # Assume sklearn is available
    with patch(f"{STRATEGY_PATH}.SKLEARN_AVAILABLE_FOR_REGIME", True):
        # Setup mocks
        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.rand(
            len(sample_regime_features), 3
        )  # Scaled data
        MockStandardScaler.return_value = mock_scaler_instance

        mock_kmeans_instance = MagicMock()
        mock_kmeans_instance.fit.return_value = None
        # Simulate prediction for the last point -> regime 1
        mock_kmeans_instance.predict.return_value = np.array([1])
        # Simulate distances for confidence calculation
        mock_kmeans_instance.transform.return_value = np.array(
            [[0.5, 0.1, 0.8]]
        )  # Closest to cluster 1
        mock_kmeans_instance.cluster_centers_ = np.random.rand(3, 3)  # Dummy centers
        MockKMeans.return_value = mock_kmeans_instance

        detector = MarketRegimeDetector(
            n_regimes=3, lookback_window=50
        )  # Use lookback < data length
        regime_info = detector.detect_regime(sample_regime_features)

        assert detector.is_fitted is True  # Should be fitted now
        mock_scaler_instance.fit_transform.assert_called_once()
        mock_kmeans_instance.fit.assert_called_once()
        mock_kmeans_instance.predict.assert_called_once()
        mock_kmeans_instance.transform.assert_called_once()

        assert regime_info is not None
        assert regime_info["regime"] == 1
        assert regime_info["timestamp"] == sample_regime_features.index[-1]
        assert "confidence" in regime_info
        assert "regime_name" in regime_info  # Check default name exists
        assert len(detector.regime_history) == 1  # Check history stored


@patch(f"{STRATEGY_PATH}.SKLEARN_AVAILABLE_FOR_REGIME", False)
@patch(f"{STRATEGY_PATH}.logger")
def test_regime_detector_clustering_unavailable(mock_logger, sample_regime_features):
    """Test regime detection when clustering is unavailable."""
    detector = MarketRegimeDetector(use_clustering=True)
    regime_info = detector.detect_regime(sample_regime_features)
    assert regime_info is None
    mock_logger.error.assert_called_with("Clustering requires scikit-learn.")


@patch(f"{STRATEGY_PATH}.logger")
def test_regime_detector_insufficient_data(mock_logger, sample_regime_features):
    """Test regime detection with insufficient data for lookback."""
    detector = MarketRegimeDetector(lookback_window=200)  # Window > data length
    regime_info = detector.detect_regime(sample_regime_features)
    assert regime_info is None
    mock_logger.warning.assert_any_call(
        "Not enough data (100) for regime lookback window (200)."
    )


@patch(f"{STRATEGY_PATH}.logger")
def test_regime_detector_missing_features(mock_logger, sample_regime_features):
    """Test regime detection when required features are missing."""
    detector = MarketRegimeDetector(
        regime_features=["volatility_ratio_20_50", "MISSING_FEATURE"]
    )
    regime_info = detector.detect_regime(sample_regime_features)
    assert regime_info is None
    mock_logger.error.assert_called_with(
        "Missing required features for regime detection: ['MISSING_FEATURE']"
    )


# --- Tests for EnhancedTradingStrategy ---


@pytest.fixture
def mock_strategy_config(tmp_path):
    """Provides a StrategyConfig instance with mocks."""
    # Mock os.makedirs within the fixture scope
    with patch(f"{STRATEGY_PATH}.os.makedirs"):
        config = StrategyConfig(output_dir=str(tmp_path / "strategy_run"))
    return config


@pytest.fixture
def mock_data_loader_instance():
    """Provides a mock DataLoader instance."""
    loader = MagicMock()
    loader.load_data.return_value = pd.DataFrame(
        {"close": [100, 101, 102]}
    )  # Dummy data
    loader.config = MagicMock()  # Mock config attribute if accessed
    loader.config.default_start_date = "2020-01-01"
    return loader


@patch(f"{STRATEGY_PATH}.DataLoader")
def test_enhanced_strategy_init(MockDataLoader, mock_strategy_config):
    """Test EnhancedTradingStrategy initialization."""
    mock_loader_instance = MagicMock()
    MockDataLoader.return_value = mock_loader_instance

    strategy = EnhancedTradingStrategy(mock_strategy_config)

    assert strategy.config is mock_strategy_config
    assert strategy.data_loader is mock_loader_instance
    assert isinstance(
        strategy.regime_detector, MarketRegimeDetector
    )  # Check detector created
    assert strategy.models == {}
    assert strategy.predictors == {}
    assert strategy.signal_generators == {}
    assert strategy.results == {}
    MockDataLoader.assert_called_once()  # Check DataLoader was instantiated


@patch(f"{STRATEGY_PATH}.DataLoader")
def test_enhanced_strategy_init_no_regime(MockDataLoader, mock_strategy_config):
    """Test EnhancedTradingStrategy initialization with regime detection disabled."""
    mock_strategy_config.model_config.regime_adaptation_enabled = False
    mock_loader_instance = MagicMock()
    MockDataLoader.return_value = mock_loader_instance

    strategy = EnhancedTradingStrategy(mock_strategy_config)
    assert strategy.regime_detector is None  # Should be None


@patch(f"{STRATEGY_PATH}.DataLoader")
def test_enhanced_strategy_load_data_success(
    MockDataLoader, mock_strategy_config, mock_data_loader_instance
):
    """Test successful data loading via strategy."""
    MockDataLoader.return_value = mock_data_loader_instance
    strategy = EnhancedTradingStrategy(mock_strategy_config)
    asset_cfg = AssetConfig("AAPL", timeframe="1h", data_source="test_source")

    data = strategy.load_data(asset_cfg)

    assert data is not None
    assert not data.empty
    # Check that the mock loader's load_data was called with correct args
    mock_data_loader_instance.load_data.assert_called_once_with(
        ticker="AAPL",
        start_date=mock_strategy_config.data_start_date,
        end_date=mock_strategy_config.data_end_date,
        interval="1h",
        data_source="test_source",
    )
    # Check basic preprocessing (column lowercasing)
    assert "close" in data.columns


@patch(f"{STRATEGY_PATH}.DataLoader")
def test_enhanced_strategy_load_data_fail(
    MockDataLoader, mock_strategy_config, mock_data_loader_instance
):
    """Test data loading failure."""
    mock_data_loader_instance.load_data.return_value = None  # Simulate failure
    MockDataLoader.return_value = mock_data_loader_instance
    strategy = EnhancedTradingStrategy(mock_strategy_config)
    asset_cfg = AssetConfig("FAIL")

    data = strategy.load_data(asset_cfg)
    assert data is None
    mock_data_loader_instance.load_data.assert_called_once()
