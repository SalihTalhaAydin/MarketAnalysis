import pytest
from unittest.mock import patch, MagicMock, ANY
import sys

# Import the factory function and specific model classes for type checking/mocking
# Import MODEL_FACTORY directly to access registered functions in tests
from market_ml_model.models.factory.model_factory import create_model, MODEL_FACTORY
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
# Import other necessary classes if needed for spec in mocks, handling potential ImportError
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
# Import TF components conditionally for spec, but patch target will be within factory module
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.optimizers import Adam
    TF_SPEC_AVAILABLE = True
except ImportError:
    Sequential = None
    Dense = None
    Dropout = None
    LSTM = None
    Adam = None
    TF_SPEC_AVAILABLE = False


# Define path for patching availability flags and model classes
FACTORY_PATH = 'market_ml_model.models.factory.model_factory'

# --- Mocks for External Libraries (Scoped to this module) ---

@pytest.fixture(autouse=True)
def mock_imports(mocker):
    """Ensure all library availability flags are True and mock imports."""
    mocker.patch(f'{FACTORY_PATH}.SKLEARN_AVAILABLE', True)
    mocker.patch(f'{FACTORY_PATH}.XGBOOST_AVAILABLE', True)
    mocker.patch(f'{FACTORY_PATH}.LIGHTGBM_AVAILABLE', True)
    mocker.patch(f'{FACTORY_PATH}.CATBOOST_AVAILABLE', True)
    mocker.patch(f'{FACTORY_PATH}.TENSORFLOW_AVAILABLE', True)

    # Mock the actual classes being instantiated within the factory module
    mocker.patch(f'{FACTORY_PATH}.RandomForestClassifier', return_value=MagicMock(spec=RandomForestClassifier))
    mocker.patch(f'{FACTORY_PATH}.GradientBoostingClassifier', return_value=MagicMock(spec=GradientBoostingClassifier))
    mocker.patch(f'{FACTORY_PATH}.LogisticRegression', return_value=MagicMock(spec=LogisticRegression))
    mocker.patch(f'{FACTORY_PATH}.VotingClassifier', return_value=MagicMock(spec=VotingClassifier))

    # Mock libraries imported conditionally using sys.modules and patching the alias
    mock_xgb_module = MagicMock()
    mock_xgb_module.XGBClassifier.return_value = MagicMock() # No spec needed here
    mocker.patch.dict(sys.modules, {'xgboost': mock_xgb_module})
    mocker.patch(f'{FACTORY_PATH}.xgb', mock_xgb_module, create=True)

    mock_lgbm_module = MagicMock()
    mock_lgbm_module.LGBMClassifier.return_value = MagicMock() # No spec needed here
    mocker.patch.dict(sys.modules, {'lightgbm': mock_lgbm_module})
    mocker.patch(f'{FACTORY_PATH}.lgb', mock_lgbm_module, create=True)

    mock_catboost_module = MagicMock()
    mock_catboost_module.CatBoostClassifier.return_value = MagicMock() # No spec needed here
    mocker.patch.dict(sys.modules, {'catboost': mock_catboost_module})
    # Patch the CatBoostClassifier name directly in the factory module's namespace
    mocker.patch(f'{FACTORY_PATH}.CatBoostClassifier', mock_catboost_module.CatBoostClassifier, create=True)

    # Mock TensorFlow/Keras components by patching them *within* the factory module
    mock_sequential_instance = MagicMock(spec=Sequential if TF_SPEC_AVAILABLE else object)
    mocker.patch(f'{FACTORY_PATH}.Sequential', return_value=mock_sequential_instance, create=True)
    mocker.patch(f'{FACTORY_PATH}.Dense', MagicMock(spec=Dense if TF_SPEC_AVAILABLE else object), create=True)
    mocker.patch(f'{FACTORY_PATH}.Dropout', MagicMock(spec=Dropout if TF_SPEC_AVAILABLE else object), create=True)
    mocker.patch(f'{FACTORY_PATH}.LSTM', MagicMock(spec=LSTM if TF_SPEC_AVAILABLE else object), create=True)
    mocker.patch(f'{FACTORY_PATH}.Adam', MagicMock(spec=Adam if TF_SPEC_AVAILABLE else object), create=True)
    # Mock other potentially imported layers if needed by tests later
    mocker.patch(f'{FACTORY_PATH}.Input', MagicMock(), create=True)
    mocker.patch(f'{FACTORY_PATH}.Concatenate', MagicMock(), create=True)
    mocker.patch(f'{FACTORY_PATH}.BatchNormalization', MagicMock(), create=True)
    mocker.patch(f'{FACTORY_PATH}.Bidirectional', MagicMock(), create=True)
    mocker.patch(f'{FACTORY_PATH}.GRU', MagicMock(), create=True)
    mocker.patch(f'{FACTORY_PATH}.Model', MagicMock(), create=True)


# --- Tests ---

@pytest.mark.parametrize("model_type, expected_class_path", [
    ('random_forest', f'{FACTORY_PATH}.RandomForestClassifier'),
    ('gradient_boosting', f'{FACTORY_PATH}.GradientBoostingClassifier'),
    ('logistic_regression', f'{FACTORY_PATH}.LogisticRegression'),
    ('xgboost', f'{FACTORY_PATH}.xgb.XGBClassifier'),
    ('lightgbm', f'{FACTORY_PATH}.lgb.LGBMClassifier'),
    ('catboost', f'{FACTORY_PATH}.CatBoostClassifier'),
    ('ensemble', f'{FACTORY_PATH}.VotingClassifier'),
    ('neural_network', f'{FACTORY_PATH}.Sequential'),
    ('lstm', f'{FACTORY_PATH}.Sequential'),
])
def test_create_model_types(model_type, expected_class_path):
    """Test creating different model types with default parameters."""
    # For ensemble, need to mock the sub-models it tries to create *before* calling create_model
    if model_type == 'ensemble':
        # Mock the global MODEL_FACTORY temporarily for this test
        # Include the actual ensemble factory function but mock sub-factories
        factory_dict_mock = MODEL_FACTORY.copy() # Start with original factory
        factory_dict_mock.update({
            'random_forest': MagicMock(return_value=MagicMock()), # Mock factory function
            'gradient_boosting': MagicMock(return_value=MagicMock()), # Mock factory function
        })
        # Patch the factory dict used internally by create_model/create_ensemble
        with patch.dict(f'{FACTORY_PATH}.MODEL_FACTORY', factory_dict_mock, clear=True):
             # Patch the VotingClassifier class itself to check instantiation
             with patch(expected_class_path) as mock_class:
                model = create_model(model_type) # Call the main factory function
                mock_class.assert_called_once() # Check VotingClassifier called
    else:
         # Patch the specific model class being created
         with patch(expected_class_path) as mock_class:
            model = create_model(model_type)
            # For XGBoost/LightGBM/CatBoost, check the call on the mocked module
            if model_type == 'xgboost':
                mock_class.assert_called_once() # Checks XGBClassifier on the mocked xgb module
            elif model_type == 'lightgbm':
                 mock_class.assert_called_once() # Checks LGBMClassifier on the mocked lgb module
            elif model_type == 'catboost':
                 mock_class.assert_called_once() # Checks CatBoostClassifier in the factory namespace
            elif model_type in ['neural_network', 'lstm']:
                 mock_class.assert_called_once() # Checks Sequential in the factory namespace
            else:
                 mock_class.assert_called_once() # Checks sklearn classes directly


def test_create_model_custom_params_rf():
    """Test passing custom parameters to RandomForest."""
    custom_params = {'n_estimators': 555, 'max_depth': 15}
    with patch(f'{FACTORY_PATH}.RandomForestClassifier') as mock_rf:
        model = create_model('random_forest', params=custom_params)
        # Check if RF was called with custom params overriding defaults
        call_args, call_kwargs = mock_rf.call_args
        assert call_kwargs['n_estimators'] == 555
        assert call_kwargs['max_depth'] == 15
        assert call_kwargs['random_state'] == 42 # Default still applied

@patch(f'{FACTORY_PATH}.VotingClassifier') # Patch the class being tested
def test_create_model_custom_params_ensemble(mock_voting_class): # Renamed mock
    """Test passing custom parameters to Ensemble."""
    custom_params = {
        'models': ['logistic_regression', 'random_forest'],
        'voting': 'hard',
        'weights': [0.3, 0.7],
        'logistic_regression': {'C': 10}, # Params for sub-model
        'random_forest': {'n_estimators': 50}
    }
    # Mock the sub-model factory functions within the global MODEL_FACTORY
    mock_lr = MagicMock()
    mock_rf = MagicMock()
    # We need the actual registered functions in the factory dict for the lookup
    factory_dict_mock = MODEL_FACTORY.copy() # Start with original
    factory_dict_mock.update({
        'logistic_regression': MagicMock(return_value=mock_lr),
        'random_forest': MagicMock(return_value=mock_rf),
    })

    # Patch the global factory dict *before* calling create_model
    with patch.dict(f'{FACTORY_PATH}.MODEL_FACTORY', factory_dict_mock, clear=True):
        # Call the main create_model function for 'ensemble'
        model = create_model('ensemble', params=custom_params)

        # Check sub-model factories were called with correct params
        factory_dict_mock['logistic_regression'].assert_called_once_with({'C': 10})
        factory_dict_mock['random_forest'].assert_called_once_with({'n_estimators': 50})

        # Check VotingClassifier instantiation
        mock_voting_class.assert_called_once()
        call_args, call_kwargs = mock_voting_class.call_args
        assert call_kwargs['voting'] == 'hard'
        assert call_kwargs['weights'] == [0.3, 0.7]
        # Check estimators passed correctly
        passed_estimators = call_kwargs['estimators']
        assert len(passed_estimators) == 2
        assert passed_estimators[0] == ('logistic_regression', mock_lr)
        assert passed_estimators[1] == ('random_forest', mock_rf)


@patch(f'{FACTORY_PATH}.logger')
def test_create_model_unsupported(mock_logger):
    """Test creating an unsupported model type."""
    model = create_model('unsupported_model_type')
    assert model is None
    mock_logger.error.assert_called_with("Unsupported model type: unsupported_model_type")

@patch(f'{FACTORY_PATH}.SKLEARN_AVAILABLE', False)
@patch(f'{FACTORY_PATH}.logger')
def test_create_model_sklearn_unavailable(mock_logger):
    """Test creating sklearn model when library is unavailable."""
    model = create_model('random_forest')
    assert model is None
    mock_logger.error.assert_called_with("scikit-learn not available for Random Forest")

@patch(f'{FACTORY_PATH}.XGBOOST_AVAILABLE', False)
@patch(f'{FACTORY_PATH}.logger')
def test_create_model_xgboost_unavailable(mock_logger):
    """Test creating xgboost model when library is unavailable."""
    model = create_model('xgboost')
    assert model is None
    mock_logger.error.assert_called_with("XGBoost not available")

# Similar tests can be added for LightGBM, CatBoost, TensorFlow unavailability