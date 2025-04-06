# Refactoring Plan for market_ml_model

This document outlines the proposed refactoring plan to improve the modularity and maintainability of the `market_ml_model` Python package by breaking down large files into smaller, more focused modules.

## 1. Analysis Summary

The following files were identified as overly large and containing multiple distinct responsibilities:

*   `market_ml_model/strategy/main.py` (Config, Regime Detection, Strategy Core, Runner)
*   `market_ml_model/models/prediction.py` (Prediction, Management, Signals, Scheduling)
*   `market_ml_model/trading/simulation.py` (Trade Representation, Trade Management)

## 2. Proposed New Structure and Content Migration

### Original File: `market_ml_model/strategy/main.py`

*   **New File:** `market_ml_model/strategy/config.py`
    *   **Content:** All configuration classes (`AssetConfig`, `DifferencingConfig`, `ScalingConfig`, `FeatureConfig`, `ModelConfig`, `WalkForwardConfig`, `TradingConfig`, `MarketRegimeConfig`, `StrategyConfig`)
*   **New File:** `market_ml_model/strategy/regime.py`
    *   **Content:** `MarketRegimeDetector` class
*   **New File:** `market_ml_model/strategy/core.py`
    *   **Content:** `EnhancedTradingStrategy` class
*   **New File:** `market_ml_model/strategy/runner.py`
    *   **Content:** `run_trading_strategy` function

### Original File: `market_ml_model/models/prediction.py`

*   **New File:** `market_ml_model/models/predictor.py`
    *   **Content:** `ModelPredictorBase` class, `load_model`, `preprocess_features`, `predict_proba` (standalone function), `predict_with_threshold`, `get_confidence_levels` functions.
*   **New File:** `market_ml_model/models/manager.py`
    *   **Content:** `PredictionManager` class.
*   **New File:** `market_ml_model/models/signals.py`
    *   **Content:** `SignalGenerator` class, `calibrate_probabilities`, `validate_model_predictions` functions.
*   **New File:** `market_ml_model/models/scheduler.py`
    *   **Content:** `PredictionScheduler` class.
*   **New File:** `market_ml_model/models/utils.py`
    *   **Content:** `predict_with_model` function.

### Original File: `market_ml_model/trading/simulation.py`

*   **New File:** `market_ml_model/trading/trade.py`
    *   **Content:** `Trade` class.
*   **New File:** `market_ml_model/trading/manager.py`
    *   **Content:** `TradeManager` class.

## 3. Proposed Directory Structure Visualization

```mermaid
graph TD
    A[market_ml_model] --> B(strategy)
    A --> C(models)
    A --> D(trading)
    A --> E(data)
    A --> F(features)
    A --> G(utils)
    A --> H[__init__.py]

    B --> B1[__init__.py]
    B --> B2[config.py]
    B --> B3[core.py]
    B --> B4[regime.py]
    B --> B5[runner.py]

    C --> C1[__init__.py]
    C --> C2[manager.py]
    C --> C3[predictor.py]
    C --> C4[scheduler.py]
    C --> C5[signals.py]
    C --> C6[utils.py]

    D --> D1[__init__.py]
    D --> D2[manager.py]
    D --> D3[trade.py]

    subgraph Strategy Components
        B2(config.py - All *Config classes)
        B3(core.py - EnhancedTradingStrategy)
        B4(regime.py - MarketRegimeDetector)
        B5(runner.py - run_trading_strategy)
    end

    subgraph Model Components
        C2(manager.py - PredictionManager)
        C3(predictor.py - ModelPredictorBase, helpers)
        C4(scheduler.py - PredictionScheduler)
        C5(signals.py - SignalGenerator, helpers)
        C6(utils.py - predict_with_model)
    end

    subgraph Trading Components
        D2(manager.py - TradeManager)
        D3(trade.py - Trade)
    end
