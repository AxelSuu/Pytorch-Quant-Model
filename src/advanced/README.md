# Advanced Features

This directory contains optional advanced features for power users who need more sophisticated analysis tools beyond the core `pystock` CLI.

## 📁 Files

### `advanced_metrics.py`
Comprehensive evaluation metrics suite including:
- **AdvancedMetrics**: Financial metrics (Sharpe ratio, Maximum Drawdown, VaR)
- **AdvancedBacktester**: Sophisticated backtesting strategies
- **ModelComplexityAnalyzer**: Model parameter and efficiency analysis
- **TimeSeriesCV**: Cross-validation for time series data

**Usage Example:**
```python
from src.advanced.advanced_metrics import AdvancedMetrics, AdvancedBacktester

# Calculate advanced metrics
metrics = AdvancedMetrics.calculate_all(y_true, y_pred, prices)
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

# Run backtest
backtester = AdvancedBacktester(model, data_loader, device)
results = backtester.backtest_with_strategy(
    strategy='buy_and_hold',
    initial_capital=10000
)
```

### `enhanced_prediction.py`
Enhanced prediction utilities with:
- Proper scaling and inverse scaling
- Technical indicator recalculation for multi-step predictions
- Advanced confidence intervals
- Trend analysis and market regime detection

**Usage Example:**
```python
from src.advanced.enhanced_prediction import EnhancedPredictor

predictor = EnhancedPredictor(model, data_loader, device)
forecast = predictor.predict_multi_step(
    steps=10,
    return_confidence=True,
    recalculate_indicators=True
)
```

### `interpretability.py`
Model interpretability and explainability tools:
- **FeatureImportanceAnalyzer**: Permutation importance, gradient-based importance
- **AttentionVisualizer**: Visualize attention weights (for models with attention)
- **SaliencyAnalyzer**: Gradient-based saliency maps
- **ShapleyValueAnalyzer**: SHAP-like attribution for time series

**Usage Example:**
```python
from src.advanced.interpretability import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer(model, device)
importance = analyzer.permutation_importance(
    test_loader,
    feature_names=['Open', 'High', 'Low', 'Close', 'Volume', ...]
)
analyzer.plot_feature_importance(importance)
```

## 🚀 When to Use These Tools

Use the advanced features when you need:
- Detailed financial performance metrics beyond RMSE/MAE
- Backtesting with different trading strategies
- Model interpretability for research or regulatory requirements
- Advanced multi-step forecasting with indicator recalculation
- Cross-validation with proper time series splitting

## 📝 Note

These tools are **not required** for basic usage of PyStock. The core CLI (`pystock train/evaluate/predict`) works independently without these modules.

For basic usage, stick to the main `pystock` commands in the root directory.
