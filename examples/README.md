# Examples

This directory contains example scripts and legacy code for reference.

## 📁 Files

### `quick_demo.py`
A complete end-to-end example showing how to use PyStock programmatically:
- Data loading
- Model creation and training
- Evaluation and prediction
- Result visualization

**Usage:**
```bash
python examples/quick_demo.py
```

### `legacy_realtime_predict.py`
Legacy standalone script for real-time stock predictions. This was the original CLI interface before `pystock.py` was created.

**Note:** This functionality is now integrated into the main `pystock predict` command. This file is kept for reference and backward compatibility.

**Legacy Usage:**
```bash
python examples/legacy_realtime_predict.py \
    --model_path checkpoints/AAPL_lstm_128h_2l_60s_best.pth \
    --symbol AAPL
```

**Modern Equivalent:**
```bash
pystock predict --symbol AAPL
```

### `industrial_evaluation.py`
Standalone comprehensive evaluation script that uses the advanced metrics from `src/advanced/`.

Provides detailed analysis including:
- Advanced financial metrics
- Backtesting with multiple strategies
- Model interpretability analysis
- Cross-validation results
- Complexity analysis

**Usage:**
```bash
python examples/industrial_evaluation.py \
    --model_path checkpoints/AAPL_lstm_128h_2l_60s_best.pth \
    --symbol AAPL
```

This script is useful for in-depth model analysis beyond what the basic `pystock evaluate` provides.

## 🎯 Recommendation

**For new users:** Start with the main `pystock` CLI commands:
```bash
pystock train --symbol AAPL
pystock evaluate --symbol AAPL
pystock predict --symbol AAPL
```

**For programmatic usage:** Refer to `quick_demo.py` as a template.

**For advanced analysis:** Use `industrial_evaluation.py` for comprehensive metrics.
