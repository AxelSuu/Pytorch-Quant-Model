### Current day Stock Price Prediction with PyTorch for any Yahoo Finance ticker for the next trading day and 5 day forecasts.

Easy to use CLI interface for training, evaluating, and predicting real time stock prices using Yahoo finance with 3 simple commands. 

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

## CLI Reference
### 2. Run PyStock

PyStock provides three simple commands:

**Windows (PowerShell or CMD):**
```bash
# Default
pystock.bat train --symbol TICKER
# Given time
pystock.bat train --symbol TICKER [--start yyyy-mm-dd] [--end yyyy-mm-dd]
# Evaluate
pystock.bat evaluate --symbol AAPL
# Predict next 5 days
pystock.bat predict --symbol AAPL
# Powershell
pystock.ps1 train --symbol TICKER
```

### 3. Or activate virtual environment first:
```bash
# PowerShell
.venv\Scripts\Activate.ps1
python pystock.py train --symbol AAPL

# CMD
.venv\Scripts\activate.bat
python pystock.py train --symbol AAPL

# Linux/Mac
source .venv/bin/activate
python pystock.py train --symbol AAPL
```
