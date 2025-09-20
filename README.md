# Real-Time Stock Price Prediction with PyTorch

**Real-time stock forecasting system** powered by PyTorch deep learning models with live market data integration for any Yahoo Finance ticker.

<table>
  <tr>
    <td><img src="results/AAPL_lstm_128h_2l_60s_predictions.png" /></td>
    <td><img src="results/AAPL_lstm_128h_2l_60s_training_history.png" /></td>
  </tr>
</table>
<img src="results/Prediction.png" />

## What This System Does

This project delivers **real-time stock price prediction** for any stock listed on Yahoo Finance. Simply provide a ticker symbol (AAPL, TSLA, GOOGL, etc.) and get instant predictions for the next trading day and multi-day forecasts.

## Key Features

### **Neural Networks**
- **LSTM Models**: Long Short-Term Memory for capturing long-term patterns
- **GRU Networks**: Gated Recurrent Units for efficient sequence learning
- **Transformer Architecture**: Attention-based models for complex relationships
- **Ensemble Predictions**: Multiple model comparison and selection

### **Analysis**
- **Technical Analysis**: 15 financial indicators including RSI, Bollinger Bands, volatility
- **Advanced Backtesting**: Multiple trading strategies (momentum, mean reversion)
- **Risk Metrics**: Sharpe ratio, Maximum Drawdown, VaR, Calmar ratio

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Stock Predictions
```bash
# Real-time prediction for Apple stock
python src/realtime_predict.py --model_path checkpoints/AAPL_lstm_128h_2l_60s_best.pth --symbol AAPL

# Tesla stock with 3-day forecast
python src/realtime_predict.py --model_path checkpoints/AAPL_gru_128h_2l_60s_best.pth --model_type gru --symbol TSLA --days 3

# Any Yahoo Finance ticker (Google, Microsoft, etc.)
python src/realtime_predict.py --model_path checkpoints/AAPL_lstm_128h_2l_60s_best.pth --symbol GOOGL --days 5
```

### **Cross-Stock Compatibility**
Models trained on one stock can predict others:

```bash
# Model trained on AAPL, predicting TSLA
python src/realtime_predict.py --model_path checkpoints/AAPL_lstm_128h_2l_60s_best.pth --symbol TSLA

# Popular tickers that work out-of-the-box
--symbol AAPL    # Apple
--symbol TSLA    # Tesla  
--symbol GOOGL   # Google
--symbol MSFT    # Microsoft
--symbol AMZN    # Amazon
--symbol NVDA    # NVIDIA
--symbol META    # Meta
--symbol NFLX    # Netflix
```
## Training Your Own Models

### Train a New Model
```bash
# Train LSTM model on Apple stock
python src/main.py --symbol AAPL --model_type lstm --epochs 50 --hidden_size 128 --num_layers 2

# Train GRU model on Tesla
python src/main.py --symbol TSLA --model_type gru --epochs 30 --hidden_size 64 --sequence_length 60

# Train Transformer model with custom parameters
python src/main.py --symbol GOOGL --model_type transformer --epochs 40 --hidden_size 256 --num_heads 8
```

## Project Structure

```
pytorch-ml-project/
├── README.md                    # This documentation
├── REALTIME_IMPLEMENTATION.md  # Real-time implementation guide
├── CLAUDE.md                   # Development guidelines
├── requirements.txt            # Python dependencies
├── src/
│   ├── realtime_predict.py     # Main real-time prediction script
│   ├── industrial_evaluation.py # Industrial-grade evaluation framework
│   ├── main.py                 # Model training entry point
│   ├── evaluate.py             # Model evaluation and backtesting
│   ├── model.py                # Neural network models (LSTM, GRU, Transformer)
│   ├── train.py                # Training infrastructure
│   ├── utils.py                # Utility functions and visualization
│   ├── advanced_metrics.py     # Comprehensive metrics and backtesting
│   ├── interpretability.py     # Model interpretability tools
│   └── data/
│       └── stock_data.py       # Real-time data fetching and preprocessing
├── tests/
│   └── test_model.py           # Unit tests
├── checkpoints/                # Trained model files
├── results/                    # Prediction results and plots
└── examples/                   # Demo scripts
```

### Real-Time Data Pipeline

1. **Data Fetching**: Yahoo Finance API integration
2. **Technical Indicators**: 15 indicators calculated in real-time
3. **Normalization**: MinMaxScaler fitted on training data
4. **Sequence Creation**: Rolling 60-day windows for model input
5. **Prediction**: Neural network inference in <5 seconds

### Supported Technical Indicators
- **Price**: Open, High, Low, Close, Volume
- **Moving Averages**: 5-day, 10-day, 20-day MA
- **Momentum**: Relative Strength Index (RSI)
- **Volatility**: Bollinger Bands (upper/lower), rolling volatility
- **Volume**: Volume moving average
- **Change**: Price change percentage, high-low percentage

### Model Architectures

#### LSTM (Long Short-Term Memory)
```python
# Architecture: Input(15) → LSTM(128, 2 layers) → Dense(64) → Output(1)
# Parameters: ~163,000
# Best for: Long-term pattern recognition
```

#### GRU (Gated Recurrent Unit)  
```python
# Architecture: Input(15) → GRU(128, 2 layers) → Dense(64) → Output(1)
# Parameters: ~139,000
# Best for: Efficient training, good performance
```

#### Transformer
```python
# Architecture: Input(15) → Multi-Head Attention → Dense → Output(1)
# Parameters: ~256,000
# Best for: Complex pattern relationships
```

## API Reference

### RealTimePredictor Class

```python
from src.evaluate import StockPredictor

# Initialize predictor
predictor = StockPredictor("checkpoints/model.pth", "lstm")

# Next-day prediction
price = predictor.predict_next_price("AAPL", use_realtime=True)

# Multi-day forecast
prices = predictor.predict_sequence("AAPL", steps=5, use_realtime=True)
```

### Command Line Interface

#### Real-Time Prediction
```bash
python src/realtime_predict.py [OPTIONS]

Options:
  --model_path PATH     Path to trained model [REQUIRED]
  --symbol TICKER       Stock ticker symbol [default: AAPL]
  --model_type TYPE     Model type (lstm/gru/transformer) [default: lstm]
  --days INTEGER        Days to predict [default: 5]
  --sequence_length INT Sequence length [default: 60]
```

#### Model Training
```bash
python src/main.py [OPTIONS]

Options:
  --symbol TICKER       Stock ticker [default: AAPL]
  --model_type TYPE     Model architecture [default: lstm]
  --epochs INT          Training epochs [default: 100]
  --hidden_size INT     Hidden layer size [default: 128]
  --num_layers INT      Number of layers [default: 2]
  --sequence_length INT Input sequence length [default: 60]
  --period PERIOD       Data period (1y/2y/5y) [default: 2y]
  --batch_size INT      Batch size [default: 32]
  --lr FLOAT           Learning rate [default: 0.001]
```

#### Model Evaluation
```bash
python src/evaluate.py [OPTIONS]

Options:
  --model_path PATH     Path to model [REQUIRED]
  --symbol TICKER       Stock to evaluate [default: AAPL]
  --model_type TYPE     Model type [default: lstm]
  --use_realtime        Use real-time data [default: True]
  --historical_only     Use only historical data
  --save_results        Save results to JSON
  --predict_steps INT   Future steps to predict [default: 5]
```

```python
model = create_model(
    "transformer",
    input_size=15,
    d_model=128,
    num_heads=8,
    num_layers=4,
    dropout=0.1
)
```

## Available Stock Symbols

The system supports any stock symbol available on Yahoo Finance. Popular examples:

### US Stocks
- **AAPL** - Apple Inc.
- **GOOGL** - Alphabet Inc.
- **MSFT** - Microsoft Corporation
- **TSLA** - Tesla Inc.
- **AMZN** - Amazon.com Inc.
- **NVDA** - NVIDIA Corporation
- **META** - Meta Platforms Inc.

### Indices
- **^GSPC** - S&P 500
- **^DJI** - Dow Jones Industrial Average
- **^IXIC** - NASDAQ Composite

### Cryptocurrencies
- **BTC-USD** - Bitcoin
- **ETH-USD** - Ethereum
