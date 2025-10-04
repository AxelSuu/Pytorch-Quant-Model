Current day Stock Price Prediction with PyTorch for any Yahoo Finance ticker for the next trading day and 5 day forecasts.

Fun side project when I was learning to use pytorch and combining stock price prediction. There already is a trained model included and you can train and evaliuate new models with the CLI API.

<table>
  <tr>
    <td><img src="results/AAPL_lstm_128h_2l_60s_predictions.png" /></td>
    <td><img src="results/AAPL_lstm_128h_2l_60s_training_history.png" /></td>
  </tr>
</table>
<img src="results/Prediction.png" />

Quick Start

1. Install Dependencies
```bash
pip install -r requirements.txt
```

2. Get Stock Predictions
```bash
# Real-time prediction for Apple stock
python src/realtime_predict.py --model_path checkpoints/AAPL_lstm_128h_2l_60s_best.pth --symbol AAPL

# Tesla stock with 3-day forecast
python src/realtime_predict.py --model_path checkpoints/AAPL_gru_128h_2l_60s_best.pth --model_type gru --symbol TSLA --days 3

# Any Yahoo Finance ticker (Google, Microsoft, etc.)
python src/realtime_predict.py --model_path checkpoints/AAPL_lstm_128h_2l_60s_best.pth --symbol GOOGL --days 5
```

Train a New Model
```bash
# Train LSTM model on Apple stock
python src/main.py --symbol AAPL --model_type lstm --epochs 50 --hidden_size 128 --num_layers 2

# Train GRU model on Tesla
python src/main.py --symbol TSLA --model_type gru --epochs 30 --hidden_size 64 --sequence_length 60

# Train Transformer model with custom parameters
python src/main.py --symbol GOOGL --model_type transformer --epochs 40 --hidden_size 256 --num_heads 8
```




API Reference

RealTimePredictor Class

```python
from src.evaluate import StockPredictor

# Initialize predictor
predictor = StockPredictor("checkpoints/model.pth", "lstm")

# Next-day prediction
price = predictor.predict_next_price("AAPL", use_realtime=True)

# Multi-day forecast
prices = predictor.predict_sequence("AAPL", steps=5, use_realtime=True)
```

Command Line Interface

Real-Time Prediction
```bash
python src/realtime_predict.py [OPTIONS]

Options:
  --model_path PATH     Path to trained model [REQUIRED]
  --symbol TICKER       Stock ticker symbol [default: AAPL]
  --model_type TYPE     Model type (lstm/gru/transformer) [default: lstm]
  --days INTEGER        Days to predict [default: 5]
  --sequence_length INT Sequence length [default: 60]
```

Model Training
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

Model Evaluation
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

