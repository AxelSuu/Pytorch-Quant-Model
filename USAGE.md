# Stock Price Prediction - Installation & Usage Guide

This guide will help you set up and use the PyTorch Stock Price Prediction project.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run a Quick Demo

```bash
python examples/quick_demo.py
```

### 3. Train Your First Model

```bash
python src/main.py --symbol AAPL --epochs 50 --model_type lstm
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for downloading stock data)

### Step-by-Step Installation

1. **Clone or download the project:**
   ```bash
   git clone <repository-url>
   cd pytorch-ml-project
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import yfinance; print('yfinance installed successfully')"
   ```

## 🎯 Usage Examples

### Basic Usage

#### Training a Model

```bash
# Train LSTM model on Apple stock
python src/main.py --symbol AAPL --period 2y --epochs 100

# Train GRU model on Tesla stock
python src/main.py --symbol TSLA --model_type gru --epochs 50 --lr 0.001

# Train Transformer model
python src/main.py --symbol GOOGL --model_type transformer --hidden_size 256
```

#### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--symbol` | Stock symbol (e.g., AAPL, GOOGL, TSLA) | AAPL |
| `--period` | Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y) | 2y |
| `--model_type` | Model type (lstm, gru, transformer) | lstm |
| `--epochs` | Number of training epochs | 100 |
| `--lr` | Learning rate | 0.001 |
| `--batch_size` | Batch size | 32 |
| `--sequence_length` | Input sequence length | 60 |
| `--hidden_size` | Hidden layer size | 128 |
| `--num_layers` | Number of layers | 2 |
| `--dropout` | Dropout rate | 0.2 |
| `--save_dir` | Directory to save models | checkpoints |

### Evaluation

#### Evaluate a Trained Model

```bash
python src/evaluate.py --model_path checkpoints/AAPL_lstm_best.pth --symbol AAPL
```

#### Make Predictions

```bash
# Predict next day price
python src/evaluate.py --model_path checkpoints/AAPL_lstm_best.pth --symbol AAPL --predict_steps 1

# Predict next 5 days
python src/evaluate.py --model_path checkpoints/AAPL_lstm_best.pth --symbol AAPL --predict_steps 5
```

### Python API Usage

#### Training in Python

```python
from src.data.stock_data import create_dataloaders
from src.model import create_model
from src.train import StockTrainer
from src.utils import get_device

# Set up
device = get_device()

# Load data
train_loader, val_loader, data_loader = create_dataloaders(
    symbol="AAPL",
    period="2y",
    sequence_length=60,
    batch_size=32
)

# Create model
model = create_model(
    "lstm",
    input_size=15,
    hidden_size=128,
    num_layers=2,
    dropout=0.2
)

# Train model
trainer = StockTrainer(model, device, "my_model")
history = trainer.train(train_loader, val_loader, epochs=100)
```

#### Making Predictions

```python
from src.evaluate import StockPredictor

# Load trained model
predictor = StockPredictor("checkpoints/AAPL_lstm_best.pth", "lstm")

# Predict next day price
next_price = predictor.predict_next_price("AAPL")
print(f"Predicted next day price: ${next_price:.2f}")

# Predict next 5 days
future_prices = predictor.predict_sequence("AAPL", steps=5)
print(f"Future prices: {future_prices}")
```

## 🔧 Advanced Configuration

### Using Configuration Files

Create a `config.yaml` file:

```yaml
data:
  symbol: "AAPL"
  period: "2y"
  sequence_length: 60
  batch_size: 32

model:
  type: "lstm"
  hidden_size: 128
  num_layers: 2
  dropout: 0.2

training:
  epochs: 100
  learning_rate: 0.001
  patience: 15
```

### Model Architectures

#### LSTM Configuration

```python
model = create_model(
    "lstm",
    input_size=15,
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    bidirectional=True  # Optional bidirectional LSTM
)
```

#### GRU Configuration

```python
model = create_model(
    "gru",
    input_size=15,
    hidden_size=128,
    num_layers=2,
    dropout=0.2
)
```

#### Transformer Configuration

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

## 📊 Available Stock Symbols

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

## 🎨 Visualization

The project automatically generates various plots:

1. **Stock Data Visualization**: Price charts with technical indicators
2. **Training History**: Loss curves and metrics over time
3. **Predictions**: Actual vs predicted price comparisons
4. **Model Performance**: Evaluation metrics and backtesting results

## 📈 Performance Tips

### For Better Training
- Use longer sequences (60-120 days) for better pattern recognition
- Increase hidden size for more complex patterns
- Use bidirectional LSTM for better context
- Apply proper regularization (dropout, weight decay)

### For Better Predictions
- Train on longer periods (2+ years) for more data
- Use multiple models in ensemble
- Include more technical indicators
- Regular retraining with new data

### Memory Optimization
- Reduce batch size if running out of memory
- Use gradient checkpointing for large models
- Enable mixed precision training

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Run specific tests:

```bash
python -m pytest tests/test_model.py -v
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the project directory
   cd pytorch-ml-project
   # Install missing packages
   pip install -r requirements.txt
   ```

2. **Data Download Issues**
   ```bash
   # Check internet connection
   # Try different stock symbol
   # Use shorter period
   ```

3. **Memory Issues**
   ```bash
   # Reduce batch size
   python src/main.py --batch_size 16
   # Use CPU instead of GPU
   python src/main.py --device cpu
   ```

4. **CUDA Issues**
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   # Force CPU usage
   python src/main.py --device cpu
   ```

### Getting Help

1. Check the logs in the console output
2. Review the error messages carefully
3. Make sure all dependencies are installed
4. Try with default parameters first
5. Check that the stock symbol exists

## 🚀 Next Steps

1. **Experiment with different stocks and time periods**
2. **Try different model architectures**
3. **Implement your own trading strategies**
4. **Add new technical indicators**
5. **Explore ensemble methods**

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [Technical Analysis Indicators](https://python-ta-lib.readthedocs.io/)
- [Time Series Forecasting](https://otexts.com/fpp2/)

---

Happy predicting! 🎉📈
