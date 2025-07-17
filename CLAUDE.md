# CLAUDE.md - Development Guide & AI Assistant Context

This document serves as a development guide and provides context for AI assistants working on this PyTorch stock price prediction project.

## 🤖 AI Assistant Context

### Project Overview
This is a comprehensive PyTorch-based stock price forecasting system with a modular architecture designed for flexibility and scalability. The project is fully implemented and production-ready, specializing in time series prediction using deep learning models.

### Current State
- **Framework**: PyTorch 2.0+ with modern features
- **Domain**: Stock price prediction and financial time series forecasting
- **Architecture**: Fully modular design with separate components for data, models, training, and evaluation
- **Testing**: Complete pytest test suite with 17+ passing tests
- **Dependencies**: Modern ML stack (PyTorch, yfinance, scikit-learn, matplotlib, seaborn)

### Key Components
1. **src/main.py**: CLI entry point with full argument parsing and demo mode
2. **src/model.py**: Three neural architectures (LSTM, GRU, Transformer) with ensemble support
3. **src/train.py**: Advanced training infrastructure with StockTrainer class
4. **src/utils.py**: Comprehensive utilities for visualization, evaluation, and device management
5. **src/data/stock_data.py**: Complete stock data pipeline with technical indicators
6. **src/evaluate.py**: Model evaluation, prediction, and backtesting tools
7. **tests/**: Full test coverage for all components
8. **examples/**: Demo scripts and usage examples

## 🎯 Development Status

### Phase 1: Foundation ✅ COMPLETED
- [x] Project structure setup
- [x] Comprehensive documentation (README.md, CLAUDE.md, USAGE.md, PROJECT_SUMMARY.md)
- [x] Advanced model implementations (LSTM, GRU, Transformer)
- [x] Complete training pipeline with StockTrainer
- [x] Data loading utilities with Yahoo Finance integration

### Phase 2: Core Features ✅ COMPLETED
- [x] Multiple model architectures (LSTM, GRU, Transformer)
- [x] Configuration management (config.yaml)
- [x] Logging and monitoring (training metrics, visualization)
- [x] Model checkpointing and saving (best model persistence)
- [x] Evaluation metrics (MSE, MAE, RMSE, directional accuracy)

### Phase 3: Advanced Features ✅ COMPLETED
- [x] Technical indicators (RSI, Bollinger Bands, moving averages)
- [x] Model comparison tools (ensemble methods)
- [x] Visualization utilities (training plots, prediction charts)
- [x] Performance optimization (GPU support, efficient data loading)
- [x] Production-ready code (error handling, type hints, documentation)

## 🔧 Technical Implementation

### Stock Price Prediction System
The project implements a complete stock price forecasting system using:

#### Data Pipeline
- **Yahoo Finance Integration**: Real-time stock data fetching via yfinance
- **Technical Indicators**: RSI, Bollinger Bands, moving averages, volatility metrics
- **Data Preprocessing**: Min-max scaling, sequence generation, train/validation splits
- **Feature Engineering**: 15 technical features per time step

#### Model Architectures
- **StockLSTM**: Bidirectional LSTM with dropout and batch normalization
- **StockGRU**: Efficient GRU-based architecture for faster training
- **StockTransformer**: Multi-head attention mechanism for complex patterns
- **Ensemble Methods**: Combine predictions from multiple models

#### Training Infrastructure
- **StockTrainer Class**: Complete training workflow with early stopping
- **Advanced Metrics**: MSE, MAE, RMSE, directional accuracy tracking
- **Checkpointing**: Automatic saving of best models during training
- **GPU Support**: Automatic device detection and optimization

### Code Quality Standards
- **Type Hints**: Complete type annotation throughout codebase
- **Documentation**: Comprehensive docstrings for all functions and classes
- **Error Handling**: Robust exception handling and user feedback
- **Testing**: 17+ unit tests covering all major components
- **Logging**: Detailed training progress and evaluation metrics

### Model Development Best Practices
- All models inherit from `nn.Module` with proper initialization
- Forward methods handle variable sequence lengths
- Device-agnostic implementations (CPU/GPU compatibility)
- Batch processing support for efficient training
- Proper gradient handling and memory management

## 📊 Implemented Features

### Current Implementation: Stock Price Prediction System
The project has evolved from a generic ML framework to a specialized stock price forecasting system with:

#### Data Features
1. **Real-time Data**: Yahoo Finance integration for live stock data
2. **Technical Analysis**: RSI, Bollinger Bands, moving averages, volatility
3. **Flexible Timeframes**: Support for 1d to 10y historical data periods
4. **Data Preprocessing**: Advanced scaling and sequence generation
5. **Feature Engineering**: 15 technical indicators per time step

#### Model Architectures
1. **LSTM Networks**: Bidirectional support with configurable layers
2. **GRU Networks**: Efficient alternative to LSTM for faster training
3. **Transformer Models**: Multi-head attention for complex pattern recognition
4. **Ensemble Methods**: Combine multiple models for improved accuracy
5. **Model Factory**: Easy creation of different architectures

#### Training & Evaluation
1. **Advanced Training Loop**: Early stopping, learning rate scheduling
2. **Comprehensive Metrics**: MSE, MAE, RMSE, directional accuracy
3. **Model Checkpointing**: Automatic saving of best performing models
4. **Visualization Tools**: Training curves, prediction plots, stock charts
5. **Backtesting**: Historical performance evaluation

#### CLI Interface
1. **Full Argument Parsing**: Extensive command-line options
2. **Demo Mode**: Quick start with `python src/main.py demo`
3. **Flexible Configuration**: Support for different stocks, timeframes, models
4. **Real-time Predictions**: Latest sequence prediction capabilities
5. **Batch Processing**: Efficient handling of multiple predictions

## 🚀 Usage Examples

### Basic Stock Prediction
```bash
# Quick demo with default settings
python src/main.py demo

# Train LSTM model for Apple stock
python src/main.py --symbol AAPL --model_type lstm --epochs 100

# Train Transformer model with custom parameters
python src/main.py --symbol TSLA --model_type transformer --hidden_size 256 --num_layers 4

# Evaluate model with different timeframes
python src/main.py --symbol MSFT --period 5y --sequence_length 90
```

### Advanced Usage
```bash
# Train with specific configuration
python src/main.py --symbol GOOGL --model_type gru --epochs 200 --lr 0.0005 --batch_size 64

# Use bidirectional LSTM
python src/main.py --symbol NVDA --model_type lstm --bidirectional --hidden_size 128

# Custom training parameters
python src/main.py --symbol AMZN --epochs 150 --patience 20 --weight_decay 1e-4
```

### Model Development Templates

#### Adding a New Stock Model
```python
class NewStockModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
```

#### Custom Training Loop
```python
def train_stock_model(model, train_loader, val_loader, device, epochs=100):
    trainer = StockTrainer(model, device, "my_model", "checkpoints")
    history = trainer.train(train_loader, val_loader, epochs=epochs)
    return trainer, history
```

#### Model Evaluation
```python
from src.evaluate import evaluate_model, backtest_model

# Evaluate model performance
metrics = evaluate_model(trainer.model, test_loader, device)

# Run backtesting
backtest_results = backtest_model(trainer.model, data_loader, device)
```

## 📝 Documentation Standards

### Function Documentation
```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model to train
        dataloader: Training data loader
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to run training on
        
    Returns:
        float: Average loss for the epoch
    """
```

### Class Documentation
```python
class ModelTrainer:
    """
    Handles model training with logging and checkpointing.
    
    Attributes:
        model: The PyTorch model to train
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        
    Example:
        trainer = ModelTrainer(model, optimizer)
        trainer.train(train_loader, val_loader, epochs=10)
    """
```

## 🔄 Development Workflow

1. **Planning**: Define the specific ML problem to solve
2. **Data**: Prepare and understand the dataset
3. **Model**: Design appropriate architecture
4. **Training**: Implement training loop with monitoring
5. **Evaluation**: Test model performance
6. **Optimization**: Tune hyperparameters and architecture
7. **Documentation**: Update README and add comments
8. **Testing**: Write comprehensive tests

## 🛠️ Commands & Testing

### Development Commands
```bash
# Run stock prediction demo
python src/main.py demo

# Train model with specific parameters
python src/main.py --symbol AAPL --model_type lstm --epochs 100

# Run comprehensive tests
pytest tests/test_model.py -v

# Run specific test
pytest tests/test_model.py::test_lstm_model -v

# Check model performance
python src/evaluate.py --model_path checkpoints/best_model.pth
```

### Testing Results
The project includes comprehensive testing:
- ✅ **17+ unit tests** covering all model types
- ✅ **Model architecture validation** (LSTM, GRU, Transformer)
- ✅ **Training pipeline tests** (StockTrainer functionality)
- ✅ **Data loading tests** (Yahoo Finance integration)
- ✅ **Utility function tests** (visualization, evaluation metrics)

### Configuration Management
```yaml
# config.yaml example
data:
  symbol: "AAPL"
  period: "2y"
  sequence_length: 60
  
model:
  type: "lstm"
  hidden_size: 128
  num_layers: 2
  dropout: 0.2
  
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
```

## 🎯 Success Metrics (ACHIEVED)

- **Code Quality**: ✅ Clean, fully documented, extensively tested codebase
- **Performance**: ✅ Efficient training with GPU support and optimized data loading
- **Flexibility**: ✅ Easy to add new models, stocks, and timeframes
- **Reproducibility**: ✅ Consistent results with seed management and checkpointing
- **Usability**: ✅ Comprehensive documentation, CLI interface, and demo mode

## 🔄 Extension Opportunities

The project is complete and ready for extensions such as:

### Advanced Features
- **Sentiment Analysis**: Integrate news sentiment with price prediction
- **Multi-Asset Support**: Portfolio-level predictions across multiple stocks
- **Real-time Trading**: Live trading signal generation
- **Options Pricing**: Extend to derivatives and volatility prediction
- **Crypto Support**: Cryptocurrency price prediction

### Technical Enhancements
- **Hyperparameter Tuning**: Automated optimization with Optuna
- **Model Interpretability**: SHAP values and attention visualization
- **Distributed Training**: Multi-GPU and multi-node support
- **Model Compression**: Quantization and pruning for deployment
- **REST API**: Web service for real-time predictions

### Research Directions
- **Advanced Architectures**: Graph neural networks for market structure
- **Reinforcement Learning**: Trading agents with RL algorithms
- **Multi-modal Learning**: Combine price, volume, and text data
- **Federated Learning**: Distributed model training across institutions

## 🤝 Collaboration Guidelines

- Keep commits atomic and well-described
- Update documentation with code changes
- Add tests for new functionality
- Follow the established project structure for stock prediction
- Communicate changes that affect the model APIs
- Maintain backward compatibility with existing model interfaces

## 📊 Current Performance

The system has been tested and validated with:
- **Multiple Stock Symbols**: AAPL, TSLA, MSFT, GOOGL, NVDA, AMZN
- **Various Timeframes**: 1d to 10y historical data
- **All Model Types**: LSTM, GRU, and Transformer architectures
- **Comprehensive Metrics**: MSE, MAE, RMSE, directional accuracy
- **Real-time Predictions**: Latest sequence forecasting capabilities

---

**This project is production-ready for stock price prediction. The documentation reflects the current fully-implemented state with all core features operational and tested.**
