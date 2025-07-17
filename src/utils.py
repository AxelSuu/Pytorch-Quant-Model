"""
Utility functions for stock price prediction project.

This module contains helper functions for data visualization, model evaluation,
and other utility operations.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_stock_data(data: pd.DataFrame, 
                   title: str = "Stock Price Data",
                   save_path: Optional[str] = None,
                   figsize: Tuple[int, int] = (15, 10)):
    """
    Plot stock price data with technical indicators.
    
    Args:
        data: DataFrame containing stock data
        title: Title for the plot
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Price data
    axes[0, 0].plot(data.index, data['Close'], label='Close Price', linewidth=2)
    if 'MA_20' in data.columns:
        axes[0, 0].plot(data.index, data['MA_20'], label='MA 20', alpha=0.7)
    if 'BB_upper' in data.columns and 'BB_lower' in data.columns:
        axes[0, 0].fill_between(data.index, data['BB_upper'], data['BB_lower'], 
                               alpha=0.2, label='Bollinger Bands')
    axes[0, 0].set_title('Price with Technical Indicators')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Volume
    axes[0, 1].bar(data.index, data['Volume'], alpha=0.7, width=1)
    if 'Volume_MA' in data.columns:
        axes[0, 1].plot(data.index, data['Volume_MA'], color='red', linewidth=2, label='Volume MA')
    axes[0, 1].set_title('Trading Volume')
    axes[0, 1].set_ylabel('Volume')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # RSI
    if 'RSI' in data.columns:
        axes[1, 0].plot(data.index, data['RSI'], linewidth=2)
        axes[1, 0].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        axes[1, 0].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        axes[1, 0].set_title('RSI (Relative Strength Index)')
        axes[1, 0].set_ylabel('RSI')
        axes[1, 0].set_ylim(0, 100)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Price change
    if 'Price_Change' in data.columns:
        colors = ['red' if x < 0 else 'green' for x in data['Price_Change']]
        axes[1, 1].bar(data.index, data['Price_Change'], color=colors, alpha=0.7, width=1)
        axes[1, 1].set_title('Daily Price Change (%)')
        axes[1, 1].set_ylabel('Price Change')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_predictions(actual: np.ndarray,
                    predicted: np.ndarray,
                    dates: Optional[List] = None,
                    title: str = "Stock Price Predictions",
                    save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (15, 8)):
    """
    Plot actual vs predicted stock prices.
    
    Args:
        actual: Actual stock prices
        predicted: Predicted stock prices
        dates: Optional dates for x-axis
        title: Title for the plot
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    x_axis = dates if dates is not None else range(len(actual))
    
    # Time series plot
    ax1.plot(x_axis, actual, label='Actual', linewidth=2, alpha=0.8)
    ax1.plot(x_axis, predicted, label='Predicted', linewidth=2, alpha=0.8)
    ax1.set_title('Actual vs Predicted Prices')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot
    ax2.scatter(actual, predicted, alpha=0.6, s=20)
    
    # Perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual Price ($)')
    ax2.set_ylabel('Predicted Price ($)')
    ax2.set_title('Actual vs Predicted Scatter Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_metrics(train_losses: List[float],
                         val_losses: List[float],
                         metrics_history: Optional[Dict[str, List[float]]] = None,
                         title: str = "Training Metrics",
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (15, 6)):
    """
    Plot training metrics and loss curves.
    
    Args:
        train_losses: Training loss history
        val_losses: Validation loss history
        metrics_history: Optional dictionary of metric histories
        title: Title for the plot
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    if metrics_history:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    fig.suptitle(title, fontsize=16)
    
    # Loss curves
    axes[0].plot(train_losses, label='Training Loss', linewidth=2)
    axes[0].plot(val_losses, label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss curves (log scale)
    axes[1].plot(train_losses, label='Training Loss', linewidth=2)
    axes[1].plot(val_losses, label='Validation Loss', linewidth=2)
    axes[1].set_title('Training and Validation Loss (Log Scale)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Metrics
    if metrics_history:
        metric_names = list(metrics_history.keys())
        for i, metric in enumerate(metric_names[:4]):  # Show up to 4 metrics
            if i < len(axes) - 2:
                axes[i + 2].plot(metrics_history[metric], linewidth=2)
                axes[i + 2].set_title(f'{metric} Over Time')
                axes[i + 2].set_xlabel('Epoch')
                axes[i + 2].set_ylabel(metric)
                axes[i + 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def calculate_portfolio_metrics(returns: np.ndarray,
                               risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate portfolio performance metrics.
    
    Args:
        returns: Array of portfolio returns
        risk_free_rate: Risk-free rate (annual)
        
    Returns:
        Dictionary of portfolio metrics
    """
    returns = np.array(returns)
    
    # Annual returns
    annual_returns = np.mean(returns) * 252
    
    # Volatility
    volatility = np.std(returns) * np.sqrt(252)
    
    # Sharpe ratio
    sharpe_ratio = (annual_returns - risk_free_rate) / volatility
    
    # Maximum drawdown
    cumulative_returns = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = np.min(drawdown)
    
    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_volatility = np.std(downside_returns) * np.sqrt(252)
    sortino_ratio = (annual_returns - risk_free_rate) / downside_volatility if downside_volatility > 0 else np.inf
    
    # Calmar ratio
    calmar_ratio = annual_returns / abs(max_drawdown) if max_drawdown != 0 else np.inf
    
    return {
        'Annual_Returns': annual_returns,
        'Volatility': volatility,
        'Sharpe_Ratio': sharpe_ratio,
        'Max_Drawdown': max_drawdown,
        'Sortino_Ratio': sortino_ratio,
        'Calmar_Ratio': calmar_ratio
    }


def backtest_strategy(predictions: np.ndarray,
                     actual_prices: np.ndarray,
                     initial_capital: float = 10000,
                     transaction_cost: float = 0.001) -> Dict[str, Union[float, List[float]]]:
    """
    Backtest a simple trading strategy based on predictions.
    
    Args:
        predictions: Array of price predictions
        actual_prices: Array of actual prices
        initial_capital: Initial capital amount
        transaction_cost: Transaction cost as a fraction
        
    Returns:
        Dictionary containing backtest results
    """
    capital = initial_capital
    position = 0  # 0 = cash, 1 = long position
    portfolio_values = [initial_capital]
    trades = []
    
    for i in range(1, len(predictions)):
        current_price = actual_prices[i]
        predicted_direction = 1 if predictions[i] > actual_prices[i-1] else -1
        
        # Simple strategy: buy if prediction is up, sell if down
        if predicted_direction > 0 and position == 0:
            # Buy signal
            shares = capital / current_price
            capital = 0
            position = 1
            trades.append(('BUY', current_price, shares, i))
            
        elif predicted_direction < 0 and position == 1:
            # Sell signal
            capital = current_price * shares * (1 - transaction_cost)
            position = 0
            trades.append(('SELL', current_price, shares, i))
            
        # Calculate portfolio value
        if position == 1:
            portfolio_value = current_price * shares
        else:
            portfolio_value = capital
            
        portfolio_values.append(portfolio_value)
    
    # Calculate returns
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Calculate metrics
    total_return = (portfolio_values[-1] - initial_capital) / initial_capital
    
    return {
        'Total_Return': total_return,
        'Final_Value': portfolio_values[-1],
        'Portfolio_Values': portfolio_values,
        'Returns': returns,
        'Trades': trades,
        'Number_of_Trades': len(trades)
    }


def save_results(results: Dict,
                filename: str,
                save_dir: str = "results"):
    """
    Save results to JSON file.
    
    Args:
        results: Dictionary of results to save
        filename: Name of the file
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value
    
    filepath = os.path.join(save_dir, f"{filename}.json")
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"Results saved to {filepath}")


def load_results(filename: str,
                save_dir: str = "results") -> Dict:
    """
    Load results from JSON file.
    
    Args:
        filename: Name of the file
        save_dir: Directory containing results
        
    Returns:
        Dictionary of loaded results
    """
    filepath = os.path.join(save_dir, f"{filename}.json")
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch operations.
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_model_summary(model: torch.nn.Module, input_size: Tuple[int, ...]):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input size tuple (without batch dimension)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 50)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 50)
    
    # Try to print model structure
    try:
        print(model)
    except:
        print("Could not print model structure")


# Legacy functions for backward compatibility
def load_data(file_path):
    """
    Legacy function for loading data from CSV.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        pandas DataFrame
    """
    import pandas as pd
    data = pd.read_csv(file_path)
    return data


def save_model(model, file_path):
    """
    Legacy function for saving model state dict.
    
    Args:
        model: PyTorch model
        file_path: Path to save model
    """
    torch.save(model.state_dict(), file_path)


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for stock data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical indicators
    """
    df = df.copy()
    
    # Simple Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean() if len(df) >= 50 else df['close'].rolling(window=min(20, len(df))).mean()
    
    # Exponential Moving Averages
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    bb_window = min(20, len(df))
    bb_std = df['close'].rolling(window=bb_window).std()
    df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    
    # Stochastic Oscillator
    stoch_window = min(14, len(df))
    lowest_low = df['low'].rolling(window=stoch_window).min()
    highest_high = df['high'].rolling(window=stoch_window).max()
    df['stoch_k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Williams %R
    williams_window = min(14, len(df))
    highest_high_w = df['high'].rolling(window=williams_window).max()
    lowest_low_w = df['low'].rolling(window=williams_window).min()
    df['williams_r'] = -100 * ((highest_high_w - df['close']) / (highest_high_w - lowest_low_w))
    
    # Momentum
    df['momentum'] = df['close'] / df['close'].shift(10) - 1
    
    # Fill NaN values with forward fill and then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df


def create_sequences(data: np.ndarray, sequence_length: int = 30, target_column: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction.
    
    Args:
        data: Input data array
        sequence_length: Length of input sequences
        target_column: Index of target column
        
    Returns:
        Tuple of (X, y) arrays for training
    """
    X, y = [], []
    
    for i in range(sequence_length, len(data)):
        # Input sequence
        X.append(data[i-sequence_length:i])
        
        # Target (next value of target column)
        y.append(data[i, target_column])
    
    return np.array(X), np.array(y)