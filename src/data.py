"""Data fetching and preprocessing module."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def fetch_stock_data(symbol: str, start: str = None, end: str = None) -> np.ndarray:
    """Fetch stock data from Yahoo Finance."""
    ticker = yf.Ticker(symbol)
    
    if start and end:
        df = ticker.history(start=start, end=end)
    else:
        df = ticker.history(period="2y")
    
    if df.empty:
        raise ValueError(f"No data found for {symbol}")
    
    # Use OHLCV features
    data = df[["Open", "High", "Low", "Close", "Volume"]].values
    return data, df.index[-1]


def create_sequences(data: np.ndarray, seq_length: int) -> tuple:
    """Create sequences for LSTM training."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 3])  # Close price
    return np.array(X), np.array(y)


def prepare_data(symbol: str, seq_length: int, batch_size: int, train_split: float,
                 start: str = None, end: str = None) -> tuple:
    """Prepare data loaders for training."""
    data, _ = fetch_stock_data(symbol, start, end)
    
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = create_sequences(scaled_data, seq_length)
    
    # Split data
    split_idx = int(len(X) * train_split)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, scaler


def prepare_prediction_data(symbol: str, seq_length: int) -> tuple:
    """Prepare data for prediction."""
    data, last_date = fetch_stock_data(symbol)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Get last sequence
    last_sequence = torch.FloatTensor(scaled_data[-seq_length:]).unsqueeze(0)
    current_price = data[-1, 3]  # Last close price
    
    return last_sequence, scaler, current_price, last_date
