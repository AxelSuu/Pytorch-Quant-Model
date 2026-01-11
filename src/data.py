"""Data fetching and preprocessing module."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Compute MACD, Signal line, and Histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple:
    """Compute Bollinger Bands."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    # Return band width and %B indicator
    band_width = (upper - lower) / (sma + 1e-10)
    percent_b = (series - lower) / (upper - lower + 1e-10)
    return band_width, percent_b


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to OHLCV dataframe.
    
    Adds:
        - SMA_10, SMA_20, SMA_50: Simple Moving Averages
        - EMA_12, EMA_26: Exponential Moving Averages
        - RSI_14: Relative Strength Index
        - MACD, MACD_Signal, MACD_Hist: MACD indicators
        - BB_Width, BB_PercentB: Bollinger Band indicators
        - Price_Change: Daily price change percentage
        - Volume_Change: Daily volume change percentage
    """
    close = df["Close"]
    volume = df["Volume"]
    
    # Moving Averages
    df["SMA_10"] = close.rolling(window=10).mean()
    df["SMA_20"] = close.rolling(window=20).mean()
    df["SMA_50"] = close.rolling(window=50).mean()
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    
    # RSI
    df["RSI_14"] = compute_rsi(close, 14)
    
    # MACD
    macd, signal, hist = compute_macd(close)
    df["MACD"] = macd
    df["MACD_Signal"] = signal
    df["MACD_Hist"] = hist
    
    # Bollinger Bands
    bb_width, bb_pctb = compute_bollinger_bands(close)
    df["BB_Width"] = bb_width
    df["BB_PercentB"] = bb_pctb
    
    # Price and Volume momentum
    df["Price_Change"] = close.pct_change()
    df["Volume_Change"] = volume.pct_change()
    
    # Fill NaN values from rolling windows
    df = df.ffill().bfill()
    
    return df


def fetch_stock_data(symbol: str, start: str = None, end: str = None, 
                     use_indicators: bool = False) -> tuple:
    """Fetch stock data from Yahoo Finance.
    
    Args:
        symbol: Stock ticker symbol
        start: Start date (optional)
        end: End date (optional)
        use_indicators: Whether to include technical indicators
    
    Returns:
        tuple: (data array, last date)
    """
    ticker = yf.Ticker(symbol)
    
    if start and end:
        df = ticker.history(start=start, end=end)
    else:
        df = ticker.history(period="2y")
    
    if df.empty:
        raise ValueError(f"No data found for {symbol}")
    
    if use_indicators:
        df = add_technical_indicators(df)
        # Return all columns except Dividends and Stock Splits
        cols = ["Open", "High", "Low", "Close", "Volume",
                "SMA_10", "SMA_20", "SMA_50", "EMA_12", "EMA_26",
                "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
                "BB_Width", "BB_PercentB", "Price_Change", "Volume_Change"]
        data = df[cols].values
    else:
        # Use OHLCV features only
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
                 start: str = None, end: str = None, use_indicators: bool = False) -> tuple:
    """Prepare data loaders for training.
    
    Args:
        symbol: Stock ticker symbol
        seq_length: Sequence length for LSTM
        batch_size: Batch size for DataLoader
        train_split: Fraction of data for training
        start: Start date (optional)
        end: End date (optional)
        use_indicators: Whether to include technical indicators
    
    Returns:
        tuple: (train_loader, val_loader, scaler, input_size)
    """
    data, _ = fetch_stock_data(symbol, start, end, use_indicators)
    input_size = data.shape[1]
    
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
    
    return train_loader, val_loader, scaler, input_size


def prepare_prediction_data(symbol: str, seq_length: int, scaler: MinMaxScaler = None,
                            use_indicators: bool = False) -> tuple:
    """Prepare data for prediction.
    
    Args:
        symbol: Stock ticker symbol
        seq_length: Sequence length for LSTM
        scaler: Pre-fitted scaler from checkpoint (optional, will fit new if None)
        use_indicators: Whether to include technical indicators
    
    Returns:
        tuple: (sequence_tensor, scaler, current_price, last_date)
    """
    data, last_date = fetch_stock_data(symbol, use_indicators=use_indicators)
    
    if scaler is None:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
    else:
        scaled_data = scaler.transform(data)
    
    # Get last sequence
    last_sequence = torch.FloatTensor(scaled_data[-seq_length:]).unsqueeze(0)
    current_price = data[-1, 3]  # Last close price
    
    return last_sequence, scaler, current_price, last_date
