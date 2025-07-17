"""
Stock data handling module for fetching, processing, and preparing stock data for training.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import os
import warnings
warnings.filterwarnings('ignore')


class StockDataset(Dataset):
    """
    Custom PyTorch Dataset for stock price time series data.
    
    Args:
        data: Processed stock data array
        sequence_length: Length of input sequences
        target_column: Index of the target column (default: 3 for 'Close')
    """
    
    def __init__(self, data: np.ndarray, sequence_length: int = 60, target_column: int = 3):
        self.data = data
        self.sequence_length = sequence_length
        self.target_column = target_column
        
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence of features (all columns)
        sequence = self.data[idx:idx + self.sequence_length]
        
        # Get target value (next day's closing price)
        target = self.data[idx + self.sequence_length, self.target_column]
        
        return torch.FloatTensor(sequence), torch.FloatTensor([target])


class StockDataLoader:
    """
    Handles stock data fetching, preprocessing, and dataset creation.
    """
    
    def __init__(self, symbol: str = "AAPL", period: str = "5y"):
        """
        Initialize the stock data loader.
        
        Args:
            symbol: Stock symbol (e.g., "AAPL", "GOOGL")
            period: Time period for data ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
        """
        self.symbol = symbol
        self.period = period
        self.scaler = MinMaxScaler()
        self.data = None
        self.scaled_data = None
        
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.
        
        Returns:
            Raw stock data DataFrame
        """
        try:
            print(f"Fetching data for {self.symbol} for period {self.period}...")
            stock = yf.Ticker(self.symbol)
            data = stock.history(period=self.period)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
                
            print(f"Fetched {len(data)} days of data")
            return data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            raise
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the stock data.
        
        Args:
            data: Raw stock data
            
        Returns:
            Data with technical indicators
        """
        df = data.copy()
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # Relative Strength Index (RSI)
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
            
        df['RSI'] = calculate_rsi(df['Close'])
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        
        # Price change indicators
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        
        # Volatility (rolling standard deviation)
        df['Volatility'] = df['Close'].rolling(window=10).std()
        
        return df
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Preprocess the stock data for training.
        
        Args:
            data: Raw stock data with technical indicators
            
        Returns:
            Scaled data array and feature names
        """
        # Select features for training
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'RSI',
            'BB_upper', 'BB_lower', 'Volume_MA',
            'Price_Change', 'High_Low_Pct', 'Volatility'
        ]
        
        # Remove rows with NaN values
        df = data[feature_columns].dropna()
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(df.values)
        
        print(f"Preprocessed data shape: {scaled_data.shape}")
        print(f"Features: {feature_columns}")
        
        return scaled_data, feature_columns
    
    def preprocess_realtime_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess real-time stock data using the existing fitted scaler.
        
        Args:
            data: Raw stock data with technical indicators
            
        Returns:
            Scaled data array using the existing scaler
        """
        # Select features for training (same as training data)
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'RSI',
            'BB_upper', 'BB_lower', 'Volume_MA',
            'Price_Change', 'High_Low_Pct', 'Volatility'
        ]
        
        # Remove rows with NaN values
        df = data[feature_columns].dropna()
        
        # Scale the data using the EXISTING scaler (no refitting!)
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Load and preprocess training data first.")
        
        scaled_data = self.scaler.transform(df.values)  # transform only, no fit!
        
        print(f"Real-time preprocessed data shape: {scaled_data.shape}")
        print(f"Features: {feature_columns}")
        
        return scaled_data
    
    def create_datasets(self, sequence_length: int = 60, train_ratio: float = 0.8) -> Tuple[StockDataset, StockDataset]:
        """
        Create training and testing datasets.
        
        Args:
            sequence_length: Length of input sequences
            train_ratio: Ratio of data to use for training
            
        Returns:
            Training and testing datasets
        """
        if self.scaled_data is None:
            raise ValueError("Data must be preprocessed first. Call load_and_preprocess_data()")
        
        # Split data
        train_size = int(len(self.scaled_data) * train_ratio)
        train_data = self.scaled_data[:train_size]
        test_data = self.scaled_data[train_size:]
        
        # Create datasets
        train_dataset = StockDataset(train_data, sequence_length)
        test_dataset = StockDataset(test_data, sequence_length)
        
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Testing dataset size: {len(test_dataset)}")
        
        return train_dataset, test_dataset
    
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, List[str]]:
        """
        Complete data loading and preprocessing pipeline.
        
        Returns:
            Scaled data array and feature names
        """
        # Fetch data
        self.data = self.fetch_data()
        
        # Add technical indicators
        self.data = self.add_technical_indicators(self.data)
        
        # Preprocess data
        self.scaled_data, feature_names = self.preprocess_data(self.data)
        
        return self.scaled_data, feature_names
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Inverse transform predictions to original scale.
        
        Args:
            predictions: Scaled predictions
            
        Returns:
            Predictions in original scale
        """
        # Create dummy array with same shape as original features
        dummy = np.zeros((predictions.shape[0], self.scaled_data.shape[1]))
        # Place predictions in the 'Close' price column (index 3)
        dummy[:, 3] = predictions.flatten()
        
        # Inverse transform
        inverse_transformed = self.scaler.inverse_transform(dummy)
        
        return inverse_transformed[:, 3]
    
    def get_latest_sequence(self, sequence_length: int = 60) -> torch.FloatTensor:
        """
        Get the latest sequence for prediction.
        
        Args:
            sequence_length: Length of the sequence
            
        Returns:
            Latest sequence tensor
        """
        if self.scaled_data is None:
            raise ValueError("Data must be preprocessed first")
            
        latest_sequence = self.scaled_data[-sequence_length:]
        return torch.FloatTensor(latest_sequence).unsqueeze(0)  # Add batch dimension
    
    def get_realtime_sequence(self, sequence_length: int = 60) -> torch.FloatTensor:
        """
        Get the latest real-time sequence for prediction by fetching fresh data.
        
        Args:
            sequence_length: Length of the sequence needed
            
        Returns:
            Latest real-time sequence tensor
        """
        print(f"📡 Fetching real-time data for {self.symbol}...")
        
        # Fetch recent data (more than needed to ensure we have enough for technical indicators)
        recent_period = f"{sequence_length + 50}d"  # Get extra days for technical indicators
        
        try:
            # Get fresh data from yfinance
            ticker = yf.Ticker(self.symbol)
            recent_data = ticker.history(period=recent_period)
            
            if recent_data.empty:
                raise ValueError(f"No recent data available for {self.symbol}")
            
            # Add technical indicators to the fresh data
            recent_data_with_indicators = self.add_technical_indicators(recent_data)
            
            # Process the recent data using the EXISTING scaler (no refitting!)
            scaled_recent = self.preprocess_realtime_data(recent_data_with_indicators)
            
            # Get the last sequence_length points
            if len(scaled_recent) < sequence_length:
                raise ValueError(f"Not enough recent data. Got {len(scaled_recent)}, need {sequence_length}")
            
            latest_sequence = scaled_recent[-sequence_length:]
            
            # Get the actual latest close price for reference
            latest_close = recent_data['Close'].iloc[-1]
            print(f"💰 Latest {self.symbol} close price: ${latest_close:.2f}")
            print(f"📅 Latest data date: {recent_data.index[-1].strftime('%Y-%m-%d')}")
            
            return torch.FloatTensor(latest_sequence).unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            print(f"❌ Error fetching real-time data: {e}")
            print("🔄 Falling back to historical data...")
            return self.get_latest_sequence(sequence_length)
    
    def is_market_open(self) -> bool:
        """
        Check if the market is currently open (basic implementation).
        
        Returns:
            True if market is likely open, False otherwise
        """
        import datetime
        now = datetime.datetime.now()
        
        # Basic check: Monday-Friday, 9:30 AM - 4:00 PM ET
        # This is a simplified check and doesn't account for holidays
        if now.weekday() >= 5:  # Weekend
            return False
            
        # Convert to ET (this is simplified - doesn't handle DST properly)
        # In a production system, you'd use pytz for proper timezone handling
        hour = now.hour
        return 9 <= hour <= 16


def create_dataloaders(symbol: str = "AAPL", 
                      period: str = "5y",
                      sequence_length: int = 60,
                      batch_size: int = 32,
                      train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader, StockDataLoader]:
    """
    Create training and testing data loaders for stock price prediction.
    
    Args:
        symbol: Stock symbol
        period: Time period for data
        sequence_length: Length of input sequences
        batch_size: Batch size for training
        train_ratio: Ratio of data to use for training
        
    Returns:
        Training DataLoader, Testing DataLoader, and StockDataLoader instance
    """
    # Initialize data loader
    stock_data_loader = StockDataLoader(symbol, period)
    
    # Load and preprocess data
    stock_data_loader.load_and_preprocess_data()
    
    # Create datasets
    train_dataset, test_dataset = stock_data_loader.create_datasets(sequence_length, train_ratio)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, stock_data_loader


if __name__ == "__main__":
    # Example usage
    train_loader, test_loader, data_loader = create_dataloaders("AAPL", "2y", 60, 32)
    
    # Test data loading
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Data shape: {data.shape}, Target shape: {target.shape}")
        if batch_idx == 0:  # Just show first batch
            break
