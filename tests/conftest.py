"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import torch


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_days = 200
    
    # Generate realistic-looking price data
    base_price = 100
    returns = np.random.randn(n_days) * 0.02  # 2% daily volatility
    close = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    data = np.column_stack([
        close * (1 + np.random.randn(n_days) * 0.005),  # Open
        close * (1 + np.abs(np.random.randn(n_days) * 0.01)),  # High
        close * (1 - np.abs(np.random.randn(n_days) * 0.01)),  # Low
        close,  # Close
        np.abs(np.random.randn(n_days)) * 1000000,  # Volume
    ])
    
    return data


@pytest.fixture
def sample_model():
    """Create a sample StockLSTM model."""
    from src.model import StockLSTM
    return StockLSTM(input_size=5, hidden_size=32, num_layers=2, dropout=0.1)


@pytest.fixture
def sample_sequence_data():
    """Generate sample sequence data for model input."""
    np.random.seed(42)
    # Batch size 4, sequence length 60, 5 features
    X = np.random.randn(4, 60, 5).astype(np.float32)
    y = np.random.randn(4).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
