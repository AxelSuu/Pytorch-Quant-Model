"""Tests for data module."""

import pytest
import numpy as np
import pandas as pd
import torch

from src.data import (
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    add_technical_indicators,
    create_sequences,
)


class TestTechnicalIndicators:
    """Tests for technical indicator calculations."""
    
    def test_compute_rsi_range(self):
        """RSI should be between 0 and 100."""
        # Create sample price data
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                           110, 108, 106, 105, 107, 109, 111, 113, 112, 114])
        rsi = compute_rsi(prices, period=14)
        
        # RSI should be between 0 and 100
        assert rsi.min() >= 0, "RSI should not be negative"
        assert rsi.max() <= 100, "RSI should not exceed 100"
    
    def test_compute_rsi_extreme_values(self):
        """RSI should approach extremes for consistent moves."""
        # Consistently rising prices
        rising = pd.Series(list(range(100, 150)))
        rsi_rising = compute_rsi(rising, period=14)
        assert rsi_rising.iloc[-1] > 70, "RSI should be high for rising prices"
        
        # Consistently falling prices
        falling = pd.Series(list(range(150, 100, -1)))
        rsi_falling = compute_rsi(falling, period=14)
        assert rsi_falling.iloc[-1] < 30, "RSI should be low for falling prices"
    
    def test_compute_macd_returns_three_series(self):
        """MACD should return macd line, signal, and histogram."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        macd, signal, hist = compute_macd(prices)
        
        assert len(macd) == len(prices), "MACD length should match input"
        assert len(signal) == len(prices), "Signal length should match input"
        assert len(hist) == len(prices), "Histogram length should match input"
    
    def test_compute_macd_histogram_is_difference(self):
        """Histogram should be MACD - Signal."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        macd, signal, hist = compute_macd(prices)
        
        expected_hist = macd - signal
        np.testing.assert_array_almost_equal(hist.values, expected_hist.values)
    
    def test_compute_bollinger_bands_returns_two_series(self):
        """Bollinger bands should return band width and %B."""
        prices = pd.Series(np.random.randn(50).cumsum() + 100)
        band_width, percent_b = compute_bollinger_bands(prices, period=20)
        
        assert len(band_width) == len(prices)
        assert len(percent_b) == len(prices)
    
    def test_add_technical_indicators_columns(self):
        """Adding indicators should create expected columns."""
        df = pd.DataFrame({
            "Open": np.random.randn(100).cumsum() + 100,
            "High": np.random.randn(100).cumsum() + 102,
            "Low": np.random.randn(100).cumsum() + 98,
            "Close": np.random.randn(100).cumsum() + 100,
            "Volume": np.abs(np.random.randn(100)) * 1000000,
        })
        
        result = add_technical_indicators(df)
        
        expected_cols = [
            "SMA_10", "SMA_20", "SMA_50", "EMA_12", "EMA_26",
            "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
            "BB_Width", "BB_PercentB", "Price_Change", "Volume_Change"
        ]
        
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_add_technical_indicators_no_nan(self):
        """After adding indicators, there should be no NaN values."""
        df = pd.DataFrame({
            "Open": np.random.randn(100).cumsum() + 100,
            "High": np.random.randn(100).cumsum() + 102,
            "Low": np.random.randn(100).cumsum() + 98,
            "Close": np.random.randn(100).cumsum() + 100,
            "Volume": np.abs(np.random.randn(100)) * 1000000,
        })
        
        result = add_technical_indicators(df)
        
        assert not result.isna().any().any(), "Should have no NaN values after indicators"


class TestSequenceCreation:
    """Tests for sequence creation."""
    
    def test_create_sequences_shape(self):
        """Sequences should have correct shape."""
        data = np.random.randn(100, 5)  # 100 timesteps, 5 features
        seq_length = 10
        
        X, y = create_sequences(data, seq_length)
        
        assert X.shape == (90, 10, 5), f"Expected (90, 10, 5), got {X.shape}"
        assert y.shape == (90,), f"Expected (90,), got {y.shape}"
    
    def test_create_sequences_content(self):
        """Sequence content should be correct."""
        data = np.arange(20).reshape(4, 5)  # 4 timesteps, 5 features
        seq_length = 2
        
        X, y = create_sequences(data, seq_length)
        
        # First sequence should be rows 0-1, target is Close (index 3) from row 2
        np.testing.assert_array_equal(X[0], data[0:2])
        assert y[0] == data[2, 3]
        
        # Second sequence should be rows 1-2, target is Close from row 3
        np.testing.assert_array_equal(X[1], data[1:3])
        assert y[1] == data[3, 3]
    
    def test_create_sequences_empty_with_short_data(self):
        """Should handle data shorter than sequence length."""
        data = np.random.randn(5, 5)
        seq_length = 10
        
        X, y = create_sequences(data, seq_length)
        
        assert len(X) == 0, "Should return empty array for short data"
        assert len(y) == 0
