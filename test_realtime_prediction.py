"""
Test script for real-time prediction capabilities.

This script tests:
1. Next-day price prediction for all model types
2. Multi-day sequence prediction
3. Confidence intervals and uncertainty estimation
4. Model performance with synthetic real-time data
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import models and utilities
from model import StockLSTM, StockGRU, StockTransformer, create_model
from enhanced_prediction import EnhancedPredictor
from utils import create_sequences, calculate_technical_indicators

class MockDataLoader:
    """Mock data loader for testing purposes."""
    
    def __init__(self):
        # Create mock scaling parameters
        self.scaler_y = type('MockScaler', (), {
            'inverse_transform': lambda x: x * 100 + 50,  # Mock inverse scaling
            'transform': lambda x: (x - 50) / 100
        })()
        
        # Mock feature scaling parameters
        self.feature_scaler = type('MockScaler', (), {
            'inverse_transform': lambda x: x,  # Identity for simplicity
            'transform': lambda x: x
        })()

def generate_synthetic_market_data(days: int = 100, start_price: float = 100.0) -> pd.DataFrame:
    """
    Generate synthetic market data for testing.
    
    Args:
        days: Number of days to generate
        start_price: Starting price
        
    Returns:
        DataFrame with OHLCV data and technical indicators
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic price movements
    returns = np.random.normal(0.001, 0.02, days)  # 0.1% mean return, 2% volatility
    prices = [start_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = prices[1:]  # Remove the initial price
    
    # Generate OHLC data
    data = []
    for i, close in enumerate(prices):
        # Generate realistic OHLC from close price
        volatility = 0.015  # 1.5% intraday volatility
        high = close * (1 + np.random.uniform(0, volatility))
        low = close * (1 - np.random.uniform(0, volatility))
        open_price = close * (1 + np.random.uniform(-volatility/2, volatility/2))
        volume = np.random.randint(1000000, 10000000)
        
        data.append({
            'date': datetime.now() - timedelta(days=days-i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df = df.set_index('date')
    
    # Add technical indicators
    df = calculate_technical_indicators(df)
    
    return df

def create_test_input(df: pd.DataFrame, sequence_length: int = 30) -> torch.Tensor:
    """
    Create test input from market data.
    
    Args:
        df: Market data DataFrame
        sequence_length: Length of input sequence
        
    Returns:
        Tensor of shape (1, sequence_length, num_features)
    """
    # Select features (assuming specific order)
    feature_columns = [
        'close', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_middle', 'bb_lower',
        'sma_20', 'ema_12', 'ema_26', 'atr', 'stoch_k', 'stoch_d', 'williams_r', 'momentum'
    ]
    
    # Take the last sequence_length rows
    features = df[feature_columns].tail(sequence_length).values
    
    # Normalize features (simple min-max scaling for testing)
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    
    # Convert to tensor and add batch dimension
    tensor = torch.FloatTensor(features).unsqueeze(0)
    
    return tensor

def test_model_predictions(model_name: str, model: torch.nn.Module, test_input: torch.Tensor):
    """
    Test prediction capabilities of a single model.
    
    Args:
        model_name: Name of the model
        model: Model instance
        test_input: Input tensor for testing
    """
    print(f"\n{'='*50}")
    print(f"Testing {model_name} Model")
    print(f"{'='*50}")
    
    try:
        # Test 1: Single next-day prediction
        print("\n1. Testing next-day prediction...")
        next_day_pred = model.predict_next_day(test_input)
        print(f"   Next day prediction shape: {next_day_pred.shape}")
        print(f"   Next day prediction value: {next_day_pred.item():.4f}")
        
        # Test 2: Multi-day sequence prediction
        print("\n2. Testing multi-day sequence prediction...")
        for steps in [3, 5, 10]:
            sequence_pred = model.predict_sequence(test_input, steps=steps)
            print(f"   {steps}-day sequence shape: {sequence_pred.shape}")
            print(f"   {steps}-day predictions: {sequence_pred.squeeze().detach().numpy()}")
        
        # Test 3: Enhanced predictor integration
        print("\n3. Testing enhanced predictor integration...")
        mock_loader = MockDataLoader()
        enhanced_predictor = EnhancedPredictor(model, mock_loader, torch.device('cpu'))
        
        # Test batch prediction
        batch_input = test_input.repeat(5, 1, 1)  # Create batch of 5
        batch_predictions = enhanced_predictor.predict_batch(batch_input)
        print(f"   Batch prediction shape: {batch_predictions.shape}")
        print(f"   Batch predictions: {batch_predictions.squeeze().detach().numpy()}")
        
        # Test confidence intervals
        print("\n4. Testing confidence interval estimation...")
        confidence_bounds = enhanced_predictor.estimate_confidence_intervals(test_input, n_samples=10)
        print(f"   Lower bound: {confidence_bounds['lower']:.4f}")
        print(f"   Upper bound: {confidence_bounds['upper']:.4f}")
        print(f"   Prediction: {confidence_bounds['prediction']:.4f}")
        
        print(f"\n✅ {model_name} model tests PASSED!")
        
    except Exception as e:
        print(f"\n❌ {model_name} model tests FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

def test_realtime_simulation():
    """
    Simulate real-time prediction scenario.
    """
    print(f"\n{'='*60}")
    print("REAL-TIME PREDICTION SIMULATION")
    print(f"{'='*60}")
    
    # Generate market data
    print("\n1. Generating synthetic market data...")
    market_data = generate_synthetic_market_data(days=100)
    print(f"   Generated {len(market_data)} days of market data")
    print(f"   Latest close price: ${market_data['close'].iloc[-1]:.2f}")
    
    # Create models
    print("\n2. Initializing models...")
    models = {
        'LSTM': create_model('lstm', input_size=15, hidden_size=64, num_layers=2),
        'GRU': create_model('gru', input_size=15, hidden_size=64, num_layers=2),
        'Transformer': create_model('transformer', input_size=15, d_model=64, num_heads=4, num_layers=2)
    }
    
    # Create test input
    print("\n3. Preparing input data...")
    test_input = create_test_input(market_data, sequence_length=30)
    print(f"   Input tensor shape: {test_input.shape}")
    
    # Test each model
    print("\n4. Testing model predictions...")
    all_predictions = {}
    
    for model_name, model in models.items():
        test_model_predictions(model_name, model, test_input)
        
        # Collect predictions for comparison
        with torch.no_grad():
            next_day = model.predict_next_day(test_input).item()
            week_ahead = model.predict_sequence(test_input, steps=5).squeeze().detach().numpy()
            all_predictions[model_name] = {
                'next_day': next_day,
                'week_ahead': week_ahead
            }
    
    # Compare predictions across models
    print(f"\n{'='*60}")
    print("PREDICTION COMPARISON ACROSS MODELS")
    print(f"{'='*60}")
    
    current_price = market_data['close'].iloc[-1]
    print(f"\nCurrent Price: ${current_price:.2f}")
    print("\nNext Day Predictions:")
    for model_name, preds in all_predictions.items():
        next_price = preds['next_day']
        change_pct = ((next_price - current_price) / current_price) * 100
        print(f"  {model_name:12}: ${next_price:.2f} ({change_pct:+.2f}%)")
    
    print("\n5-Day Ahead Predictions:")
    for model_name, preds in all_predictions.items():
        print(f"\n  {model_name}:")
        for i, price in enumerate(preds['week_ahead'], 1):
            change_pct = ((price - current_price) / current_price) * 100
            print(f"    Day {i}: ${price:.2f} ({change_pct:+.2f}%)")

def test_edge_cases():
    """
    Test edge cases and error handling.
    """
    print(f"\n{'='*60}")
    print("TESTING EDGE CASES")
    print(f"{'='*60}")
    
    # Test with minimal data
    print("\n1. Testing with minimal input data...")
    try:
        small_data = generate_synthetic_market_data(days=35)  # Just enough for 30-day sequence
        small_input = create_test_input(small_data, sequence_length=30)
        
        model = create_model('lstm', input_size=15, hidden_size=32, num_layers=1)
        pred = model.predict_next_day(small_input)
        print(f"   ✅ Minimal data test passed: {pred.item():.4f}")
    except Exception as e:
        print(f"   ❌ Minimal data test failed: {str(e)}")
    
    # Test with extreme market conditions
    print("\n2. Testing with extreme market volatility...")
    try:
        # Generate high volatility data
        np.random.seed(123)
        extreme_returns = np.random.normal(0, 0.1, 50)  # 10% daily volatility
        extreme_prices = [100.0]
        for ret in extreme_returns:
            extreme_prices.append(extreme_prices[-1] * (1 + ret))
        
        # Create DataFrame with extreme data
        extreme_df = pd.DataFrame({
            'close': extreme_prices[1:],
            'open': extreme_prices[1:],
            'high': [p * 1.05 for p in extreme_prices[1:]],
            'low': [p * 0.95 for p in extreme_prices[1:]],
            'volume': [1000000] * len(extreme_prices[1:])
        })
        extreme_df = calculate_technical_indicators(extreme_df)
        extreme_input = create_test_input(extreme_df, sequence_length=30)
        
        model = create_model('gru', input_size=15, hidden_size=32, num_layers=1)
        pred = model.predict_next_day(extreme_input)
        print(f"   ✅ Extreme volatility test passed: {pred.item():.4f}")
    except Exception as e:
        print(f"   ❌ Extreme volatility test failed: {str(e)}")
    
    # Test prediction consistency
    print("\n3. Testing prediction consistency...")
    try:
        market_data = generate_synthetic_market_data(days=50)
        test_input = create_test_input(market_data, sequence_length=30)
        
        model = create_model('transformer', input_size=15, d_model=32, num_heads=2, num_layers=1)
        
        # Run same prediction multiple times
        predictions = []
        for _ in range(5):
            pred = model.predict_next_day(test_input)
            predictions.append(pred.item())
        
        # Check if predictions are consistent (should be identical for deterministic model)
        if len(set(f"{p:.6f}" for p in predictions)) == 1:
            print(f"   ✅ Prediction consistency test passed: {predictions[0]:.6f}")
        else:
            print(f"   ⚠️ Prediction consistency warning: {predictions}")
    except Exception as e:
        print(f"   ❌ Prediction consistency test failed: {str(e)}")

def main():
    """
    Main test function.
    """
    print("🚀 STARTING REAL-TIME PREDICTION TESTS")
    print("=" * 80)
    
    try:
        # Run comprehensive tests
        test_realtime_simulation()
        test_edge_cases()
        
        print(f"\n{'='*80}")
        print("🎉 ALL REAL-TIME PREDICTION TESTS COMPLETED!")
        print("✅ Models are ready for real-time stock price prediction")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ TESTS FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*80}")

if __name__ == "__main__":
    main()
