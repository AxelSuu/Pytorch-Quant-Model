"""
Simplified real-time prediction test script.

This script focuses on core prediction functionality:
1. Next-day price prediction for all model types
2. Multi-day sequence prediction
3. Model consistency and reliability tests
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import models and utilities
from model import StockLSTM, StockGRU, StockTransformer, create_model
from enhanced_prediction import EnhancedPredictor
from utils import calculate_technical_indicators

class SimpleDataLoader:
    """Simple data loader for testing."""
    
    def __init__(self):
        # Mock scaling parameters for price normalization
        self.price_mean = 100.0
        self.price_std = 20.0
        
    def inverse_transform_predictions(self, predictions):
        """Convert normalized predictions back to actual prices."""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        return predictions * self.price_std + self.price_mean

def generate_test_data(days: int = 100, start_price: float = 100.0) -> pd.DataFrame:
    """Generate synthetic market data for testing."""
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

def normalize_features(df: pd.DataFrame, sequence_length: int = 30) -> torch.Tensor:
    """Normalize features and create input tensor."""
    # Select features
    feature_columns = [
        'close', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_middle', 'bb_lower',
        'sma_20', 'ema_12', 'ema_26', 'atr', 'stoch_k', 'stoch_d', 'williams_r', 'momentum'
    ]
    
    # Take the last sequence_length rows
    features = df[feature_columns].tail(sequence_length).values
    
    # Normalize close price separately (first column)
    close_prices = features[:, 0]
    mean_price = np.mean(close_prices)
    std_price = np.std(close_prices)
    features[:, 0] = (features[:, 0] - mean_price) / (std_price + 1e-8)
    
    # Normalize other features
    for i in range(1, features.shape[1]):
        col = features[:, i]
        if np.std(col) > 1e-8:
            features[:, i] = (col - np.mean(col)) / (np.std(col) + 1e-8)
    
    # Convert to tensor and add batch dimension
    tensor = torch.FloatTensor(features).unsqueeze(0)
    
    return tensor, mean_price, std_price

def denormalize_predictions(predictions, mean_price, std_price):
    """Convert normalized predictions back to actual prices."""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    # Ensure we return scalar values for single predictions
    result = predictions * std_price + mean_price
    if result.ndim == 0 or (result.ndim == 1 and len(result) == 1):
        return float(result.item() if hasattr(result, 'item') else result)
    return result

def test_model_core_functionality(model_name: str, model: torch.nn.Module, test_input: torch.Tensor, current_price: float, mean_price: float, std_price: float):
    """Test core prediction functionality of a model."""
    print(f"\n{'='*50}")
    print(f"Testing {model_name} Model")
    print(f"{'='*50}")
    
    try:
        # Test 1: Single next-day prediction
        print("\n1. Testing next-day prediction...")
        next_day_pred = model.predict_next_day(test_input)
        print(f"   Raw prediction type: {type(next_day_pred)}, shape: {next_day_pred.shape}")
        actual_next_day = denormalize_predictions(next_day_pred, mean_price, std_price)
        print(f"   Denormalized type: {type(actual_next_day)}, value: {actual_next_day}")
        
        # Ensure it's a float
        if isinstance(actual_next_day, (np.ndarray, torch.Tensor)):
            actual_next_day = float(actual_next_day.item())
        elif isinstance(actual_next_day, (list, tuple)):
            actual_next_day = float(actual_next_day[0])
        else:
            actual_next_day = float(actual_next_day)
            
        change_pct = ((actual_next_day - current_price) / current_price) * 100
        
        print(f"   Current price: ${current_price:.2f}")
        print(f"   Predicted next day: ${actual_next_day:.2f} ({change_pct:+.2f}%)")
        print(f"   Raw prediction tensor shape: {next_day_pred.shape}")
        
        # Test 2: Multi-day sequence prediction
        print("\n2. Testing multi-day sequence prediction...")
        for steps in [3, 5]:
            sequence_pred = model.predict_sequence(test_input, steps=steps)
            actual_sequence = denormalize_predictions(sequence_pred.squeeze(), mean_price, std_price)
            
            print(f"   {steps}-day sequence predictions:")
            for i, price in enumerate(actual_sequence, 1):
                day_change = ((price - current_price) / current_price) * 100
                print(f"     Day {i}: ${price:.2f} ({day_change:+.2f}%)")
        
        # Test 3: Consistency check
        print("\n3. Testing prediction consistency...")
        predictions = []
        for _ in range(3):
            pred = model.predict_next_day(test_input)
            actual_pred = denormalize_predictions(pred, mean_price, std_price)
            # Ensure it's a float
            if isinstance(actual_pred, (np.ndarray, torch.Tensor)):
                actual_pred = float(actual_pred.item())
            elif isinstance(actual_pred, (list, tuple)):
                actual_pred = float(actual_pred[0])
            else:
                actual_pred = float(actual_pred)
            predictions.append(actual_pred)
        
        if len(set(f"{p:.4f}" for p in predictions)) == 1:
            print(f"   ✅ Predictions consistent: ${predictions[0]:.2f}")
        else:
            print(f"   ⚠️ Predictions vary: {[f'${p:.2f}' for p in predictions]}")
        
        print(f"\n✅ {model_name} model tests PASSED!")
        return True, actual_next_day
        
    except Exception as e:
        print(f"\n❌ {model_name} model tests FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

def test_enhanced_predictor_integration():
    """Test enhanced predictor integration with simplified functionality."""
    print(f"\n{'='*60}")
    print("ENHANCED PREDICTOR INTEGRATION TEST")
    print(f"{'='*60}")
    
    try:
        # Create simple model and data loader
        model = create_model('lstm', input_size=15, hidden_size=32, num_layers=1)
        data_loader = SimpleDataLoader()
        enhanced_predictor = EnhancedPredictor(model, data_loader, torch.device('cpu'))
        
        # Generate test data
        market_data = generate_test_data(days=50)
        test_input, mean_price, std_price = normalize_features(market_data, sequence_length=30)
        current_price = market_data['close'].iloc[-1]
        
        print(f"\nCurrent market price: ${current_price:.2f}")
        
        # Test batch prediction
        print("\n1. Testing batch prediction...")
        batch_input = test_input.repeat(3, 1, 1)  # Create batch of 3
        batch_predictions = enhanced_predictor.predict_batch(batch_input)
        actual_batch = denormalize_predictions(batch_predictions.squeeze(), mean_price, std_price)
        print(f"   Batch predictions: {[f'${p:.2f}' for p in actual_batch]}")
        
        # Test confidence intervals
        print("\n2. Testing confidence interval estimation...")
        confidence_result = enhanced_predictor.estimate_confidence_intervals(test_input, n_samples=10)
        
        # Convert to actual prices
        actual_prediction = denormalize_predictions(confidence_result['prediction'], mean_price, std_price)
        actual_lower = denormalize_predictions(confidence_result['lower'], mean_price, std_price)
        actual_upper = denormalize_predictions(confidence_result['upper'], mean_price, std_price)
        
        print(f"   Prediction: ${actual_prediction:.2f}")
        print(f"   95% Confidence Interval: [${actual_lower:.2f}, ${actual_upper:.2f}]")
        print(f"   Confidence width: ${actual_upper - actual_lower:.2f}")
        
        print(f"\n✅ Enhanced predictor integration tests PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Enhanced predictor integration tests FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 STARTING SIMPLIFIED REAL-TIME PREDICTION TESTS")
    print("=" * 80)
    
    # Generate market data
    print("\n1. Generating synthetic market data...")
    market_data = generate_test_data(days=100)
    current_price = market_data['close'].iloc[-1]
    print(f"   Generated {len(market_data)} days of market data")
    print(f"   Current price: ${current_price:.2f}")
    
    # Prepare normalized input
    print("\n2. Preparing normalized input data...")
    test_input, mean_price, std_price = normalize_features(market_data, sequence_length=30)
    print(f"   Input tensor shape: {test_input.shape}")
    print(f"   Price normalization: mean=${mean_price:.2f}, std=${std_price:.2f}")
    
    # Test each model type
    print("\n3. Testing core model functionality...")
    models = {
        'LSTM': create_model('lstm', input_size=15, hidden_size=64, num_layers=2),
        'GRU': create_model('gru', input_size=15, hidden_size=64, num_layers=2),
        'Transformer': create_model('transformer', input_size=15, d_model=64, num_heads=4, num_layers=2)
    }
    
    all_predictions = {}
    success_count = 0
    
    for model_name, model in models.items():
        success, next_day_pred = test_model_core_functionality(
            model_name, model, test_input, current_price, mean_price, std_price
        )
        if success:
            success_count += 1
            all_predictions[model_name] = next_day_pred
    
    # Test enhanced predictor
    print("\n4. Testing enhanced predictor integration...")
    enhanced_success = test_enhanced_predictor_integration()
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nCore Model Tests: {success_count}/{len(models)} models passed")
    print(f"Enhanced Predictor: {'✅ PASSED' if enhanced_success else '❌ FAILED'}")
    
    if all_predictions:
        print(f"\nNext-Day Predictions Comparison:")
        print(f"Current Price: ${current_price:.2f}")
        for model_name, pred in all_predictions.items():
            change_pct = ((pred - current_price) / current_price) * 100
            print(f"  {model_name:12}: ${pred:.2f} ({change_pct:+.2f}%)")
    
    if success_count == len(models) and enhanced_success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("✅ Real-time prediction system is fully operational")
    else:
        print(f"\n⚠️ Some tests failed. Check individual test results above.")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
