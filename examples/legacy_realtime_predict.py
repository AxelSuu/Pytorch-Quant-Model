"""
Real-time stock price prediction script.

This script fetches the latest market data and makes predictions for next day and future prices.
"""

import torch
import argparse
import datetime
from evaluate import StockPredictor
from utils import set_seed, get_device


def display_market_status():
    """Display current market status."""
    now = datetime.datetime.now()
    print(f"🕒 Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Basic market hours check (simplified)
    weekday = now.weekday()
    hour = now.hour
    
    if weekday >= 5:  # Weekend
        print("📴 Market Status: CLOSED (Weekend)")
        next_monday = now + datetime.timedelta(days=(7-weekday))
        print(f"📅 Next trading day: {next_monday.strftime('%Y-%m-%d')}")
    elif 9 <= hour <= 16:  # Rough market hours
        print("🟢 Market Status: LIKELY OPEN")
        print("💡 Predictions will use the most recent available data")
    else:
        print("🔴 Market Status: CLOSED")
        print("💡 Predictions will use latest available closing data")


def main():
    """Main real-time prediction function."""
    parser = argparse.ArgumentParser(description='Real-time Stock Price Prediction')
    
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Stock symbol to predict (default: AAPL)')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--days', type=int, default=5,
                       help='Number of future days to predict (default: 5)')
    parser.add_argument('--sequence_length', type=int, default=60,
                       help='Sequence length for model input (default: 60)')
    parser.add_argument('--confidence', action='store_true',
                       help='Show confidence intervals for predictions')
    
    args = parser.parse_args()
    
    print("🚀 REAL-TIME STOCK PRICE PREDICTION")
    print("=" * 50)
    
    # Display market status
    display_market_status()
    print("=" * 50)
    
    # Set up predictor
    set_seed(42)
    device = get_device()
    print(f"🖥️  Using device: {device}")
    
    try:
        predictor = StockPredictor(args.model_path, device)
        
        print(f"📊 Loading LSTM model for {args.symbol} predictions...")
        print("=" * 50)
        
        # Next day prediction
        print("\n🎯 NEXT TRADING DAY PREDICTION")
        print("-" * 30)
        
        if args.confidence:
            next_price, confidence = predictor.predict_next_price(
                symbol=args.symbol,
                sequence_length=args.sequence_length,
                use_realtime=True,
                return_confidence=True
            )
            print(f"💰 Predicted price for {args.symbol}: ${next_price:.2f}")
            print(f"📊 95% Confidence Interval: ±${confidence:.2f}")
            print(f"   Range: ${next_price - confidence:.2f} - ${next_price + confidence:.2f}")
        else:
            next_price = predictor.predict_next_price(
                symbol=args.symbol,
                sequence_length=args.sequence_length,
                use_realtime=True
            )
            print(f"💰 Predicted price for {args.symbol}: ${next_price:.2f}")
        
        # Multi-day predictions
        print(f"\n📈 {args.days}-DAY PRICE FORECAST")
        print("-" * 30)
        
        if args.confidence:
            future_prices_with_ci = predictor.predict_sequence(
                symbol=args.symbol,
                steps=args.days,
                sequence_length=args.sequence_length,
                use_realtime=True,
                return_confidence=True
            )
            
            print("📅 Future price predictions with confidence intervals:")
            for i, (price, ci) in enumerate(future_prices_with_ci, 1):
                print(f"   Day {i}: ${price:.2f} (±${ci:.2f}) [{price-ci:.2f} - {price+ci:.2f}]")
            
            future_prices = [p[0] for p in future_prices_with_ci]
        else:
            future_prices = predictor.predict_sequence(
                symbol=args.symbol,
                steps=args.days,
                sequence_length=args.sequence_length,
                use_realtime=True
            )
            
            print("📅 Future price predictions:")
            for i, price in enumerate(future_prices, 1):
                print(f"   Day {i}: ${price:.2f}")
        
        # Calculate trend
        if len(future_prices) > 1:
            trend = "📈 BULLISH" if future_prices[-1] > future_prices[0] else "📉 BEARISH"
            change = future_prices[-1] - future_prices[0]
            change_pct = (change / future_prices[0]) * 100
            print(f"\n🎯 {args.days}-day trend: {trend}")
            print(f"💹 Expected change: ${change:+.2f} ({change_pct:+.2f}%)")
        
        print("\n" + "=" * 50)
        print("⚠️  DISCLAIMER: These are model predictions, not financial advice!")
        print("💡 Always do your own research before making investment decisions.")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
